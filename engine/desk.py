"""Adaptive dispatcher (SmartDesk) for Spread Levels 1-8 and the Live Final."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from spread.client import World
from spread.types import (
    Action,
    ActionResult,
    Audience,
    DeskEvent,
    DeskEventKind,
    DeskSnapshot,
    DriverProfile,
    FeeStep,
    Gig,
    Order,
    PostAction,
    PriceRequest,
    StopRef,
    TimeWindow,
    WorldMap,
)

from .network import NetworkIndex, site_day_hours
from .notes import ParsedNote, apply_note_to_window, fmt_min, parse_note

HOUR, DAY = 60, 24 * 60

# Wake only on events that can trigger an action so practice weeks run in ~1,200-1,600 HTTP calls.
WAKE_ON: list[DeskEventKind] = [
    "price_request",
    "order_won",
    "order_changed",
    "order_cancelled",
    "release",
    "visit_failed",
    "gig_expired",
    "gig_dropped",
    "no_show",
    "driver_late",
    "deadline_close",
    "forecast",
]


@dataclass
class LegState:
    manual_licence: bool = False
    needs_plates: bool = False
    summer_tyres: bool = False
    pickup_from: int = 0
    released: bool = False
    failed_count: int = 0


@dataclass
class QuoteMeta:
    request_id: str
    dealer_id: str
    pickup_place: str
    delivery_place: str
    declined: bool
    primary_mult: float
    comp_only_mult: float
    was_floor_bound: bool
    is_bulk: bool
    est_cost_cents: int
    at: int
    was_panic: bool = False


class SmartDesk:
    """State-rebuilding, supply-aware, competitor-learning dispatcher."""

    def __init__(
        self,
        world: World,
        level: int,
        now: int,
        world_map: WorldMap,
        roster: list[DriverProfile],
    ):
        self.world = world
        self.level = level
        self.now = now
        self.net = NetworkIndex(world_map, roster)

        self.outbox: list[Action] = []
        self.requests: dict[str, PriceRequest] = {}
        self.parsed_notes: dict[str, ParsedNote] = {}
        self.quote_meta: dict[str, QuoteMeta] = {}

        self.orders: dict[str, Order] = {}
        self.leg_states: dict[tuple[str, int], LegState] = {}
        self.leg_gigs: dict[str, dict[int, str]] = {}  # order_id -> leg_idx -> gig_id
        self.posts: dict[str, PostAction] = {}  # gig_id -> PostAction
        self.gig_posted_at: dict[str, int] = {}
        self.booked: set[str] = set()
        self.started: set[str] = set()
        self.picked_up_legs: set[tuple[str, int]] = set()
        self.doomed_late_orders: set[str] = set()
        self.gig_driver: dict[str, str] = {}
        self.gig_viewers: dict[str, set[str]] = {}
        self.urgent: set[str] = set()
        self.early_boosted: set[str] = set()
        self._last_sweep_min: int = -60

        # Driver activity & finishing telemetry
        self.active_day: int = -1
        self.active_drivers_today: set[str] = set()
        self.driver_offers_today: dict[str, int] = {}
        self.gig_open_offers: dict[str, int] = {}
        self.offered_pairs: set[tuple[str, str]] = set()
        self.finishing_places: dict[str, tuple[str, int]] = {}  # driver_id -> (place_id, est_finish_min)
        self.drops_today: int = 0
        self.bookings_today: int = 0
        self.price_war_day: Optional[int] = None

        # Calibrated against true normal-world roster (~2,400 drivers; L7/L8 shortage = 26%..40% -> 620..960 drivers)
        self.shortage_ratio = max(0.20, min(1.30, self.net.total_drivers / 2400.0))
        self.shortage_Mode = self.level >= 7 or self.shortage_ratio < 0.55

        if self.level <= 3:
            self.world_price_mult = 1.00
        elif self.level <= 5:
            self.world_price_mult = 1.01
        elif self.level == 6:
            self.world_price_mult = 1.04
        elif self.level == 7:
            self.world_price_mult = 1.27
        else:
            # Balanced L8 opening mult (1.37x): captures healthy Monday-Tuesday loyalty without dumping to 70% win-rate overload
            self.world_price_mult = 1.37
        self._day_start_world_mult = self.world_price_mult

        if self.level == 1:
            self.fee_mult_est = 1.01
        elif self.shortage_Mode:
            # 0.35 ratio (~840 drivers) -> 0.97; 0.26 ratio (~625 drivers) -> 1.07
            self.fee_mult_est = min(1.16, max(0.94, 0.97 + max(0.0, 0.35 - self.shortage_ratio) * 1.15))
        else:
            self.fee_mult_est = 0.94

        avg_sup = sum(self.net.supply_score.values()) / max(1, len(self.net.supply_score))
        self.avg_supply_score = max(1.0, avg_sup)

        self.day_price_adj: dict[int, float] = {}
        self.day_outcomes: dict[int, list[int]] = {}  # day -> recent [1/0]
        self.dest_price_adj: dict[str, float] = {}  # delivery_place -> learned lane residual
        self.place_outage_boost: dict[str, tuple[float, int]] = {}  # place_id -> (boost, day)
        self.region_recent_wins: dict[str, list[int]] = {}  # place_id -> recent [1/0] today
        self.dealer_wins: dict[str, list[int]] = {}  # dealer_id -> recent [1/0]
        self.dealer_trust: dict[str, int] = {}  # L8 trust count 0..25
        self.order_had_late_refund: set[str] = set()
        self.order_scored: set[str] = set()
        self.pending_quotes: deque[tuple[int, str]] = deque()  # (respond_by_min, pickup_place_id)
        self.batch_counts: dict[str, int] = {}  # batchId -> quoted count

        # SLA, Predictive Defense & Panic-Mode tracking (tier 1=Caution, 2=Pre-Dip Shield, 3=Panic <=92%)
        self.delivered_orders: int = 0
        self.on_time_orders: int = 0
        self.late_orders: int = 0
        self.panic_mode: bool = False
        self.sla_tier: int = 0

        # Human-readable brain feed for UI
        self.decisions: deque[dict[str, Any]] = deque(maxlen=150)

    def _log(self, category: str, title: str, detail: str, tone: str = "info") -> None:
        self.decisions.appendleft({
            "at": self.now,
            "time": fmt_min(self.now),
            "category": category,
            "title": title,
            "detail": detail,
            "tone": tone,
        })

    def _roll_day(self) -> None:
        d = self.now // DAY
        if d != self.active_day:
            self.active_day = d
            self._day_start_world_mult = self.world_price_mult
            self.active_drivers_today.clear()
            self.driver_offers_today.clear()
            self.region_recent_wins.clear()
            self.drops_today = 0
            self.bookings_today = 0

    def _note_of(self, req: PriceRequest) -> ParsedNote:
        rid = req["id"]
        if rid not in self.parsed_notes:
            p0 = req["legs"][0]["pickup"]["window"]["from"]
            d0 = max(o["deadline"] for o in req["options"])
            self.parsed_notes[rid] = parse_note(req.get("note"), req["arrivedAt"], p0, d0)
        return self.parsed_notes[rid]

    def _ensure_leg_states(self, order: Order) -> None:
        oid = order["id"]
        req = self.requests.get(oid)
        note = self._note_of(req) if req else ParsedNote()
        for idx in range(len(order["legs"])):
            key = (oid, idx)
            if key not in self.leg_states:
                self.leg_states[key] = LegState(
                    manual_licence=note.manual_gearbox,
                    needs_plates=note.needs_plates,
                    summer_tyres=note.summer_tyres,
                    pickup_from=note.pickup_after or 0,
                )

    # ---------- Rejoin / State Recovery ----------
    def rejoin(self, snapshot: DeskSnapshot, history: list[DeskEvent]) -> None:
        self.now = snapshot["now"]
        self._roll_day()
        for ev in history:
            k = ev["kind"]
            if k == "price_request":
                req = ev["request"]
                self.requests[req["id"]] = req
                self._note_of(req)
            elif k == "forecast":
                self.net.add_forecast(ev["from"], ev["to"], ev["placeIds"], ev["text"])
            elif k == "deadline_close":
                self.urgent.add(ev["orderId"])
            elif k == "picked_up":
                self.picked_up_legs.add((ev["orderId"], ev["leg"]))
            elif k == "gig_viewed":
                self.gig_viewers.setdefault(ev["gigId"], set()).add(ev["driverId"])
                if ev["at"] // DAY == self.active_day:
                    self.active_drivers_today.add(ev["driverId"])
            elif k == "ledger":
                self._ingest_ledger(ev["line"])

        self.booked = {g["id"] for g in snapshot["gigs"] if g["status"] in ("booked", "working", "done")}
        self.started = {g["id"] for g in snapshot["gigs"] if g["status"] in ("working", "done")}
        posted_by_order: dict[str, list[Gig]] = {}
        for g in snapshot["gigs"]:
            if g["stops"]:
                for st in g["stops"]:
                    posted_by_order.setdefault(st["orderId"], []).append(g)
            if g["status"] in ("open", "booked", "working"):
                act: PostAction = {
                    "type": "post",
                    "stops": g["stops"],
                    "audience": g["audience"],
                    "schedule": g["schedule"],
                }
                self.carry(act, g["id"], g["postedAt"])
                if g.get("driverId"):
                    self.gig_driver[g["id"]] = g["driverId"]

        self.orders = {
            o["id"]: o
            for o in snapshot["orders"]
            if o["id"] in self.requests and o["status"] != "cancelled"
        }
        for o in self.orders.values():
            self._ensure_leg_states(o)
            for idx, d_at in enumerate(o["deliveredAt"]):
                if d_at is not None:
                    self.picked_up_legs.add((o["id"], idx))

        for ev in history:
            if ev["kind"] == "visit_failed":
                oid, leg = ev["orderId"], ev["leg"]
                o = self.orders.get(oid)
                if not o or o["deliveredAt"][leg] is not None:
                    continue
                self._record_failure(oid, leg, ev["reason"], ev["at"])
                later = [
                    g for g in posted_by_order.get(oid, [])
                    if g["postedAt"] >= ev["at"] and g["id"] != ev["gigId"]
                ]
                if not any(s["orderId"] == oid and s["leg"] == leg for g in later for s in g["stops"]):
                    self.post_order_legs(o, [leg])

        for o in self.orders.values():
            missing = [
                i for i in self._undelivered(o)
                if i not in self.leg_gigs.get(o["id"], {})
            ]
            if missing and o["status"] == "waiting":
                self.post_order_legs(o, missing)

        for req in snapshot["openRequests"]:
            self.requests[req["id"]] = req
            self.quote(req)

    # ---------- Event Dispatch ----------
    def handle(self, event: DeskEvent) -> None:
        self.now = max(self.now, event["at"])
        self._roll_day()
        if self.now - self._last_sweep_min >= 15:
            self._last_sweep_min = self.now
            self._sweep_aging_orders()
        k = event["kind"]

        if k == "price_request":
            self.quote(event["request"])
        elif k == "order_won":
            self._on_win(event["order"])
        elif k == "order_lost":
            self._on_loss(event["requestId"])
        elif k == "order_changed":
            self._on_change(event["order"], event["message"]["text"])
        elif k == "order_cancelled":
            self._on_cancel(event["orderId"], event["gigs"])
        elif k == "release":
            self._on_payment_release(event["orderId"], event["leg"])
        elif k == "gig_viewed":
            did = event["driverId"]
            gid = event["gigId"]
            self.active_drivers_today.add(did)
            self.gig_viewers.setdefault(gid, set()).add(did)
        elif k == "gig_booked":
            self._on_gig_booked(event["gigId"], event["driverId"], event["feeCents"])
        elif k == "gig_started":
            self.started.add(event["gigId"])
        elif k in ("gig_dropped", "no_show"):
            self.drops_today += 1
            self._on_gig_lost_driver(event["gigId"], event["driverId"], k)
        elif k == "driver_late":
            self._on_driver_late(event["gigId"], event["driverId"], event["pickupAt"])
        elif k == "gig_expired":
            self._on_gig_expired(event["gigId"])
        elif k == "offer_answered":
            gid = event["gigId"]
            self.gig_open_offers[gid] = max(0, self.gig_open_offers.get(gid, 1) - 1)
            if event.get("status") != "accepted" and gid in self.posts and gid not in self.booked:
                act = self.posts[gid]
                first_oid = act["stops"][0]["orderId"]
                self._maybe_send_offers(gid, act, urgent=(first_oid in self.urgent))
        elif k == "picked_up":
            self._on_picked_up(event["orderId"], event["leg"], event["at"])
        elif k == "visit_failed":
            self._on_visit_failed(event["orderId"], event["leg"], event["reason"])
        elif k == "delivered":
            self._on_delivered(event["orderId"], event["leg"], event["at"])
            self._update_panic_mode()
        elif k == "deadline_close":
            self._on_deadline_close(event["orderId"])
        elif k == "forecast":
            self.net.add_forecast(event["from"], ev_to := event["to"], event["placeIds"], event["text"])
            places_str = ", ".join(self.net.place_name(p) for p in event["placeIds"][:4])
            self._log("weather", "Snow Forecast", f"{places_str} ({fmt_min(event['from'])}–{fmt_min(ev_to)})", "warn")
        elif k == "ledger":
            self._ingest_ledger(event["line"])
            self._update_panic_mode()

    def _ingest_ledger(self, line: dict[str, Any]) -> None:
        oid = line.get("orderId")
        if not oid:
            return
        if line["kind"] in ("late_refund", "undelivered_penalty"):
            self.order_had_late_refund.add(oid)
            self.doomed_late_orders.add(oid)
            if oid in self.order_scored:
                self.on_time_orders = max(0, self.on_time_orders - 1)
                self.late_orders += 1
                req = self.requests.get(oid)
                if req:
                    did = req["dealerId"]
                    self.dealer_trust[did] = max(0, self.dealer_trust.get(did, 0) - 1) // 2
        elif line["kind"] == "revenue" and oid not in self.order_scored:
            self.order_scored.add(oid)
            self.delivered_orders += 1
            req = self.requests.get(oid)
            did = req["dealerId"] if req else ""
            if oid in self.order_had_late_refund or oid in self.doomed_late_orders:
                self.late_orders += 1
                if did:
                    self.dealer_trust[did] = self.dealer_trust.get(did, 0) // 2
            else:
                self.on_time_orders += 1
                if did:
                    self.dealer_trust[did] = min(25, self.dealer_trust.get(did, 0) + 1)

    # ---------- Cost & Feasibility Modeling ----------
    def _open_order_counts(self, near_place: Optional[str] = None) -> tuple[int, int]:
        """Returns (unbooked_waiting_cnt, total_open_waiting_cnt) including expected in-flight quote wins."""
        while self.pending_quotes and self.pending_quotes[0][0] <= self.now:
            self.pending_quotes.popleft()

        unbooked = 0
        total_open = 0
        for oid, o in self.orders.items():
            if o["status"] != "waiting":
                continue
            if near_place is not None:
                p0 = o["legs"][0]["pickup"]["placeId"]
                if self.net.straight_km(p0, near_place) > 40.0:
                    continue
            total_open += 1
            if not any(g in self.booked for g in self.gigs_of(oid)):
                unbooked += 1

        exp_win_w = 1.0 if self.level <= 3 else 0.42
        inflight = 0.0
        for _, p_q in self.pending_quotes:
            if near_place is None or self.net.straight_km(p_q, near_place) <= 40.0:
                inflight += exp_win_w
        add = int(inflight)
        return (unbooked + add, total_open + add)

    def _active_unbooked_orders(self, near_place: Optional[str] = None) -> int:
        return self._open_order_counts(near_place)[0]

    def _sla_snapshot(self) -> tuple[int, float, float, float]:
        """Returns (effective_total, headroom_orders, actual_on_time_share, projected_on_time_share)."""
        doomed_waiting = 0
        at_risk_waiting = 0
        unbooked_safe = 0
        booked_safe = 0

        for oid, o in self.orders.items():
            if o["status"] != "waiting" or oid in self.order_scored:
                continue
            gids = self.gigs_of(oid)
            is_booked = any(g in self.booked for g in gids)
            picked_first = (oid, 0) in self.picked_up_legs

            if (
                oid in self.doomed_late_orders
                or oid in self.order_had_late_refund
                or (not picked_first and self.now > o["lastPickupAt"])
                or any(d is not None and d > o["deadline"] for d in o["deliveredAt"])
            ):
                self.doomed_late_orders.add(oid)
                doomed_waiting += 1
            elif not is_booked and (self.now >= o["lastPickupAt"] - 180 or oid in self.urgent):
                at_risk_waiting += 1
            elif not is_booked:
                unbooked_safe += 1
            else:
                booked_safe += 1

        total_waiting = doomed_waiting + at_risk_waiting + unbooked_safe + booked_safe
        effective_total = self.delivered_orders + total_waiting
        actual_ot = (self.on_time_orders / self.delivered_orders) if self.delivered_orders > 0 else 1.0
        if effective_total == 0:
            return (0, 3.0, actual_ot, 1.0)

        # Predictive failure weighting: unbooked shortage backlog, realized late velocity, flu drops, and Thu/Fri crunch
        day = self.now // DAY
        tod = self.now % DAY
        obs_late_rate = (self.late_orders / self.delivered_orders) if self.delivered_orders >= 20 else 0.0
        unbooked_risk = (0.048 if self.shortage_Mode else 0.015) + 0.35 * obs_late_rate
        booked_risk = (0.018 if self.shortage_Mode else 0.008) + 0.15 * obs_late_rate

        if self.drops_today >= 6 or (day >= 3 and self.drops_today >= 4):
            unbooked_risk += 0.025
            booked_risk += 0.012
        if self.shortage_Mode and total_waiting >= max(65, int(self.net.total_drivers * 0.11)):
            unbooked_risk += 0.025
        if self.shortage_Mode and ((day == 3 and tod >= 12 * HOUR) or day == 4):
            unbooked_risk += 0.030

        effective_late = (
            self.late_orders
            + doomed_waiting
            + 0.55 * at_risk_waiting
            + min(0.22, unbooked_risk) * unbooked_safe
            + min(0.10, booked_risk) * booked_safe
        )
        headroom = (0.082 * effective_total) - effective_late if effective_total >= 10 else 3.0
        proj_ot = max(0.0, 1.0 - (effective_late / effective_total))
        return (effective_total, headroom, actual_ot, proj_ot)

    def _sla_headroom(self) -> float:
        """Positive = safe orders above 91.8% target buffer (ranking floor is 90.0%); avoids phantom-late throttling on fresh orders."""
        _, headroom, _, _ = self._sla_snapshot()
        return headroom

    def _update_panic_mode(self) -> bool:
        """Updates predictive SLA tier (0..2) and latches Tier 3 Panic Mode at <=92.0% OT."""
        effective_total, _, actual_ot, proj_ot = self._sla_snapshot()
        if effective_total < 12:
            self.sla_tier = 0
            return self.panic_mode

        if not self.panic_mode:
            if (self.delivered_orders >= 12 and actual_ot <= 0.920) or (effective_total >= 15 and proj_ot <= 0.920):
                self.panic_mode = True
                self.sla_tier = 3
                self._log(
                    "urgent",
                    f"PANIC MODE ACTIVATED ({actual_ot*100:.1f}% deliv / {proj_ot*100:.1f}% proj OT)",
                    "Switching to <=92% OT rescue: banning risky options, fast-clearing waiting gigs, padding safe 3d volume",
                    "warn",
                )
                self._activate_panic_rescue()
                return True
        else:
            if proj_ot >= 0.935 and (self.delivered_orders < 12 or actual_ot >= 0.935):
                self.panic_mode = False
                self._log(
                    "urgent",
                    f"Exited Panic Mode ({actual_ot*100:.1f}% deliv / {proj_ot*100:.1f}% proj OT)",
                    "On-time cushion recovered above 93.5%; resuming predictive defense control",
                    "good",
                )
            else:
                self.sla_tier = 3
                return True

        # Predictive first lines of defense (BEFORE OT dips to 92%)
        day = self.now // DAY
        tod = self.now % DAY
        _, global_total = self._open_order_counts()
        eow = self.shortage_Mode and (day == 4 or (day == 3 and tod >= 12 * HOUR))
        open_cap = max(24, int(self.net.total_drivers * (0.145 if eow else 0.19))) if self.shortage_Mode else max(36, int(self.net.total_drivers * 0.30))

        prev_tier = self.sla_tier
        new_tier = 0
        if effective_total >= 25 and (proj_ot <= 0.930 or (day == 4 and proj_ot <= 0.936)):
            new_tier = 2
        elif effective_total >= 20 and (proj_ot <= 0.943 or (day >= 3 and global_total >= int(open_cap * 0.80)) or self.drops_today >= 6):
            new_tier = 1

        self.sla_tier = new_tier
        if new_tier == 2 and prev_tier < 2:
            self._log(
                "urgent",
                f"PRE-DIP SHIELD ({actual_ot*100:.1f}% deliv / {proj_ot*100:.1f}% proj OT)",
                f"Projected dip incoming (backlog {global_total}/{open_cap}); halting same-day/risky next-day & accelerating open pickups",
                "warn",
            )
            self._activate_panic_rescue()
        return False

    def _split_or_reprice_gig(
        self,
        order: Order,
        gid: str,
        *,
        force_urgent: bool = True,
        schedule: Optional[list[FeeStep]] = None,
    ) -> None:
        """Withdraws any multi-order chain into solo fast-clearing legs, or reprices a solo gig and fires direct offers."""
        act = self.posts.get(gid)
        if not act or gid in self.booked:
            return
        order_set = list(dict.fromkeys(s["orderId"] for s in act["stops"]))
        if len(order_set) > 1:
            self.outbox.append({"type": "withdraw", "gigId": gid})
            self.posts.pop(gid, None)
            for s in act["stops"]:
                self.leg_gigs.get(s["orderId"], {}).pop(s["leg"], None)
            for oid2 in order_set:
                o2 = self.orders.get(oid2)
                if o2 and o2["status"] != "cancelled":
                    self.post_order_legs(
                        o2,
                        self._undelivered(o2),
                        allow_windows=True,
                        allow_delivery_window=False,
                        allow_chain=False,
                    )
            return

        cars = sum(1 for s in act["stops"] if s["kind"] == "pickup")
        new_sched = schedule if schedule is not None else self._schedule_for(order, cars, force_urgent=force_urgent)
        self.outbox.append({"type": "reprice", "gigId": gid, "schedule": new_sched})
        act["schedule"] = new_sched
        self._maybe_send_offers(gid, act, urgent=True)

    def _activate_panic_rescue(self) -> None:
        """On entry into Panic Mode, split open chains and boost non-late unbooked gigs so waiting cars clear on time."""
        for oid, o in list(self.orders.items()):
            if o["status"] != "waiting" or oid in self.doomed_late_orders or oid in self.order_had_late_refund:
                continue
            for gid in list(self.gigs_of(oid)):
                if gid in self.booked:
                    continue
                is_urg_now = (o["lastPickupAt"] - self.now) <= 360
                self._split_or_reprice_gig(o, gid, force_urgent=is_urg_now)

    def _estimate_fulfillment_cost(self, req: PriceRequest, note: ParsedNote) -> tuple[int, Audience, float, float]:
        """Returns (expected_driver_fee_cents, required_audience, supply_quality, dest_ratio)."""
        legs = req["legs"]
        experienced = any(l["car"]["premium"] for l in legs)
        manual = note.manual_gearbox
        audience: Audience = {"experienced": experienced, "manualLicence": manual}

        p_first = legs[0]["pickup"]["placeId"]
        d_last = legs[-1]["delivery"]["placeId"]

        local_notif = self.net.matching_notif_count(p_first, audience, region=False)
        region_pool = self.net.matching_notif_count(p_first, audience, region=True)
        dest_notif = self.net.matching_notif_count(d_last, audience, region=False)

        sup_pick = self.net.supply_score.get(p_first, 0.5) / self.avg_supply_score
        sup_dest = self.net.supply_score.get(d_last, 0.5) / self.avg_supply_score
        supply_quality = 0.55 * sup_pick + 0.45 * sup_dest

        mult = self.fee_mult_est
        # Town-level scarcity vs hub-destination discount (homebound drivers at delivery hub need no return ticket!)
        if supply_quality < 0.85:
            mult *= min(1.42, (0.88 / max(0.12, supply_quality)) ** 0.25)
        elif supply_quality > 1.20:
            mult *= max(0.86, (1.18 / supply_quality) ** 0.16)

        total_road_min = sum(self.net.minutes(l["pickup"]["placeId"], l["delivery"]["placeId"]) for l in legs)
        if self.shortage_Mode and total_road_min > 240:
            # Mild long-haul shortage adjustment without compounding into extreme opening fees
            mult *= 1.0 + min(0.20, ((total_road_min - 240) / 320.0) * 0.18)

        if len(legs) == 1:
            if dest_notif >= 12 and local_notif >= 2 and total_road_min <= 220:
                # Car heads straight into a driver hub -> homebound drivers take this cheaply!
                mult *= 0.92
            elif dest_notif == 0:
                mult *= 1.18
            elif dest_notif <= 2:
                # Dead-end delivery town costs driver an expensive day-of return ticket
                mult *= 1.12
            elif dest_notif <= 5:
                mult *= 1.04

        if local_notif == 0:
            mult *= 1.18 if dest_notif >= 10 else 1.26
        elif local_notif <= 2:
            mult *= 1.08

        # Local unbooked congestion surcharge
        local_waiting = self._active_unbooked_orders(p_first)
        if local_waiting > 2:
            mult *= 1.0 + min(0.12, ((local_waiting - 2) / max(2.5, local_notif * 0.65)) * 0.08)

        # Flu / drop shock surcharge
        if self.drops_today >= 6:
            mult *= 1.0 + min(0.10, (self.drops_today - 4) * 0.014)

        if experienced:
            mult *= 1.07
        if manual:
            mult *= 1.05
        if any(l["pickup"]["privateAddress"] or l["delivery"]["privateAddress"] for l in legs):
            mult *= 1.07

        extra_cents = 0
        if note.needs_plates:
            extra_cents += 1800
            mult *= 1.03

        for l in legs:
            km = self.net.km(l["pickup"]["placeId"], l["delivery"]["placeId"])
            car = l["car"]
            if car["electric"] and car["chargePercent"] is not None and car["chargePercent"] < 30 and km > 195:
                extra_cents += 2200
                mult *= 1.04
            if self.net.has_snow(l["pickup"]["placeId"], req["arrivedAt"], req["arrivedAt"] + 2 * DAY) or self.net.has_snow(
                l["delivery"]["placeId"], req["arrivedAt"], req["arrivedAt"] + 2 * DAY
            ):
                mult *= 1.10

        # Backhaul / finishing driver / chainable corridor synergy!
        has_finisher = any(
            self.net.straight_km(fin_place, p_first) <= 35.0 and self.now - 30 <= fin_at <= req["arrivedAt"] + 340
            for fin_place, fin_at in self.finishing_places.values()
        )
        has_pair_order = any(
            o["status"] == "waiting"
            and len(o["legs"]) == 1
            and (
                self.net.minutes(o["legs"][0]["delivery"]["placeId"], p_first) <= 35
                or self.net.minutes(d_last, o["legs"][0]["pickup"]["placeId"]) <= 35
            )
            for o in self.orders.values()
        )
        if has_finisher:
            mult *= 0.91
        elif has_pair_order:
            mult *= 0.95

        if len(legs) == 1:
            min_leg_mult = 0.80 if (dest_notif >= 12 and total_road_min <= 180) else 0.84
            mult = max(mult, min_leg_mult)

        est_cents = int(round(req["referenceFeeCents"] * mult + extra_cents))
        if region_pool == 0:
            supply_quality = 0.02
        return max(1200, est_cents), audience, supply_quality, sup_dest

    def _option_slack_hours(self, req: PriceRequest, opt: dict[str, Any], note: ParsedNote) -> float:
        legs = req["legs"]
        deadline = opt["deadline"]
        if note.delivery_before is not None:
            deadline = min(deadline, note.delivery_before)

        ready = max(legs[0]["pickup"]["window"]["from"], req["respondBy"], note.pickup_after or 0)
        if note.summer_tyres:
            for pid in (legs[0]["pickup"]["placeId"], legs[-1]["delivery"]["placeId"]):
                se = self.net.snow_end_after(pid, ready)
                if se is not None:
                    ready = max(ready, se)

        drive = 0
        for idx, l in enumerate(legs):
            u, v = l["pickup"]["placeId"], l["delivery"]["placeId"]
            leg_m = self.net.minutes(u, v)
            if self.net.has_snow(u, ready, deadline) or self.net.has_snow(v, ready, deadline):
                leg_m = int(leg_m * 1.28)
            drive += leg_m
            if note.needs_plates:
                drive += 30
            if l["car"]["electric"] and l["car"]["chargePercent"] is not None and l["car"]["chargePercent"] < 30 and self.net.km(u, v) > 195:
                drive += 40
            if idx + 1 < len(legs):
                drive += self.net.minutes(v, legs[idx + 1]["pickup"]["placeId"]) + 15

        last_del = legs[-1]["delivery"]
        eff_due = deadline
        if not last_del["privateAddress"]:
            st_dh = site_day_hours(last_del.get("hours"), deadline // DAY)
            if note.delivery_hours:
                dh: Optional[tuple[int, int]] = (
                    note.delivery_hours[0] if note.delivery_hours[0] > 0 else (st_dh[0] if st_dh else 8 * HOUR),
                    note.delivery_hours[1],
                )
                if (deadline // DAY) % 7 >= 5 and st_dh is None:
                    dh = None
            else:
                dh = st_dh
            if dh is None:
                for back in (1, 2):
                    prev_d = (deadline // DAY) - back
                    prev_h = site_day_hours(last_del.get("hours"), prev_d)
                    if prev_h is not None:
                        eff_close = min(prev_h[1], note.delivery_hours[1]) if note.delivery_hours else prev_h[1]
                        eff_due = min(eff_due, prev_d * DAY + eff_close)
                        break
            else:
                eff_due = min(eff_due, (deadline // DAY) * DAY + dh[1])

        first_pick = legs[0]["pickup"]
        latest_pick = eff_due - drive
        if first_pick["privateAddress"]:
            latest_pick = min(latest_pick, first_pick["window"]["to"])
        elif note.pickup_before is not None:
            pb = note.pickup_before
            if pb < ready and (pb % DAY) > (ready % DAY):
                pb = (ready // DAY) * DAY + (pb % DAY)
            latest_pick = min(latest_pick, pb)

        if latest_pick <= ready:
            return (latest_pick - ready) / 60.0

        open_mins = 0
        for d_idx in range(ready // DAY, (latest_pick // DAY) + 1):
            if first_pick["privateAddress"]:
                sh: Optional[tuple[int, int]] = (7 * HOUR + 30, 20 * HOUR + 30)
            else:
                st_ph = site_day_hours(first_pick.get("hours"), d_idx)
                if st_ph is None:
                    continue
                if note.pickup_hours and (d_idx % 7) < 5:
                    sh = (
                        note.pickup_hours[0] if note.pickup_hours[0] > 0 else st_ph[0],
                        note.pickup_hours[1],
                    )
                else:
                    sh = st_ph
            seg_s = max(ready, d_idx * DAY + max(7 * HOUR + 30, sh[0]))
            seg_e = min(latest_pick, d_idx * DAY + min(21 * HOUR, sh[1]))
            if seg_e > seg_s:
                open_mins += seg_e - seg_s
        open_h = open_mins / 60.0
        min_slot_mins = 90 if self.shortage_Mode else 75
        if open_mins < min_slot_mins:
            return open_h
        # Pre-booking notice helps drivers plan, but cap it relative to open_h so a 1.5h slot never mimics 8h slack
        advance_notice_h = max(0.0, (ready - req["respondBy"]) / 60.0)
        return open_h + min(open_h * 0.60, 0.50 * min(8.0, advance_notice_h))

    # ---------- Quoting Strategy ----------
    def _is_price_war_today(self, day: int) -> bool:
        if self.level < 8:
            return False
        tod = self.now % DAY
        if not (8 * HOUR <= tod <= 18 * HOUR + 15):
            return False
        if self.price_war_day == day:
            return True
        hist = self.day_outcomes.get(day, [])
        if (
            (len(hist) >= 5 and sum(hist[-5:]) == 0)
            or (len(hist) >= 9 and sum(hist[-9:]) <= 1)
            or (len(hist) >= 14 and (sum(hist) / len(hist)) <= 0.16)
        ):
            self.price_war_day = day
            self.world_price_mult = max(self.world_price_mult, self._day_start_world_mult)
            war_floor_adj = 1.26 - self.world_price_mult
            self.day_price_adj[day] = max(war_floor_adj, min(self.day_price_adj.get(day, 0.0), -0.18))
            return True
        return False

    def quote(self, req: PriceRequest) -> None:
        rid = req["id"]
        self.requests[rid] = req
        note = self._note_of(req)
        est_cost, audience, sup_q, dest_ratio = self._estimate_fulfillment_cost(req, note)

        p_first = req["legs"][0]["pickup"]["placeId"]
        d_last = req["legs"][-1]["delivery"]["placeId"]
        local_notif = self.net.matching_notif_count(p_first, audience, region=False)
        region_notif = self.net.matching_notif_count(p_first, audience, region=True)
        dest_notif = self.net.matching_notif_count(d_last, audience, region=False)
        total_road_min = sum(self.net.minutes(l["pickup"]["placeId"], l["delivery"]["placeId"]) for l in req["legs"])

        # Capacity & Bulk-Order Throttling (tracks BOTH unbooked and already-booked active backlog!)
        day = self.now // DAY
        tod = self.now % DAY
        global_waiting, global_total = self._open_order_counts()
        local_waiting, local_total = self._open_order_counts(p_first)
        flu_shock = self.drops_today >= 6
        eow_crunch = self.shortage_Mode and (day == 4 or (day == 3 and tod >= 12 * HOUR))
        if self.level == 1:
            global_cap = min(52, max(16, int(self.net.total_drivers * 0.25)))
            local_cap = max(2, int(max(local_notif, region_notif * 0.25) * 0.42))
            global_total_cap = global_cap + 12
            local_total_cap = local_cap + 4
        else:
            early_ok = (day <= 2 and not flu_shock)
            cap_factor = (0.18 if (flu_shock or eow_crunch) else (0.25 if early_ok else 0.22)) if self.shortage_Mode else 0.42
            loc_factor = (0.42 if (flu_shock or eow_crunch) else (0.56 if early_ok else 0.52)) if self.shortage_Mode else 0.72
            open_factor = (0.125 if flu_shock else (0.145 if eow_crunch else (0.22 if early_ok else 0.19))) if self.shortage_Mode else 0.30
            global_cap = max(16, int(self.net.total_drivers * cap_factor))
            local_cap = max(4, int(max(local_notif, region_notif * 0.35) * loc_factor))
            global_total_cap = max(24, int(self.net.total_drivers * open_factor))
            local_total_cap = max(6, int(max(local_notif, region_notif * 0.35) * (0.68 if (eow_crunch or flu_shock) else (0.94 if early_ok else 0.88))))

        overloaded = (
            global_waiting >= global_cap
            or local_waiting >= local_cap
            or global_total >= global_total_cap
            or local_total >= local_total_cap
        )
        in_panic = self._update_panic_mode()
        sla_tier = self.sla_tier
        sla_caution = sla_tier >= 1
        sla_shield = sla_tier >= 2
        sla_head = self._sla_headroom()
        sla_tight = sla_head < 1.0 or sla_caution
        sla_critical = sla_head < -0.4 or sla_shield

        # Levels 1-3: Every quote wins first option at listPriceCents; no goodwill penalty for not quoting!
        if self.level <= 3:
            first_opt = req["options"][0]
            slack = self._option_slack_hours(req, first_opt, note)
            min_slack = 6.5 if in_panic else ((3.2 if self.level == 1 else 2.2) if not sla_tight else 4.5)
            eff_l_cost = max(est_cost, int(req["referenceFeeCents"] * 0.96)) if self.level == 1 else est_cost
            min_margin = 650 if self.level == 1 else (500 if self.level == 2 else 350)
            if (
                slack >= min_slack
                and region_notif >= (2 if self.level == 1 else 1)
                and first_opt["listPriceCents"] >= eff_l_cost + min_margin
                and not overloaded
                and (not sla_critical or (in_panic and slack >= 8.0 and local_notif >= 4))
            ):
                price = int(first_opt["listPriceCents"])
                self.pending_quotes.append((req["respondBy"], p_first))
                self.outbox.append({"type": "quote", "requestId": rid, "prices": {first_opt["kind"]: price}})
                self._log(
                    "quote",
                    f"Accepted {rid} ({self.net.place_name(p_first)} → {self.net.place_name(d_last)})",
                    f"L{self.level} list €{price/100:.0f} vs est. fee €{est_cost/100:.0f} (+€{(price-est_cost)/100:.0f} exp)",
                    "good",
                )
            else:
                self._log(
                    "decline",
                    f"Skipped {rid} ({self.net.place_name(p_first)} → {self.net.place_name(d_last)})",
                    f"Unprofitable/risky at L{self.level} (list €{first_opt['listPriceCents']/100:.0f} vs est €{est_cost/100:.0f}, slack {slack:.1f}h)",
                    "mute",
                )
            return

        # Levels 4-8: Competitor-aware, lane-difficulty-aware, loyalty-building pricing
        in_price_war = self._is_price_war_today(day)
        day_adj = max(1.26 - self.world_price_mult, self.day_price_adj.get(day, 0.0)) if in_price_war else self.day_price_adj.get(day, 0.0)
        outage_boost = 0.0
        ob = self.place_outage_boost.get(p_first)
        if ob and ob[1] == day and 8 * HOUR <= (self.now % DAY) <= 18 * HOUR + 15 and not in_price_war:
            outage_boost = ob[0]

        dealer_id = req["dealerId"]
        trust_count = self.dealer_trust.get(dealer_id, 0)
        # RULES.md line 164: +1% accepted price per on-time order at Level 8 (up to +25%)
        trust_bonus = (trust_count * 0.0095) if self.level >= 8 else 0.0
        # Mild early loyalty investment ONLY on Monday or early Tuesday for new dealers (< 3 orders)
        loyalty_invest = 0.0
        if self.level >= 8 and not in_price_war and ((day == 0 and trust_count < 4) or (day == 1 and trust_count < 3)):
            loyalty_invest = -0.012 if day == 0 else -0.008

        d_hist = self.dealer_wins.get(dealer_id, [])
        dealer_win_rate = (sum(d_hist) / len(d_hist)) if len(d_hist) >= 4 else 0.45
        anti_war_pad = 0.045 if dealer_win_rate > 0.68 else (-0.012 if len(d_hist) >= 6 and dealer_win_rate < 0.18 else 0.0)

        # Empirical competitor lane factor: preserve full 1.00× floor on easy routes, charge steeper on long/thin lanes
        ref_to_list = req["referenceFeeCents"] / max(1, req["options"][0]["listPriceCents"])
        if ref_to_list > 0.94:
            # Expensive / long-haul lane: competitor charges a steeper lane factor here
            comp_lane_mult = min(1.28, 1.00 + (ref_to_list - 0.94) * 0.50)
        else:
            comp_lane_mult = 1.00
        if dest_ratio < 0.55:
            comp_lane_mult *= 1.06

        has_finisher = any(
            self.net.straight_km(fin_place, p_first) <= 35.0 and self.now - 30 <= fin_at <= req["arrivedAt"] + 340
            for fin_place, fin_at in self.finishing_places.values()
        )
        has_active_homebound = any(
            d["id"] in self.active_drivers_today
            for d in self.net.notif_drivers.get(d_last, [])
            if (not audience["experienced"] or (d["rating"] >= 4.8 and d["completedGigs"] >= 50))
            and (not audience["manualLicence"] or d["licence"] == "manual")
        )

        # Out-of-the-way / Remote lane detection ONLY for genuine thin/remote towns (e.g. Rovaniemi, Kittilä, Kemi, Kajaani):
        is_remote_lane = (dest_notif <= 3 or local_notif <= 2 or dest_ratio < 0.45)
        remote_loyalty_mode = is_remote_lane and (trust_count < 12 or has_finisher or has_active_homebound)

        # Only block when physically unfulfillable (never blanket-ban long 3-day or short hub hops during price war!)
        toxic_lane = False
        if self.shortage_Mode and not has_finisher and not has_active_homebound:
            if (local_notif == 0 and region_notif < 3) or (in_price_war and local_notif == 0):
                toxic_lane = True

        # Bulk-order (`batchId`) progressive surge + shared EOW/caution yard throttle across bursts
        batch_surge = 0.0
        batch_cap_exceeded = False
        bid = req.get("batchId")
        if bid:
            b_idx = self.batch_counts.get(bid, 0)
            self.batch_counts[bid] = b_idx + 1
            step_rate = 0.006 if local_notif >= 80 else (0.012 if local_notif >= 15 else 0.022)
            batch_surge = min(0.55, b_idx * step_rate)
            batch_ratio = 0.40 if (eow_crunch or sla_caution) else ((0.72 if (day <= 2 and not flu_shock) else 0.62) if self.shortage_Mode else 0.94)
            max_batch_take = max(5, int(max(local_notif, region_notif * 0.25) * batch_ratio))
            if b_idx >= max_batch_take or ((eow_crunch or sla_caution) and not in_panic and local_total >= int(local_total_cap * 0.85)):
                batch_cap_exceeded = True

        dest_res = self.dest_price_adj.get(d_last, 0.0)
        remote_comp_shade = -0.015 if (remote_loyalty_mode and trust_count < 8 and not in_price_war) else 0.0
        # In Panic Mode (<= 92% OT): never block safe 3-day capacity padding behind high batch surge, keep yard cap active
        eff_batch_surge = min(0.08, batch_surge) if in_panic else batch_surge
        raw_comp_mult = max(
            0.88,
            (self.world_price_mult + day_adj + outage_boost + dest_res + anti_war_pad + loyalty_invest + remote_comp_shade)
            * comp_lane_mult,
        )
        base_comp_mult = raw_comp_mult * (1.0 + trust_bonus) * (1.0 + eff_batch_surge)

        has_multi_opts = len(req["options"]) > 1

        prices: dict[str, int] = {}
        floor_bound_by_opt: dict[str, bool] = {}
        best_summary = ""
        if not batch_cap_exceeded and not toxic_lane:
            for opt in req["options"]:
                kind = opt["kind"]
                list_c = opt["listPriceCents"]
                slack = self._option_slack_hours(req, opt, note)

                if sla_shield or in_panic:
                    # Tier 2 Pre-Dip Shield & Tier 3 Panic: ban same_day; shift multi-opt to within_3_days; restrict solo next_day
                    if kind == "same_day":
                        continue
                    if kind == "next_day":
                        if has_multi_opts:
                            continue
                        min_loc_req = 5 if in_panic else 4
                        max_nd_road = 210 if in_panic else 220
                        if (
                            (local_notif < min_loc_req and not has_finisher)
                            or (dest_notif < 3 and not has_active_homebound)
                            or total_road_min > max_nd_road
                            or note.has_constraints
                        ):
                            continue
                    elif kind == "within_3_days":
                        if (local_notif < 2 and not has_finisher) or region_notif < 4 or total_road_min > 420:
                            continue
                elif sla_caution:
                    # Tier 1 Early Caution: cut same_day risk on multi-opts & trim long next_day into within_3_days before dip hits
                    if kind == "same_day":
                        if has_multi_opts or total_road_min > 155 or local_notif < 5 or note.has_constraints or slack < 5.5:
                            continue
                    elif kind == "next_day" and has_multi_opts and total_road_min > 290 and not has_finisher:
                        continue

                req_slack = 2.5 if not self.shortage_Mode else 3.4
                if kind == "same_day":
                    if self.shortage_Mode:
                        max_loc_wait = max(1, int(local_notif * 0.14))
                        flu_block = (flu_shock or self.drops_today >= 4) and (total_road_min > 120 or local_notif < 6 or slack < 5.0)
                        if has_multi_opts and (
                            (local_notif < 3 and not has_finisher)
                            or (local_waiting > max_loc_wait and not has_finisher)
                            or slack < 4.9
                            or total_road_min > 180
                            or sla_head < 1.0
                            or flu_block
                        ):
                            continue
                        if not has_multi_opts and (
                            (local_notif < 2 and not has_finisher)
                            or slack < 4.7
                            or total_road_min > 205
                            or sla_tight
                            or flu_block
                        ):
                            continue
                    req_slack = 4.3 if self.shortage_Mode else 2.8
                elif kind == "next_day" and self.shortage_Mode and (
                    (total_road_min > 380 and has_multi_opts and not has_finisher and not has_active_homebound)
                    or (total_road_min > 460 and ((local_notif <= 2 and not has_finisher) or day == 4 or (not has_active_homebound and not has_finisher and slack < 12.0)))
                ):
                    # Block extreme 7.5h+ next_day when origin is thin (e.g. Rovaniemi -> Vantaa) or on Friday even with hub homebound!
                    continue
                if total_road_min > 240:
                    req_slack += (total_road_min - 240) / 70.0
                if local_notif == 0 and not has_finisher:
                    if self.shortage_Mode and slack < 9.0:
                        continue
                    req_slack += 2.6
                elif local_notif <= 2 and not has_finisher:
                    req_slack += 1.0
                if dest_notif <= 2 and self.shortage_Mode and not has_active_homebound:
                    req_slack += 0.9
                if in_panic:
                    req_slack = max(req_slack, 7.5 if kind == "next_day" else 7.0)
                    if note.has_constraints:
                        req_slack += 2.0
                elif sla_shield:
                    req_slack = max(req_slack, 6.8 if kind == "next_day" else 6.0)
                    if note.has_constraints:
                        req_slack += 1.5
                else:
                    if sla_caution and kind == "next_day":
                        req_slack += 1.2
                    if sla_tight:
                        req_slack += 1.2
                    if sla_critical:
                        req_slack += 2.6
                        if kind != "within_3_days" and ((local_notif < 3 and not has_finisher) or kind == "same_day" or total_road_min > 220):
                            continue
                if overloaded:
                    if (in_panic or sla_shield) and kind == "within_3_days":
                        continue
                    req_slack += 2.6

                if slack < req_slack or region_notif == 0:
                    continue

                safe_pad_opt = (
                    in_panic
                    and not overloaded
                    and not note.has_constraints
                    and local_notif >= 4
                    and total_road_min <= 360
                    and (kind == "within_3_days" or (total_road_min <= 180 and dest_notif >= 3 and slack >= 8.0))
                )

                opt_spread = (1.04 if kind == "same_day" else (0.95 if (kind == "within_3_days" and sla_shield) else (0.985 if kind == "within_3_days" else 1.00))) if has_multi_opts else 1.00
                urgency_risk = 1.16 if kind == "same_day" else (1.08 if slack < 6.0 else 1.04)
                if overloaded:
                    urgency_risk += 0.14
                if sla_tight and not safe_pad_opt:
                    urgency_risk += 0.06
                if in_price_war and not safe_pad_opt:
                    urgency_risk += 0.05

                opt_cost = int(est_cost * (1.06 if kind == "same_day" else (0.94 if kind == "within_3_days" else 0.99)))
                if safe_pad_opt:
                    # Panic denominator-padding: trim profit cushion to +3% / +€2.50 & shade comp bid so safe OT volume clears reliably
                    pad_risk = 1.03 if kind == "within_3_days" else 1.06
                    floor_price = int(round(opt_cost * pad_risk + 250))
                elif remote_loyalty_mode and (
                    (kind == "next_day" and total_road_min <= 420)
                    or (kind == "within_3_days" and (total_road_min <= 420 or (total_road_min <= 600 and slack >= 10.0 and sla_tier <= 1)))
                ):
                    # Remote loyalty priority (e.g. Rovaniemi, incl. relaxed within_3_days): thin profitable floor to earn dealer_trust without losses
                    rem_risk = 1.06 if (kind == "next_day" or slack < 6.0) else 1.03
                    floor_price = int(round(opt_cost * rem_risk + 300))
                else:
                    min_euro_pad = (
                        (1200 if note.has_constraints else 600)
                        if self.shortage_Mode
                        else (750 if note.has_constraints else 350)
                    )
                    floor_price = int(round(opt_cost * urgency_risk + min_euro_pad))

                comp_price = int(round(list_c * base_comp_mult * opt_spread))
                if safe_pad_opt:
                    comp_price = int(round(comp_price * 0.90))
                q_price = max(floor_price, comp_price)
                if q_price <= list_c * 2.40:
                    prices[kind] = max(100, q_price)
                    floor_bound_by_opt[kind] = floor_price > int(comp_price * 1.05)
                    if not best_summary:
                        panic_tag = " [PANIC-SAFE]" if safe_pad_opt else ""
                        best_summary = f"{kind} €{q_price/100:.0f} ({q_price/list_c:.2f}× list, est €{opt_cost/100:.0f}, {slack:.1f}h slack){panic_tag}"

        declined = False
        if not prices:
            declined = True
            safest = req["options"][-1]
            prices[safest["kind"]] = max(100, int(round(safest["listPriceCents"] * 4.5)))
            reason = (
                "PANIC MODE (<=92% OT shield)"
                if in_panic
                else (
                    "PRE-DIP SLA SHIELD"
                    if sla_shield
                    else (
                        "capacity full"
                        if (overloaded or batch_cap_exceeded)
                        else ("unfulfillable remote lane" if toxic_lane else ("SLA shield" if sla_tight else f"tight slack / thin drivers ({local_notif} local)"))
                    )
                )
            )
            self._log(
                "decline",
                f"Polite Decline {rid} ({self.net.place_name(p_first)} → {self.net.place_name(d_last)})",
                f"Quoted 4.5× list to preserve dealer goodwill — {reason}",
                "mute",
            )
        else:
            self.pending_quotes.append((req["respondBy"], p_first))
            note_tag = f" · [{', '.join(note.summary)}]" if note.summary else ""
            self._log(
                "quote",
                f"Quoted {rid} ({self.net.place_name(p_first)} → {self.net.place_name(d_last)})",
                best_summary + note_tag,
                "info",
            )

        ref_k = "next_day" if "next_day" in prices else next(iter(prices))
        was_floor_bound = floor_bound_by_opt.get(ref_k, False)
        ref_list = next(o["listPriceCents"] for o in req["options"] if o["kind"] == ref_k)
        norm_mult = (prices[ref_k] / max(1, ref_list)) / (max(0.85, comp_lane_mult) * (1.0 + eff_batch_surge) * (1.0 + trust_bonus))
        self.quote_meta[rid] = QuoteMeta(
            request_id=rid,
            dealer_id=dealer_id,
            pickup_place=p_first,
            delivery_place=d_last,
            declined=declined,
            primary_mult=norm_mult,
            comp_only_mult=raw_comp_mult / max(0.85, comp_lane_mult),
            was_floor_bound=was_floor_bound,
            is_bulk=bool(bid),
            est_cost_cents=est_cost,
            at=self.now,
            was_panic=in_panic,
        )
        self.outbox.append({"type": "quote", "requestId": rid, "prices": prices})  # type: ignore[typeddict-item]

    # ---------- Competitor, Outage & Price-War Learning ----------
    def _record_dealer_outcome(self, dealer_id: str, won: int) -> None:
        hist = self.dealer_wins.setdefault(dealer_id, [])
        hist.append(won)
        if len(hist) > 12:
            del hist[0]

    def _world_mult_bounds(self) -> tuple[float, float]:
        if self.level >= 8:
            return (1.26, 1.62)
        if self.level == 7:
            return (1.12, 1.46)
        return (0.88, 1.22)

    def _on_win(self, order: Order) -> None:
        oid = order["id"]
        meta = self.quote_meta.get(oid)
        if meta and not meta.declined and not meta.was_panic and self.level >= 4:
            day = self.now // DAY
            self._record_dealer_outcome(order["dealerId"], 1)
            self.day_outcomes.setdefault(day, []).append(1)

            in_war = self._is_price_war_today(day)
            _, hi = self._world_mult_bounds()
            if not in_war and meta.primary_mult >= self.world_price_mult - 0.12:
                self.world_price_mult = min(hi, self.world_price_mult + 0.012)
            self.day_price_adj[day] = min(0.18, self.day_price_adj.get(day, 0.0) + 0.009)
            if not in_war:
                self.dest_price_adj[meta.delivery_place] = min(
                    0.11, self.dest_price_adj.get(meta.delivery_place, 0.0) + 0.008
                )

            # Regional Outage detection (08:00-18:00 within 80 km): ignore bulk orders, trigger on 4/4 streak or high win
            if (
                self.level >= 6
                and not in_war
                and not meta.is_bulk
                and 8 * HOUR <= (self.now % DAY) <= 18 * HOUR
            ):
                rw = self.region_recent_wins.setdefault(meta.pickup_place, [])
                rw.append(1)
                if len(rw) > 6:
                    del rw[0]
                high_win = not meta.was_floor_bound and meta.primary_mult >= (self.world_price_mult + 0.05)
                streak = len(rw) >= 4 and sum(rw[-4:]) == 4
                if high_win or streak:
                    bump = 0.040 if streak else 0.022
                    for pid in self.net.places:
                        if self.net.straight_km(meta.pickup_place, pid) <= 80.0:
                            cur_b, cur_d = self.place_outage_boost.get(pid, (0.0, day))
                            nb = min(0.24, (cur_b if cur_d == day else 0.0) + bump)
                            self.place_outage_boost[pid] = (nb, day)

        if oid in self.requests and oid not in self.orders:
            self.orders[oid] = order
            self._ensure_leg_states(order)
            p0 = self.net.place_name(order["legs"][0]["pickup"]["placeId"])
            d0 = self.net.place_name(order["legs"][-1]["delivery"]["placeId"])
            self._log(
                "win",
                f"Won {oid} · €{order['priceCents']/100:.0f} ({order['option']})",
                f"{p0} → {d0} · due {fmt_min(order['deadline'])}",
                "good",
            )
            self.post_order_legs(order, self._undelivered(order))

    def _on_loss(self, request_id: str) -> None:
        meta = self.quote_meta.get(request_id)
        if not meta or meta.declined or meta.was_panic or self.level <= 3:
            return
        day = self.now // DAY
        self._record_dealer_outcome(meta.dealer_id, 0)
        if not meta.was_floor_bound:
            self.day_outcomes.setdefault(day, []).append(0)

        if self.level >= 6 and 8 * HOUR <= (self.now % DAY) <= 18 * HOUR:
            for pid in self.net.places:
                if self.net.straight_km(meta.pickup_place, pid) <= 80.0 and pid in self.region_recent_wins:
                    rw = self.region_recent_wins[pid]
                    rw.append(0)
                    if len(rw) > 6:
                        del rw[0]
                    cur_b, cur_d = self.place_outage_boost.get(pid, (0.0, day))
                    if cur_d == day and cur_b > 0:
                        self.place_outage_boost[pid] = (max(0.0, cur_b - 0.05), day)

        in_war = self._is_price_war_today(day)
        lo, _ = self._world_mult_bounds()
        day_hist = self.day_outcomes.get(day, [])
        # Guard against dragging world_price_mult down during an unconfirmed price-war morning (<25% win rate or first 3 quotes)
        low_morning_wr = (len(day_hist) >= 4 and (sum(day_hist) / len(day_hist)) < 0.25) or (
            len(day_hist) <= 3 and 8 * HOUR <= (self.now % DAY) <= 13 * HOUR
        )
        if not in_war and not low_morning_wr and meta.primary_mult <= self.world_price_mult + 0.14:
            self.world_price_mult = max(lo, self.world_price_mult - 0.008)
        min_day_adj = min(-0.22, 1.26 - self.world_price_mult) if in_war else -0.15
        self.day_price_adj[day] = max(min_day_adj, self.day_price_adj.get(day, 0.0) - 0.007)
        if not in_war:
            self.dest_price_adj[meta.delivery_place] = max(
                -0.09, self.dest_price_adj.get(meta.delivery_place, 0.0) - 0.007
            )

    # ---------- Gig Posting, Single-Day Window Pinning & Chaining ----------
    def _undelivered(self, order: Order) -> list[int]:
        return [i for i, d in enumerate(order["deliveredAt"]) if d is None]

    def _stops_road_minutes(self, stops: list[StopRef]) -> int:
        places: list[str] = []
        for s in stops:
            o = self.orders.get(s["orderId"])
            if not o:
                return 9999
            leg = o["legs"][s["leg"]]
            places.append(leg["pickup"]["placeId"] if s["kind"] == "pickup" else leg["delivery"]["placeId"])
        total = 0
        for i in range(len(places) - 1):
            total += self.net.minutes(places[i], places[i + 1])
        return total

    def _build_stops(
        self,
        order: Order,
        legs: list[int],
        allow_windows: bool = True,
        allow_delivery_window: bool = True,
    ) -> tuple[list[StopRef], Audience]:
        oid = order["id"]
        req = self.requests[oid]
        note = self._note_of(req)
        self._ensure_leg_states(order)

        exp = any(order["legs"][i]["car"]["premium"] for i in legs)
        man = any(self.leg_states[(oid, i)].manual_licence for i in legs)
        audience: Audience = {"experienced": exp, "manualLicence": man}

        stops: list[StopRef] = []
        for i in legs:
            leg_obj = order["legs"][i]
            st = self.leg_states[(oid, i)]
            car, pick, deliv = leg_obj["car"], leg_obj["pickup"], leg_obj["delivery"]
            km = self.net.km(pick["placeId"], deliv["placeId"])
            low_ev = bool(car["electric"] and car["chargePercent"] is not None and car["chargePercent"] < 30 and km > 195)

            p_stop: StopRef = {
                "orderId": oid,
                "leg": i,
                "kind": "pickup",
                "withPlates": st.needs_plates,
                "withCharging": low_ev,
            }
            d_stop: StopRef = {"orderId": oid, "leg": i, "kind": "delivery"}

            if allow_windows:
                p_win = apply_note_to_window(note, pick["window"], "pickup")
                p_start = max(p_win["from"], st.pickup_from, pick["window"]["from"])
                if st.summer_tyres:
                    for pid in (pick["placeId"], deliv["placeId"]):
                        se = self.net.snow_end_after(pid, max(self.now, p_start))
                        if se is not None:
                            p_start = max(p_start, se)
                p_end = p_win["to"]
                drive_est = (
                    self.net.minutes(pick["placeId"], deliv["placeId"])
                    + (30 if st.needs_plates else 0)
                    + (40 if low_ev else 0)
                    + 25
                )

                if pick["privateAddress"]:
                    p_end = min(p_end, pick["window"]["to"])
                elif note.pickup_hours is not None:
                    # Pin dealer pickup stop window to the next single open day's [eff_o, eff_c].
                    # Per RULES.md line 61, dealer stop windows may end after the order's window so late cars still respect note hours!
                    search_from = max(self.now + 65, p_start)
                    chosen_pair: Optional[tuple[int, int]] = None
                    fallback_pair: Optional[tuple[int, int]] = None
                    for d_off in range(0, 7):
                        d_idx = (search_from // DAY) + d_off
                        if d_idx > 10:
                            break
                        st_ph = site_day_hours(pick.get("hours"), d_idx)
                        if st_ph is None:
                            continue
                        if (d_idx % 7) < 5:
                            eff_o = max(st_ph[0], note.pickup_hours[0] if note.pickup_hours[0] > 0 else st_ph[0])
                            eff_c = min(st_ph[1], note.pickup_hours[1])
                        else:
                            eff_o, eff_c = st_ph
                        cand_s = max(p_start, d_idx * DAY + eff_o)
                        cand_e = d_idx * DAY + eff_c
                        if note.pickup_before is not None and note.pickup_before >= self.now + 90:
                            cand_e = min(cand_e, note.pickup_before)
                        if cand_e >= max(cand_s + 35, self.now + 85):
                            if fallback_pair is None:
                                fallback_pair = (cand_s, cand_e)
                            # If remaining time today is too short (< 130m) and another open day exists before deadline, prefer when enough time exists
                            if cand_e >= self.now + 130 or (d_idx + 1) * DAY > order["deadline"]:
                                chosen_pair = (cand_s, cand_e)
                                break
                    if chosen_pair is not None:
                        p_start, p_end = chosen_pair
                    elif fallback_pair is not None:
                        p_start, p_end = fallback_pair
                else:
                    # Dealer site without daily pickup_hours narrowing: if p_start moved later (e.g. not_ready/closed/note), ensure p_end stays open
                    if p_start > pick["window"]["from"] and p_end < max(p_start + 90, self.now + 150):
                        p_end = max(p_end, p_start + 2 * DAY, self.now + 2 * DAY)

                if p_start <= p_end and p_end > self.now + 25 and (
                    p_start > pick["window"]["from"] or p_end != pick["window"]["to"]
                ):
                    p_stop["window"] = {"from": int(p_start), "to": int(p_end)}

                d_base: TimeWindow = {
                    "from": deliv["window"]["from"],
                    "to": max(deliv["window"]["from"], order["deadline"]),
                }
                earliest_arr = max(self.now + 40, p_start) + drive_est
                if allow_delivery_window and self.now < order["lastPickupAt"] - 90:
                    d_win = apply_note_to_window(note, d_base, "delivery")
                    if note.delivery_hours is not None and not deliv["privateAddress"]:
                        arr_day = earliest_arr // DAY
                        st_dh = site_day_hours(deliv.get("hours"), arr_day)
                        if st_dh is not None and (arr_day % 7) < 5:
                            eff_do = max(st_dh[0], note.delivery_hours[0] if note.delivery_hours[0] > 0 else st_dh[0])
                            eff_dc = min(st_dh[1], note.delivery_hours[1])
                            cand_ds = max(d_win["from"], arr_day * DAY + eff_do)
                            cand_de = min(d_win["to"], arr_day * DAY + eff_dc)
                            if cand_de >= max(cand_ds + 30, earliest_arr + 20):
                                d_win = {"from": cand_ds, "to": cand_de}

                    if (
                        d_win["from"] <= d_win["to"]
                        and d_win["to"] >= max(self.now + 60, earliest_arr + 20)
                        and (d_win["from"] > d_base["from"] or d_win["to"] < d_base["to"])
                    ):
                        d_stop["window"] = {"from": int(d_win["from"]), "to": int(d_win["to"])}

                # Crucial fix for RULES.md lines 61 & 73: if p_stop has a custom window that pushes earliest arrival past
                # the order's stated delivery window (while the world still considers the stated window active),
                # extend d_stop["window"]["to"] explicitly so drivers can still book the gig!
                if "window" in p_stop and "window" not in d_stop and not deliv["privateAddress"]:
                    st_del_h = site_day_hours(deliv.get("hours"), order["deadline"] // DAY)
                    stated_del_end = (
                        min(order["deadline"], (order["deadline"] // DAY) * DAY + st_del_h[1])
                        if st_del_h is not None
                        else order["deadline"]
                    )
                    if earliest_arr + 15 > stated_del_end:
                        ext_to = max(order["deadline"] + DAY, p_end + drive_est + DAY)
                        d_stop["window"] = {"from": int(deliv["window"]["from"]), "to": int(ext_to)}

            stops.extend([p_stop, d_stop])

        return stops, audience

    def _sweep_aging_orders(self) -> None:
        """Graduated predictive rescue (Tier 1 Caution, Tier 2 Pre-Dip Shield, Tier 3 Panic) + viewer offer sniping."""
        in_panic = self._update_panic_mode()
        tier = self.sla_tier
        day = self.now // DAY

        # 1. Snipe any active viewers on open unbooked gigs immediately via direct offers
        if self.level >= 3:
            for gid, act in list(self.posts.items()):
                if gid in self.booked or not self.gig_viewers.get(gid):
                    continue
                if self.gig_open_offers.get(gid, 0) >= 3:
                    continue
                first_oid = act["stops"][0]["orderId"]
                self._maybe_send_offers(gid, act, urgent=(first_oid in self.urgent or tier >= 2))

        # 2. Two-stage aging & pre-deadline rescue
        tod = self.now % DAY
        is_daytime = 7 * HOUR + 30 <= tod <= 20 * HOUR + 30
        for oid, o in list(self.orders.items()):
            if o["status"] != "waiting":
                continue
            gids = self.gigs_of(oid)
            if any(g in self.booked for g in gids):
                continue
            rem = o["lastPickupAt"] - self.now
            if rem < -120:
                continue

            req = self.requests.get(oid)
            p_place = o["legs"][0]["pickup"]["placeId"]
            aud: Audience = {
                "experienced": any(l["car"]["premium"] for l in o["legs"]),
                "manualLicence": False,
            }
            local_notif = self.net.matching_notif_count(p_place, aud, region=False)
            est_road = sum(self.net.minutes(l["pickup"]["placeId"], l["delivery"]["placeId"]) for l in o["legs"])
            age = max((self.now - self.gig_posted_at.get(g, self.now) for g in gids), default=0)

            # Stage 2: widened proactively in Tier 1/2/3 & Friday so unbooked gigs clear before window expiration
            if in_panic:
                urg_rem = 300 + min(150, int(est_road * 0.28))
            elif tier >= 2:
                urg_rem = 260 + min(140, int(est_road * 0.25))
            elif tier == 1 or (day == 4 and self.shortage_Mode):
                urg_rem = 235 + min(135, int(est_road * 0.24))
            else:
                urg_rem = 215 + min(130, int(est_road * 0.22))
            if rem <= urg_rem and oid not in self.urgent:
                self.urgent.add(oid)
                self.early_boosted.add(oid)
                if self.level == 1:
                    if self._withdraw_order_gigs(oid):
                        self.post_order_legs(
                            o, self._undelivered(o), allow_windows=True, allow_delivery_window=False, allow_chain=False
                        )
                    continue
                for gid in list(gids):
                    self._split_or_reprice_gig(o, gid, force_urgent=True)
                continue

            # Stage 1: road-aware early gentle nudge + direct offers
            if oid not in self.urgent and oid not in self.early_boosted:
                road_pad = min(180, int(est_road * 0.35))
                base_s1 = 520 if in_panic else (480 if tier >= 2 else (430 if (tier == 1 or day == 4) else 350))
                s1_thresh = base_s1 + road_pad
                trigger_stage1 = (rem <= s1_thresh) or (rem <= s1_thresh + 60 and local_notif <= 2)
                if not trigger_stage1:
                    offer_age = 10 if tier >= 2 else (15 if (tier == 1 or day == 4) else 25)
                    if is_daytime and age >= offer_age:
                        for gid in list(gids):
                            act = self.posts.get(gid)
                            if act and gid not in self.booked:
                                self._maybe_send_offers(gid, act, urgent=(tier >= 2))
                    continue
                self.early_boosted.add(oid)
                if self.level == 1:
                    self.urgent.add(oid)
                    if self._withdraw_order_gigs(oid):
                        self.post_order_legs(
                            o, self._undelivered(o), allow_windows=True, allow_delivery_window=False, allow_chain=False
                        )
                    continue
                for gid in list(gids):
                    act = self.posts.get(gid)
                    if not act or gid in self.booked:
                        continue
                    cars = sum(1 for s in act["stops"] if s["kind"] == "pickup")
                    est_base = (
                        self._estimate_fulfillment_cost(req, self._note_of(req))[0] * (cars / max(1, len(o["legs"])))
                        if req
                        else 15000.0
                    )
                    if in_panic:
                        s1_steps, s1_step_m = [1.14, 1.24, 1.36], 25
                    elif tier >= 2:
                        s1_steps, s1_step_m = [1.10, 1.19, 1.30], 25
                    elif tier == 1 or (day == 4 and self.shortage_Mode):
                        s1_steps, s1_step_m = [1.08, 1.15, 1.23], 30
                    else:
                        s1_steps, s1_step_m = [1.06, 1.12, 1.18], 35
                    mid_sched = self._monotonic_steps(est_base, s1_steps, step_mins=s1_step_m)
                    self._split_or_reprice_gig(o, gid, force_urgent=False, schedule=mid_sched)

    def _schedule_for(self, order: Order, legs_count: int, force_urgent: bool = False) -> list[FeeStep]:
        oid = order["id"]
        req = self.requests[oid]
        note = self._note_of(req)
        est_total, aud, _, _ = self._estimate_fulfillment_cost(req, note)
        base = est_total * (legs_count / max(1, len(order["legs"])))

        p_place = order["legs"][0]["pickup"]["placeId"]
        local_notif = self.net.matching_notif_count(p_place, aud, region=False)
        if local_notif <= 1:
            base *= 1.04

        time_to_last = order["lastPickupAt"] - self.now
        est_road = sum(self.net.minutes(l["pickup"]["placeId"], l["delivery"]["placeId"]) for l in order["legs"])
        tier = max(self.sla_tier, 3 if self.panic_mode else 0)
        day = self.now // DAY
        urg_base = 300 if tier >= 3 else (260 if tier == 2 else (235 if (tier == 1 or day == 4) else 215))
        urg_window = urg_base + min(150, int(est_road * 0.25))
        is_urg = force_urgent or (oid in self.urgent) or (time_to_last <= urg_window)
        is_late = self.now >= order["lastPickupAt"]

        if self.level == 1:
            ref_part = req["referenceFeeCents"] * (legs_count / max(1, len(order["legs"])))
            flat_base = max(base, ref_part * 0.98)
            mult = 1.45 if is_urg else 1.02
            return [{"at": int(self.now), "feeCents": max(100, int(round(flat_base * mult)))}]

        if is_late:
            return self._monotonic_steps(base, [1.40, 1.62, 1.88, 2.15], step_mins=35)

        if is_urg:
            steps = [1.35, 1.54, 1.76, 2.00] if tier >= 2 else [1.30, 1.46, 1.65, 1.88]
            return self._monotonic_steps(base, steps, step_mins=20 if tier >= 2 else 25)

        if tier >= 3:
            # Panic Mode (<= 92% OT): pay modest clearing premium immediately so non-late orders book fast
            return self._monotonic_steps(base, [1.06, 1.15, 1.25], step_mins=25)
        if tier == 2:
            # Pre-Dip Shield: accelerate booking cleanly before OT hits 92%
            return self._monotonic_steps(base, [1.01, 1.09, 1.18], step_mins=30)
        if tier == 1 or (day == 4 and self.shortage_Mode):
            # Caution / Friday Crunch: prevent closing-time holdouts while staying steady
            return self._monotonic_steps(base, [0.97, 1.03, 1.10], step_mins=35)

        # Steady, low-fluctuation public schedule (~9% span): keeps booked_fee / first_viewed_fee flat so drivers don't hold out!
        steps = [0.94, 0.98, 1.03] if self.shortage_Mode else [0.90, 0.94, 0.99]
        interval = 55 if time_to_last > 450 else 40
        return self._monotonic_steps(base, steps, step_mins=interval)

    def _monotonic_steps(
        self, base: float, mults: list[float], step_mins: int, cap_cents: Optional[int] = None
    ) -> list[FeeStep]:
        out: list[FeeStep] = []
        prev = 100
        for idx, m in enumerate(mults):
            raw = int(round(base * m))
            if cap_cents is not None:
                raw = min(raw, max(prev, cap_cents))
            fee = max(prev, raw)
            prev = fee
            out.append({"at": int(self.now + idx * step_mins), "feeCents": fee})
        return out

    def _try_chain_with_open_gig(self, new_order: Order, new_stops: list[StopRef], new_aud: Audience) -> bool:
        """Pairs two compatible 1-leg orders into a cheap 2-car chain before drivers lock onto the single gig."""
        if self.panic_mode or self.sla_tier >= 2 or self.level < 3 or len(new_order["legs"]) != 1 or new_order["id"] in self.urgent:
            return False
        min_rem = 540 if self.sla_tier == 1 else 360
        if new_order["lastPickupAt"] - self.now < min_rem:
            return False

        b_pick = new_order["legs"][0]["pickup"]["placeId"]
        b_del = new_order["legs"][0]["delivery"]["placeId"]
        if new_order["legs"][0]["pickup"]["privateAddress"] or new_order["legs"][0]["delivery"]["privateAddress"]:
            return False

        best_match: Optional[tuple[float, str, Order, list[StopRef], Audience]] = None

        for gid, act in list(self.posts.items()):
            max_v = 15 if self.shortage_Mode else 18
            if gid in self.booked or len(act["stops"]) != 2 or len(self.gig_viewers.get(gid, ())) > max_v:
                continue
            oid_a = act["stops"][0]["orderId"]
            if oid_a == new_order["id"] or oid_a in self.urgent:
                continue
            ord_a = self.orders.get(oid_a)
            if not ord_a or ord_a["status"] != "waiting" or len(ord_a["legs"]) != 1:
                continue
            if self.leg_gigs.get(oid_a, {}).get(0) != gid:
                continue
            if ord_a["lastPickupAt"] - self.now < 360:
                continue
            if ord_a["legs"][0]["pickup"]["privateAddress"] or ord_a["legs"][0]["delivery"]["privateAddress"]:
                continue

            a_pick = ord_a["legs"][0]["pickup"]["placeId"]
            a_del = ord_a["legs"][0]["delivery"]["placeId"]
            comb_aud: Audience = {
                "experienced": bool(act["audience"]["experienced"] or new_aud["experienced"]),
                "manualLicence": bool(act["audience"]["manualLicence"] or new_aud["manualLicence"]),
            }

            for first_o, second_o, first_stops, second_stops, f_pick, f_del, s_pick, s_del in (
                (ord_a, new_order, list(act["stops"]), list(new_stops), a_pick, a_del, b_pick, b_del),
                (new_order, ord_a, list(new_stops), list(act["stops"]), b_pick, b_del, a_pick, a_del),
            ):
                conn = self.net.minutes(f_del, s_pick)
                if conn > 40:
                    continue
                cand_stops = first_stops + second_stops
                total_road = self._stops_road_minutes(cand_stops)
                return_gap_km = self.net.straight_km(s_del, f_pick)
                is_rt = return_gap_km <= 35.0
                max_road = (540 if is_rt else 480) if self.shortage_Mode else 620
                if total_road > max_road:
                    continue
                if "same_day" in (first_o["option"], second_o["option"]) and not (is_rt and total_road <= 230):
                    continue
                if self.net.matching_notif_count(f_pick, comb_aud, region=True) < 2:
                    continue

                f_ready = max(self.now + 30, first_o["legs"][0]["pickup"]["window"]["from"])
                if "window" in first_stops[0]:
                    f_ready = max(f_ready, first_stops[0]["window"]["from"])
                f_arr = f_ready + self.net.minutes(f_pick, f_del) + 30 + conn

                s_ready = second_o["legs"][0]["pickup"]["window"]["from"]
                if "window" in second_stops[0]:
                    s_ready = max(s_ready, second_stops[0]["window"]["from"])
                if f_arr > second_o["lastPickupAt"] - 170 or s_ready > f_arr + 120:
                    continue

                if self.shortage_Mode and return_gap_km > 85.0 and conn > 28:
                    continue
                quality = 100.0 - conn * 1.2 - return_gap_km * 0.25
                if best_match is None or quality > best_match[0]:
                    best_match = (quality, gid, ord_a, cand_stops, comb_aud)

        if best_match is None:
            return False

        _, gid, ord_a, combined_stops, comb_aud = best_match
        oid_a = ord_a["id"]
        f_pick = self.orders[combined_stops[0]["orderId"]]["legs"][0]["pickup"]["placeId"]
        s_del = self.orders[combined_stops[-1]["orderId"]]["legs"][0]["delivery"]["placeId"]
        is_round_trip = self.net.straight_km(f_pick, s_del) <= 35.0

        chain_disc = 0.89 if is_round_trip else 0.93
        sched_a = self._schedule_for(ord_a, 1)
        sched_b = self._schedule_for(new_order, 1)
        n_steps = min(len(sched_a), len(sched_b))
        comb_sched: list[FeeStep] = []
        prev = 100
        for k in range(n_steps):
            f = max(prev, int(round((sched_a[k]["feeCents"] + sched_b[k]["feeCents"]) * chain_disc)))
            prev = f
            comb_sched.append({"at": sched_a[k]["at"], "feeCents": f})

        self.leg_gigs.pop(oid_a, None)
        self.posts.pop(gid, None)
        self.outbox.append({"type": "withdraw", "gigId": gid})
        self.outbox.append({
            "type": "post",
            "stops": combined_stops,
            "audience": comb_aud,
            "schedule": comb_sched,
        })
        total_road = self._stops_road_minutes(combined_stops)
        self._log(
            "chain",
            f"Chained {oid_a} + {new_order['id']} ({'Round-Trip' if is_round_trip else 'Corridor'})",
            f"{self.net.place_name(f_pick)} → … → {self.net.place_name(s_del)} ({total_road//60}h{total_road%60:02d}m, €{comb_sched[0]['feeCents']/100:.0f})",
            "good",
        )
        return True

    def post_order_legs(
        self,
        order: Order,
        legs: list[int],
        allow_windows: bool = True,
        allow_delivery_window: bool = True,
        allow_chain: bool = True,
    ) -> None:
        if not legs:
            return
        stops, audience = self._build_stops(
            order,
            legs,
            allow_windows=allow_windows,
            allow_delivery_window=allow_delivery_window,
        )

        if len(legs) > 1 and self._stops_road_minutes(stops) > 715:
            for lg in legs:
                self.post_order_legs(
                    order,
                    [lg],
                    allow_windows=allow_windows,
                    allow_delivery_window=allow_delivery_window,
                    allow_chain=False,
                )
            return

        if len(legs) == 1 and allow_chain and not self.panic_mode and self.sla_tier < 2 and self._try_chain_with_open_gig(order, stops, audience):
            return

        sched = self._schedule_for(order, len(legs))
        self.outbox.append({"type": "post", "stops": stops, "audience": audience, "schedule": sched})

    # ---------- Direct Offers (Finishers, Homebound Hub Drivers, Viewers & Urgent Rescue) ----------
    def _maybe_send_offers(self, gig_id: str, action: PostAction, urgent: bool = False) -> None:
        if self.level < 3 or gig_id in self.booked:
            return
        tod = self.now % DAY
        if tod < 7 * HOUR or tod > 22 * HOUR + 45:
            return

        open_cnt = self.gig_open_offers.get(gig_id, 0)
        if open_cnt >= 3:
            return

        first_stop = action["stops"][0]
        last_stop = action["stops"][-1]
        o_first = self.orders.get(first_stop["orderId"])
        o_last = self.orders.get(last_stop["orderId"])
        if not o_first or not o_last:
            return

        p_place = o_first["legs"][first_stop["leg"]]["pickup"]["placeId"]
        d_place = o_last["legs"][last_stop["leg"]]["delivery"]["placeId"]
        lead_mins = max(0, o_first["lastPickupAt"] - self.now)
        viewers = self.gig_viewers.get(gig_id, set())

        candidates = self.net.candidate_drivers(
            p_place,
            d_place,
            action["audience"],
            self.finishing_places,
            self.active_drivers_today,
            self.now,
            lead_mins=lead_mins,
            gig_viewers=viewers,
            urgent=urgent,
        )
        if not candidates:
            return

        local_cnt = self.net.matching_notif_count(p_place, action["audience"], region=False)
        top_score, top_did, top_kind = candidates[0]
        if not urgent and top_kind == "local" and top_did not in viewers and local_cnt >= 6 and top_score < 32.0:
            return

        cur_fee = action["schedule"][0]["feeCents"]
        for step in action["schedule"]:
            if step["at"] <= self.now:
                cur_fee = step["feeCents"]

        sent = 0
        max_to_send = min(3 - open_cnt, 3 if urgent else 2)
        last_fee = cur_fee
        for score, did, c_kind in candidates:
            if sent >= max_to_send:
                break
            if not urgent and c_kind == "local" and did not in viewers and local_cnt >= 6 and score < 32.0:
                continue
            if (gig_id, did) in self.offered_pairs:
                continue
            if self.driver_offers_today.get(did, 0) >= 3:
                continue

            if urgent:
                mult = 1.06
            elif c_kind == "finisher":
                mult = 0.91
            elif c_kind == "homebound":
                mult = 0.93
            else:
                mult = 0.98
            offer_fee = max(100, int(round(cur_fee * mult)))
            last_fee = offer_fee

            self.outbox.append({"type": "offer", "gigId": gig_id, "driverId": did, "feeCents": offer_fee})
            self.offered_pairs.add((gig_id, did))
            self.driver_offers_today[did] = self.driver_offers_today.get(did, 0) + 1
            self.gig_open_offers[gig_id] = self.gig_open_offers.get(gig_id, 0) + 1
            sent += 1

        if sent:
            self._log(
                "offer",
                f"Sent {sent} Direct Offer(s) for {gig_id}",
                f"{self.net.place_name(p_place)} → {self.net.place_name(d_place)} @ ~€{last_fee/100:.0f}",
                "info",
            )

    # ---------- Booking & Exception Handlers ----------
    def _on_gig_booked(self, gig_id: str, driver_id: str, fee_cents: int) -> None:
        self.booked.add(gig_id)
        self.gig_driver[gig_id] = driver_id
        self.active_drivers_today.add(driver_id)
        self.bookings_today += 1
        self.outbox = [
            a for a in self.outbox if not (a.get("gigId") == gig_id and a["type"] in ("reprice", "withdraw", "offer"))
        ]

        act = self.posts.get(gig_id)
        if not act:
            return
        last_s = act["stops"][-1]
        o_last = self.orders.get(last_s["orderId"])
        if o_last:
            dest_place = o_last["legs"][last_s["leg"]]["delivery"]["placeId"]
            first_s = act["stops"][0]
            o_first = self.orders.get(first_s["orderId"])
            p_from = o_first["legs"][first_s["leg"]]["pickup"]["window"]["from"] if o_first else self.now
            win_from = first_s["window"]["from"] if "window" in first_s else 0
            est_fin = max(self.now, p_from, win_from) + self._stops_road_minutes(act["stops"]) + 45
            self.finishing_places[driver_id] = (dest_place, est_fin)

            # Immediately check if we already have an open unbooked gig starting near dest_place around est_fin!
            for open_gid, open_act in list(self.posts.items()):
                if open_gid in self.booked or open_gid == gig_id:
                    continue
                fs = open_act["stops"][0]
                fo = self.orders.get(fs["orderId"])
                if not fo:
                    continue
                op_pick = fo["legs"][fs["leg"]]["pickup"]["placeId"]
                if self.net.straight_km(dest_place, op_pick) <= 45.0 and est_fin <= fo["lastPickupAt"] - 50:
                    self._maybe_send_offers(open_gid, open_act, urgent=(fo["id"] in self.urgent))

        # Online update of empirical clearing fee multiplier (excluding emergency/aging boosts & 2-car discounted chains)
        order_ids = list(dict.fromkeys(s["orderId"] for s in act["stops"]))
        if len(order_ids) == 1 and order_ids[0] not in self.urgent and order_ids[0] not in self.early_boosted:
            oid = order_ids[0]
            req = self.requests.get(oid)
            ord_o = self.orders.get(oid)
            if req and ord_o:
                cars = sum(1 for s in act["stops"] if s["kind"] == "pickup")
                est_tot, _, _, _ = self._estimate_fulfillment_cost(req, self._note_of(req))
                est_base = est_tot * (cars / max(1, len(ord_o["legs"])))
                if est_base > 0:
                    ratio = fee_cents / est_base
                    target_mult = self.fee_mult_est * max(0.88, min(1.16, ratio))
                    lo_f, hi_f = (0.90, 1.12) if self.shortage_Mode else (0.85, 1.06)
                    self.fee_mult_est = max(lo_f, min(hi_f, 0.94 * self.fee_mult_est + 0.06 * target_mult))

    def _on_gig_lost_driver(self, gig_id: str, driver_id: str, kind: str) -> None:
        self.booked.discard(gig_id)
        self.started.discard(gig_id)
        self.gig_driver.pop(gig_id, None)
        if self.finishing_places.get(driver_id):
            self.finishing_places.pop(driver_id, None)

        act = self.posts.get(gig_id)
        if not act:
            return
        order_ids = list(dict.fromkeys(s["orderId"] for s in act["stops"]))
        orders_here = [self.orders[oid] for oid in order_ids if oid in self.orders]
        if not orders_here:
            return

        min_slack = min(o["lastPickupAt"] - self.now for o in orders_here)
        if (len(order_ids) > 1 and min_slack < 300) or self.level == 1:
            self.outbox.append({"type": "withdraw", "gigId": gig_id})
            for s in act["stops"]:
                self.leg_gigs.get(s["orderId"], {}).pop(s["leg"], None)
            self.posts.pop(gig_id, None)
            for o in orders_here:
                self.post_order_legs(
                    o,
                    self._undelivered(o),
                    allow_windows=True,
                    allow_delivery_window=(self.now < o["lastPickupAt"] - 90),
                    allow_chain=False,
                )
            return

        primary = orders_here[0]
        force_urg = (min_slack <= 270) or (kind == "no_show")
        if force_urg:
            self.urgent.add(primary["id"])
        cars = sum(1 for s in act["stops"] if s["kind"] == "pickup")
        new_sched = self._schedule_for(primary, cars, force_urgent=force_urg)
        if new_sched:
            bump = int(round(new_sched[0]["feeCents"] * 1.08))
            new_sched[0]["feeCents"] = bump
            for i in range(1, len(new_sched)):
                new_sched[i]["feeCents"] = max(new_sched[i - 1]["feeCents"], new_sched[i]["feeCents"])

        self.outbox.append({"type": "reprice", "gigId": gig_id, "schedule": new_sched})
        act["schedule"] = new_sched
        self._log(
            "recover",
            f"Recovered {gig_id} after {kind}",
            f"Repriced from €{new_sched[0]['feeCents']/100:.0f} ({min_slack//60}h to last pickup)",
            "warn",
        )
        self._maybe_send_offers(gig_id, act, urgent=force_urg)

    def _on_driver_late(self, gig_id: str, driver_id: str, pickup_at: int) -> None:
        if gig_id in self.started:
            return
        act = self.posts.get(gig_id)
        if not act:
            return
        first_s = act["stops"][0]
        o = self.orders.get(first_s["orderId"])
        if not o:
            return

        road_m = self._stops_road_minutes(act["stops"])
        est_deliv = pickup_at + road_m + 20
        will_miss_deadline = est_deliv > o["deadline"]
        p_place = o["legs"][first_s["leg"]]["pickup"]["placeId"]
        local_alts = self.net.matching_notif_count(p_place, act["audience"], region=False)

        if will_miss_deadline and self.now < o["lastPickupAt"] - 45 and pickup_at - self.now > 90 and local_alts >= 3:
            self.booked.discard(gig_id)
            self.posts.pop(gig_id, None)
            affected_orders: set[str] = set()
            for s in act["stops"]:
                self.leg_gigs.get(s["orderId"], {}).pop(s["leg"], None)
                affected_orders.add(s["orderId"])
            self.outbox.append({"type": "release", "gigId": gig_id})
            for oid in affected_orders:
                ord_obj = self.orders.get(oid)
                if ord_obj:
                    self.urgent.add(oid)
                    self.post_order_legs(
                        ord_obj,
                        self._undelivered(ord_obj),
                        allow_windows=True,
                        allow_delivery_window=False,
                        allow_chain=False,
                    )
            self._log(
                "recover",
                f"Free-Released Late Driver on {gig_id}",
                f"Pickup {fmt_min(pickup_at)} would miss {fmt_min(o['deadline'])}; emergency reposting",
                "warn",
            )

    def _on_gig_expired(self, gig_id: str) -> None:
        act = self.posts.pop(gig_id, None)
        self.booked.discard(gig_id)
        if not act:
            return
        still_owned: set[str] = set()
        for s in act["stops"]:
            if self.leg_gigs.get(s["orderId"], {}).get(s["leg"]) == gig_id:
                self.leg_gigs[s["orderId"]].pop(s["leg"], None)
                still_owned.add(s["orderId"])
        for oid in still_owned:
            o = self.orders.get(oid)
            if o and o["status"] != "cancelled":
                missing = [i for i in self._undelivered(o) if i not in self.leg_gigs.get(oid, {})]
                if missing:
                    # Always preserve pickup windows (so notes like `nouto 9-14` advance to the next open day instead of failing closed!)
                    keep_deliv_win = self.now < o["lastPickupAt"] - 90
                    self.post_order_legs(
                        o,
                        missing,
                        allow_windows=True,
                        allow_delivery_window=keep_deliv_win,
                        allow_chain=False,
                    )
                    self._log(
                        "recover",
                        f"Reposted Expired Gig for {oid}",
                        "Advanced pickup window to next open day",
                        "warn",
                    )

    def _on_picked_up(self, order_id: str, leg: int, at: int) -> None:
        self.picked_up_legs.add((order_id, leg))
        o = self.orders.get(order_id)
        if not o or leg >= len(o["legs"]):
            return
        l_obj = o["legs"][leg]
        rem_drive = self.net.minutes(l_obj["pickup"]["placeId"], l_obj["delivery"]["placeId"])
        if at + rem_drive > o["deadline"]:
            self.doomed_late_orders.add(order_id)

    def _record_failure(self, order_id: str, leg: int, reason: str, at: int) -> None:
        o = self.orders.get(order_id)
        if not o:
            return
        self._ensure_leg_states(o)
        st = self.leg_states[(order_id, leg)]
        st.failed_count += 1
        if reason == "licence":
            st.manual_licence = True
        elif reason == "no_plates":
            st.needs_plates = True
        elif reason in ("not_ready", "closed"):
            req = self.requests.get(order_id)
            note = self._note_of(req) if req else ParsedNote()
            delay = 0
            if req:
                promised = next((op["deadline"] for op in req["options"] if op["kind"] == o["option"]), o["deadline"])
                delay = max(0, o["deadline"] - promised)
            next_try = max(at + 2 * HOUR, o["legs"][leg]["pickup"]["window"]["from"] + delay, note.pickup_after or 0)
            pick = o["legs"][leg]["pickup"]
            if reason == "closed" and not pick["privateAddress"]:
                for d_off in range(0, 4):
                    d_idx = (next_try // DAY) + d_off
                    sh = site_day_hours(pick.get("hours"), d_idx)
                    if sh is not None:
                        eff_o = max(
                            sh[0],
                            note.pickup_hours[0]
                            if (note.pickup_hours and note.pickup_hours[0] > 0 and (d_idx % 7) < 5)
                            else sh[0],
                        )
                        eff_c = min(
                            sh[1],
                            note.pickup_hours[1] if (note.pickup_hours and (d_idx % 7) < 5) else sh[1],
                        )
                        cand_open = d_idx * DAY + eff_o
                        cand_close = d_idx * DAY + eff_c
                        if next_try <= cand_close - 45:
                            next_try = max(next_try, cand_open)
                            break
            st.pickup_from = max(st.pickup_from, next_try)

    def _on_visit_failed(self, order_id: str, leg: int, reason: str) -> None:
        if order_id not in self.orders:
            return
        o = self.orders[order_id]
        self.leg_gigs.get(order_id, {}).pop(leg, None)
        self._record_failure(order_id, leg, reason, self.now)
        self.post_order_legs(
            o,
            [leg],
            allow_windows=True,
            allow_delivery_window=(self.now < o["lastPickupAt"] - 90),
            allow_chain=False,
        )
        self._log("recover", f"Retrying {order_id} leg {leg} ({reason})", "Updated constraints & reposted", "bad")

    def _on_payment_release(self, order_id: str, leg: int) -> None:
        if order_id not in self.orders:
            return
        o = self.orders[order_id]
        self._ensure_leg_states(o)
        st = self.leg_states[(order_id, leg)]
        st.released = True
        st.pickup_from = self.now
        gid = self.leg_gigs.get(order_id, {}).get(leg)
        if gid and gid not in self.booked:
            act = self.posts.get(gid)
            if act and len(set(s["orderId"] for s in act["stops"])) == 1:
                if self._withdraw_order_gigs(order_id):
                    self.post_order_legs(o, self._undelivered(o), allow_chain=False)

    def _on_change(self, order: Order, msg_text: str) -> None:
        oid = order["id"]
        if oid not in self.orders:
            return
        self.orders[oid] = order
        if self._withdraw_order_gigs(oid):
            self.post_order_legs(order, self._undelivered(order), allow_chain=False)
            self._log("change", f"Order {oid} Changed", f"{msg_text} · new due {fmt_min(order['deadline'])}", "warn")

    def _on_cancel(self, order_id: str, gigs: list[dict[str, Any]]) -> None:
        self.orders.pop(order_id, None)
        self.leg_gigs.pop(order_id, None)
        self.urgent.discard(order_id)
        self.doomed_late_orders.discard(order_id)
        for cg in gigs:
            gid, outcome = cg["gigId"], cg["outcome"]
            act = self.posts.get(gid)
            if outcome in ("withdrawn", "released"):
                self.booked.discard(gid)
                self.posts.pop(gid, None)
                if act:
                    other_orders = {s["orderId"] for s in act["stops"] if s["orderId"] != order_id}
                    for other_id in other_orders:
                        self.leg_gigs.pop(other_id, None)
                        other_o = self.orders.get(other_id)
                        if other_o and other_o["status"] != "cancelled":
                            self.post_order_legs(other_o, self._undelivered(other_o), allow_chain=False)

    def _on_delivered(self, order_id: str, leg: int, at: int) -> None:
        self.picked_up_legs.add((order_id, leg))
        self.leg_gigs.get(order_id, {}).pop(leg, None)
        if order_id in self.orders:
            self.orders[order_id]["deliveredAt"][leg] = at
            if all(d is not None for d in self.orders[order_id]["deliveredAt"]):
                self.orders[order_id]["status"] = "delivered"
                self.doomed_late_orders.discard(order_id)

    def _on_deadline_close(self, order_id: str) -> None:
        o = self.orders.get(order_id)
        if not o:
            return
        self.urgent.add(order_id)
        self.send_post_of(order_id)
        gids = self.gigs_of(order_id)

        if not gids:
            missing = self._undelivered(o)
            if missing:
                self.post_order_legs(
                    o, missing, allow_windows=True, allow_delivery_window=False, allow_chain=False
                )
            return

        if self.level == 1:
            if self._withdraw_order_gigs(order_id):
                self.post_order_legs(
                    o, self._undelivered(o), allow_windows=True, allow_delivery_window=False, allow_chain=False
                )
            return

        for gid in list(gids):
            self._split_or_reprice_gig(o, gid, force_urgent=True)

        self._log(
            "urgent",
            f"Deadline Warning on {order_id}",
            f"Last pickup {fmt_min(o['lastPickupAt'])} — boosted fee & sent direct offers",
            "warn",
        )

    # ---------- Outbox & Post Result Plumbing ----------
    def gigs_of(self, order_id: str) -> list[str]:
        return list(dict.fromkeys(self.leg_gigs.get(order_id, {}).values()))

    def carry(self, action: PostAction, gig_id: str, posted_at: Optional[int] = None) -> None:
        for stop in action["stops"]:
            self.leg_gigs.setdefault(stop["orderId"], {})[stop["leg"]] = gig_id
        self.posts[gig_id] = action
        self.gig_posted_at[gig_id] = self.now if posted_at is None else posted_at

    def _withdraw_order_gigs(self, order_id: str) -> bool:
        self.send_post_of(order_id)
        gids = self.gigs_of(order_id)
        if any(g in self.booked for g in gids):
            return False
        for g in gids:
            act = self.posts.get(g)
            if act and len(set(s["orderId"] for s in act["stops"])) > 1:
                return False
        self.leg_gigs.pop(order_id, None)
        for g in gids:
            self.posts.pop(g, None)
            self.outbox = [
                a for a in self.outbox if not (a.get("gigId") == g and a["type"] in ("reprice", "withdraw", "offer"))
            ]
            self.outbox.append({"type": "withdraw", "gigId": g})
        return True

    def send_post_of(self, order_id: str) -> None:
        if any(
            a["type"] == "post" and any(s["orderId"] == order_id for s in a["stops"])
            for a in self.outbox
        ):
            self.send()

    def send(self) -> None:
        actions, self.outbox = self.outbox, []
        if actions:
            self.read_results(actions, self.world.act(actions))

    def read_results(self, actions: list[Action], results: list[ActionResult]) -> None:
        for action, result in zip(actions, results):
            if action["type"] != "post":
                continue
            if result["ok"] is True:
                gid = result["id"]
                self.carry(action, gid)
                first_oid = action["stops"][0]["orderId"]
                is_urg = first_oid in self.urgent or self.panic_mode
                self._maybe_send_offers(gid, action, urgent=is_urg)
            else:
                had_deliv_win = any(s["kind"] == "delivery" and "window" in s for s in action["stops"])
                had_any_win = any("window" in s for s in action["stops"])
                pickups_cnt = sum(1 for s in action["stops"] if s["kind"] == "pickup")
                if pickups_cnt > 1:
                    seen: set[tuple[str, int]] = set()
                    for s in action["stops"]:
                        key = (s["orderId"], s["leg"])
                        if key in seen:
                            continue
                        seen.add(key)
                        o = self.orders.get(s["orderId"])
                        if o and o["deliveredAt"][s["leg"]] is None:
                            self.post_order_legs(
                                o, [s["leg"]], allow_windows=True, allow_delivery_window=False, allow_chain=False
                            )
                elif had_deliv_win:
                    # Strip delivery window first while preserving critical pickup windows (`nouto 9-14`, etc.)!
                    keep_pickup_stops: list[StopRef] = [
                        ({k: v for k, v in s.items() if k != "window"} if s["kind"] == "delivery" else s)  # type: ignore[misc]
                        for s in action["stops"]
                    ]
                    self.outbox.append({**action, "stops": keep_pickup_stops})
                elif had_any_win:
                    clean_stops: list[StopRef] = [
                        {k: v for k, v in s.items() if k != "window"}  # type: ignore[misc]
                        for s in action["stops"]
                    ]
                    self.outbox.append({**action, "stops": clean_stops})

    def telemetry(self) -> dict[str, Any]:
        """Structured live state of the bot's internal brain for the web dashboard."""
        day = self.now // DAY
        return {
            "level": self.level,
            "totalDrivers": self.net.total_drivers,
            "shortageMode": self.shortage_Mode,
            "worldPriceMult": round(self.world_price_mult + self.day_price_adj.get(day, 0.0), 3),
            "clearingFeeMult": round(self.fee_mult_est, 3),
            "activeDriversToday": len(self.active_drivers_today),
            "waitingUnbooked": self._active_unbooked_orders(),
            "waitingTotal": self._open_order_counts()[1],
            "slaHeadroomOrders": round(self._sla_headroom(), 1),
            "slaTier": self.sla_tier,
            "panicMode": self.panic_mode,
            "topTrustedDealers": sorted(
                [{"dealerId": k, "trust": v} for k, v in self.dealer_trust.items() if v > 0],
                key=lambda x: -x["trust"],
            )[:8],
            "snowForecasts": [
                {
                    "from": sw.start,
                    "to": sw.end,
                    "places": [self.net.place_name(p) for p in list(sw.places)[:6]],
                    "text": sw.text,
                }
                for sw in self.net.snow_windows[-5:]
            ],
            "decisions": list(self.decisions)[:60],
        }
