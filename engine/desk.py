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
    est_cost_cents: int
    at: int


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
        self.gig_driver: dict[str, str] = {}
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
            # Level 8 competitor baseline is 1.26x..1.54x list (midpoint ~1.42x)
            self.world_price_mult = 1.42

        if self.level == 1:
            self.fee_mult_est = 1.01
        elif self.shortage_Mode:
            # 0.35 ratio (~840 drivers) -> 1.00; 0.26 ratio (~625 drivers) -> 1.12
            self.fee_mult_est = min(1.22, max(0.98, 1.00 + max(0.0, 0.35 - self.shortage_ratio) * 1.25))
        else:
            self.fee_mult_est = 0.95

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

        # SLA tracking
        self.delivered_orders: int = 0
        self.on_time_orders: int = 0
        self.late_orders: int = 0

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
            self.active_drivers_today.clear()
            self.driver_offers_today.clear()
            self.region_recent_wins.clear()

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
            elif k == "gig_viewed":
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
        if self.now - self._last_sweep_min >= 20:
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
            self.active_drivers_today.add(event["driverId"])
        elif k == "gig_booked":
            self._on_gig_booked(event["gigId"], event["driverId"], event["feeCents"])
        elif k == "gig_started":
            self.started.add(event["gigId"])
        elif k in ("gig_dropped", "no_show"):
            self._on_gig_lost_driver(event["gigId"], event["driverId"], k)
        elif k == "driver_late":
            self._on_driver_late(event["gigId"], event["driverId"], event["pickupAt"])
        elif k == "gig_expired":
            self._on_gig_expired(event["gigId"])
        elif k == "offer_answered":
            self.gig_open_offers[event["gigId"]] = max(0, self.gig_open_offers.get(event["gigId"], 1) - 1)
        elif k == "visit_failed":
            self._on_visit_failed(event["orderId"], event["leg"], event["reason"])
        elif k == "delivered":
            self._on_delivered(event["orderId"], event["leg"], event["at"])
        elif k == "deadline_close":
            self._on_deadline_close(event["orderId"])
        elif k == "forecast":
            self.net.add_forecast(event["from"], ev_to := event["to"], event["placeIds"], event["text"])
            places_str = ", ".join(self.net.place_name(p) for p in event["placeIds"][:4])
            self._log("weather", "Snow Forecast", f"{places_str} ({fmt_min(event['from'])}–{fmt_min(ev_to)})", "warn")
        elif k == "ledger":
            self._ingest_ledger(event["line"])

    def _ingest_ledger(self, line: dict[str, Any]) -> None:
        oid = line.get("orderId")
        if not oid:
            return
        if line["kind"] in ("late_refund", "undelivered_penalty"):
            self.order_had_late_refund.add(oid)
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
            if oid in self.order_had_late_refund:
                self.late_orders += 1
                if did:
                    self.dealer_trust[did] = self.dealer_trust.get(did, 0) // 2
            else:
                self.on_time_orders += 1
                if did:
                    self.dealer_trust[did] = min(25, self.dealer_trust.get(did, 0) + 1)

    # ---------- Cost & Feasibility Modeling ----------
    def _active_unbooked_orders(self, near_place: Optional[str] = None) -> int:
        while self.pending_quotes and self.pending_quotes[0][0] <= self.now:
            self.pending_quotes.popleft()

        cnt = 0
        for oid, o in self.orders.items():
            if o["status"] != "waiting":
                continue
            gids = self.gigs_of(oid)
            if any(g in self.booked for g in gids):
                continue
            if near_place is None:
                cnt += 1
            else:
                p0 = o["legs"][0]["pickup"]["placeId"]
                if self.net.straight_km(p0, near_place) <= 40.0:
                    cnt += 1

        # Include expected wins from quotes currently in their 15-minute respondBy window so bulk bursts never blindside us!
        exp_win_w = 1.0 if self.level <= 3 else 0.45
        inflight = 0.0
        for _, p_q in self.pending_quotes:
            if near_place is None or self.net.straight_km(p_q, near_place) <= 40.0:
                inflight += exp_win_w
        return cnt + int(inflight)

    def _sla_headroom(self) -> float:
        """Positive = safe orders above 92.0% target buffer (ranking floor is 90.0%); counts jeopardized orders immediately."""
        at_risk = sum(
            1
            for oid, o in self.orders.items()
            if o["status"] == "waiting"
            and not any(g in self.booked for g in self.gigs_of(oid))
            and self.now >= o["lastPickupAt"] - 150
        )
        total = self.delivered_orders + at_risk
        if total < 10:
            return 3.0
        return self.on_time_orders - 0.922 * total

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
            mult *= min(1.46, (0.88 / max(0.12, supply_quality)) ** 0.25)
        elif supply_quality > 1.20:
            mult *= max(0.82, (1.18 / supply_quality) ** 0.18)

        if len(legs) == 1:
            if dest_notif >= 12 and local_notif >= 2:
                # Car heads straight into a driver hub -> homebound drivers take this cheaply!
                mult *= 0.92
            elif dest_notif <= 1:
                # Dead-end delivery town costs driver an expensive day-of return ticket
                mult *= 1.13

        if local_notif == 0:
            mult *= 1.18 if dest_notif >= 10 else 1.26
        elif local_notif <= 2:
            mult *= 1.08

        if experienced:
            mult *= 1.08
        if manual:
            mult *= 1.06
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
            self.net.straight_km(fin_place, p_first) <= 35.0 and self.now - 30 <= fin_at <= req["arrivedAt"] + 360
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
        if has_finisher or has_pair_order:
            mult *= 0.88

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
        # Future-ready & private-appointment pickups give drivers pre-booking notice on the board before `ready`!
        advance_notice_h = max(0.0, (ready - req["respondBy"]) / 60.0)
        return open_h + 0.65 * min(10.0, advance_notice_h)

    # ---------- Quoting Strategy ----------
    def _is_price_war_today(self, day: int) -> bool:
        if self.level < 8:
            return False
        tod = self.now % DAY
        if not (8 * HOUR <= tod <= 18 * HOUR + 15):
            return False
        hist = self.day_outcomes.get(day, [])
        return len(hist) >= 12 and (sum(hist[-12:]) / 12.0) <= 0.17 and self.day_price_adj.get(day, 0.0) <= -0.09

    def quote(self, req: PriceRequest) -> None:
        rid = req["id"]
        self.requests[rid] = req
        note = self._note_of(req)
        est_cost, audience, sup_q, dest_ratio = self._estimate_fulfillment_cost(req, note)

        p_first = req["legs"][0]["pickup"]["placeId"]
        d_last = req["legs"][-1]["delivery"]["placeId"]
        local_notif = self.net.matching_notif_count(p_first, audience, region=False)
        region_notif = self.net.matching_notif_count(p_first, audience, region=True)

        # Capacity & Bulk-Order Throttling (includes in-flight respondBy quotes!)
        global_waiting = self._active_unbooked_orders()
        local_waiting = self._active_unbooked_orders(p_first)
        if self.level == 1:
            global_cap = min(52, max(16, int(self.net.total_drivers * 0.25)))
            local_cap = max(2, int(max(local_notif, region_notif * 0.25) * 0.42))
        else:
            global_cap = max(14, int(self.net.total_drivers * (0.20 if self.shortage_Mode else 0.45)))
            local_cap = max(4, int(max(local_notif, region_notif * 0.35) * (0.42 if self.shortage_Mode else 0.75)))

        overloaded = global_waiting >= global_cap or local_waiting >= local_cap
        sla_head = self._sla_headroom()
        sla_tight = sla_head < 1.2
        sla_critical = sla_head < -0.5

        # Levels 1-3: Every quote wins first option at listPriceCents; no goodwill penalty for not quoting!
        if self.level <= 3:
            first_opt = req["options"][0]
            slack = self._option_slack_hours(req, first_opt, note)
            min_slack = (3.2 if self.level == 1 else 2.2) if not sla_tight else 4.5
            eff_l_cost = max(est_cost, int(req["referenceFeeCents"] * 0.96)) if self.level == 1 else est_cost
            min_margin = 650 if self.level == 1 else (500 if self.level == 2 else 350)
            if (
                slack >= min_slack
                and region_notif >= (2 if self.level == 1 else 1)
                and first_opt["listPriceCents"] >= eff_l_cost + min_margin
                and not overloaded
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

        # Levels 4-8: Competitor-aware, lane-difficulty-aware, multi-option pricing
        day = self.now // DAY
        day_adj = self.day_price_adj.get(day, 0.0)
        outage_boost = 0.0
        ob = self.place_outage_boost.get(p_first)
        if ob and ob[1] == day and 8 * HOUR <= (self.now % DAY) <= 18 * HOUR + 15:
            outage_boost = ob[0]

        dealer_id = req["dealerId"]
        # RULES.md line 164: +1% accepted price per on-time order at Level 8 (up to +25%)
        trust_bonus = (self.dealer_trust.get(dealer_id, 0) * 0.0095) if self.level >= 8 else 0.0
        d_hist = self.dealer_wins.get(dealer_id, [])
        dealer_win_rate = (sum(d_hist) / len(d_hist)) if len(d_hist) >= 4 else 0.45
        anti_war_pad = 0.05 if dealer_win_rate > 0.70 else (-0.02 if len(d_hist) >= 6 and dealer_win_rate < 0.14 else 0.0)

        # Empirical competitor lane factor: scales directly with refFee / listPrice[0] + remote destination scarcity!
        ref_to_list = req["referenceFeeCents"] / max(1, req["options"][0]["listPriceCents"])
        if ref_to_list < 0.78:
            # High-margin easy lane: quote sharper so we win 65-75% of these 50%+ margin layups!
            comp_lane_mult = max(0.90, 0.97 - (0.78 - ref_to_list) * 0.24)
        elif ref_to_list > 0.96:
            # Expensive lane: competitor prices noticeably higher here too
            comp_lane_mult = min(1.16, 1.00 + (ref_to_list - 0.96) * 0.40)
        else:
            comp_lane_mult = 1.00
        if dest_ratio < 0.55:
            comp_lane_mult *= 1.05

        # Bulk-order (`batchId`) progressive surge: win the best few cars normally, surge on the rest so 40-car bursts never swamp a yard!
        batch_surge = 0.0
        batch_cap_exceeded = False
        bid = req.get("batchId")
        if bid:
            b_idx = self.batch_counts.get(bid, 0)
            self.batch_counts[bid] = b_idx + 1
            step_rate = 0.006 if local_notif >= 80 else (0.012 if local_notif >= 15 else 0.022)
            batch_surge = min(0.55, b_idx * step_rate)
            max_batch_take = max(6, int(max(local_notif, region_notif * 0.25) * (0.55 if self.shortage_Mode else 0.95)))
            if b_idx >= max_batch_take:
                batch_cap_exceeded = True

        dest_res = self.dest_price_adj.get(d_last, 0.0)
        base_comp_mult = max(
            0.88,
            (self.world_price_mult + day_adj + outage_boost + dest_res + anti_war_pad)
            * comp_lane_mult
            * (1.0 + trust_bonus)
            * (1.0 + batch_surge),
        )

        has_multi_opts = len(req["options"]) > 1
        in_price_war = self._is_price_war_today(day)

        prices: dict[str, int] = {}
        best_summary = ""
        if not batch_cap_exceeded:
            for opt in req["options"]:
                kind = opt["kind"]
                list_c = opt["listPriceCents"]
                slack = self._option_slack_hours(req, opt, note)

                req_slack = 2.5 if not self.shortage_Mode else 3.5
                if kind == "same_day":
                    if self.shortage_Mode and local_notif == 0:
                        continue
                    if self.shortage_Mode and local_notif < 2 and slack < 5.6:
                        continue
                    req_slack = 4.3 if self.shortage_Mode else 2.9
                if local_notif == 0:
                    if self.shortage_Mode and slack < 9.2:
                        continue
                    req_slack += 2.8
                elif local_notif <= 2:
                    req_slack += 1.2
                if sla_tight:
                    req_slack += 1.2
                if sla_critical:
                    req_slack += 2.0
                    if local_notif < 2 or kind == "same_day":
                        continue
                if overloaded:
                    req_slack += 3.0

                if slack < req_slack or region_notif == 0:
                    continue

                opt_spread = (1.02 if kind == "same_day" else (0.99 if kind == "within_3_days" else 1.00)) if has_multi_opts else 1.00
                urgency_risk = 1.22 if kind == "same_day" else (1.13 if slack < 6.0 else 1.08)
                if overloaded:
                    urgency_risk += 0.16
                if sla_tight:
                    urgency_risk += 0.08
                if in_price_war:
                    urgency_risk += 0.08

                opt_cost = int(est_cost * (1.08 if kind == "same_day" else (0.94 if kind == "within_3_days" else 0.99)))
                floor_price = int(round(opt_cost * urgency_risk + (1400 if note.has_constraints else 600)))
                comp_price = int(round(list_c * base_comp_mult * opt_spread))
                q_price = max(floor_price, comp_price)

                if q_price <= list_c * 2.40:
                    prices[kind] = max(100, q_price)
                    if not best_summary:
                        best_summary = f"{kind} €{q_price/100:.0f} ({q_price/list_c:.2f}× list, est €{opt_cost/100:.0f}, {slack:.1f}h slack)"

        declined = False
        if not prices:
            declined = True
            safest = req["options"][-1]
            prices[safest["kind"]] = max(100, int(round(safest["listPriceCents"] * 4.5)))
            reason = "capacity full" if (overloaded or batch_cap_exceeded) else ("SLA shield" if sla_tight else f"tight slack / thin drivers ({local_notif} local)")
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
        ref_list = next(o["listPriceCents"] for o in req["options"] if o["kind"] == ref_k)
        norm_mult = (prices[ref_k] / max(1, ref_list)) / (max(0.85, comp_lane_mult) * (1.0 + batch_surge))
        self.quote_meta[rid] = QuoteMeta(
            request_id=rid,
            dealer_id=dealer_id,
            pickup_place=p_first,
            delivery_place=d_last,
            declined=declined,
            primary_mult=norm_mult,
            est_cost_cents=est_cost,
            at=self.now,
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
            return (1.26, 1.66)
        if self.level == 7:
            return (1.12, 1.46)
        return (0.88, 1.22)

    def _on_win(self, order: Order) -> None:
        oid = order["id"]
        meta = self.quote_meta.get(oid)
        if meta and not meta.declined and self.level >= 4:
            day = self.now // DAY
            self._record_dealer_outcome(order["dealerId"], 1)
            self.day_outcomes.setdefault(day, []).append(1)

            # Balanced quantile step targeting ~47% win rate on feasible requests (~380-410 won orders/week!)
            _, hi = self._world_mult_bounds()
            if meta.primary_mult >= self.world_price_mult - 0.12:
                self.world_price_mult = min(hi, self.world_price_mult + 0.011)
            self.day_price_adj[day] = min(0.18, self.day_price_adj.get(day, 0.0) + 0.007)
            self.dest_price_adj[meta.delivery_place] = min(
                0.12, self.dest_price_adj.get(meta.delivery_place, 0.0) + 0.009
            )

            # Regional Outage detection (08:00-18:00 within 80 km): trigger only on genuinely elevated wins!
            if self.level >= 6 and 8 * HOUR <= (self.now % DAY) <= 18 * HOUR:
                rw = self.region_recent_wins.setdefault(meta.pickup_place, [])
                rw.append(1)
                if len(rw) > 6:
                    del rw[0]
                high_win = meta.primary_mult >= (self.world_price_mult + 0.05)
                streak = len(rw) >= 4 and sum(rw[-4:]) == 4
                if high_win or streak:
                    bump = 0.045 if streak else 0.025
                    for pid in self.net.places:
                        if self.net.straight_km(meta.pickup_place, pid) <= 80.0:
                            cur_b, cur_d = self.place_outage_boost.get(pid, (0.0, day))
                            nb = min(0.28, (cur_b if cur_d == day else 0.0) + bump)
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
        if not meta or meta.declined or self.level <= 3:
            return
        day = self.now // DAY
        self._record_dealer_outcome(meta.dealer_id, 0)
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
                        self.place_outage_boost[pid] = (max(0.0, cur_b - 0.045), day)

        in_war = self._is_price_war_today(day)
        lo, _ = self._world_mult_bounds()
        if not in_war and meta.primary_mult <= self.world_price_mult + 0.14:
            self.world_price_mult = max(lo, self.world_price_mult - 0.010)
        self.day_price_adj[day] = max(-0.14 if in_war else -0.16, self.day_price_adj.get(day, 0.0) - 0.007)
        if not in_war:
            self.dest_price_adj[meta.delivery_place] = max(
                -0.10, self.dest_price_adj.get(meta.delivery_place, 0.0) - 0.008
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

    def _build_stops(self, order: Order, legs: list[int], allow_windows: bool = True) -> tuple[list[StopRef], Audience]:
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
                if pick["privateAddress"]:
                    p_end = min(p_end, pick["window"]["to"])
                elif note.pickup_hours is not None:
                    # CRITICAL FIX: if a note narrows daily pickup hours (e.g. "Nouto arkisin 10-17" when stated is 08-18),
                    # a multi-day [Day2 10:00, Day5 17:00] window still allows drivers at Day3 08:00 -> `closed` failure!
                    # Pin the stop window to the next single open day's [eff_open, eff_close] so every minute in window is truly open.
                    search_from = max(self.now + 75, p_start)
                    for d_off in range(0, 6):
                        d_idx = (search_from // DAY) + d_off
                        st_ph = site_day_hours(pick.get("hours"), d_idx)
                        if st_ph is None:
                            continue
                        if (d_idx % 7) < 5:
                            eff_o = max(st_ph[0], note.pickup_hours[0] if note.pickup_hours[0] > 0 else st_ph[0])
                            eff_c = min(st_ph[1], note.pickup_hours[1])
                        else:
                            eff_o, eff_c = st_ph
                        cand_s = max(p_start, d_idx * DAY + eff_o)
                        cand_e = min(p_end, d_idx * DAY + eff_c)
                        if cand_e >= max(cand_s + 30, self.now + 90):
                            p_start, p_end = cand_s, cand_e
                            break

                if p_start <= p_end and p_end > self.now + 25 and (p_start > pick["window"]["from"] or p_end < pick["window"]["to"]):
                    p_stop["window"] = {"from": int(p_start), "to": int(p_end)}

                d_base: TimeWindow = {"from": deliv["window"]["from"], "to": max(deliv["window"]["from"], order["deadline"])}
                d_win = apply_note_to_window(note, d_base, "delivery")
                if note.delivery_hours is not None and not deliv["privateAddress"]:
                    # Ensure single-day or final-day delivery doesn't exceed note's closing hour
                    p_day = (p_start // DAY)
                    drive_est = self.net.minutes(pick["placeId"], deliv["placeId"]) + 30
                    arr_day = (max(self.now, p_start) + drive_est) // DAY
                    st_dh = site_day_hours(deliv.get("hours"), arr_day)
                    if st_dh is not None and (arr_day % 7) < 5:
                        eff_do = max(st_dh[0], note.delivery_hours[0] if note.delivery_hours[0] > 0 else st_dh[0])
                        eff_dc = min(st_dh[1], note.delivery_hours[1])
                        cand_ds = max(d_win["from"], arr_day * DAY + eff_do)
                        cand_de = min(d_win["to"], arr_day * DAY + eff_dc)
                        if cand_de >= max(cand_ds + 30, max(self.now, p_start) + drive_est + 20):
                            d_win = {"from": cand_ds, "to": cand_de}

                if d_win["from"] <= d_win["to"] and d_win["to"] > self.now + 45 and (d_win["from"] > d_base["from"] or d_win["to"] < d_base["to"]):
                    d_stop["window"] = {"from": int(d_win["from"]), "to": int(d_win["to"])}

            stops.extend([p_stop, d_stop])

        return stops, audience

    def _sweep_aging_orders(self) -> None:
        """Proactively boosts unbooked gigs at 5.5 hours before lastPickupAt so drivers still have transit time."""
        for oid, o in list(self.orders.items()):
            if o["status"] != "waiting" or oid in self.urgent or oid in self.early_boosted:
                continue
            rem = o["lastPickupAt"] - self.now
            if rem > 330 or rem < -60:
                continue
            gids = self.gigs_of(oid)
            if any(g in self.booked for g in gids):
                continue
            self.early_boosted.add(oid)
            if self.level == 1:
                self.urgent.add(oid)
                if self._withdraw_order_gigs(oid):
                    self.post_order_legs(o, self._undelivered(o), allow_windows=False, allow_chain=False)
                continue
            for gid in list(gids):
                act = self.posts.get(gid)
                if not act or gid in self.booked:
                    continue
                order_set = list(dict.fromkeys(s["orderId"] for s in act["stops"]))
                if len(order_set) > 1:
                    self.outbox.append({"type": "withdraw", "gigId": gid})
                    self.posts.pop(gid, None)
                    for s in act["stops"]:
                        self.leg_gigs.get(s["orderId"], {}).pop(s["leg"], None)
                    for oid2 in order_set:
                        o2 = self.orders.get(oid2)
                        if o2 and o2["status"] != "cancelled":
                            self.post_order_legs(o2, self._undelivered(o2), allow_windows=False, allow_chain=False)
                    continue
                cars = sum(1 for s in act["stops"] if s["kind"] == "pickup")
                mid_sched = self._monotonic_steps(
                    self._estimate_fulfillment_cost(self.requests[oid], self._note_of(self.requests[oid]))[0]
                    * (cars / max(1, len(o["legs"]))),
                    [1.18, 1.34, 1.52, 1.72],
                    step_mins=35,
                )
                self.outbox.append({"type": "reprice", "gigId": gid, "schedule": mid_sched})
                act["schedule"] = mid_sched
                self._maybe_send_offers(gid, act, urgent=True)

    def _schedule_for(self, order: Order, legs_count: int, force_urgent: bool = False) -> list[FeeStep]:
        oid = order["id"]
        req = self.requests[oid]
        note = self._note_of(req)
        est_total, _, _, _ = self._estimate_fulfillment_cost(req, note)
        base = est_total * (legs_count / max(1, len(order["legs"])))

        time_to_last = order["lastPickupAt"] - self.now
        is_urg = force_urgent or (oid in self.urgent) or (time_to_last <= 240)
        is_late = self.now >= order["lastPickupAt"]

        if self.level == 1:
            # Level 1 allows only 1 flat fee step (no reprice, no offer), so open right near typical clearing fee!
            ref_part = req["referenceFeeCents"] * (legs_count / max(1, len(order["legs"])))
            flat_base = max(base, ref_part * 0.96)
            mult = 1.45 if is_urg else 1.00
            return [{"at": int(self.now), "feeCents": max(100, int(round(flat_base * mult)))}]

        if is_late:
            return self._monotonic_steps(base, [1.40, 1.65, 1.90, 2.20], step_mins=50)

        if is_urg:
            steps = [1.36, 1.56, 1.80, 2.08, 2.38]
            return self._monotonic_steps(base, steps, step_mins=30)

        # Reliable clearing schedule: opens sharp so cheap drivers take it immediately, steps high enough so 95%+ deliver on time!
        steps = [0.98, 1.08, 1.19, 1.32] if self.shortage_Mode else [0.95, 1.03, 1.12, 1.22]
        interval = 70 if time_to_last > 480 else 45
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
        """Pairs two compatible 1-leg orders (checking BOTH A->B and B->A directions!) into a cheap 2-car chain."""
        if self.level < 3 or len(new_order["legs"]) != 1 or new_order["id"] in self.urgent:
            return False
        if new_order["lastPickupAt"] - self.now < 450:
            return False

        b_pick = new_order["legs"][0]["pickup"]["placeId"]
        b_del = new_order["legs"][0]["delivery"]["placeId"]
        if new_order["legs"][0]["pickup"]["privateAddress"] or new_order["legs"][0]["delivery"]["privateAddress"]:
            return False

        best_match: Optional[tuple[float, str, Order, list[StopRef], Audience]] = None

        for gid, act in list(self.posts.items()):
            if gid in self.booked or len(act["stops"]) != 2:
                continue
            oid_a = act["stops"][0]["orderId"]
            if oid_a == new_order["id"] or oid_a in self.urgent:
                continue
            ord_a = self.orders.get(oid_a)
            if not ord_a or ord_a["status"] != "waiting" or len(ord_a["legs"]) != 1:
                continue
            if self.leg_gigs.get(oid_a, {}).get(0) != gid:
                continue
            if ord_a["lastPickupAt"] - self.now < 450:
                continue
            if ord_a["legs"][0]["pickup"]["privateAddress"] or ord_a["legs"][0]["delivery"]["privateAddress"]:
                continue

            a_pick = ord_a["legs"][0]["pickup"]["placeId"]
            a_del = ord_a["legs"][0]["delivery"]["placeId"]
            comb_aud: Audience = {
                "experienced": bool(act["audience"]["experienced"] or new_aud["experienced"]),
                "manualLicence": bool(act["audience"]["manualLicence"] or new_aud["manualLicence"]),
            }

            # Try both directions: (ord_a -> new_order) AND (new_order -> ord_a)
            for first_o, second_o, first_stops, second_stops, f_pick, f_del, s_pick, s_del in (
                (ord_a, new_order, list(act["stops"]), list(new_stops), a_pick, a_del, b_pick, b_del),
                (new_order, ord_a, list(new_stops), list(act["stops"]), b_pick, b_del, a_pick, a_del),
            ):
                conn = self.net.minutes(f_del, s_pick)
                if conn > 40:
                    continue
                cand_stops = first_stops + second_stops
                total_road = self._stops_road_minutes(cand_stops)
                if total_road > 630:
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
                if f_arr > second_o["lastPickupAt"] - 160 or s_ready > f_arr + 150:
                    continue

                # Prefer true round-trips (s_del near f_pick) because driver needs ZERO public transit tickets!
                return_gap_km = self.net.straight_km(s_del, f_pick)
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

        # Round-trip eliminates BOTH transit tickets -> 0.89x sum; corridor eliminates 1 ticket -> 0.93x sum!
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

    def post_order_legs(self, order: Order, legs: list[int], allow_windows: bool = True, allow_chain: bool = True) -> None:
        if not legs:
            return
        stops, audience = self._build_stops(order, legs, allow_windows=allow_windows)

        if len(legs) > 1 and self._stops_road_minutes(stops) > 715:
            for lg in legs:
                self.post_order_legs(order, [lg], allow_windows=allow_windows, allow_chain=False)
            return

        if len(legs) == 1 and allow_chain and self._try_chain_with_open_gig(order, stops, audience):
            return

        sched = self._schedule_for(order, len(legs))
        self.outbox.append({"type": "post", "stops": stops, "audience": audience, "schedule": sched})

    # ---------- Direct Offers (Finishers, Homebound Hub Drivers & Urgent Rescue) ----------
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

        candidates = self.net.candidate_drivers(
            p_place,
            d_place,
            action["audience"],
            self.finishing_places,
            self.active_drivers_today,
            self.now,
            lead_mins=lead_mins,
        )
        if not candidates:
            return

        local_cnt = self.net.matching_notif_count(p_place, action["audience"], region=False)
        top_score, _, top_kind = candidates[0]
        # Even in busy towns, ALWAYS snipe finishers & active homebound drivers because they get NO push notification and take cheaper fees!
        if not urgent and top_kind == "local" and local_cnt >= 5 and top_score < 32.0:
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
            if not urgent and c_kind == "local" and local_cnt >= 5 and score < 32.0:
                continue
            if (gig_id, did) in self.offered_pairs:
                continue
            if self.driver_offers_today.get(did, 0) >= 3:
                continue

            if urgent:
                mult = 1.08
            elif c_kind == "finisher":
                mult = 0.92
            elif c_kind == "homebound":
                mult = 0.94
            else:
                mult = 0.99
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
            est_fin = max(self.now, p_from) + self._stops_road_minutes(act["stops"]) + 45
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
                if self.net.straight_km(dest_place, op_pick) <= 40.0 and est_fin <= fo["lastPickupAt"] - 60:
                    self._maybe_send_offers(open_gid, open_act, urgent=False)

        # Online update of empirical clearing fee multiplier (excluding emergency rescues & 2-car discounted chains)
        order_ids = list(dict.fromkeys(s["orderId"] for s in act["stops"]))
        if len(order_ids) == 1 and order_ids[0] not in self.urgent:
            oid = order_ids[0]
            req = self.requests.get(oid)
            ord_o = self.orders.get(oid)
            if req and ord_o:
                cars = sum(1 for s in act["stops"] if s["kind"] == "pickup")
                ref_sum = req["referenceFeeCents"] * (cars / max(1, len(ord_o["legs"])))
                if ref_sum > 0:
                    obs_mult = fee_cents / ref_sum
                    if 0.60 <= obs_mult <= 1.85:
                        self.fee_mult_est = 0.92 * self.fee_mult_est + 0.08 * obs_mult

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
                self.post_order_legs(o, self._undelivered(o), allow_chain=False)
            return

        primary = orders_here[0]
        force_urg = (min_slack <= 270) or (kind == "no_show")
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
                    self.post_order_legs(ord_obj, self._undelivered(ord_obj), allow_chain=False)
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
                    # Keep narrowed daily hours if we still have time before lastPickupAt; otherwise clear windows so late cars still deliver!
                    req = self.requests.get(oid)
                    note = self._note_of(req) if req else ParsedNote()
                    keep_win = bool(note.pickup_hours and self.now < o["lastPickupAt"] - 120)
                    self.post_order_legs(o, missing, allow_windows=keep_win, allow_chain=False)
                    self._log("recover", f"Reposted Expired Gig for {oid}", "Advanced window for next open day" if keep_win else "Cleared custom stop windows", "warn")

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
                        eff_o = max(sh[0], note.pickup_hours[0] if (note.pickup_hours and note.pickup_hours[0] > 0 and (d_idx % 7) < 5) else sh[0])
                        eff_c = min(sh[1], note.pickup_hours[1] if (note.pickup_hours and (d_idx % 7) < 5) else sh[1])
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
        self.post_order_legs(o, [leg], allow_windows=(reason in ("not_ready", "closed")), allow_chain=False)
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
        self.leg_gigs.get(order_id, {}).pop(leg, None)
        if order_id in self.orders:
            self.orders[order_id]["deliveredAt"][leg] = at
            if all(d is not None for d in self.orders[order_id]["deliveredAt"]):
                self.orders[order_id]["status"] = "delivered"

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
                self.post_order_legs(o, missing, allow_windows=False, allow_chain=False)
            return

        if self.level == 1:
            # Level 1 prohibits `reprice`, so withdraw & repost at urgent fee!
            if self._withdraw_order_gigs(order_id):
                self.post_order_legs(o, self._undelivered(o), allow_windows=False, allow_chain=False)
            return

        for gid in list(gids):
            if gid in self.booked:
                continue
            act = self.posts.get(gid)
            if not act:
                continue
            order_set = list(dict.fromkeys(s["orderId"] for s in act["stops"]))
            if len(order_set) > 1:
                self.outbox.append({"type": "withdraw", "gigId": gid})
                self.posts.pop(gid, None)
                for s in act["stops"]:
                    self.leg_gigs.get(s["orderId"], {}).pop(s["leg"], None)
                for oid2 in order_set:
                    o2 = self.orders.get(oid2)
                    if o2 and o2["status"] != "cancelled":
                        self.post_order_legs(o2, self._undelivered(o2), allow_windows=False, allow_chain=False)
                continue

            cars = sum(1 for s in act["stops"] if s["kind"] == "pickup")
            steep_sched = self._schedule_for(o, cars, force_urgent=True)
            self.outbox.append({"type": "reprice", "gigId": gid, "schedule": steep_sched})
            act["schedule"] = steep_sched
            self._maybe_send_offers(gid, act, urgent=True)

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
                is_urg = first_oid in self.urgent
                self._maybe_send_offers(gid, action, urgent=is_urg)
            else:
                had_win = any("window" in s for s in action["stops"])
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
                            self.post_order_legs(o, [s["leg"]], allow_windows=False, allow_chain=False)
                elif had_win:
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
            "slaHeadroomOrders": round(self._sla_headroom(), 1),
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
