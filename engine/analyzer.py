"""Post-run forensic and visual analytics engine for single and multi-week (5-week) Spread runs."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import statistics
from typing import Any, Optional

from spread.client import Client, SpreadError

from .namer import enrich_worlds, get_world_name, record_world_score
from .network import NetworkIndex
from .notes import fmt_min, parse_note

HOUR, DAY = 60, 24 * 60
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Mon (W2)", "Tue (W2)", "Wed (W2)"]


def _fetch_board_benchmark(client: Client, level: int, projected_5_euros: float) -> dict[str, Any]:
    board_benchmark: dict[str, Any] = {
        "top1PerRun": None,
        "top10PerRun": None,
        "projected5Run": round(projected_5_euros, 2),
    }
    try:
        bd = client.board(level)
        ranked_teams = [
            t for t in bd.get("current", {}).get("teams", [])
            if t.get("ranked") and t.get("played", 0) > 0 and not str(t.get("team", "")).startswith("house-")
        ]
        if ranked_teams:
            t1 = ranked_teams[0]
            board_benchmark["top1PerRun"] = round(t1["marginCents"] / t1["played"] / 100, 2)
            board_benchmark["top1Name"] = t1["name"]
            t10 = ranked_teams[min(9, len(ranked_teams) - 1)]
            board_benchmark["top10PerRun"] = round(t10["marginCents"] / t10["played"] / 100, 2)
            board_benchmark["top10Name"] = t10["name"]
    except Exception:
        pass
    return board_benchmark


def analyze_world(
    client: Client,
    world_id: str,
    world_map: dict[str, Any],
    worlds_list: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Computes comprehensive P&L, quoting, driver-fee, corridor, and note-truth analytics for a single world."""
    raw_worlds = worlds_list if worlds_list is not None else client.worlds()
    worlds = enrich_worlds(raw_worlds)
    w_meta = next((w for w in worlds if w["worldId"] == world_id), None)
    level = w_meta["level"] if w_meta else 8
    kind = w_meta["kind"] if w_meta else "practice"
    board_run = w_meta.get("boardRun") if w_meta else None
    world_name = w_meta.get("name") if w_meta else get_world_name(world_id)
    seed_name = w_meta.get("seed") if w_meta else None
    batch_id = w_meta.get("batchId") if w_meta else None
    batch_name = w_meta.get("batchName") if w_meta else None

    w = client.world(world_id)
    snap = w.snapshot()
    evs = w.events()
    try:
        roster = w.roster()
    except Exception:
        roster = []

    record_world_score(world_id, snap["score"])
    net = NetworkIndex(world_map, roster)  # type: ignore[arg-type]

    # Try loading post-game reveal (available on ended non-ranking practice worlds)
    reveal: Optional[dict[str, Any]] = None
    if snap["now"] >= snap["endsAt"] and not board_run:
        try:
            reveal = w.reveal()  # type: ignore[assignment]
        except SpreadError:
            reveal = None

    board_benchmark = _fetch_board_benchmark(client, level, snap["score"]["marginCents"] * 5 / 100)

    # --- Index requests, orders, gigs, events, ledger ---
    requests_by_id: dict[str, dict[str, Any]] = {}
    orders_by_id: dict[str, dict[str, Any]] = {o["id"]: o for o in snap.get("orders", [])}
    gigs_by_id: dict[str, dict[str, Any]] = {g["id"]: g for g in snap.get("gigs", [])}

    ev_counts: Counter[str] = Counter()
    failed_by_reason: Counter[str] = Counter()
    failed_by_order: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ledger_totals: Counter[str] = Counter()
    ledger_counts: Counter[str] = Counter()
    order_ledger: dict[str, Counter[str]] = defaultdict(Counter)
    gig_booked_fee: dict[str, int] = {}
    gig_first_view_fee: dict[str, int] = {}
    drops_by_day: Counter[int] = Counter()

    # 4-hour timeline buckets (0 .. endsAt)
    bucket_mins = 240
    n_buckets = max(1, (snap["endsAt"] // bucket_mins) + 1)
    tl_rev = [0] * n_buckets
    tl_fee = [0] * n_buckets
    tl_pen = [0] * n_buckets
    tl_ontime = [0] * n_buckets
    tl_late = [0] * n_buckets
    tl_drops = [0] * n_buckets

    order_had_late_line: set[str] = set()
    for e in evs:
        if e["kind"] == "ledger" and e["line"]["kind"] in ("late_refund", "undelivered_penalty"):
            if e["line"].get("orderId"):
                order_had_late_line.add(e["line"]["orderId"])

    scored_orders_seen: set[str] = set()
    for e in evs:
        k = e["kind"]
        at = e["at"]
        b_idx = min(n_buckets - 1, max(0, at // bucket_mins))
        ev_counts[k] += 1

        if k == "price_request":
            rq = e["request"]
            requests_by_id[rq["id"]] = rq
        elif k == "order_won":
            orders_by_id.setdefault(e["order"]["id"], e["order"])
        elif k == "gig_viewed":
            gid = e["gigId"]
            if gid not in gig_first_view_fee and "feeCents" in e:
                gig_first_view_fee[gid] = e["feeCents"]
        elif k == "gig_booked":
            gig_booked_fee[e["gigId"]] = e["feeCents"]
        elif k == "gig_dropped":
            drops_by_day[at // DAY] += 1
            tl_drops[b_idx] += 1
        elif k == "visit_failed":
            failed_by_reason[e["reason"]] += 1
            failed_by_order[e["orderId"]].append({"at": at, "leg": e["leg"], "reason": e["reason"]})
        elif k == "ledger":
            ln = e["line"]
            lk = ln["kind"]
            cents = ln["cents"]
            ledger_totals[lk] += cents
            ledger_counts[lk] += 1
            oid = ln.get("orderId")
            if oid:
                order_ledger[oid][lk] += cents
            elif ln.get("gigId") and ln["gigId"] in gigs_by_id:
                g_obj = gigs_by_id[ln["gigId"]]
                g_oids = [
                    o2 for o2 in dict.fromkeys(s["orderId"] for s in g_obj.get("stops", []))
                    if o2 in orders_by_id and orders_by_id[o2]["status"] != "cancelled"
                ]
                if not g_oids:
                    g_oids = list(dict.fromkeys(s["orderId"] for s in g_obj.get("stops", [])))
                if g_oids:
                    weights = [max(100, requests_by_id.get(o2, {}).get("referenceFeeCents", 5000)) for o2 in g_oids]
                    tot_w = sum(weights)
                    rem_c = cents
                    for idx_o, o2 in enumerate(g_oids):
                        share = rem_c if idx_o == len(g_oids) - 1 else int(round(cents * weights[idx_o] / tot_w))
                        rem_c -= share
                        order_ledger[o2][lk] += share
            if lk == "revenue":
                tl_rev[b_idx] += cents
                if oid and oid not in scored_orders_seen:
                    scored_orders_seen.add(oid)
                    if oid in order_had_late_line:
                        tl_late[b_idx] += 1
                    else:
                        tl_ontime[b_idx] += 1
            elif lk == "driver_fee":
                tl_fee[b_idx] += cents
            else:
                tl_pen[b_idx] += cents

    if reveal:
        for v in reveal.get("views", []):
            gid = v["gigId"]
            if gid not in gig_first_view_fee:
                gig_first_view_fee[gid] = v["feeCents"]
        for r_it in reveal.get("requests", []):
            rq = r_it["request"]
            requests_by_id[rq["id"]] = rq

    # Build cumulative timeline series
    timeline: list[dict[str, Any]] = []
    cum_rev = cum_fee = cum_pen = cum_ot = cum_lt = 0
    for idx in range(n_buckets):
        t_min = min(snap["endsAt"], (idx + 1) * bucket_mins)
        if t_min > snap["now"] and idx > 0 and (idx * bucket_mins) > snap["now"]:
            break
        cum_rev += tl_rev[idx]
        cum_fee += tl_fee[idx]
        cum_pen += tl_pen[idx]
        cum_ot += tl_ontime[idx]
        cum_lt += tl_late[idx]
        tot_del = cum_ot + cum_lt
        timeline.append({
            "minute": t_min,
            "label": fmt_min(t_min),
            "cumMarginEuros": round((cum_rev + cum_fee + cum_pen) / 100, 2),
            "cumRevenueEuros": round(cum_rev / 100, 2),
            "cumDriverFeeEuros": round(-cum_fee / 100, 2),
            "cumPenaltiesEuros": round(-cum_pen / 100, 2),
            "onTimePct": round((cum_ot / tot_del * 100), 2) if tot_del > 0 else 100.0,
            "cumOnTimeOrders": cum_ot,
            "cumTotalOrders": tot_del,
            "drops": tl_drops[idx],
        })

    # P&L Waterfall
    waterfall = [
        {"key": "revenue", "label": "Gross Order Revenue", "euros": round(ledger_totals["revenue"] / 100, 2), "count": ledger_counts["revenue"], "type": "pos"},
        {"key": "driver_fee", "label": "Driver Gig Fees", "euros": round(ledger_totals["driver_fee"] / 100, 2), "count": ledger_counts["driver_fee"], "type": "neg"},
        {"key": "late_refund", "label": "Late Delivery Refunds", "euros": round(ledger_totals["late_refund"] / 100, 2), "count": ledger_counts["late_refund"], "type": "neg"},
        {"key": "undelivered_penalty", "label": "Undelivered 2× Penalties", "euros": round(ledger_totals["undelivered_penalty"] / 100, 2), "count": ledger_counts["undelivered_penalty"], "type": "neg"},
        {"key": "failed_visit", "label": "Failed Visit Fees", "euros": round(ledger_totals["failed_visit"] / 100, 2), "count": ledger_counts["failed_visit"], "type": "neg"},
        {"key": "fine", "label": "Compliance Fines (EV/Prem)", "euros": round(ledger_totals["fine"] / 100, 2), "count": ledger_counts["fine"], "type": "neg"},
        {"key": "release_fee", "label": "Driver Release Fees", "euros": round(ledger_totals["release_fee"] / 100, 2), "count": ledger_counts["release_fee"], "type": "neg"},
        {"key": "waiting", "label": "Closed-Site Waiting (€15/h)", "euros": round(ledger_totals["waiting"] / 100, 2), "count": ledger_counts["waiting"], "type": "neg"},
        {"key": "net", "label": "Net Gross Margin", "euros": round(snap["score"]["marginCents"] / 100, 2), "count": snap["score"]["wonOrders"], "type": "total"},
    ]

    # --- Quoting, Daily Regimes & Option Breakdown ---
    daily_stats: dict[int, dict[str, Any]] = {
        d: {"day": d, "name": DAY_NAMES[d], "requests": 0, "declined": 0, "quoted": 0, "won": 0, "winMults": [], "lossMults": [], "marginCents": 0, "drops": drops_by_day[d], "badges": []}
        for d in range(5)
    }
    quote_bins: dict[float, dict[str, Any]] = {
        round(0.9 + i * 0.1, 1): {"mult": round(0.9 + i * 0.1, 1), "quoted": 0, "won": 0, "marginCents": 0}
        for i in range(15)
    }
    option_stats: dict[str, dict[str, Any]] = {
        k: {"option": k, "won": 0, "onTime": 0, "revenueCents": 0, "feeCents": 0, "penaltyCents": 0, "marginCents": 0, "priceToList": [], "feeToRef": []}
        for k in ("same_day", "next_day", "within_3_days")
    }

    day_place_reqs: dict[tuple[int, str], int] = Counter()
    day_place_high_wins: dict[tuple[int, str], int] = Counter()

    if reveal:
        for item in reveal.get("requests", []):
            rq = item["request"]
            rid = rq["id"]
            d = min(4, max(0, rq["arrivedAt"] // DAY))
            p0 = rq["legs"][0]["pickup"]["placeId"]
            day_place_reqs[(d, p0)] += 1
            daily_stats[d]["requests"] += 1

            q = item.get("quote") or {}
            if not q:
                daily_stats[d]["declined"] += 1
                continue
            ref_k = "next_day" if "next_day" in q else next(iter(q))
            list_c = next((o["listPriceCents"] for o in rq["options"] if o["kind"] == ref_k), rq["options"][0]["listPriceCents"])
            mult = q[ref_k] / max(1, list_c)
            if mult >= 3.0:
                daily_stats[d]["declined"] += 1
                continue

            daily_stats[d]["quoted"] += 1
            b_key = min(2.3, max(0.9, round(mult, 1)))
            quote_bins[b_key]["quoted"] += 1

            if item["outcome"] == "won":
                if rid in orders_by_id and orders_by_id[rid]["status"] == "cancelled":
                    continue
                daily_stats[d]["won"] += 1
                daily_stats[d]["winMults"].append(mult)
                ord_net = sum(order_ledger[rid].values())
                daily_stats[d]["marginCents"] += ord_net
                quote_bins[b_key]["won"] += 1
                quote_bins[b_key]["marginCents"] += ord_net
                if mult >= (1.52 if level >= 8 else 1.18):
                    day_place_high_wins[(d, p0)] += 1
            elif item["outcome"] == "lost":
                daily_stats[d]["lossMults"].append(mult)
    else:
        for rid, rq in requests_by_id.items():
            d = min(4, max(0, rq["arrivedAt"] // DAY))
            p0 = rq["legs"][0]["pickup"]["placeId"]
            day_place_reqs[(d, p0)] += 1
            daily_stats[d]["requests"] += 1
            if rid in orders_by_id and orders_by_id[rid]["status"] != "cancelled":
                o = orders_by_id[rid]
                daily_stats[d]["won"] += 1
                opt_list = next((op["listPriceCents"] for op in rq["options"] if op["kind"] == o["option"]), rq["options"][0]["listPriceCents"])
                m = o["priceCents"] / max(1, opt_list)
                daily_stats[d]["winMults"].append(m)
                daily_stats[d]["marginCents"] += sum(order_ledger[rid].values())

    for (d, pid), cnt in day_place_reqs.items():
        if cnt >= 28:
            daily_stats[d]["badges"].append(f"📦 Bulk Order: {cnt} cars at {net.place_name(pid)}")
    for d in range(5):
        ds = daily_stats[d]
        wm = statistics.mean(ds["winMults"]) if ds["winMults"] else 0.0
        lm = statistics.mean(ds["lossMults"]) if ds["lossMults"] else 0.0
        wr = (ds["won"] / ds["quoted"] * 100) if ds["quoted"] else (ds["won"] / max(1, ds["requests"]) * 100)
        ds["avgWinMult"] = round(wm, 3)
        ds["avgLossMult"] = round(lm, 3)
        ds["winRatePct"] = round(wr, 1)
        ds["marginEuros"] = round(ds["marginCents"] / 100, 2)
        del ds["winMults"], ds["lossMults"], ds["marginCents"]

        top_outage = max(((cnt, pid) for (dd, pid), cnt in day_place_high_wins.items() if dd == d), default=(0, ""))
        if top_outage[0] >= 8:
            ds["badges"].append(f"⚡ Outage Captured near {net.place_name(top_outage[1])} ({top_outage[0]} high-mult wins)")
        if level >= 8 and ((ds["avgLossMult"] > 0 and ds["avgLossMult"] < 1.38 and wr < 34.0) or (ds["avgWinMult"] > 0 and ds["avgWinMult"] < 1.24)):
            ds["badges"].append("⚔️ Competitor Price War (-20%)")
        if ds["drops"] >= 18:
            ds["badges"].append(f"🤒 High Driver Drops ({ds['drops']} drops — Flu/Snow)")

    # --- Per-Order, Corridor & Town Forensics ---
    corridors: dict[tuple[str, str], dict[str, Any]] = {}
    town_stats: dict[str, dict[str, Any]] = {
        pid: {
            "id": pid,
            "name": p["name"],
            "lat": p["lat"],
            "lon": p["lon"],
            "localDrivers": len(net.notif_drivers.get(pid, [])),
            "pickups": 0,
            "deliveries": 0,
            "marginCents": 0,
            "lateOrders": 0,
            "feeRatios": [],
        }
        for pid, p in net.places.items()
    }

    order_rows: list[dict[str, Any]] = []
    fee_to_ref_all: list[float] = []

    rev_orders_truth: dict[str, dict[str, Any]] = {}
    if reveal:
        rev_orders_truth = {o["orderId"]: o for o in reveal.get("orders", [])}

    for oid, o in orders_by_id.items():
        if o["status"] == "cancelled" or oid not in requests_by_id:
            continue
        rq = requests_by_id[oid]
        led = order_ledger[oid]
        rev_c = led["revenue"]
        fee_c = led["driver_fee"]
        pen_c = sum(v for k, v in led.items() if k not in ("revenue", "driver_fee"))
        net_c = rev_c + fee_c + pen_c

        p_id = o["legs"][0]["pickup"]["placeId"]
        d_id = o["legs"][-1]["delivery"]["placeId"]
        opt = o["option"]
        opt_list = next((op["listPriceCents"] for op in rq["options"] if op["kind"] == opt), rq["options"][0]["listPriceCents"])
        ref_fee = rq["referenceFeeCents"]
        p_to_list = o["priceCents"] / max(1, opt_list)
        f_to_ref = (-fee_c / max(1, ref_fee)) if fee_c < 0 else 0.0
        if f_to_ref > 0:
            fee_to_ref_all.append(f_to_ref)

        is_late = oid in order_had_late_line or any(d is None or d > o["deadline"] for d in o["deliveredAt"])

        if opt in option_stats:
            os_d = option_stats[opt]
            os_d["won"] += 1
            if not is_late:
                os_d["onTime"] += 1
            os_d["revenueCents"] += rev_c
            os_d["feeCents"] += fee_c
            os_d["penaltyCents"] += pen_c
            os_d["marginCents"] += net_c
            os_d["priceToList"].append(p_to_list)
            if f_to_ref > 0:
                os_d["feeToRef"].append(f_to_ref)

        ckey = (p_id, d_id)
        if ckey not in corridors:
            corridors[ckey] = {
                "fromId": p_id,
                "toId": d_id,
                "route": f"{net.place_name(p_id)} → {net.place_name(d_id)}",
                "km": round(net.km(p_id, d_id), 0),
                "orders": 0,
                "late": 0,
                "revenueCents": 0,
                "feeCents": 0,
                "marginCents": 0,
            }
        cd = corridors[ckey]
        cd["orders"] += 1
        if is_late:
            cd["late"] += 1
        cd["revenueCents"] += rev_c
        cd["feeCents"] += fee_c
        cd["marginCents"] += net_c

        if p_id in town_stats:
            town_stats[p_id]["pickups"] += 1
            town_stats[p_id]["marginCents"] += net_c
            if is_late:
                town_stats[p_id]["lateOrders"] += 1
            if f_to_ref > 0:
                town_stats[p_id]["feeRatios"].append(f_to_ref)
        if d_id in town_stats:
            town_stats[d_id]["deliveries"] += 1

        note_obj = rq.get("note")
        truth_leg = rev_orders_truth[oid]["legs"][0] if oid in rev_orders_truth else None
        order_rows.append({
            "id": oid,
            "worldId": world_id,
            "worldName": world_name,
            "route": f"{net.place_name(p_id)} → {net.place_name(d_id)}",
            "legs": len(o["legs"]),
            "option": opt,
            "arrived": fmt_min(rq["arrivedAt"]),
            "deadline": fmt_min(o["deadline"]),
            "onTime": not is_late,
            "priceEuros": round(o["priceCents"] / 100, 2),
            "priceToList": round(p_to_list, 2),
            "refFeeEuros": round(ref_fee / 100, 2),
            "driverFeeEuros": round(-fee_c / 100, 2),
            "feeToRef": round(f_to_ref, 2),
            "penaltyEuros": round(-pen_c / 100, 2),
            "marginEuros": round(net_c / 100, 2),
            "failedVisits": [f["reason"] for f in failed_by_order.get(oid, [])],
            "note": note_obj["text"] if note_obj else "",
            "truth": {
                "manual": truth_leg["manualGearbox"],
                "plates": truth_leg["needsPlates"],
                "summerTyres": truth_leg["summerTyres"],
                "readyAt": fmt_min(truth_leg["readyAt"]),
                "noteDeadline": fmt_min(truth_leg["noteDeadline"]) if truth_leg.get("noteDeadline") else None,
            } if truth_leg else None,
        })

    order_rows.sort(key=lambda r: r["marginEuros"])
    worst_orders = order_rows[:40]
    best_orders = list(reversed(order_rows[-25:]))
    late_or_failed_orders = [r for r in order_rows if (not r["onTime"]) or r["failedVisits"]][:80]

    # Format Option Breakdown
    option_out = []
    for k, od in option_stats.items():
        won = od["won"]
        option_out.append({
            "option": k,
            "won": won,
            "onTime": od["onTime"],
            "onTimePct": round(od["onTime"] / won * 100, 1) if won else 0.0,
            "marginEuros": round(od["marginCents"] / 100, 2),
            "avgMarginEuros": round(od["marginCents"] / won / 100, 2) if won else 0.0,
            "avgPriceToList": round(statistics.mean(od["priceToList"]), 3) if od["priceToList"] else 0.0,
            "avgFeeToRef": round(statistics.mean(od["feeToRef"]), 3) if od["feeToRef"] else 0.0,
        })

    # Format Quote Elasticity Curve
    quote_curve = []
    for b_val in sorted(quote_bins):
        qb = quote_bins[b_val]
        q_cnt = qb["quoted"]
        w_cnt = qb["won"]
        quote_curve.append({
            "multLabel": f"{b_val:.1f}×",
            "mult": b_val,
            "quoted": q_cnt,
            "won": w_cnt,
            "winRatePct": round(w_cnt / q_cnt * 100, 1) if q_cnt else 0.0,
            "avgMarginEuros": round(qb["marginCents"] / w_cnt / 100, 2) if w_cnt else 0.0,
            "totalMarginEuros": round(qb["marginCents"] / 100, 2),
        })

    # --- Driver Fee & Chaining Efficiency ---
    fee_ref_hist = [
        {"bucket": "< 0.85×", "count": sum(1 for r in fee_to_ref_all if r < 0.85)},
        {"bucket": "0.85–0.95×", "count": sum(1 for r in fee_to_ref_all if 0.85 <= r < 0.95)},
        {"bucket": "0.95–1.05×", "count": sum(1 for r in fee_to_ref_all if 0.95 <= r < 1.05)},
        {"bucket": "1.05–1.20×", "count": sum(1 for r in fee_to_ref_all if 1.05 <= r < 1.20)},
        {"bucket": "1.20–1.50×", "count": sum(1 for r in fee_to_ref_all if 1.20 <= r < 1.50)},
        {"bucket": "≥ 1.50×", "count": sum(1 for r in fee_to_ref_all if r >= 1.50)},
    ]

    booked_vs_open: list[float] = []
    chained_gigs = 0
    chained_cars = 0
    solo_gigs = 0
    for gid, g in gigs_by_id.items():
        if g["status"] == "done":
            cars = sum(1 for s in g["stops"] if s["kind"] == "pickup")
            orders_in_gig = len({s["orderId"] for s in g["stops"]})
            if orders_in_gig > 1 or cars > 1:
                chained_gigs += 1
                chained_cars += cars
            else:
                solo_gigs += 1
        if gid in gig_booked_fee:
            open_f = gig_first_view_fee.get(gid) or (g["schedule"][0]["feeCents"] if g.get("schedule") else 0)
            if open_f > 0:
                booked_vs_open.append(gig_booked_fee[gid] / open_f)

    below_open = sum(1 for r in booked_vs_open if r < 0.995)
    at_open = sum(1 for r in booked_vs_open if 0.995 <= r <= 1.005)
    stepped_up = sum(1 for r in booked_vs_open if r > 1.005)

    # --- Note Parser Ground-Truth Audit (when reveal is available) ---
    note_audit: Optional[dict[str, Any]] = None
    if reveal:
        checks = {"manual": [0, 0], "plates": [0, 0], "summerTyres": [0, 0], "readyTime": [0, 0], "deadline": [0, 0]}
        mismatches: list[dict[str, Any]] = []
        for o_rev in reveal.get("orders", []):
            oid = o_rev["orderId"]
            rq = requests_by_id.get(oid)
            if not rq or not rq.get("note"):
                continue
            p0 = rq["legs"][0]["pickup"]["window"]["from"]
            d0 = max(op["deadline"] for op in rq["options"])
            parsed = parse_note(rq["note"], rq["arrivedAt"], p0, d0)
            leg = o_rev["legs"][0]

            for key, truth_val, pred_val in (
                ("manual", bool(leg["manualGearbox"]), parsed.manual_gearbox),
                ("plates", bool(leg["needsPlates"]), parsed.needs_plates),
                ("summerTyres", bool(leg["summerTyres"]), parsed.summer_tyres),
            ):
                checks[key][1] += 1
                if truth_val == pred_val:
                    checks[key][0] += 1
                else:
                    mismatches.append({"orderId": oid, "worldName": world_name, "field": key, "note": rq["note"]["text"], "truth": str(truth_val), "parsed": str(pred_val)})

            eff_ready = max(leg["readyAt"], leg["releasedAt"] or 0)
            if eff_ready > p0:
                checks["readyTime"][1] += 1
                if parsed.pickup_after is not None and abs(parsed.pickup_after - eff_ready) <= 5:
                    checks["readyTime"][0] += 1
                else:
                    mismatches.append({"orderId": oid, "worldName": world_name, "field": "readyTime", "note": rq["note"]["text"], "truth": fmt_min(eff_ready), "parsed": fmt_min(parsed.pickup_after) if parsed.pickup_after else "None"})

            if leg.get("noteDeadline") is not None:
                checks["deadline"][1] += 1
                if parsed.delivery_before == leg["noteDeadline"]:
                    checks["deadline"][0] += 1
                else:
                    mismatches.append({"orderId": oid, "worldName": world_name, "field": "noteDeadline", "note": rq["note"]["text"], "truth": fmt_min(leg["noteDeadline"]), "parsed": fmt_min(parsed.delivery_before) if parsed.delivery_before else "None"})

        note_audit = {
            "metrics": {
                k: {"correct": v[0], "total": v[1], "accuracyPct": round(v[0] / v[1] * 100, 1) if v[1] else 100.0}
                for k, v in checks.items()
            },
            "mismatches": mismatches[:25],
        }

    # Format Corridors & Towns
    corridor_list = [
        {
            "fromId": c["fromId"],
            "toId": c["toId"],
            "route": c["route"],
            "km": c["km"],
            "orders": c["orders"],
            "late": c["late"],
            "marginEuros": round(c["marginCents"] / 100, 2),
            "avgMarginEuros": round(c["marginCents"] / c["orders"] / 100, 2),
        }
        for c in corridors.values()
    ]
    corridor_list.sort(key=lambda x: x["marginEuros"], reverse=True)

    town_list = [
        {
            "id": t["id"],
            "name": t["name"],
            "lat": t["lat"],
            "lon": t["lon"],
            "localDrivers": t["localDrivers"],
            "pickups": t["pickups"],
            "deliveries": t["deliveries"],
            "lateOrders": t["lateOrders"],
            "marginEuros": round(t["marginCents"] / 100, 2),
            "avgFeeToRef": round(statistics.mean(t["feeRatios"]), 2) if t["feeRatios"] else 0.0,
        }
        for t in town_stats.values()
    ]
    town_list.sort(key=lambda x: x["marginEuros"], reverse=True)

    return {
        "isMultiWeek": False,
        "weeksCount": 1,
        "weeksSummary": [
            {
                "worldId": world_id,
                "name": world_name,
                "seed": seed_name,
                "level": level,
                "ended": snap["now"] >= snap["endsAt"],
                "totalDrivers": len(roster),
                "shortagePctOfNormal": round(len(roster) / 2400.0 * 100, 1),
                "requests": len(requests_by_id),
                "wonOrders": snap["score"]["wonOrders"],
                "revenueEuros": round(snap["score"]["revenueCents"] / 100, 2),
                "driverFeeEuros": round(-ledger_totals["driver_fee"] / 100, 2),
                "penaltiesEuros": round(-(sum(v for k, v in ledger_totals.items() if k not in ("revenue", "driver_fee"))) / 100, 2),
                "marginEuros": round(snap["score"]["marginCents"] / 100, 2),
                "onTimePct": round(snap["score"]["onTimeShare"] * 100, 1),
                "undeliveredPct": round(snap["score"]["undeliveredShare"] * 100, 1),
                "ranked": snap["score"]["ranked"],
                "chainedGigs": chained_gigs,
                "failedVisits": ev_counts["visit_failed"],
            }
        ],
        "world": {
            "worldId": world_id,
            "name": world_name,
            "seed": seed_name,
            "batchId": batch_id,
            "batchName": batch_name,
            "level": level,
            "kind": kind,
            "boardRun": board_run,
            "ended": snap["now"] >= snap["endsAt"],
            "now": snap["now"],
            "endsAt": snap["endsAt"],
            "hasReveal": reveal is not None,
            "totalDrivers": len(roster),
            "shortagePctOfNormal": round(len(roster) / 2400.0 * 100, 1),
            "totalRequests": len(requests_by_id),
        },
        "score": snap["score"],
        "benchmark": board_benchmark,
        "waterfall": waterfall,
        "timeline": timeline,
        "dailyMarket": [daily_stats[d] for d in range(5)],
        "quoteCurve": quote_curve,
        "optionBreakdown": option_out,
        "driverEfficiency": {
            "avgFeeToRef": round(statistics.mean(fee_to_ref_all), 3) if fee_to_ref_all else 0.0,
            "medianFeeToRef": round(statistics.median(fee_to_ref_all), 3) if fee_to_ref_all else 0.0,
            "avgBookedToOpen": round(statistics.mean(booked_vs_open), 3) if booked_vs_open else 1.0,
            "bookedBelowOpen": below_open,
            "bookedAtOpen": at_open,
            "bookedSteppedUp": stepped_up,
            "soloDoneGigs": solo_gigs,
            "chainedDoneGigs": chained_gigs,
            "chainedCarsMoved": chained_cars,
            "feeToRefHistogram": fee_ref_hist,
            "events": {
                "gigBooked": ev_counts["gig_booked"],
                "gigDropped": ev_counts["gig_dropped"],
                "deadlineClose": ev_counts["deadline_close"],
                "gigExpired": ev_counts["gig_expired"],
                "driverLate": ev_counts["driver_late"],
                "visitFailed": ev_counts["visit_failed"],
                "failedReasons": dict(failed_by_reason),
            },
        },
        "allCorridors": corridor_list,
        "topCorridors": corridor_list[:15],
        "bottomCorridors": list(reversed(corridor_list[-15:])),
        "towns": town_list,
        "worstOrders": worst_orders,
        "bestOrders": best_orders,
        "lateOrders": late_or_failed_orders,
        "noteAudit": note_audit,
    }


def analyze_multi_worlds(
    client: Client,
    world_ids: list[str],
    world_map: dict[str, Any],
    *,
    title: Optional[str] = None,
    single_cache: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Aggregates forensic analytics across N (e.g. 5) weeks in parallel for instant multi-week reporting."""
    clean_ids = list(dict.fromkeys(w.strip() for w in world_ids if w.strip()))
    if not clean_ids:
        raise ValueError("No worldIds provided for multi-world analysis")
    if len(clean_ids) == 1:
        if single_cache and clean_ids[0] in single_cache:
            return single_cache[clean_ids[0]]
        rep = analyze_world(client, clean_ids[0], world_map)
        if single_cache is not None and rep.get("world", {}).get("ended"):
            single_cache[clean_ids[0]] = rep
        return rep

    raw_worlds = client.worlds()

    def _get_one(wid: str) -> dict[str, Any]:
        if single_cache is not None and wid in single_cache:
            return single_cache[wid]
        sub_client = Client(client.url, client.key)
        rep = analyze_world(sub_client, wid, world_map, worlds_list=raw_worlds)
        if single_cache is not None and rep.get("world", {}).get("ended"):
            single_cache[wid] = rep
        return rep

    with ThreadPoolExecutor(max_workers=min(5, len(clean_ids))) as pool:
        reports = list(pool.map(_get_one, clean_ids))

    n_weeks = len(reports)
    level = reports[0]["world"]["level"]
    any_reveal = any(r["world"]["hasReveal"] for r in reports)
    all_ended = all(r["world"]["ended"] for r in reports)
    weeks_summary = [r["weeksSummary"][0] for r in reports]

    # Combined Score
    tot_margin_c = sum(r["score"]["marginCents"] for r in reports)
    tot_rev_c = sum(r["score"]["revenueCents"] for r in reports)
    tot_won = sum(r["score"]["wonOrders"] for r in reports)
    tot_ontime_orders = sum(round(r["score"]["onTimeShare"] * r["score"]["wonOrders"]) for r in reports)
    tot_undeliv_orders = sum(round(r["score"]["undeliveredShare"] * r["score"]["wonOrders"]) for r in reports)
    comb_ontime_share = (tot_ontime_orders / tot_won) if tot_won > 0 else 1.0
    comb_undeliv_share = (tot_undeliv_orders / tot_won) if tot_won > 0 else 0.0
    ranked_weeks = sum(1 for r in reports if r["score"]["ranked"])
    all_ranked = (ranked_weeks == n_weeks) and tot_won > 0

    projected_5_euros = (tot_margin_c / 100.0) * (5.0 / n_weeks)
    benchmark = _fetch_board_benchmark(client, level, projected_5_euros)

    # Combined Waterfall
    wf_keys = ["revenue", "driver_fee", "late_refund", "undelivered_penalty", "failed_visit", "fine", "release_fee", "waiting", "net"]
    wf_by_key: dict[str, dict[str, Any]] = {}
    for r in reports:
        for item in r["waterfall"]:
            k = item["key"]
            if k not in wf_by_key:
                wf_by_key[k] = {"key": k, "label": item["label"], "euros": 0.0, "count": 0, "type": item["type"]}
            wf_by_key[k]["euros"] = round(wf_by_key[k]["euros"] + item["euros"], 2)
            wf_by_key[k]["count"] += item["count"]
    waterfall = [wf_by_key[k] for k in wf_keys if k in wf_by_key]

    # Combined Timeline (align by bucket index)
    max_tl_len = max((len(r["timeline"]) for r in reports), default=0)
    timeline: list[dict[str, Any]] = []
    for idx in range(max_tl_len):
        pts = [r["timeline"][idx] for r in reports if idx < len(r["timeline"])]
        if not pts:
            continue
        tot_ot = sum(p.get("cumOnTimeOrders", 0) for p in pts)
        tot_ord = sum(p.get("cumTotalOrders", 0) for p in pts)
        ot_pct = round(tot_ot / tot_ord * 100, 2) if tot_ord > 0 else round(statistics.mean(p["onTimePct"] for p in pts), 2)
        timeline.append({
            "minute": pts[0]["minute"],
            "label": pts[0]["label"],
            "cumMarginEuros": round(sum(p["cumMarginEuros"] for p in pts), 2),
            "cumRevenueEuros": round(sum(p["cumRevenueEuros"] for p in pts), 2),
            "cumDriverFeeEuros": round(sum(p["cumDriverFeeEuros"] for p in pts), 2),
            "cumPenaltiesEuros": round(sum(p["cumPenaltiesEuros"] for p in pts), 2),
            "onTimePct": ot_pct,
            "drops": sum(p["drops"] for p in pts),
        })

    # Combined Daily Market Regimes (Mon-Fri across all N weeks)
    daily_market: list[dict[str, Any]] = []
    for d in range(5):
        d_items = [r["dailyMarket"][d] for r in reports if d < len(r["dailyMarket"])]
        reqs = sum(x["requests"] for x in d_items)
        decls = sum(x["declined"] for x in d_items)
        quoted = sum(x["quoted"] for x in d_items)
        won = sum(x["won"] for x in d_items)
        win_mults = [x["avgWinMult"] for x in d_items if x["avgWinMult"] > 0]
        loss_mults = [x["avgLossMult"] for x in d_items if x["avgLossMult"] > 0]
        badges: list[str] = []
        for w_idx, r in enumerate(reports, 1):
            if d < len(r["dailyMarket"]):
                for b in r["dailyMarket"][d]["badges"]:
                    badges.append(f"[W{w_idx}] {b}")
        daily_market.append({
            "day": d,
            "name": DAY_NAMES[d],
            "requests": reqs,
            "declined": decls,
            "quoted": quoted,
            "won": won,
            "winRatePct": round(won / quoted * 100, 1) if quoted else round(won / max(1, reqs) * 100, 1),
            "avgWinMult": round(statistics.mean(win_mults), 3) if win_mults else 0.0,
            "avgLossMult": round(statistics.mean(loss_mults), 3) if loss_mults else 0.0,
            "marginEuros": round(sum(x["marginEuros"] for x in d_items), 2),
            "drops": sum(x["drops"] for x in d_items),
            "badges": badges[:8],
        })

    # Combined Quote Elasticity Curve
    qc_by_mult: dict[float, dict[str, Any]] = {}
    for r in reports:
        for b in r["quoteCurve"]:
            m = b["mult"]
            if m not in qc_by_mult:
                qc_by_mult[m] = {"multLabel": b["multLabel"], "mult": m, "quoted": 0, "won": 0, "totalMarginEuros": 0.0}
            qc_by_mult[m]["quoted"] += b["quoted"]
            qc_by_mult[m]["won"] += b["won"]
            qc_by_mult[m]["totalMarginEuros"] += b["totalMarginEuros"]
    quote_curve = []
    for m in sorted(qc_by_mult):
        qb = qc_by_mult[m]
        q_c, w_c, tot_m = qb["quoted"], qb["won"], round(qb["totalMarginEuros"], 2)
        quote_curve.append({
            "multLabel": qb["multLabel"],
            "mult": m,
            "quoted": q_c,
            "won": w_c,
            "winRatePct": round(w_c / q_c * 100, 1) if q_c else 0.0,
            "avgMarginEuros": round(tot_m / w_c, 2) if w_c else 0.0,
            "totalMarginEuros": tot_m,
        })

    # Combined Option Breakdown
    opt_out = []
    for opt_k in ("same_day", "next_day", "within_3_days"):
        items = [next((o for o in r["optionBreakdown"] if o["option"] == opt_k), None) for r in reports]
        items = [x for x in items if x is not None and x["won"] > 0]
        won = sum(x["won"] for x in items)
        ot_cnt = sum(x.get("onTime", round(x["onTimePct"] / 100 * x["won"])) for x in items)
        mar_e = round(sum(x["marginEuros"] for x in items), 2)
        p_mult = sum(x["avgPriceToList"] * x["won"] for x in items) / won if won else 0.0
        f_mult = sum(x["avgFeeToRef"] * x["won"] for x in items) / won if won else 0.0
        opt_out.append({
            "option": opt_k,
            "won": won,
            "onTime": ot_cnt,
            "onTimePct": round(ot_cnt / won * 100, 1) if won else 0.0,
            "marginEuros": mar_e,
            "avgMarginEuros": round(mar_e / won, 2) if won else 0.0,
            "avgPriceToList": round(p_mult, 3),
            "avgFeeToRef": round(f_mult, 3),
        })

    # Combined Driver Efficiency
    de_list = [r["driverEfficiency"] for r in reports]
    hist_buckets = [h["bucket"] for h in de_list[0]["feeToRefHistogram"]] if de_list else []
    comb_hist = [
        {"bucket": bk, "count": sum(next((h["count"] for h in de["feeToRefHistogram"] if h["bucket"] == bk), 0) for de in de_list)}
        for bk in hist_buckets
    ]
    failed_reasons_comb: Counter[str] = Counter()
    for de in de_list:
        for rk, rv in de["events"]["failedReasons"].items():
            failed_reasons_comb[rk] += rv

    # Combined Corridors
    corr_map: dict[str, dict[str, Any]] = {}
    for r in reports:
        for c in r.get("allCorridors", r["topCorridors"] + r["bottomCorridors"]):
            rt = c["route"]
            if rt not in corr_map:
                corr_map[rt] = {"route": rt, "km": c["km"], "orders": 0, "late": 0, "marginEuros": 0.0}
            corr_map[rt]["orders"] += c["orders"]
            corr_map[rt]["late"] += c["late"]
            corr_map[rt]["marginEuros"] = round(corr_map[rt]["marginEuros"] + c["marginEuros"], 2)
    corr_all = [
        {**c, "avgMarginEuros": round(c["marginEuros"] / max(1, c["orders"]), 2)}
        for c in corr_map.values()
    ]
    corr_all.sort(key=lambda x: x["marginEuros"], reverse=True)

    # Combined Towns
    town_map: dict[str, dict[str, Any]] = {}
    for r in reports:
        for t in r["towns"]:
            tid = t["id"]
            if tid not in town_map:
                town_map[tid] = {
                    "id": tid,
                    "name": t["name"],
                    "lat": t["lat"],
                    "lon": t["lon"],
                    "localDrivers": 0,
                    "pickups": 0,
                    "deliveries": 0,
                    "lateOrders": 0,
                    "marginEuros": 0.0,
                    "feeSum": 0.0,
                    "feeWeight": 0,
                }
            tm = town_map[tid]
            tm["localDrivers"] += t["localDrivers"]
            tm["pickups"] += t["pickups"]
            tm["deliveries"] += t["deliveries"]
            tm["lateOrders"] += t["lateOrders"]
            tm["marginEuros"] = round(tm["marginEuros"] + t["marginEuros"], 2)
            if t["avgFeeToRef"] > 0 and t["pickups"] > 0:
                tm["feeSum"] += t["avgFeeToRef"] * t["pickups"]
                tm["feeWeight"] += t["pickups"]
    town_list = [
        {
            "id": tm["id"],
            "name": tm["name"],
            "lat": tm["lat"],
            "lon": tm["lon"],
            "localDrivers": round(tm["localDrivers"] / n_weeks),
            "pickups": tm["pickups"],
            "deliveries": tm["deliveries"],
            "lateOrders": tm["lateOrders"],
            "marginEuros": tm["marginEuros"],
            "avgFeeToRef": round(tm["feeSum"] / tm["feeWeight"], 2) if tm["feeWeight"] else 0.0,
        }
        for tm in town_map.values()
    ]
    town_list.sort(key=lambda x: x["marginEuros"], reverse=True)

    # Combined Orders (Worst 50, Best 35, Late/Failed 100)
    all_worst = [o for r in reports for o in r["worstOrders"]]
    all_worst.sort(key=lambda x: x["marginEuros"])
    all_best = [o for r in reports for o in r["bestOrders"]]
    all_best.sort(key=lambda x: x["marginEuros"], reverse=True)
    all_late = [o for r in reports for o in r.get("lateOrders", [])]
    all_late.sort(key=lambda x: x["marginEuros"])

    # Combined Note Audit
    note_audit: Optional[dict[str, Any]] = None
    na_list = [r["noteAudit"] for r in reports if r.get("noteAudit")]
    if na_list:
        comb_metrics: dict[str, dict[str, Any]] = {}
        for k in ("manual", "plates", "summerTyres", "readyTime", "deadline"):
            c_sum = sum(na["metrics"][k]["correct"] for na in na_list if k in na["metrics"])
            t_sum = sum(na["metrics"][k]["total"] for na in na_list if k in na["metrics"])
            comb_metrics[k] = {
                "correct": c_sum,
                "total": t_sum,
                "accuracyPct": round(c_sum / t_sum * 100, 1) if t_sum else 100.0,
            }
        comb_mismatches = [m for na in na_list for m in na.get("mismatches", [])][:40]
        note_audit = {"metrics": comb_metrics, "mismatches": comb_mismatches}

    avg_drivers = round(statistics.mean(r["world"]["totalDrivers"] for r in reports))
    group_title = title or f"{n_weeks}-Week Combined Set (L{level})"

    return {
        "isMultiWeek": True,
        "weeksCount": n_weeks,
        "rankedWeeksCount": ranked_weeks,
        "weeksSummary": weeks_summary,
        "world": {
            "worldId": ",".join(clean_ids),
            "name": group_title,
            "level": level,
            "kind": "multi",
            "boardRun": None,
            "ended": all_ended,
            "now": max(r["world"]["now"] for r in reports),
            "endsAt": max(r["world"]["endsAt"] for r in reports),
            "hasReveal": any_reveal,
            "totalDrivers": avg_drivers,
            "shortagePctOfNormal": round(avg_drivers / 2400.0 * 100, 1),
            "totalRequests": sum(r["world"]["totalRequests"] for r in reports),
        },
        "score": {
            "marginCents": tot_margin_c,
            "revenueCents": tot_rev_c,
            "marginShare": (tot_margin_c / tot_rev_c) if tot_rev_c > 0 else 0.0,
            "wonOrders": tot_won,
            "onTimeShare": comb_ontime_share,
            "undeliveredShare": comb_undeliv_share,
            "ranked": all_ranked,
        },
        "benchmark": benchmark,
        "waterfall": waterfall,
        "timeline": timeline,
        "dailyMarket": daily_market,
        "quoteCurve": quote_curve,
        "optionBreakdown": opt_out,
        "driverEfficiency": {
            "avgFeeToRef": round(statistics.mean(de["avgFeeToRef"] for de in de_list if de["avgFeeToRef"] > 0), 3) if any(de["avgFeeToRef"] > 0 for de in de_list) else 0.0,
            "medianFeeToRef": round(statistics.mean(de["medianFeeToRef"] for de in de_list if de["medianFeeToRef"] > 0), 3) if any(de["medianFeeToRef"] > 0 for de in de_list) else 0.0,
            "avgBookedToOpen": round(statistics.mean(de["avgBookedToOpen"] for de in de_list), 3) if de_list else 1.0,
            "bookedBelowOpen": sum(de["bookedBelowOpen"] for de in de_list),
            "bookedAtOpen": sum(de["bookedAtOpen"] for de in de_list),
            "bookedSteppedUp": sum(de["bookedSteppedUp"] for de in de_list),
            "soloDoneGigs": sum(de["soloDoneGigs"] for de in de_list),
            "chainedDoneGigs": sum(de["chainedDoneGigs"] for de in de_list),
            "chainedCarsMoved": sum(de["chainedCarsMoved"] for de in de_list),
            "feeToRefHistogram": comb_hist,
            "events": {
                "gigBooked": sum(de["events"]["gigBooked"] for de in de_list),
                "gigDropped": sum(de["events"]["gigDropped"] for de in de_list),
                "deadlineClose": sum(de["events"]["deadlineClose"] for de in de_list),
                "gigExpired": sum(de["events"]["gigExpired"] for de in de_list),
                "driverLate": sum(de["events"]["driverLate"] for de in de_list),
                "visitFailed": sum(de["events"]["visitFailed"] for de in de_list),
                "failedReasons": dict(failed_reasons_comb),
            },
        },
        "allCorridors": corr_all,
        "topCorridors": corr_all[:15],
        "bottomCorridors": list(reversed(corr_all[-15:])),
        "towns": town_list,
        "worstOrders": all_worst[:50],
        "bestOrders": all_best[:35],
        "lateOrders": all_late[:100],
        "noteAudit": note_audit,
    }
