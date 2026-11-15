"""Human-readable world naming and multi-week batch registry."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES_FILE = os.path.join(HERE, "world_names.json")

_lock = threading.Lock()

CODENAMES = [
    "Aurora", "Pohjola", "Sampo", "Tapio", "Kaleva", "Otava", "Sisu", "Aino",
    "Ilmari", "Vellamo", "Saimaa", "Inari", "Polaris", "Koli", "Pallaksen", "Ruka",
    "Levi", "Yllas", "Kilpis", "Nuuksio", "Bothnia", "Kantele", "Mielikki", "Tellervo",
    "Ahti", "Pellervo", "Jouko", "Louhi", "Untamo", "Rauni", "Arctic", "Baltic",
]

RANK_RUN_CODENAMES = ["Sisu", "Sampo", "Otava", "Aino", "Ukko", "Tapio", "Kaleva", "Vellamo"]


def _load_db() -> dict[str, Any]:
    if not os.path.isfile(NAMES_FILE):
        return {"worlds": {}, "batches": {}, "counters": {"practice": 0, "pack": 0}}
    try:
        with open(NAMES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            data.setdefault("worlds", {})
            data.setdefault("batches", {})
            data.setdefault("counters", {"practice": 0, "pack": 0})
            return data
    except Exception:
        return {"worlds": {}, "batches": {}, "counters": {"practice": 0, "pack": 0}}


def _save_db(db: dict[str, Any]) -> None:
    tmp = NAMES_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
        os.replace(tmp, NAMES_FILE)
    except Exception:
        pass


def get_world_name(world_id: str) -> str:
    with _lock:
        db = _load_db()
        info = db["worlds"].get(world_id)
        if info and info.get("name"):
            return str(info["name"])
        return world_id[:10]


def create_practice_pack(
    level: int = 8,
    count: int = 5,
    custom_name: Optional[str] = None,
    seed_prefix: Optional[str] = None,
) -> dict[str, Any]:
    """Allocates a new human-readable N-week practice pack and pre-computes week names & seeds."""
    with _lock:
        db = _load_db()
        pack_num = int(db["counters"].get("pack", 0)) + 1
        db["counters"]["pack"] = pack_num
        codename = (custom_name or "").strip() or CODENAMES[(pack_num - 1) % len(CODENAMES)]
        batch_id = f"pack-{pack_num}-{int(time.time()) % 100000}"
        batch_name = f"Pack #{pack_num} · {codename} (L{level} · {count} Weeks)"
        base_seed = (seed_prefix or "").strip() or f"{codename.lower()}-p{pack_num}"

        specs = []
        for idx in range(1, count + 1):
            w_seed = f"{base_seed}-w{idx}"
            w_name = f"{codename} · Week {idx}/{count} (L{level})"
            specs.append({
                "index": idx,
                "total": count,
                "name": w_name,
                "seed": w_seed,
            })

        db["batches"][batch_id] = {
            "batchId": batch_id,
            "batchName": batch_name,
            "codename": codename,
            "level": level,
            "count": count,
            "createdAt": int(time.time()),
            "worldIds": [],
        }
        _save_db(db)
        return {
            "batchId": batch_id,
            "batchName": batch_name,
            "codename": codename,
            "level": level,
            "count": count,
            "specs": specs,
        }


def register_world(
    world_id: str,
    *,
    level: int = 8,
    kind: str = "practice",
    name: Optional[str] = None,
    seed: Optional[str] = None,
    board_run: Optional[int] = None,
    season: Optional[int] = None,
    batch_id: Optional[str] = None,
    batch_name: Optional[str] = None,
    batch_index: Optional[int] = None,
    batch_total: Optional[int] = None,
) -> dict[str, Any]:
    """Registers or updates human-readable metadata for a world."""
    with _lock:
        db = _load_db()
        existing = db["worlds"].get(world_id, {})

        eff_seed = seed or existing.get("seed")
        eff_batch_id = batch_id or existing.get("batchId")
        eff_batch_name = batch_name or existing.get("batchName")
        eff_batch_idx = batch_index or existing.get("batchIndex")
        eff_batch_tot = batch_total or existing.get("batchTotal")

        if board_run is not None:
            s_num = season or existing.get("season") or 8
            r_code = RANK_RUN_CODENAMES[(board_run - 1) % len(RANK_RUN_CODENAMES)]
            eff_name = name or existing.get("name") or f"S{s_num} Rank #{board_run} · {r_code} (L{level})"
            eff_batch_id = eff_batch_id or f"season-{s_num}-L{level}"
            eff_batch_name = eff_batch_name or f"Season {s_num} Ranking Set (L{level})"
            eff_batch_idx = board_run
            eff_batch_tot = 5
        elif kind == "final":
            eff_name = name or existing.get("name") or f"Live Final · Championship (L{level})"
        else:
            if name:
                eff_name = name.strip()
            elif existing.get("name"):
                eff_name = existing["name"]
            else:
                p_num = int(db["counters"].get("practice", 0)) + 1
                db["counters"]["practice"] = p_num
                code = CODENAMES[(p_num - 1) % len(CODENAMES)]
                if eff_seed:
                    eff_name = f"Practice #{p_num} · {eff_seed} (L{level})"
                else:
                    eff_name = f"Practice #{p_num} · {code} (L{level})"

        entry = {
            **existing,
            "worldId": world_id,
            "name": eff_name,
            "level": level,
            "kind": kind,
            "seed": eff_seed,
            "boardRun": board_run if board_run is not None else existing.get("boardRun"),
            "season": season if season is not None else existing.get("season"),
            "batchId": eff_batch_id,
            "batchName": eff_batch_name,
            "batchIndex": eff_batch_idx,
            "batchTotal": eff_batch_tot,
            "updatedAt": int(time.time()),
        }
        db["worlds"][world_id] = entry

        if eff_batch_id:
            b_obj = db["batches"].setdefault(
                eff_batch_id,
                {
                    "batchId": eff_batch_id,
                    "batchName": eff_batch_name or eff_batch_id,
                    "level": level,
                    "count": eff_batch_tot or 5,
                    "createdAt": int(time.time()),
                    "worldIds": [],
                },
            )
            if world_id not in b_obj["worldIds"]:
                b_obj["worldIds"].append(world_id)

        _save_db(db)
        return entry


def record_world_score(world_id: str, score: dict[str, Any]) -> None:
    """Stores a world's final or current score summary so selectors can show profit & SLA immediately."""
    with _lock:
        db = _load_db()
        if world_id not in db["worlds"]:
            db["worlds"][world_id] = {"worldId": world_id}
        db["worlds"][world_id]["score"] = {
            "marginCents": score.get("marginCents", 0),
            "revenueCents": score.get("revenueCents", 0),
            "wonOrders": score.get("wonOrders", 0),
            "onTimeShare": score.get("onTimeShare", 0.0),
            "ranked": bool(score.get("ranked", False)),
        }
        _save_db(db)


def rename_world(world_id: str, new_name: str) -> dict[str, Any]:
    with _lock:
        db = _load_db()
        entry = db["worlds"].setdefault(world_id, {"worldId": world_id})
        entry["name"] = new_name.strip()
        _save_db(db)
        return entry


def enrich_worlds(raw_worlds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attaches human-readable `name`, `seed`, `batchId`, `batchName`, and `score` to every world in `/v1/worlds`."""
    with _lock:
        db = _load_db()
        dirty = False
        out: list[dict[str, Any]] = []

        for w in raw_worlds:
            wid = w["worldId"]
            existing = db["worlds"].get(wid)
            lvl = w.get("level", 8)
            kind = w.get("kind", "practice")
            b_run = w.get("boardRun")
            season = w.get("season")

            if not existing or not existing.get("name"):
                dirty = True
                if b_run is not None:
                    s_num = season or 8
                    r_code = RANK_RUN_CODENAMES[(b_run - 1) % len(RANK_RUN_CODENAMES)]
                    w_name = f"S{s_num} Rank #{b_run} · {r_code} (L{lvl})"
                    b_id = f"season-{s_num}-L{lvl}"
                    b_name = f"Season {s_num} Ranking Set (L{lvl} · 5 Weeks)"
                    b_idx = b_run
                    b_tot = 5
                elif kind == "final":
                    w_name = f"Live Final · Championship (L{lvl})"
                    b_id = None
                    b_name = None
                    b_idx = None
                    b_tot = None
                else:
                    p_num = int(db["counters"].get("practice", 0)) + 1
                    db["counters"]["practice"] = p_num
                    code = CODENAMES[(p_num - 1) % len(CODENAMES)]
                    seed_val = existing.get("seed") if existing else None
                    if w.get("parent"):
                        w_name = f"Fork #{p_num} · {code} (L{lvl})"
                    elif seed_val:
                        w_name = f"Practice #{p_num} · {seed_val} (L{lvl})"
                    else:
                        w_name = f"Practice #{p_num} · {code} (L{lvl})"
                    b_id = existing.get("batchId") if existing else None
                    b_name = existing.get("batchName") if existing else None
                    b_idx = existing.get("batchIndex") if existing else None
                    b_tot = existing.get("batchTotal") if existing else None

                existing = {
                    **(existing or {}),
                    "worldId": wid,
                    "name": w_name,
                    "level": lvl,
                    "kind": kind,
                    "boardRun": b_run,
                    "season": season,
                    "batchId": b_id,
                    "batchName": b_name,
                    "batchIndex": b_idx,
                    "batchTotal": b_tot,
                }
                db["worlds"][wid] = existing
                if b_id:
                    b_obj = db["batches"].setdefault(
                        b_id,
                        {
                            "batchId": b_id,
                            "batchName": b_name or b_id,
                            "level": lvl,
                            "count": b_tot or 5,
                            "createdAt": int(time.time()),
                            "worldIds": [],
                        },
                    )
                    if wid not in b_obj["worldIds"]:
                        b_obj["worldIds"].append(wid)

            enriched = {
                **w,
                "name": existing["name"],
                "seed": existing.get("seed"),
                "batchId": existing.get("batchId"),
                "batchName": existing.get("batchName"),
                "batchIndex": existing.get("batchIndex"),
                "batchTotal": existing.get("batchTotal"),
                "updatedAt": existing.get("updatedAt"),
                "score": existing.get("score"),
            }
            out.append(enriched)

        seen_ids = {w["worldId"] for w in out}
        for wid, existing in db["worlds"].items():
            if wid not in seen_ids and existing.get("score"):
                out.append({
                    "worldId": wid,
                    "name": existing.get("name") or wid[:10],
                    "level": existing.get("level", 8),
                    "kind": existing.get("kind", "practice"),
                    "boardRun": existing.get("boardRun"),
                    "season": existing.get("season"),
                    "seed": existing.get("seed"),
                    "batchId": existing.get("batchId"),
                    "batchName": existing.get("batchName"),
                    "batchIndex": existing.get("batchIndex"),
                    "batchTotal": existing.get("batchTotal"),
                    "updatedAt": existing.get("updatedAt"),
                    "score": existing.get("score"),
                    "ended": True,
                })

        if dirty:
            _save_db(db)
        return out


def get_bigdata_catalog() -> dict[str, Any]:
    """Read-only zero-overhead local catalog of all recorded worlds and batches from world_names.json (0 upstream API calls)."""
    db = _load_db()
    worlds_list: list[dict[str, Any]] = []
    for seq_idx, (wid, w) in enumerate(db.get("worlds", {}).items(), 1):
        sc = w.get("score")
        has_orders = bool(sc and (sc.get("wonOrders", 0) > 0 or sc.get("marginCents", 0) != 0 or sc.get("revenueCents", 0) > 0))
        worlds_list.append({
            "seqIndex": seq_idx,
            "worldId": wid,
            "name": w.get("name") or wid[:10],
            "level": w.get("level", 8),
            "kind": w.get("kind", "practice"),
            "seed": w.get("seed"),
            "boardRun": w.get("boardRun"),
            "season": w.get("season"),
            "batchId": w.get("batchId"),
            "batchName": w.get("batchName"),
            "batchIndex": w.get("batchIndex"),
            "batchTotal": w.get("batchTotal"),
            "updatedAt": w.get("updatedAt"),
            "score": sc,
            "ended": has_orders or bool(sc),
            "completed": has_orders,
        })
    return {
        "worlds": worlds_list,
        "batches": list_batches(worlds_list),
        "rawBatches": db.get("batches", {}),
        "counters": db.get("counters", {}),
    }


def list_batches(enriched_worlds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Returns all multi-week batches (5-week Practice Packs, Season Ranking Sets, and quick multi-world groups)."""
    by_batch: dict[str, dict[str, Any]] = {}
    for w in enriched_worlds:
        bid = w.get("batchId")
        if not bid:
            continue
        b = by_batch.setdefault(
            bid,
            {
                "batchId": bid,
                "batchName": w.get("batchName") or bid,
                "level": w.get("level", 8),
                "worldIds": [],
                "endedCount": 0,
                "totalMarginCents": 0,
                "allRanked": True,
            },
        )
        b["worldIds"].append(w["worldId"])
        if w.get("ended"):
            b["endedCount"] += 1
        sc = w.get("score")
        if sc:
            b["totalMarginCents"] += sc.get("marginCents", 0)
            if not sc.get("ranked", False):
                b["allRanked"] = False
        else:
            b["allRanked"] = False

    batches = list(by_batch.values())

    # Also build convenient virtual groups (e.g., Latest 5 Ended Practice Worlds on L8, All Ended L8 Worlds)
    l8_practice_ended = [
        w["worldId"] for w in enriched_worlds if w.get("ended") and w.get("level") == 8 and not w.get("boardRun")
    ]
    if len(l8_practice_ended) >= 2:
        last5 = l8_practice_ended[-5:]
        batches.append({
            "batchId": "virtual:last5-practice-l8",
            "batchName": f"Latest {len(last5)} L8 Practice Weeks (Combined)",
            "level": 8,
            "worldIds": last5,
            "endedCount": len(last5),
            "totalMarginCents": 0,
            "allRanked": False,
        })

    l8_all_ended = [w["worldId"] for w in enriched_worlds if w.get("ended") and w.get("level") == 8]
    if len(l8_all_ended) >= 2:
        batches.append({
            "batchId": "virtual:all-l8",
            "batchName": f"All {len(l8_all_ended)} Ended L8 Weeks (Combined)",
            "level": 8,
            "worldIds": l8_all_ended,
            "endedCount": len(l8_all_ended),
            "totalMarginCents": 0,
            "allRanked": False,
        })

    return batches
