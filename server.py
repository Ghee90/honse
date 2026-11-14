#!/usr/bin/env python3
"""Spread Command Center Server:
- Serves the redesigned dashboard at http://localhost:8765 (and /analytics)
- Proxies /api/* to SPREAD_URL with persistent HTTPS keep-alive, 429 retry handling, and human-readable world naming
- Hosts /bot/* endpoints to run the Python SmartDesk dispatcher for 1 week, 5 consecutive practice weeks, or Live Final
"""
from __future__ import annotations

import http.client
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlsplit

from engine.analyzer import analyze_multi_worlds, analyze_world
from engine.config import load_env
from engine.desk import DAY, WAKE_ON, SmartDesk
from engine.namer import (
    create_practice_pack,
    enrich_worlds,
    get_world_name,
    list_batches,
    record_world_score,
    register_world,
    rename_world,
)
from engine.notes import parse_note
from spread.client import Client, SpreadError, Unreachable, World

load_env()

BASE = os.environ.get("SPREAD_URL", "https://spread.bahnexpress.com").rstrip("/")
KEY = os.environ.get("SPREAD_KEY", "")
PORT = int(os.environ.get("PORT", "8765"))
HERE = os.path.dirname(os.path.abspath(__file__))
PASS_HEADERS = (
    "Content-Type",
    "Retry-After",
    "RateLimit-Policy",
    "RateLimit-Limit",
    "RateLimit-Remaining",
    "RateLimit-Reset",
)

_tls_local = threading.local()
_map_cache_lock = threading.Lock()
_cached_map: Optional[dict[str, Any]] = None
_analysis_cache: dict[str, dict[str, Any]] = {}


def get_cached_map() -> dict[str, Any]:
    global _cached_map
    with _map_cache_lock:
        if _cached_map is None:
            client = Client(BASE, KEY or "none")
            _cached_map = client.map()  # type: ignore[assignment]
        return _cached_map


class BotRunner:
    """Manages a background SmartDesk worker for single-world, 5-week practice pack, or live final runs."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.world_id: Optional[str] = None
        self.mode: str = "idle"  # idle | turbo | watch | day | step | pack | final
        self.running: bool = False
        self.stop_flag: bool = False
        self.steps: int = 0
        self.now: int = 0
        self.ends_at: int = 0
        self.last_error: Optional[str] = None
        self.desk: Optional[SmartDesk] = None
        self.pack_state: Optional[dict[str, Any]] = None
        self._thread: Optional[threading.Thread] = None

    def status(self, req_world_id: Optional[str] = None) -> dict[str, Any]:
        with self.lock:
            same = (req_world_id is None) or (req_world_id == self.world_id) or (self.mode == "pack")
            tel = self.desk.telemetry() if (self.desk and same) else None
            return {
                "worldId": self.world_id,
                "worldName": get_world_name(self.world_id) if self.world_id else None,
                "running": self.running and same,
                "mode": self.mode if (self.running and same) else "idle",
                "steps": self.steps if same else 0,
                "now": self.now if same else 0,
                "endsAt": self.ends_at if same else 0,
                "error": self.last_error if same else None,
                "telemetry": tel,
                "pack": self.pack_state,
            }

    def stop(self) -> None:
        t = None
        with self.lock:
            if self.running:
                self.stop_flag = True
                t = self._thread
        if t and t.is_alive():
            t.join(timeout=3.5)

    def start(self, world_id: Optional[str], mode: str) -> dict[str, Any]:
        self.stop()
        with self.lock:
            self.world_id = world_id
            self.mode = mode
            self.running = True
            self.stop_flag = False
            self.last_error = None
            t = threading.Thread(target=self._run_loop, args=(world_id, mode), daemon=True)
            self._thread = t
            t.start()
        return self.status(world_id)

    def start_pack(
        self,
        level: int = 8,
        count: int = 5,
        custom_name: Optional[str] = None,
        seed_prefix: Optional[str] = None,
    ) -> dict[str, Any]:
        self.stop()
        pack_spec = create_practice_pack(level=level, count=count, custom_name=custom_name, seed_prefix=seed_prefix)
        with self.lock:
            self.mode = "pack"
            self.running = True
            self.stop_flag = False
            self.last_error = None
            self.pack_state = {
                "batchId": pack_spec["batchId"],
                "batchName": pack_spec["batchName"],
                "codename": pack_spec["codename"],
                "level": level,
                "currentWeek": 1,
                "totalWeeks": count,
                "running": True,
                "worldIds": [],
                "completedWeeks": [],
            }
            t = threading.Thread(target=self._run_pack_loop, args=(pack_spec,), daemon=True)
            self._thread = t
            t.start()
        return self.status()

    def _run_pack_loop(self, pack_spec: dict[str, Any]) -> None:
        try:
            client = Client(BASE, KEY)
            world_map = get_cached_map()
            level = pack_spec["level"]
            batch_id = pack_spec["batchId"]
            batch_name = pack_spec["batchName"]

            for spec in pack_spec["specs"]:
                if self.stop_flag:
                    break
                idx = spec["index"]
                tot = spec["total"]
                w_name = spec["name"]
                w_seed = spec["seed"]

                w = client.create(level=level, seed=w_seed, board=False)
                register_world(
                    w.id,
                    level=level,
                    kind="practice",
                    name=w_name,
                    seed=w_seed,
                    batch_id=batch_id,
                    batch_name=batch_name,
                    batch_index=idx,
                    batch_total=tot,
                )

                snap = w.snapshot()
                roster = w.roster()
                desk = SmartDesk(w, level, snap["now"], world_map, roster)  # type: ignore[arg-type]
                ends_at = snap["endsAt"]

                with self.lock:
                    self.world_id = w.id
                    self.desk = desk
                    self.steps = 0
                    self.now = desk.now
                    self.ends_at = ends_at
                    if self.pack_state is not None:
                        self.pack_state["currentWeek"] = idx
                        self.pack_state["currentWorldId"] = w.id
                        self.pack_state["currentWorldName"] = w_name
                        self.pack_state["worldIds"].append(w.id)

                while not self.stop_flag and desk.now < ends_at:
                    midnight = (desk.now // DAY + 1) * DAY
                    target_max = min(ends_at, midnight)
                    actions, desk.outbox = desk.outbox, []
                    step = w.advance(until=WAKE_ON, max=target_max, actions=actions)
                    desk.read_results(actions, step.get("results", []))
                    desk.now = step["now"]
                    for ev in step["events"]:
                        desk.handle(ev)

                    with self.lock:
                        self.steps += 1
                        self.now = desk.now

                if desk.outbox and not self.stop_flag:
                    desk.send()

                sc = w.score()
                record_world_score(w.id, sc)
                with self.lock:
                    if self.pack_state is not None:
                        self.pack_state["completedWeeks"].append({
                            "index": idx,
                            "worldId": w.id,
                            "name": w_name,
                            "seed": w_seed,
                            "marginEuros": round(sc["marginCents"] / 100, 2),
                            "revenueEuros": round(sc["revenueCents"] / 100, 2),
                            "wonOrders": sc["wonOrders"],
                            "onTimePct": round(sc["onTimeShare"] * 100, 1),
                            "ranked": sc["ranked"],
                        })

        except BaseException as exc:
            with self.lock:
                self.last_error = str(exc)
        finally:
            with self.lock:
                self.running = False
                self.mode = "idle"
                if self.pack_state is not None:
                    self.pack_state["running"] = False

    def _run_loop(self, world_id: Optional[str], mode: str) -> None:
        try:
            client = Client(BASE, KEY)
            world_map = get_cached_map()

            if mode == "final":
                while not self.stop_flag:
                    try:
                        fin = client.final()
                        world = client.world(fin["worldId"])
                        with self.lock:
                            self.world_id = world.id
                        self._run_final_stream(world, world_map)
                        break
                    except SpreadError as err:
                        if err.status == 404:
                            with self.lock:
                                self.last_error = "Waiting for Final / Test-Final to be scheduled..."
                            time.sleep(5.0)
                            continue
                        raise
                    except Unreachable as err:
                        with self.lock:
                            self.last_error = f"{err} — reconnecting..."
                        time.sleep(4.0)
                return

            if not world_id:
                raise SpreadError(400, "No worldId selected")

            world = client.world(world_id)
            worlds_meta = client.worlds()
            w_info = next((w for w in worlds_meta if w["worldId"] == world_id), None)
            level = w_info["level"] if w_info else 8
            is_final_world = bool(w_info and w_info["kind"] == "final")

            if is_final_world:
                self._run_final_stream(world, world_map)
                return

            snap = world.snapshot()
            roster = world.roster()
            with self.lock:
                reuse = (
                    self.desk is not None
                    and self.desk.world.id == world_id
                    and self.desk.now == snap["now"]
                )
            if reuse and self.desk is not None:
                desk = self.desk
            else:
                desk = SmartDesk(world, level, snap["now"], world_map, roster)  # type: ignore[arg-type]
                history = world.events()
                if history or snap["openRequests"] or snap["orders"]:
                    desk.rejoin(snap, history)
                with self.lock:
                    self.desk = desk
                    self.steps = 0

            ends_at = snap["endsAt"]
            start_day = desk.now // DAY
            with self.lock:
                self.now = desk.now
                self.ends_at = ends_at

            while not self.stop_flag and desk.now < ends_at:
                midnight = (desk.now // DAY + 1) * DAY
                target_max = min(ends_at, midnight)
                actions, desk.outbox = desk.outbox, []
                step = world.advance(until=WAKE_ON, max=target_max, actions=actions)
                desk.read_results(actions, step.get("results", []))
                desk.now = step["now"]
                for ev in step["events"]:
                    desk.handle(ev)

                with self.lock:
                    self.steps += 1
                    self.now = desk.now

                if mode == "step":
                    if desk.outbox:
                        desk.send()
                    break
                if mode == "day" and (desk.now // DAY) > start_day:
                    if desk.outbox:
                        desk.send()
                    break
                if mode == "watch":
                    if desk.outbox:
                        desk.send()
                    time.sleep(0.28)

            if desk.outbox and not self.stop_flag:
                desk.send()

            try:
                record_world_score(world.id, world.score())
            except Exception:
                pass

        except BaseException as exc:
            with self.lock:
                self.last_error = str(exc)
        finally:
            with self.lock:
                self.running = False
                self.mode = "idle"

    def _run_final_stream(self, world: World, world_map: dict[str, Any]) -> None:
        history = world.events()
        snap = world.snapshot()
        level = next((w["level"] for w in world.client.worlds() if w["worldId"] == world.id), 8)
        roster = world.roster()
        desk = SmartDesk(world, level, snap["now"], world_map, roster)  # type: ignore[arg-type]
        desk.rejoin(snap, history)
        desk.send()
        with self.lock:
            self.desk = desk
            self.now = snap["now"]
            self.ends_at = snap["endsAt"]
            self.last_error = None

        if snap["now"] >= snap["endsAt"]:
            return

        for event in world.stream(after=history[-1]["seq"] if history else -1, idle=1.0):
            if self.stop_flag:
                break
            desk.now = world.client.final()["now"] if event is None else max(desk.now, event["at"])
            if event is not None:
                desk.handle(event)
            desk.send()
            with self.lock:
                self.now = desk.now
                self.steps += 1
            if desk.now >= snap["endsAt"]:
                break


BOT = BotRunner()


def _upstream_conn() -> http.client.HTTPConnection:
    parts = urlsplit(BASE)
    conn: Optional[http.client.HTTPConnection] = getattr(_tls_local, "conn", None)
    if conn is None:
        cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        conn = cls(parts.netloc, timeout=45)
        _tls_local.conn = conn
    return conn


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length > 0 else b""

    def _send(self, code: int, data: bytes = b"", headers: Optional[dict[str, str]] = None) -> None:
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD" and code != 204:
            self.wfile.write(data)

    def _send_json(self, code: int, obj: Any) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), {"Content-Type": "application/json; charset=utf-8"})

    def send_error(self, code: int, message: Optional[str] = None, explain: Optional[str] = None) -> None:
        self.close_connection = True
        self._send_json(code, {"error": message or explain or f"HTTP {code}"})

    def _proxy(self) -> None:
        raw_body = self._read_body()
        clean_path = self.path.split("?", 1)[0]

        # Extract custom human-readable `name` before forwarding `POST /v1/worlds` to upstream
        custom_name: Optional[str] = None
        req_seed: Optional[str] = None
        req_level: int = 8
        if self.command == "POST" and clean_path == "/api/v1/worlds" and raw_body:
            try:
                parsed_body = json.loads(raw_body)
                custom_name = parsed_body.pop("name", None)
                req_seed = parsed_body.get("seed")
                req_level = int(parsed_body.get("level", 8))
                raw_body = json.dumps(parsed_body).encode("utf-8")
            except Exception:
                pass

        body = raw_body or None
        parts = urlsplit(BASE)
        upstream_path = parts.path + self.path[4:]
        hdrs = {"User-Agent": "spread-command-center/2.0"}
        if KEY:
            hdrs["Authorization"] = "Bearer " + KEY
        if body is not None:
            hdrs["Content-Type"] = "application/json"

        for attempt in range(4):
            conn = _upstream_conn()
            try:
                conn.request(self.command, upstream_path, body=body, headers=hdrs)
                resp = conn.getresponse()
                code, data, rhdrs = resp.status, resp.read(), resp.headers
                if code == 429 and attempt < 2:
                    wait_ms = int(rhdrs.get("Retry-After") or rhdrs.get("RateLimit-Reset") or 120)
                    time.sleep(min(1.5, max(0.05, wait_ms / 1000.0)))
                    continue
                out = {h: rhdrs[h] for h in PASS_HEADERS if rhdrs.get(h)}

                # Enrich `GET /api/v1/worlds` with human-readable names, seeds, batches, and cached scores
                if code == 200 and self.command == "GET" and clean_path == "/api/v1/worlds":
                    try:
                        raw_list = json.loads(data)
                        enriched = enrich_worlds(raw_list)
                        data = json.dumps(enriched).encode("utf-8")
                    except Exception:
                        pass
                # Register newly created world with human-readable name
                elif code in (200, 201) and self.command == "POST" and clean_path == "/api/v1/worlds":
                    try:
                        created = json.loads(data)
                        wid = created.get("worldId")
                        if wid:
                            meta = register_world(wid, level=req_level, name=custom_name, seed=req_seed)
                            created["name"] = meta["name"]
                            data = json.dumps(created).encode("utf-8")
                    except Exception:
                        pass
                # Record score automatically when snapshot is fetched
                elif code == 200 and self.command == "GET" and clean_path.startswith("/api/v1/worlds/") and clean_path.count("/") == 4:
                    try:
                        snap_obj = json.loads(data)
                        wid = clean_path.rsplit("/", 1)[-1]
                        if "score" in snap_obj:
                            record_world_score(wid, snap_obj["score"])
                    except Exception:
                        pass

                return self._send(code, data, out)
            except (ConnectionResetError, BrokenPipeError, http.client.BadStatusLine):
                conn.close()
                _tls_local.conn = None
                continue
            except Exception as exc:
                conn.close()
                _tls_local.conn = None
                return self._send_json(502, {"error": f"Proxy could not reach {BASE}: {exc}"})

        self._send_json(502, {"error": f"Upstream connection to {BASE} reset repeatedly"})

    def _handle_bot(self) -> None:
        raw_body = self._read_body()
        clean_path = self.path.split("?", 1)[0]
        qs = parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")

        try:
            if self.command == "GET" and clean_path == "/bot/status":
                wid = qs.get("worldId", [None])[0]
                return self._send_json(200, BOT.status(wid))

            if self.command == "POST" and clean_path == "/bot/start":
                payload = json.loads(raw_body or b"{}")
                wid = payload.get("worldId")
                mode = payload.get("mode", "turbo")
                if mode == "pack":
                    level = int(payload.get("level", 8))
                    count = max(1, min(8, int(payload.get("count", 5))))
                    name = payload.get("name")
                    seed_prefix = payload.get("seedPrefix")
                    return self._send_json(200, BOT.start_pack(level=level, count=count, custom_name=name, seed_prefix=seed_prefix))
                if mode not in ("turbo", "watch", "day", "step", "final"):
                    return self._send_json(400, {"error": f"Unknown mode {mode}"})
                return self._send_json(200, BOT.start(wid, mode))

            if self.command == "POST" and clean_path == "/bot/start-pack":
                payload = json.loads(raw_body or b"{}")
                level = int(payload.get("level", 8))
                count = max(1, min(8, int(payload.get("count", 5))))
                name = payload.get("name")
                seed_prefix = payload.get("seedPrefix")
                return self._send_json(200, BOT.start_pack(level=level, count=count, custom_name=name, seed_prefix=seed_prefix))

            if self.command == "POST" and clean_path == "/bot/stop":
                BOT.stop()
                return self._send_json(200, BOT.status())

            if self.command == "POST" and clean_path == "/bot/rename":
                payload = json.loads(raw_body or b"{}")
                wid = payload.get("worldId")
                new_name = payload.get("name", "")
                if not wid or not new_name.strip():
                    return self._send_json(400, {"error": "worldId and name are required"})
                return self._send_json(200, rename_world(wid, new_name))

            if self.command == "GET" and clean_path == "/bot/worlds-meta":
                client = Client(BASE, KEY)
                enriched = enrich_worlds(client.worlds())
                batches = list_batches(enriched)
                return self._send_json(200, {"worlds": enriched, "batches": batches})

            if self.command == "POST" and clean_path == "/bot/parse-notes":
                payload = json.loads(raw_body or b"[]")
                out: dict[str, Any] = {}
                for item in payload:
                    rid = item.get("id", "")
                    p = parse_note(
                        item.get("note"),
                        int(item.get("arrivedAt", 0)),
                        int(item.get("pickupFrom", 0)),
                        int(item.get("deadline", 0)),
                    )
                    out[rid] = {
                        "needsPlates": p.needs_plates,
                        "manualGearbox": p.manual_gearbox,
                        "summerTyres": p.summer_tyres,
                        "waitPayment": p.wait_payment,
                        "summary": p.summary,
                    }
                return self._send_json(200, out)

            if self.command == "GET" and clean_path == "/bot/analyze":
                force = qs.get("refresh", ["0"])[0] == "1"
                if force:
                    _analysis_cache.clear()

                batch_id = qs.get("batchId", [None])[0]
                world_ids_raw = qs.get("worldIds", [None])[0]
                wid = qs.get("worldId", [None])[0]
                title = qs.get("title", [None])[0]

                client = Client(BASE, KEY)
                if batch_id:
                    enriched = enrich_worlds(client.worlds())
                    batches = list_batches(enriched)
                    b_match = next((b for b in batches if b["batchId"] == batch_id), None)
                    if not b_match or not b_match["worldIds"]:
                        return self._send_json(404, {"error": f"Batch {batch_id} not found or empty"})
                    rep = analyze_multi_worlds(
                        client,
                        b_match["worldIds"],
                        get_cached_map(),
                        title=b_match["batchName"],
                        single_cache=_analysis_cache,
                    )
                    return self._send_json(200, rep)

                if world_ids_raw:
                    wids = [x.strip() for x in world_ids_raw.split(",") if x.strip()]
                    rep = analyze_multi_worlds(
                        client,
                        wids,
                        get_cached_map(),
                        title=title,
                        single_cache=_analysis_cache,
                    )
                    return self._send_json(200, rep)

                if not wid:
                    return self._send_json(400, {"error": "Missing worldId, worldIds, or batchId"})

                if not force and wid in _analysis_cache:
                    return self._send_json(200, _analysis_cache[wid])
                report = analyze_world(client, wid, get_cached_map())
                if report.get("world", {}).get("ended"):
                    _analysis_cache[wid] = report
                return self._send_json(200, report)

            self._send_json(404, {"error": "Not found"})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            return self._proxy()
        if self.path.startswith("/bot/"):
            return self._handle_bot()
        self._read_body()
        if self.path == "/config":
            return self._send_json(200, {"base": BASE, "hasKey": bool(KEY)})

        clean_path = unquote(self.path.split("?", 1)[0])
        if clean_path in ("/analytics", "/analytics/", "/analytics.html"):
            an_file = os.path.join(HERE, "analytics.html")
            if os.path.isfile(an_file):
                with open(an_file, "rb") as f:
                    return self._send(200, f.read(), {"Content-Type": "text/html; charset=utf-8"})
        if clean_path in ("/", "/index.html", "/api interactor9000.html"):
            ui_file = os.path.join(HERE, "index.html")
            if not os.path.isfile(ui_file):
                ui_file = os.path.join(HERE, "api interactor9000.html")
            with open(ui_file, "rb") as f:
                return self._send(200, f.read(), {"Content-Type": "text/html; charset=utf-8"})

        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.startswith("/api/"):
            return self._proxy()
        if self.path.startswith("/bot/"):
            return self._handle_bot()
        self.send_error(405)

    def do_DELETE(self) -> None:
        if self.path.startswith("/api/"):
            return self._proxy()
        self.send_error(405)


def main() -> None:
    if not KEY:
        print(f"WARNING: SPREAD_KEY is not set. Add it to .env or get a key at {BASE}/join", file=sys.stderr)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Spread Command Center live at http://localhost:{PORT}  ->  {BASE}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
