#!/usr/bin/env python3
"""Spread Smart Dispatcher CLI (Practice, 5-Week Practice Pack, Multi-Seed Sweep, Ranking Runs, and Live Final).

Examples:
    python3 dispatch.py --level 8 --weeks 5 --name "Aurora"
    python3 dispatch.py --level 8 --seed spread-1 --name "Baseline-1"
    python3 dispatch.py --level 8 --sweep spread-1,spread-2,spread-3
    python3 dispatch.py --level 8 --board
    python3 dispatch.py --final
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from engine.config import load_env
from engine.desk import DAY, WAKE_ON, SmartDesk
from engine.namer import create_practice_pack, record_world_score, register_world
from spread.client import Client, SpreadError, Unreachable, World
from spread.types import Score, WorldMap

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def report(label: str, score: Score) -> None:
    orders = f"{score['wonOrders']:>4} orders  {score['onTimeShare']:6.1%} on time  {score['undeliveredShare']:5.1%} not delivered"
    rank_tag = "RANKED" if score["ranked"] else "UNRANKED"
    print(
        f"{label:<11}  margin €{score['marginCents'] / 100:>11,.2f}  revenue €{score['revenueCents'] / 100:>11,.2f}  {orders}  [{rank_tag}]",
        flush=True,
    )


def play_practice(world: World, level: int, world_map: WorldMap, quiet: bool = False) -> tuple[Score, SmartDesk]:
    snapshot = world.snapshot()
    roster = world.roster()
    desk = SmartDesk(world, level, snapshot["now"], world_map, roster)
    if snapshot["now"] > 0:
        desk.rejoin(snapshot, world.events())
    ends_at = snapshot["endsAt"]

    while desk.now < ends_at:
        day, midnight = desk.now // DAY, (desk.now // DAY + 1) * DAY
        actions, desk.outbox = desk.outbox, []
        step = world.advance(until=WAKE_ON, max=min(ends_at, midnight), actions=actions)
        desk.read_results(actions, step.get("results", []))
        desk.now = step["now"]
        for event in step["events"]:
            desk.handle(event)
        if desk.now // DAY > day and not quiet:
            report(WEEKDAYS[day % 7], world.score())

    if desk.outbox:
        desk.send()
    sc = world.score()
    record_world_score(world.id, sc)  # type: ignore[arg-type]
    return sc, desk


def next_final(client: Client) -> World:
    said = False
    while True:
        try:
            world: Optional[World] = client.world(client.final()["worldId"])
        except SpreadError as error:
            if error.status != 404:
                raise
            world = None
        if world is not None:
            snapshot = world.snapshot()
            if snapshot["now"] < snapshot["endsAt"]:
                return world
        if not said:
            print("Waiting for final / test-final world to open...", flush=True)
            said = True
        time.sleep(5)


def play_final(world: World, world_map: WorldMap) -> Score:
    history = world.events()
    snapshot = world.snapshot()
    if snapshot["now"] >= snapshot["endsAt"]:
        return snapshot["score"]
    level = next((w["level"] for w in world.client.worlds() if w["worldId"] == world.id), 8)
    register_world(world.id, level=level, kind="final")
    roster = world.roster()
    desk = SmartDesk(world, level, snapshot["now"], world_map, roster)
    desk.rejoin(snapshot, history)
    desk.send()

    last_roster_sync = time.monotonic()
    for event in world.stream(after=history[-1]["seq"] if history else -1, idle=1.0):
        day = desk.now // DAY
        desk.now = world.client.final()["now"] if event is None else max(desk.now, event["at"])
        if event is not None:
            desk.handle(event)
        desk.send()
        if time.monotonic() - last_roster_sync > 45.0:
            try:
                desk.net.update_roster(world.roster())
                last_roster_sync = time.monotonic()
            except SpreadError:
                pass
        if desk.now // DAY > day:
            report(WEEKDAYS[day % 7], world.score())
        if desk.now >= snapshot["endsAt"]:
            break
    sc = world.score()
    record_world_score(world.id, sc)  # type: ignore[arg-type]
    return sc


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Spread Smart Python Dispatcher")
    parser.add_argument("--level", type=int, choices=range(1, 9), default=8, help="practice level 1..8 (default 8)")
    parser.add_argument("--seed", help="deterministic practice seed (or seed prefix when used with --weeks)")
    parser.add_argument("--name", help="human-readable name for the world or 5-week practice pack")
    parser.add_argument("--weeks", type=int, default=1, help="run N consecutive practice weeks (e.g. --weeks 5)")
    parser.add_argument("--pack", action="store_true", help="shortcut for --weeks 5 (run 5 consecutive practice weeks)")
    parser.add_argument("--sweep", help="comma-separated seeds to benchmark and optionally delete")
    parser.add_argument("--keep", action="store_true", help="keep practice world after completion")
    parser.add_argument("--delete-after", action="store_true", help="delete practice world after completion to free clutter")
    parser.add_argument("--board", action="store_true", help="play official practice ranking run")
    parser.add_argument("--final", action="store_true", help="connect to live final / test-final stream")
    args = parser.parse_args()

    weeks_to_run = 5 if args.pack and args.weeks == 1 else args.weeks

    try:
        client = Client()
        world_map = client.map()

        if weeks_to_run > 1 and not args.board and not args.final and not args.sweep:
            pack = create_practice_pack(
                level=args.level,
                count=weeks_to_run,
                custom_name=args.name,
                seed_prefix=args.seed,
            )
            print(f"=== Starting {pack['batchName']} [batchId={pack['batchId']}] ===", flush=True)
            margins, on_times, ranked_cnt = [], [], 0
            world_ids = []
            for spec in pack["specs"]:
                t0 = time.monotonic()
                w = client.create(level=args.level, seed=spec["seed"], board=False)
                register_world(
                    w.id,
                    level=args.level,
                    kind="practice",
                    name=spec["name"],
                    seed=spec["seed"],
                    batch_id=pack["batchId"],
                    batch_name=pack["batchName"],
                    batch_index=spec["index"],
                    batch_total=spec["total"],
                )
                world_ids.append(w.id)
                print(f"\n--- Running {spec['name']} [{w.id}] (seed={spec['seed']}) ---", flush=True)
                sc, _ = play_practice(w, args.level, world_map, quiet=False)
                dt = time.monotonic() - t0
                margins.append(sc["marginCents"])
                on_times.append(sc["onTimeShare"])
                if sc["ranked"]:
                    ranked_cnt += 1
                report(f"Week {spec['index']}/{spec['total']}", sc)
                print(f"Completed {spec['name']} in {dt:.1f}s", flush=True)
                if args.delete_after:
                    try:
                        w.delete()
                    except SpreadError:
                        pass

            tot_m = sum(margins)
            avg_m = tot_m / max(1, len(margins))
            avg_ot = sum(on_times) / max(1, len(on_times))
            print(
                f"\n=== {pack['batchName']} Complete ==="
                f"\n  Total Margin: €{tot_m/100:,.2f} (Mean €{avg_m/100:,.2f}/week)"
                f"\n  Ranked Weeks: {ranked_cnt}/{len(margins)} | Mean On-Time: {avg_ot:.1%}"
                f"\n  Analytics URL: http://localhost:8765/analytics?batchId={pack['batchId']}",
                flush=True,
            )
            return

        if args.sweep:
            seeds = [s.strip() for s in args.sweep.split(",") if s.strip()]
            margins, on_times, ranked_cnt = [], [], 0
            print(f"Benchmarking {len(seeds)} seeds at Level {args.level}...", flush=True)
            for sd in seeds:
                t0 = time.monotonic()
                w = client.create(level=args.level, seed=sd, board=False)
                meta = register_world(w.id, level=args.level, seed=sd, name=f"{args.name + ' · ' if args.name else ''}{sd} (L{args.level})")
                try:
                    sc, _ = play_practice(w, args.level, world_map, quiet=True)
                    dt = time.monotonic() - t0
                    margins.append(sc["marginCents"])
                    on_times.append(sc["onTimeShare"])
                    if sc["ranked"]:
                        ranked_cnt += 1
                    print(
                        f"  {meta['name']:<26} margin €{sc['marginCents']/100:>10,.2f} | rev €{sc['revenueCents']/100:>10,.2f} "
                        f"| {sc['wonOrders']:>4} ord | {sc['onTimeShare']:6.1%} on-time | {'RANKED' if sc['ranked'] else 'FAIL'} ({dt:.1f}s)",
                        flush=True,
                    )
                finally:
                    if args.delete_after:
                        try:
                            w.delete()
                        except SpreadError:
                            pass
            avg_m = sum(margins) / max(1, len(margins))
            avg_ot = sum(on_times) / max(1, len(on_times))
            print(
                f"Summary ({len(seeds)} seeds): Ranked {ranked_cnt}/{len(seeds)} | Mean Margin €{avg_m/100:,.2f} | Mean On-Time {avg_ot:.1%}",
                flush=True,
            )
            return

        if args.final:
            world: Optional[World] = None
            while True:
                try:
                    if world is None:
                        world = next_final(client)
                        print(f"Connected to Final world {world.id}.", flush=True)
                    score = play_final(world, world_map)
                    break
                except Unreachable as err:
                    print(f"spread: {err}. Reconnecting in 5 seconds...", file=sys.stderr, flush=True)
                    time.sleep(5)
            report("End", score)
            return

        world = client.create(level=args.level, seed=args.seed, board=args.board)
        meta = register_world(world.id, level=args.level, name=args.name, seed=args.seed)
        label = f"{meta['name']} [{world.id}]"
        print(f"Started {label}...", flush=True)
        try:
            score, _ = play_practice(world, args.level, world_map)
        except BaseException as exc:
            if not args.board:
                try:
                    world.delete()
                except SpreadError:
                    pass
            if isinstance(exc, KeyboardInterrupt):
                sys.exit(f"\nStopped {label}.")
            raise

        report("End", score)
        if args.delete_after and not args.board:
            world.delete()
    except SpreadError as err:
        sys.exit(f"spread: {err}")


if __name__ == "__main__":
    main()
