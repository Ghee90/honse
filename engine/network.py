"""Road network, driver supply geography, opening hours, and snow forecast index."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from spread.types import Audience, DriverProfile, OpeningHours, Place, WorldMap

HOUR, DAY = 60, 24 * 60


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def is_experienced(drv: DriverProfile) -> bool:
    return drv["rating"] >= 4.8 and drv["completedGigs"] >= 50


def driver_reliability(drv: DriverProfile) -> float:
    done = max(1, drv["completedGigs"])
    drop_rate = drv["drops"] / (done + drv["drops"])
    return drv["rating"] - 2.5 * drop_rate


@dataclass
class SnowWindow:
    start: int
    end: int
    places: set[str]
    text: str


class NetworkIndex:
    """Fast spatial, road, driver supply, and weather queries."""

    def __init__(self, world_map: WorldMap, roster: list[DriverProfile]):
        self.places: dict[str, Place] = {p["id"]: p for p in world_map["places"]}
        self.road_min: dict[tuple[str, str], int] = {}
        self.road_km: dict[tuple[str, str], float] = {}
        for r in world_map["roads"]:
            key = (r["from"], r["to"])
            self.road_min[key] = r["minutes"]
            self.road_km[key] = r["km"]

        self.straight_dist: dict[tuple[str, str], float] = {}
        pids = list(self.places.keys())
        for i, a in enumerate(pids):
            pa = self.places[a]
            self.straight_dist[(a, a)] = 0.0
            for b in pids[i + 1 :]:
                pb = self.places[b]
                d = haversine_km(pa["lat"], pa["lon"], pb["lat"], pb["lon"])
                self.straight_dist[(a, b)] = d
                self.straight_dist[(b, a)] = d

        self.drivers: dict[str, DriverProfile] = {}
        self.notif_drivers: dict[str, list[DriverProfile]] = {p: [] for p in pids}  # <= 30 km straight line
        self.region_drivers: dict[str, list[DriverProfile]] = {p: [] for p in pids}  # <= 85 km straight line
        self.supply_score: dict[str, float] = {p: 0.0 for p in pids}
        self.snow_windows: list[SnowWindow] = []
        self.update_roster(roster)

    def update_roster(self, roster: list[DriverProfile]) -> None:
        self.drivers = {d["id"]: d for d in roster}
        for p in self.places:
            self.notif_drivers[p] = []
            self.region_drivers[p] = []
            self.supply_score[p] = 0.0

        for drv in roster:
            home = drv["homePlaceId"]
            for p in self.places:
                dist = self.straight_km(home, p)
                if dist <= 30.0:
                    self.notif_drivers[p].append(drv)
                if dist <= 85.0:
                    self.region_drivers[p].append(drv)
                # Distance-decayed supply accessibility
                if dist <= 160.0:
                    self.supply_score[p] += 1.0 / (1.0 + (dist / 35.0) ** 1.6)

        for p in self.places:
            self.notif_drivers[p].sort(key=driver_reliability, reverse=True)
            self.region_drivers[p].sort(key=driver_reliability, reverse=True)

    @property
    def total_drivers(self) -> int:
        return len(self.drivers)

    def place_name(self, pid: str) -> str:
        p = self.places.get(pid)
        return p["name"] if p else pid

    def minutes(self, u: str, v: str) -> int:
        if u == v:
            return self.road_min.get((u, v), 10)
        return self.road_min.get((u, v), 180)

    def km(self, u: str, v: str) -> float:
        if u == v:
            return self.road_km.get((u, v), 5.0)
        return self.road_km.get((u, v), 150.0)

    def straight_km(self, u: str, v: str) -> float:
        if u == v:
            return 0.0
        return self.straight_dist.get((u, v), 200.0)

    def add_forecast(self, start: int, end: int, place_ids: list[str], text: str = "") -> None:
        self.snow_windows.append(SnowWindow(start=start, end=end, places=set(place_ids), text=text))

    def has_snow(self, place_id: str, t_from: int, t_to: int) -> bool:
        for sw in self.snow_windows:
            if place_id in sw.places and sw.start <= t_to and sw.end >= t_from:
                return True
        return False

    def snow_end_after(self, place_id: str, t: int) -> Optional[int]:
        latest = None
        for sw in self.snow_windows:
            if place_id in sw.places and sw.start <= t <= sw.end:
                latest = max(latest or 0, sw.end + 15)
        return latest

    def matching_notif_count(self, place_id: str, audience: Audience, region: bool = False) -> int:
        pool = self.region_drivers.get(place_id, []) if region else self.notif_drivers.get(place_id, [])
        cnt = 0
        for d in pool:
            if audience["experienced"] and not is_experienced(d):
                continue
            if audience["manualLicence"] and d["licence"] != "manual":
                continue
            cnt += 1
        return cnt

    def candidate_drivers(
        self,
        pickup_place: str,
        delivery_place: str,
        audience: Audience,
        finishing_places: dict[str, tuple[str, int]],
        active_today: set[str],
        now: int,
        lead_mins: int = 240,
    ) -> list[tuple[float, str, str]]:
        """Scores drivers for direct offers -> (score, driver_id, kind: 'finisher'|'homebound'|'local')."""
        scored: list[tuple[float, str, str]] = []
        trip_km = self.straight_km(pickup_place, delivery_place)
        for did, drv in self.drivers.items():
            if audience["experienced"] and not is_experienced(drv):
                continue
            if audience["manualLicence"] and drv["licence"] != "manual":
                continue

            home = drv["homePlaceId"]
            d_home_dest = self.straight_km(delivery_place, home)
            fin = finishing_places.get(did)
            is_active = did in active_today
            rel = driver_reliability(drv) * 3.5

            # 1. Finisher near pickup: saves outbound ticket AND app doesn't notify them here!
            if fin and now - 60 <= fin[1] <= now + 240:
                d_fin_pick = self.straight_km(fin[0], pickup_place)
                if d_fin_pick <= 45.0:
                    # Huge bonus if this car also brings them closer to home!
                    home_saving = max(-15.0, (self.straight_km(fin[0], home) - d_home_dest) * 0.25)
                    sc = 55.0 - d_fin_pick * 0.70 + home_saving + (14.0 if is_active else 0.0) + rel
                    scored.append((sc, did, "finisher"))
                    continue

            # 2. Homebound driver living at delivery town (pays zero day-of return ticket; gets no push notif if >30km!)
            d_home_pick = self.straight_km(home, pickup_place)
            if d_home_dest <= 25.0 and d_home_pick > 30.0 and lead_mins >= max(180, int(trip_km * 0.85)):
                if d_home_pick <= 340.0 and (is_active or d_home_pick <= 180.0):
                    sc = 32.0 - d_home_dest * 0.60 - d_home_pick * 0.08 + (24.0 if is_active else 0.0) + rel
                    scored.append((sc, did, "homebound"))
                    continue

            # 3. Local pickup-area driver
            if d_home_pick <= 80.0:
                sc = 18.0 - d_home_pick * 0.45 - d_home_dest * 0.18 + (20.0 if is_active else 0.0) + rel
                scored.append((sc, did, "local"))

        scored.sort(reverse=True)
        return scored


def site_day_hours(hours: Optional[OpeningHours], day_index: int) -> Optional[tuple[int, int]]:
    """Returns (opens, closes) minute-of-day for a dealer site on `day_index`, or None if closed all day (e.g. Sunday)."""
    if hours is None:
        return (0, DAY)
    wd = day_index % 7  # 0=Mon .. 5=Sat, 6=Sun
    if wd == 6:
        return None
    if wd == 5:
        sat = hours.get("saturday")
        if not sat:
            return None
        return (sat["opens"], sat["closes"])
    # Weekdays in schema may be 0..4 or 1..5; handle both gracefully
    wds = hours.get("weekdays", [0, 1, 2, 3, 4])
    if wds and (wd not in wds) and ((wd + 1) not in wds):
        return None
    return (hours["opens"], hours["closes"])
