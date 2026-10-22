"""Follow lanes across frames, predict traffic and pick the next hop."""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

import vision as V
from vision import GRASS, LANE, RAIL, ROAD, TILE, WATER

HOP = 0.12  # seconds a hop takes
LAG = 0.15  # seconds from the frame we read to our touch reaching the phone
BODY = 0.35 * TILE  # half the character's width plus a margin
STAY = 0.35  # how long the last spot of a plan must stay safe
MAX_SPEED = 300  # px per second along a lane
BASELINE = 0.2  # seconds between the two looks used to measure a lane's speed
MOVES = {"up": (-1, 0.0), "left": (0, -TILE), "right": (0, TILE), "down": (1, 0.0), "wait": (0, 0.0)}


def shift_of(a: np.ndarray, sa: np.ndarray, b: np.ndarray, sb: np.ndarray, lo: int, hi: int) -> float | None:
    """How far profile a moved to become b, searched over lo..hi px, None when there is nothing to line up."""
    if a[sa].std() < 0.1:
        return None
    scores = {}
    for s in range(lo - 1, hi + 2):
        a0, a1 = max(0, -s), min(len(a), len(a) - s)
        both = sa[a0:a1] & sb[a0 + s : a1 + s]
        if both.sum() >= 60:
            scores[s] = np.abs(a[a0:a1][both] - b[a0 + s : a1 + s][both]).mean()
    inside = {s: v for s, v in scores.items() if lo <= s <= hi}
    if not inside:
        return None
    best = min(inside, key=inside.get)
    if inside[best] > 0.25:
        return None
    left, right = scores.get(best - 1), scores.get(best + 1)
    if left is None or right is None or left + right - 2 * inside[best] <= 0:
        return float(best)
    # vertex of the parabola through the three scores
    return best + 0.5 * (left - right) / (left + right - 2 * inside[best])


class Track:
    """One lane, followed while it is on screen."""

    def __init__(self):
        self.kinds: Counter[str] = Counter()
        self.history: deque[tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=30)
        self.rough: deque[float] = deque(maxlen=15)
        self.speeds: deque[float] = deque(maxlen=15)
        self.signal = False

    @property
    def kind(self) -> str | None:
        return self.kinds.most_common(1)[0][0] if self.kinds else None

    @property
    def speed(self) -> float:
        if len(self.speeds) >= 3:
            return float(np.median(self.speeds))
        return float(np.median(self.rough)) if self.rough else 0.0

    def add(self, lane: V.Lane, t: float) -> None:
        if lane.kind:
            self.kinds[lane.kind] += 1
        if self.kind in (ROAD, WATER) and self.history:
            # frame to frame shifts are too small to confuse one car for the next
            t0, solid, seen = self.history[-1]
            if 0 < t - t0 <= 0.1:
                reach = int(np.ceil(MAX_SPEED * (t - t0)))
                s = shift_of(solid, seen, lane.solid, lane.seen, -reach, reach)
                if s is not None:
                    self.rough.append(s / (t - t0))
            # then a longer look near the rough guess for precision
            old = next((h for h in self.history if t - h[0] <= 2 * BASELINE), None)
            if len(self.rough) >= 3 and old is not None and t - old[0] >= BASELINE:
                guess = float(np.median(self.rough)) * (t - old[0])
                s = shift_of(old[1], old[2], lane.solid, lane.seen, int(guess) - 4, int(guess) + 4)
                if s is not None:
                    self.speeds.append(s / (t - old[0]))
        self.history.append((t, lane.solid, lane.seen))
        self.signal = lane.signal

    def at(self, t: float, lo: int, hi: int) -> np.ndarray:
        """Solid fraction of columns lo..hi expected at time t, off screen counts as solid on roads."""
        t0, solid, seen = self.history[-1]
        fill = 1.0 if self.kind == ROAD else 0.0
        a = lo - int(round(self.speed * (t - t0)))
        out = np.full(hi - lo, fill, np.float32)
        s0, s1 = max(a, 0), min(a + hi - lo, len(solid))
        if s1 > s0:
            out[s0 - a : s1 - a] = np.where(seen[s0:s1], solid[s0:s1], fill)
        return out

    def train(self) -> bool:
        _, solid, seen = self.history[-1]
        return self.signal or bool((solid[seen] > 0.5).mean() > 0.15)


class World:
    """Lanes keyed by their place in the world, so they keep their history as the camera scrolls."""

    def __init__(self):
        self.scroll = 0
        self.phase: float | None = None
        self.tracks: dict[int, Track] = {}

    def update(self, view: V.View, t: float) -> None:
        if view.phase is None:
            return
        if self.phase is not None:
            d = view.phase - self.phase
            # lanes slide down as the camera moves on, when the grid wraps the same lane has a new row
            if d < -LANE / 2:
                self.scroll += 1
            elif d > LANE / 2:
                self.scroll -= 1
        self.phase = view.phase
        ids = set()
        for lane in view.lanes:
            ids.add(lane.row - self.scroll)
            self.tracks.setdefault(lane.row - self.scroll, Track()).add(lane, t)
        self.tracks = {k: v for k, v in self.tracks.items() if k in ids}

    def lane_of(self, y: float) -> int:
        return int((y - self.phase) // LANE) - self.scroll


def safe(track: Track | None, c: float, t0: float, t1: float) -> bool:
    """Whether standing at column c of this lane is safe from t0 to t1."""
    if track is None or track.kind is None or not track.history:
        return False
    lo, hi = int(c - BODY), int(c + BODY) + 1
    if lo < 0 or hi > V.COLUMNS:
        return False
    if track.kind == GRASS:
        return track.history[-1][1][int(c - BODY / 2) : int(c + BODY / 2) + 1].max(initial=0) < 0.5
    if track.kind == RAIL:
        return not track.train()
    if track.kind == WATER:
        return track.at(t0, int(c) - 4, int(c) + 5).min() > 0.5
    for t in np.arange(t0, t1 + 1e-6, 0.03):
        if track.at(t, lo, hi).max() > 0.5:
            return False
    return True


def plan(world: World, lane: int, c: float, now: float, depth: int = 4) -> str:
    """First move of the plan that gets furthest forward while staying safe."""
    best = (-1e9, "wait")

    def drift(at: int, c: float, dt: float) -> float:
        track = world.tracks.get(at)
        return c + track.speed * dt if track is not None and track.kind == WATER else c

    def go(at: int, c: float, t: float, left: int, first: str | None, cost: float):
        nonlocal best
        if not safe(world.tracks.get(at), c, t, t + STAY):
            pass
        elif first is not None:
            score = (lane - at) * 10 - cost
            if score > best[0]:
                best = (score, first)
        if left == 0:
            return
        for name, (dl, dc) in MOVES.items():
            to, tc = at + dl, drift(at, c, HOP) + dc
            here = world.tracks.get(at)
            there = world.tracks.get(to)
            if name == "wait":
                if not safe(here, c, t, t + HOP) or (here and here.kind == RAIL):
                    continue
            elif not (safe(here, c, t, t + HOP / 2) and safe(there, tc, t + HOP / 2, t + HOP + 0.05)):
                continue
            go(to, tc, t + HOP, left - 1, first or name, cost + (0 if name == "up" else 0.3 if name == "wait" else 1))

    go(lane, c, now + LAG, depth, None, 0)
    return best[1]
