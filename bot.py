"""Follow lanes across frames, predict traffic and pick the next hop."""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

import vision as V
from vision import GRASS, LANE, RAIL, ROAD, TILE, WATER

HOP = 0.12  # seconds a hop takes
STEP = 0.22  # seconds between our hops, a frame has to come in and be read before the next one
LAG = 0.15  # seconds from the frame we read to our touch reaching the phone
BODY = 0.35 * TILE  # half the character's width plus a margin
STAY = 0.35  # how long the last spot of a plan must stay safe
MAX_SPEED = 300  # px per second along a lane
WINDOW = (0.1, 0.6)  # how far back, in seconds, frames are lined up to measure a lane's speed
MOVES = {"up": (-1, 0.0), "left": (0, -TILE), "right": (0, TILE), "down": (1, 0.0), "wait": (0, 0.0)}
COSTS = {"up": 0.0, "left": 1.0, "right": 1.0, "down": 1.0, "wait": 0.3}


class Track:
    """One lane, followed while it is on screen."""

    def __init__(self):
        self.kinds: Counter[str] = Counter()
        self.history: deque[tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=30)
        self.speeds: deque[float] = deque(maxlen=20)
        self.speed = 0.0
        # columns we hopped at and bounced off, whatever was there was not seen
        self.blocked: list[float] = []
        self.signal_at = -1e9
        # since when each column has been solid, things that never move are not traffic
        self.since = np.full(V.COLUMNS, np.inf)

    @property
    def kind(self) -> str | None:
        return self.kinds.most_common(1)[0][0] if self.kinds else None

    def measure(self, v: float) -> None:
        # the planner asks for the speed thousands of times a frame, so the median is kept, not recomputed
        self.speeds.append(v)
        self.speed = float(np.median(self.speeds))

    @property
    def measured(self) -> bool:
        return len(self.speeds) >= 3

    def add(self, lane: V.Lane, t: float) -> None:
        if lane.kind:
            self.kinds[lane.kind] += 1
        if lane.signal:
            self.signal_at = t
        # a frame that read this lane as something else would describe the wrong things in it
        if self.kind is not None and lane.kind != self.kind:
            return
        observed = lane.seen.copy()
        if lane.hidden is not None:
            observed &= ~lane.hidden
            lane.seen = observed.copy()
            if self.history:
                # carry what was there before along at the lane's speed
                behind = np.flatnonzero(lane.hidden)
                t0, solid0, seen0 = self.history[-1]
                src = behind - int(round(self.speed * (t - t0)))
                ok = (src >= 0) & (src < len(solid0))
                ok[ok] = seen0[src[ok]]
                lane.solid[behind[ok]] = solid0[src[ok]]
                lane.seen[behind[ok]] = True
        self.since = np.where((lane.solid > 0.5) & lane.seen, np.minimum(self.since, t), np.inf)
        moving = observed & (t - self.since < 0.5)
        # with nothing moving in sight there is nothing to measure, only noise
        # with nothing moving in sight there is nothing to measure, only noise
        if self.kind in (ROAD, WATER) and (moving & (lane.solid > 0.5)).sum() >= 12:
            v = self.fit(t, lane.solid, moving)
            if v is not None:
                self.measure(v)
        self.history.append((t, lane.solid, lane.seen))

    def fit(self, t: float, solid: np.ndarray, moving: np.ndarray) -> float | None:
        """Speed that best lines up the last half second of this lane with what it shows now."""
        frames = [h for h in self.history if WINDOW[0] <= t - h[0] <= WINDOW[1]]
        cols = np.flatnonzero(moving)
        if len(frames) < 2 or len(cols) < 60:
            return None
        frames = frames[:: max(1, len(frames) // 4)]

        def scores(vs: np.ndarray) -> np.ndarray:
            total, count = np.zeros(len(vs)), np.zeros(len(vs))
            for t0, solid0, seen0 in frames:
                idx = cols[None, :] - np.round(vs[:, None] * (t - t0)).astype(int)
                ok = (idx >= 0) & (idx < len(solid0))
                idx = np.clip(idx, 0, len(solid0) - 1)
                ok &= seen0[idx]
                total += (np.abs(solid0[idx] - solid[cols]) * ok).sum(axis=1)
                count += ok.sum(axis=1)
            return np.where(count > 0.5 * len(cols) * len(frames), total / np.maximum(count, 1), np.inf)

        # every look searches every speed, a wrong early guess must never stick
        coarse = np.arange(-MAX_SPEED, MAX_SPEED + 1, 6.0)
        rough = scores(coarse)
        i = int(np.argmin(rough))
        finite = rough[np.isfinite(rough)]
        # a clear winner, not one of many equally poor guesses
        if not np.isfinite(rough[i]) or rough[i] > 0.6 * np.median(finite):
            return None
        vs = np.arange(coarse[i] - 6, coarse[i] + 7, 1.0)
        score = scores(vs)
        i = int(np.argmin(score))
        if not np.isfinite(score[i]) or score[i] > 0.15:
            return None
        if 0 < i < len(vs) - 1 and np.isfinite(score[i - 1]) and np.isfinite(score[i + 1]):
            curve = score[i - 1] + score[i + 1] - 2 * score[i]
            if curve > 0:
                return float(vs[i] + 0.5 * (score[i - 1] - score[i + 1]) / curve)
        return float(vs[i])

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
        t, solid, seen = self.history[-1]
        # the warning light flashes, so it counts for a while after each flash
        return t - self.signal_at < 1.0 or bool((solid[seen] > 0.5).mean() > 0.15)


class World:
    """Lanes keyed by their place in the world, so they keep their history as the camera scrolls."""

    def __init__(self):
        self.scroll = 0
        self.phase: float | None = None
        self.tracks: dict[int, Track] = {}

    def update(self, view: V.View, t: float) -> None:
        self.follow(view)
        self.add(view, t)

    def follow(self, view: V.View) -> None:
        """Keep count of lanes the camera moved past."""
        if view.phase is None:
            return
        if self.phase is not None:
            d = view.phase - self.phase
            # lanes slide down as the camera moves on, when the grid wraps the same lane has a new row
            near = self.scroll + (1 if d < -LANE / 2 else -1 if d > LANE / 2 else 0)

            def fit(scroll: int) -> int:
                kinds = [(lane.kind, self.tracks.get(lane.row - scroll)) for lane in view.lanes if lane.kind]
                return sum(1 if track.kind == kind else -1 for kind, track in kinds if track is not None and track.kind)

            # a slow frame can move the grid over half a lane, the lane kinds tell which way it went
            self.scroll = max((near, near + 1, near - 1), key=lambda s: (fit(s), s == near))
        self.phase = view.phase

    def add(self, view: V.View, t: float) -> None:
        if view.phase is None:
            return
        ids = set()
        for lane in view.lanes:
            ids.add(lane.row - self.scroll)
            self.tracks.setdefault(lane.row - self.scroll, Track()).add(lane, t)
        self.tracks = {k: v for k, v in self.tracks.items() if k in ids}


def safe(track: Track | None, c: float, t0: float, t1: float) -> bool:
    """Whether standing at column c of this lane is safe from t0 to t1."""
    if track is None or track.kind is None or not track.history:
        return False
    lo, hi = int(c - BODY), int(c + BODY) + 1
    if lo < 0 or hi > V.COLUMNS:
        return False
    _, solid, seen = track.history[-1]
    if not seen[lo:hi].all():
        return False
    if track.kind == GRASS:
        if any(abs(b - c) < TILE / 2 for b in track.blocked):
            return False
        return solid[int(c - BODY / 2) : int(c + BODY / 2) + 1].max(initial=0) < 0.5
    if track.kind == RAIL:
        return not track.train()
    if track.kind == WATER:
        return track.at(t0, int(c - BODY / 2), int(c + BODY / 2) + 1).min() > 0.5
    if not track.measured and (solid[seen] > 0.5).any():
        # cars whose speed is unknown could be anywhere soon
        return False
    for t in [*np.arange(t0, t1, 0.03), t1]:
        if track.at(t, lo, hi).max() > 0.5:
            return False
    return True


def plan(world: World, lane: int, c: float, now: float, depth: int = 4) -> str:
    """First move of the plan that gets furthest forward while staying safe."""
    best = (-1e9, "wait")

    def drift(at: int, c: float, dt: float) -> float:
        track = world.tracks.get(at)
        return c + track.speed * dt if track is not None and track.kind == WATER else c

    memo: dict[tuple, float] = {}

    def toward(at: int, c: float, t: float) -> float:
        """Credit for standing near somewhere the lane ahead can be entered, a tile closer is worth a sideways hop."""
        track = world.tracks.get(at)
        if track is None or track.kind is None or not track.history:
            return 0.0
        for d in range(0, 7):
            for side in ((0,) if d == 0 else (-1, 1)):
                if safe(track, c + side * d * TILE, t, t + HOP):
                    return -1.0 * d
        return -7.0

    def value(at: int, c: float, t: float, left: int) -> float:
        """Best score reachable from standing at c in lane at at time t, many paths meet the same spots."""
        key = (at, round(c), round(t, 3), left)
        if key in memo:
            return memo[key]
        here = world.tracks.get(at)
        out = (lane - at) * 10.0 + toward(at - 1, c, t + STEP) if safe(here, c, t, t + STAY) else -1e9
        if left > 0:
            for name, (dl, dc) in MOVES.items():
                to, tc = at + dl, drift(at, c, STEP) + dc
                if name == "wait":
                    if not safe(here, c, t, t + STEP):
                        continue
                # after landing we stay put until the next hop can go
                elif not (safe(here, c, t, t + HOP / 2) and safe(world.tracks.get(to), tc, t + HOP / 2, t + STEP)):
                    continue
                out = max(out, value(to, tc, t + STEP, left - 1) - COSTS[name])
        memo[key] = out
        return out

    t = now + LAG
    here = world.tracks.get(lane)
    for name, (dl, dc) in MOVES.items():
        to, tc = lane + dl, drift(lane, c, STEP) + dc
        if name == "wait":
            if not safe(here, c, t, t + STEP):
                continue
        elif not (safe(here, c, t, t + HOP / 2) and safe(world.tracks.get(to), tc, t + HOP / 2, t + STEP)):
            continue
        score = value(to, tc, t + STEP, depth - 1) - COSTS[name]
        if score > best[0]:
            best = (score, name)
    if best[0] > -1e8:
        return best[1]
    # nothing is safe for long, so jump to wherever is clear on landing rather than wait to be hit
    t = now + LAG
    for name in ("up", "left", "right", "down"):
        dl, dc = MOVES[name]
        if safe(world.tracks.get(lane + dl), drift(lane, c, HOP) + dc, t + HOP / 2, t + HOP + 0.05):
            return name
    return "wait"
