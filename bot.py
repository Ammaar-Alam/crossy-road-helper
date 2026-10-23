"""Follow lanes across frames, predict traffic and pick the next hop."""

from __future__ import annotations

import heapq
from collections import Counter, deque

import numpy as np

import vision as V
from vision import GRASS, LANE, RAIL, ROAD, TILE, WATER

HOP = 0.12  # seconds a hop takes
LAG = 0.15  # seconds from the frame we read to our touch reaching the phone
BODY = 0.35 * TILE  # half the character's width plus a margin
STAY = 0.35  # how long the last spot of a plan must stay safe
ROUTE_AHEAD, ROUTE_BEHIND, ROUTE_SIDE = 4, 3, 7  # lanes and tiles the route finder looks over
EDGE = 0.3  # hops a route's end is worth per tile away from the middle of the view
EXPOSED = 2.0  # score lost for ending a plan on a road or rail rather than somewhere cars cannot reach
MAX_SPEED = 300  # px per second along a lane
WINDOW = (0.1, 0.6)  # how far back, in seconds, frames are lined up to measure a lane's speed
MOVES = {"up": (-1, 0.0), "left": (0, -TILE), "right": (0, TILE), "down": (1, 0.0), "wait": (0, 0.0)}
COSTS = {"up": 0.0, "left": 1.0, "right": 1.0, "down": 1.0, "wait": 0.3}
# seconds from one move to when the next can go, sideways landings show up later than hops forward
TAKES = {"up": 0.24, "down": 0.26, "left": 0.3, "right": 0.3, "wait": 0.2}
IDLE = 0.4  # extra cost of waiting per second without new progress, the eagle comes for anyone who dawdles
WARNED = 4.0  # seconds a warning holds a rail closed when no train has come yet, trains came 0.6 to 3 s after it


class Track:
    """One lane, followed while it is on screen."""

    def __init__(self):
        self.kinds: Counter[str] = Counter()
        self.history: deque[tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=30)
        self.speeds: deque[float] = deque(maxlen=20)
        self.speed = 0.0
        # columns we hopped at and bounced off, whatever was there was not seen
        self.blocked: list[float] = []
        self.bounced: list[float] = []
        # last flash of the warning light and last time a train was in view
        self.flash = -1e9
        self.passed = -1e9
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
            self.flash = t
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
        if lane.pads:
            self.measure(0.0)
        # with nothing moving in sight there is nothing to measure, only noise
        elif self.kind in (ROAD, WATER) and (moving & (lane.solid > 0.5)).sum() >= 12:
            v = self.fit(t, lane.solid, moving)
            if v is not None:
                self.measure(v)
        if self.kind == RAIL and (lane.solid[lane.seen] > 0.5).any():
            self.passed = t
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
        # two speeds that line up about as well, like evenly spaced cars, mean it cannot tell which
        far = rough[np.abs(coarse - coarse[i]) > 30]
        if np.isfinite(far).any() and rough[i] > 0.75 * far[np.isfinite(far)].min():
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

    def train(self, t: float) -> bool:
        """Whether a train is on this rail or may be by time t: one in view, or a warning it has not yet answered."""
        _, solid, seen = self.history[-1]
        if (solid[seen] > 0.5).any() or t - self.passed < 0.3:
            return True
        # the light flashes on and off and can be missed for a while, a warning holds until its train has gone by
        return self.flash > self.passed and t - self.flash < WARNED


class World:
    """Lanes keyed by their place in the world, so they keep their history as the camera scrolls."""

    def __init__(self):
        self.scroll = 0
        self.phase: float | None = None
        self.tracks: dict[int, Track] = {}
        # how far the camera has slid sideways following the character, columns are kept fixed to the world
        self.pan = 0.0
        self.statics: list[tuple[int, float]] = []

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

    def slide(self, statics: list[tuple[int, float]]) -> None:
        """Follow the camera sliding sideways, from screen columns of things that never move, by lane row."""
        now = [(row - self.scroll, col) for row, col in statics]
        moved = [min((col - c0 for l0, c0 in self.statics if l0 == lane), key=abs, default=None) for lane, col in now]
        moved = [d for d in moved if d is not None and abs(d) < 25]
        if len(moved) >= 2:
            self.pan += float(np.median(moved))
        self.statics = now

    def add(self, view: V.View, t: float) -> None:
        if view.phase is None:
            return
        k = -int(round(self.pan))
        ids = set()
        for lane in view.lanes:
            lane.solid, lane.seen = shifted(lane.solid, k, 0.0), shifted(lane.seen, k, False)
            if lane.hidden is not None:
                lane.hidden = shifted(lane.hidden, k, False)
            ids.add(lane.row - self.scroll)
            self.tracks.setdefault(lane.row - self.scroll, Track()).add(lane, t)
        self.tracks = {k: v for k, v in self.tracks.items() if k in ids}


def shifted(a: np.ndarray, k: int, fill) -> np.ndarray:
    """a moved k places along, what comes in is fill."""
    out = np.full_like(a, fill)
    if k >= 0:
        out[k:] = a[: len(a) - k]
    else:
        out[:k] = a[-k:]
    return out


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
        return not track.train(t0)
    if track.kind == WATER:
        # a rider moves with what it stands on, it needs support the whole time and must not be carried off screen
        view = np.flatnonzero(seen)
        for t in (t0, (t0 + t1) / 2, t1):
            col = c + track.speed * (t - t0)
            if not view[0] + TILE / 2 <= col <= view[-1] - TILE / 2:
                return False
            if track.at(t, int(col - BODY / 2), int(col + BODY / 2) + 1).min() <= 0.5:
                return False
        return True
    if not track.measured and (solid[seen] > 0.5).any():
        # cars whose speed is unknown could be anywhere soon
        return False
    for t in [*np.arange(t0, t1, 0.03), t1]:
        if track.at(t, lo, hi).max() > 0.5:
            return False
    return True


def routes(world: World, lane: int, c: float, now: float) -> dict[tuple[int, int], float]:
    """Fewest hops from each spot near the character to a lane well ahead, by lane and tiles sideways.

    Trees and rocks are walls and open water costs extra since a log has to come, traffic is left to the planner.
    Being boxed in by trees then shows up as a route that steps back and goes round."""
    def cost(at: int, k: int) -> float | None:
        track = world.tracks.get(at)
        if track is None or track.kind is None or not track.history:
            return None
        col = c + k * TILE
        _, solid, seen = track.history[-1]
        if not BODY <= col < len(solid) - BODY or not seen[int(col)]:
            return None
        if track.kind == GRASS:
            return 1.0 if safe(track, col, now, now) else None
        if track.kind == WATER:
            return 1.0 if safe(track, col, now, now) else 2.0
        return 1.0

    goal = lane - ROUTE_AHEAD
    while goal < lane and world.tracks.get(goal) is None:
        goal += 1
    # ending near the middle of what can be seen leaves ways open on both sides, the edge leaves one or none
    track = world.tracks.get(goal)
    view = np.flatnonzero(track.history[-1][2]) if track is not None and track.history else np.array([c])
    middle = (view[0] + view[-1]) / 2 if len(view) else c
    hops = {(goal, k): EDGE * abs(c + k * TILE - middle) / TILE
            for k in range(-ROUTE_SIDE, ROUTE_SIDE + 1) if cost(goal, k) is not None}
    heap = [(d, cell) for cell, d in hops.items()]
    while heap:
        d, (at, k) = heapq.heappop(heap)
        if d > hops.get((at, k), np.inf):
            continue
        enter = cost(at, k)
        for nl, nk in ((at + 1, k), (at - 1, k), (at, k - 1), (at, k + 1)):
            if not (goal <= nl <= lane + ROUTE_BEHIND and abs(nk) <= ROUTE_SIDE) or cost(nl, nk) is None:
                continue
            if d + enter < hops.get((nl, nk), np.inf):
                hops[(nl, nk)] = d + enter
                heapq.heappush(heap, (d + enter, (nl, nk)))
    return hops


def plan(world: World, lane: int, c: float, now: float, depth: int = 4, idle: float = 0.0) -> str:
    """First move of the plan that gets furthest forward while staying safe, idle is seconds since the last new lane."""
    best = (-1e9, "wait")

    def drift(at: int, c: float, dt: float) -> float:
        track = world.tracks.get(at)
        return c + track.speed * dt if track is not None and track.kind == WATER else c

    memo: dict[tuple, float] = {}

    hops = routes(world, lane, c, now)

    def togo(at: int, c2: float) -> float:
        """Hops still needed to get well ahead from here, unreachable spots count as far."""
        return hops.get((at, round((c2 - c) / TILE)), 30.0)

    def value(at: int, c: float, t: float, left: int) -> float:
        """Best score reachable from standing at c in lane at at time t, many paths meet the same spots."""
        key = (at, round(c), round(t, 3), left)
        if key in memo:
            return memo[key]
        track = world.tracks.get(at)
        exposed = EXPOSED if track is not None and track.kind in (ROAD, RAIL) else 0.0
        out = -10.0 * togo(at, c) - exposed if safe(track, c, t, t + STAY) else -1e9
        if left > 0:
            for name in MOVES:
                score = attempt(at, c, t, left, name)
                if score is not None:
                    out = max(out, score)
        memo[key] = out
        return out

    def attempt(at: int, c: float, t: float, left: int, name: str) -> float | None:
        """Score of making a move from here, None if it is not safe."""
        here, takes = world.tracks.get(at), TAKES[name]
        dl, dc = MOVES[name]
        cost = COSTS[name] + (IDLE * idle * takes if name in ("wait", "down") else 0.0)
        if name == "wait":
            if not safe(here, c, t, t + takes):
                return None
            return value(at, drift(at, c, takes), t + takes, left - 1) - cost
        # what it stands on carries it until takeoff, what it lands on carries it after, and it stays put until the next move
        to, tc = at + dl, drift(at, c, HOP / 2) + dc
        if not (safe(here, c, t, t + HOP / 2) and safe(world.tracks.get(to), tc, t + HOP / 2, t + takes)):
            return None
        return value(to, drift(to, tc, takes - HOP / 2), t + takes, left - 1) - cost

    for name in MOVES:
        score = attempt(lane, c, now + LAG, depth, name)
        if score is not None and score > best[0]:
            best = (score, name)
    if best[0] > -1e8:
        return best[1]
    # nothing stays safe, so take whichever move keeps the character alive longest
    t = now + LAG

    def lasts(name: str) -> float:
        dl, dc = MOVES[name]
        if name != "wait" and not safe(world.tracks.get(lane), c, t, t + HOP / 2):
            return 0.0
        at, col, start = lane + dl, drift(lane, c, HOP / 2) + dc, t + (HOP / 2 if name != "wait" else 0.0)
        span = 0.0
        while span < 1.0 and safe(world.tracks.get(at), col, start, start + span + 0.05):
            span += 0.05
        return span

    return max(MOVES, key=lasts)
