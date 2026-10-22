"""Read lanes, traffic and the player from a canonical frame.

Work happens on a half size copy sheared so lanes run flat. In that space a
lane is LANE px tall and a column leans right by LEAN px for every px up, so
x + LEAN * y names the same column in every lane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

import screen

SCALE = 0.5
SLOPE = math.tan(math.radians(14.1))
LANE = 33.2
TILE = 41.5
LEAN = 0.46
WIDTH = round(screen.WIDTH * SCALE)
LIFT = SLOPE * WIDTH / 2
HEIGHT = round(screen.HEIGHT * SCALE + 2 * LIFT)
COLUMNS = round(WIDTH + LEAN * HEIGHT) + 1
SHEAR = np.float32([[1, 0, 0], [-SLOPE, 1, 2 * LIFT]])

GRASS, ROAD, WATER, RAIL = "grass", "road", "water", "rail"

# where the character waits on the home screen, canonical px
HOME = (335, 990)
FEET = 18  # canonical px from the middle of the body down to the feet


@dataclass(slots=True)
class Lane:
    row: int  # lane index from the top of the sheared frame
    kind: str | None
    top: float
    solid: np.ndarray  # per column, cars on roads, platforms on water, trees on grass, trains on rails
    seen: np.ndarray  # per column, whether the lane is on screen there
    signal: bool = False


@dataclass(slots=True)
class View:
    phase: float | None
    lanes: list[Lane]
    over: bool
    home: bool


def shear(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (WIDTH, round(screen.HEIGHT * SCALE)), interpolation=cv2.INTER_AREA)
    return cv2.warpAffine(small, SHEAR, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST)


def to_sheared(x: float, y: float) -> tuple[float, float]:
    """Canonical px to sheared px."""
    x, y = x * SCALE, y * SCALE
    return x, y - SLOPE * x + 2 * LIFT


def to_canonical(x: float, y: float) -> tuple[float, float]:
    return x / SCALE, (y + SLOPE * x - 2 * LIFT) / SCALE


def column(x: float, y: float) -> float:
    """Sheared px to the column coordinate shared by every lane."""
    return x + LEAN * y


def masks(img: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(np.int16) for i in range(3))
    red = (h >= 165) | (h <= 5)
    return {
        "valid": img.max(axis=2) > 0,
        "grass": (h >= 33) & (h <= 48) & (s > 100) & (v > 100),
        # olive leaves, yellow taxis and green cars sit either side of it
        "tree": (h >= 31) & (h <= 40) & (s > 130),
        "road": (h >= 104) & (h <= 125) & (s > 35) & (s < 95) & (v >= 60) & (v <= 140),
        "water": (h >= 88) & (h <= 104) & (s > 90) & (v > 190),
        "foam": (s < 60) & (v > 200),
        "dark": (v < 58) & (v > 8),
        "log": red & (s > 90) & (s < 165) & (v > 70) & (v < 150),
        "trunk": ((h >= 150) | (h <= 5)) & (s > 40) & (s < 140) & (v > 35) & (v < 75),
        "rock": (h >= 115) & (h <= 140) & (s > 15) & (s < 110) & (v > 150),
        "glow": red & (s > 120) & (v > 220),
    }


def lane_phase(m: dict[str, np.ndarray]) -> float | None:
    """Offset of the lane grid, from rows where the ground type changes, None when there are too few."""
    valid = np.maximum(m["valid"].sum(axis=1), 1)
    frac = np.stack([m[k].sum(axis=1) / valid for k in ("grass", "road", "water", "dark")], axis=1)
    edge = np.abs(np.diff(frac, axis=0)).sum(axis=1)
    edge[:60] = 0
    edge[-60:] = 0
    edge[edge < 0.4] = 0
    if edge.sum() < 2:
        return None
    y = np.arange(len(edge)) + 0.5
    return float(np.angle((edge * np.exp(2j * np.pi * y / LANE)).sum()) / (2 * np.pi) * LANE) % LANE


def kind_of(m: dict[str, np.ndarray], y0: int, y1: int) -> str | None:
    band = slice(y0, y1)
    valid = m["valid"][band].sum()
    if valid < 200:
        return None
    frac = {k: m[k][band].sum() / valid for k in ("grass", "road", "water")}
    # the rail bed is road grey, the steel rails are what give it away by running the full width
    rails = m["rock"][band].sum(axis=1) / np.maximum(m["valid"][band].sum(axis=1), 1)
    if rails.max() > 0.5:
        return RAIL
    best = max((GRASS, ROAD, WATER), key=frac.get)
    # logs can cover most of a river
    return best if frac[best] > (0.25 if best == WATER else 0.35) else None


def profile(hit: np.ndarray, valid: np.ndarray, y0: int, y1: int) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of hit pixels per column over rows y0..y1, and where those rows are on screen."""
    hits = np.zeros(COLUMNS, np.float32)
    seen = np.zeros(COLUMNS, np.float32)
    shift = round(LEAN * (y0 + y1) / 2)
    hits[shift : shift + WIDTH] = (hit[y0:y1] & valid[y0:y1]).sum(axis=0)
    seen[shift : shift + WIDTH] = valid[y0:y1].sum(axis=0)
    return np.where(seen > 0, hits / np.maximum(seen, 1), 0), seen > 0


def blobs(mask: np.ndarray, y0: int, y1: int, smallest: int) -> np.ndarray:
    """Columns covered by blobs whose bottom sits inside rows y0..y1."""
    out = np.zeros(COLUMNS, np.float32)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    for x, y, w, h, area in stats[1:n]:
        bottom = y + h
        if area >= smallest and y0 <= bottom <= y1:
            c = round(column(x, bottom))
            out[max(c - 4, 0) : c + w + 4] = 1
    return out


def read(frame: np.ndarray) -> View:
    """One canonical frame to lanes, top to bottom."""
    hsv = cv2.cvtColor(frame[1270:1380, 210:455], cv2.COLOR_BGR2HSV)
    over = ((hsv[..., 0] >= 10) & (hsv[..., 0] <= 28) & (hsv[..., 1] > 150) & (hsv[..., 2] > 200)).mean() > 0.5
    home = (frame[1385:1440].max(axis=2) < 45).mean() > 0.6

    img = shear(frame)
    m = masks(img)
    phase = lane_phase(m)
    lanes = []
    if phase is None:
        return View(None, lanes, bool(over), bool(home))
    row = 0
    while phase + (row + 1) * LANE <= HEIGHT:
        top = phase + row * LANE
        y0, y1 = int(top), int(top + LANE)
        kind = kind_of(m, int(top + 0.15 * LANE), int(top + 0.85 * LANE))
        # the top of a lane is the part least covered by things standing in the lane below
        strip = int(top + 0.1 * LANE), int(top + 0.45 * LANE)
        signal = False
        if kind == ROAD:
            # tree tops from the lane below never move, leave them out
            solid, seen = profile(~(m["road"] | m["tree"]), m["valid"], *strip)
        elif kind == WATER:
            # log tops are too patchy to find by color, anything in the river that is not water floats
            solid, seen = profile(~(m["water"] | m["foam"]), m["valid"], int(top + 0.3 * LANE), int(top + 0.7 * LANE))
        elif kind == RAIL:
            solid, seen = profile(~(m["dark"] | m["log"] | m["rock"] | m["road"] | m["tree"]), m["valid"], *strip)
            signal = bool(m["glow"][y0:y1].sum() > 40)
        else:
            _, seen = profile(m["grass"], m["valid"], y0, y1)
            solid = np.maximum(blobs(m["trunk"], y0 + LANE * 0.4, y1 + LANE * 0.2, 12),
                               blobs(m["rock"], y0 + LANE * 0.4, y1 + LANE * 0.2, 60))
        lanes.append(Lane(row, kind, top, solid, seen, signal))
        row += 1
    return View(phase, lanes, bool(over), bool(home))


class Player:
    """Finds the character by its colors, learned from the home screen."""

    def __init__(self, home_frame: np.ndarray):
        x, y = HOME
        box = cv2.cvtColor(home_frame[y - 45 : y + 40, x - 55 : x + 55], cv2.COLOR_BGR2HSV)
        grass = (box[..., 0] >= 30) & (box[..., 0] <= 50) & (box[..., 1] > 80)
        mine = cv2.calcHist([box], [0, 1], (~grass).astype(np.uint8), [30, 16], [0, 180, 0, 256])
        every = cv2.calcHist([cv2.cvtColor(home_frame, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 16], [0, 180, 0, 256])
        # colors common in the character and rare in the world score high
        ratio = (mine / max(mine.sum(), 1)) / (every / every.sum() + 1e-4)
        self.model = np.clip(ratio / max(ratio.max(), 1e-9) * 1024, 0, 255).astype(np.float32)

    def find(self, frame: np.ndarray, box: tuple[float, float, float, float] = (120, 545, 560, 1200)) -> tuple[float, float] | None:
        """Canonical px of the character's feet, with its feet inside box (x0, x1, y0, y1)."""
        pad = 40
        x0, y0 = max(int(box[0]) - pad, 0), max(int(box[2]) - pad, 0)
        crop = frame[y0 : int(box[3]) + pad, x0 : int(box[1]) + pad]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        score = cv2.blur(cv2.calcBackProject([hsv], [0, 1], self.model, [0, 180, 0, 256], 1).astype(np.float32), (21, 21))
        # feet are a little below the middle of the body, only bodies with their feet in the box count
        ys, xs = np.ogrid[: score.shape[0], : score.shape[1]]
        inside = (xs + x0 >= box[0]) & (xs + x0 <= box[1]) & (ys + y0 + FEET >= box[2]) & (ys + y0 + FEET <= box[3])
        score = np.where(inside, score, 0)
        if score.max() < 60:
            return None
        cy, cx = np.unravel_index(score.argmax(), score.shape)
        return x0 + float(cx), y0 + float(cy) + FEET
