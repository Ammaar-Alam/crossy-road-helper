import unittest
from pathlib import Path

import cv2
import numpy as np

import bot
import vision as V

FRAMES = Path(__file__).parent / "tests"
LETTER = {V.GRASS: "g", V.ROAD: "r", V.WATER: "w", V.RAIL: "T", None: "?"}


def lane(kind: str, solid: np.ndarray | None = None) -> V.Lane:
    solid = np.zeros(V.COLUMNS, np.float32) if solid is None else solid
    return V.Lane(0, kind, 0.0, solid, np.ones(V.COLUMNS, bool))


def world(lanes: dict[int, tuple[str, float, np.ndarray | None]]) -> bot.World:
    """World with fixed lanes, each given as kind, speed and solid profile."""
    w = bot.World()
    for i, (kind, speed, solid) in lanes.items():
        track = bot.Track()
        track.add(lane(kind, solid), 0.0)
        track.speeds.extend([speed] * 3)
        w.tracks[i] = track
    return w


def car(at: int, length: int = 50) -> np.ndarray:
    solid = np.zeros(V.COLUMNS, np.float32)
    solid[at : at + length] = 1
    return solid


class Screens(unittest.TestCase):
    def read(self, name: str) -> V.View:
        return V.read(cv2.imread(str(FRAMES / f"{name}.jpg")))

    def test_home_and_splat(self):
        self.assertTrue(self.read("home").home)
        self.assertTrue(self.read("over").over)
        self.assertFalse(self.read("lanes").over or self.read("lanes").home)

    def test_lane_kinds(self):
        self.assertEqual("".join(LETTER[l.kind] for l in self.read("lanes").lanes), "rrrgrrrggrrrgggwwwrrr?r")
        self.assertEqual("".join(LETTER[l.kind] for l in self.read("rails").lanes), "g??grrgrTgrrrwwTggggggg")


class Boxes(unittest.TestCase):
    def test_footprint_of_a_box(self):
        # a car 50 px long whose front edge sits at sheared y 500 starting at column 300
        front, left, length = 500.0, 300.0 - V.LEAN * 500, 50.0
        x1, y1 = V.to_canonical(left + length, front)
        y, c0, c1 = V.footprint(left / V.SCALE, y1 - 120, x1 + V.LEAN * V.DEPTH / V.SCALE, y1)
        self.assertAlmostEqual(c0, 300, delta=0.5)
        self.assertAlmostEqual(c1, 350, delta=0.5)
        self.assertAlmostEqual(y, front - V.DEPTH / 2, delta=0.5)


class Plans(unittest.TestCase):
    def test_hops_onto_an_empty_road(self):
        w = world({0: (V.ROAD, 100, None), 1: (V.GRASS, 0, None)})
        self.assertEqual(bot.plan(w, 1, 350, 0.0), "up")

    def test_waits_for_a_car_about_to_pass(self):
        # a car heading right reaches column 350 as we would land
        w = world({0: (V.ROAD, 150, car(300)), 1: (V.GRASS, 0, None)})
        self.assertNotEqual(bot.plan(w, 1, 350, 0.0, depth=1), "up")

    def test_only_hops_onto_logs(self):
        w = world({0: (V.WATER, 0, None), 1: (V.GRASS, 0, None)})
        self.assertNotEqual(bot.plan(w, 1, 350, 0.0, depth=1), "up")
        w = world({0: (V.WATER, 0, car(320, 80)), 1: (V.GRASS, 0, None)})
        self.assertEqual(bot.plan(w, 1, 350, 0.0, depth=1), "up")

    def test_escapes_a_car_it_cannot_outwait(self):
        # a car is about to hit us and the lane ahead is clear now but not for long
        w = world({0: (V.ROAD, -150, car(500)), 1: (V.ROAD, 150, car(250)), 2: (V.GRASS, 0, None)})
        self.assertEqual(bot.plan(w, 1, 350, 0.0), "up")

    def test_speed_from_a_moving_profile(self):
        track = bot.Track()
        for i in range(12):
            t = i / 30
            track.add(lane(V.ROAD, car(200 + round(120 * t))), t)
        self.assertAlmostEqual(track.speed, 120, delta=8)


if __name__ == "__main__":
    unittest.main()
