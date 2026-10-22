"""Play Crossy Road through iPhone Mirroring, or replay a recording through the same eyes."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

import bot
import capture
import screen
import vision as V

ROOT = Path(__file__).parent
TAP = (332, 500)  # clear of the score, the pause button and the banners on the splat screen
PLAY = (332, 1325)  # the big play button after a splat
SWIPE = 70  # screen points
COLORS = {V.GRASS: (60, 200, 90), V.ROAD: (120, 120, 120), V.WATER: (250, 200, 90), V.RAIL: (40, 60, 160), None: (0, 0, 255)}


def hide(view: V.View, x: float, y: float) -> None:
    """Take the character out of the lanes it covers, it is neither an obstacle nor a missing log."""
    for lane in view.lanes:
        if lane.top > y or lane.top + V.LANE < y - 1.3 * V.LANE:
            continue
        c = round(V.column(x, lane.top + V.LANE / 2))
        span = slice(max(c - round(0.45 * V.TILE), 0), c + round(0.45 * V.TILE))
        if lane.top + V.LANE > y:
            # standing on water means standing on something
            lane.solid[span] = 1 if lane.kind == V.WATER else 0
        elif lane.kind in (V.ROAD, V.RAIL):
            # its head pokes into the strip read for traffic, grass obstacles are found by their base instead
            lane.solid[span] = 0


class Brain:
    """Everything between a canonical frame and the next move, shared by live play and replays."""

    def __init__(self, home_frame: np.ndarray):
        self.player = V.Player(home_frame)
        self.world = bot.World()
        self.near: tuple[float, float] | None = None
        self.start: int | None = None
        self.best = 0
        self.view: V.View | None = None
        self.spot: tuple[int, float] | None = None

    def step(self, frame: np.ndarray, t: float) -> str | None:
        """Read a frame taken at time t and return a move, or None when the character is not in sight."""
        self.view = V.read(frame)
        if self.view.over or self.view.home:
            return None
        self.near = self.player.find(frame, self.near) or self.player.find(frame)
        if self.near is not None:
            hide(self.view, *V.to_sheared(*self.near))
        self.world.update(self.view, t)
        if self.world.phase is None or self.near is None:
            self.spot = None
            return None
        x, y = V.to_sheared(*self.near)
        lane, c = self.world.lane_of(y), V.column(x, y)
        self.spot = lane, c
        self.start = lane if self.start is None else self.start
        self.best = max(self.best, self.start - lane)
        return bot.plan(self.world, lane, c, t)

    def draw(self, frame: np.ndarray, move: str | None) -> np.ndarray:
        img = V.shear(frame)
        if self.view is None:
            return img
        for lane in self.view.lanes:
            y0 = int(lane.top)
            track = self.world.tracks.get(lane.row - self.world.scroll)
            cv2.line(img, (0, y0), (V.WIDTH, y0), (0, 0, 255) if lane.signal else (255, 255, 255), 1)
            cv2.rectangle(img, (0, y0 + 2), (5, int(lane.top + V.LANE) - 2), COLORS[track.kind if track else None], -1)
            if track and track.speeds:
                cv2.putText(img, f"{track.speed:+.0f}", (8, y0 + 14), 0, 0.3, (255, 255, 255), 1)
            yc = int(lane.top + V.LANE / 2)
            for x in range(V.WIDTH):
                col = int(round(V.column(x, yc)))
                if lane.seen[col] and lane.solid[col] > 0.5:
                    img[yc - 1 : yc + 2, x] = (0, 255, 0) if lane.kind == V.WATER else (0, 0, 255)
        if self.near is not None:
            x, y = V.to_sheared(*self.near)
            cv2.circle(img, (int(x), int(y)), 6, (255, 0, 255), 2)
            cv2.putText(img, move or "-", (int(x) + 10, int(y)), 0, 0.5, (255, 0, 255), 2)
        cv2.putText(img, str(self.best), (V.WIDTH - 60, 30), 0, 0.8, (255, 255, 255), 2)
        return img


def replay(video: Path, out: Path, fps: float) -> None:
    """Run a recording through the brain at a live-like frame rate and save what it saw."""
    cap = cv2.VideoCapture(str(video))
    # live runs keep their real frame times next to the video
    times = video.with_suffix(".txt")
    stamps = [float(line.split()[0]) for line in times.read_text().splitlines()] if times.exists() else None
    step = 1 if stamps else max(1, round(cap.get(cv2.CAP_PROP_FPS) / fps))
    writer, brain, i, spent = None, None, 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % step:
            continue
        canon = screen.normalize(frame)
        if canon is None:
            continue
        t = stamps[i - 1] if stamps else i / cap.get(cv2.CAP_PROP_FPS)
        if brain is None or V.read(canon).home:
            if V.read(canon).home:
                brain = Brain(canon)
            continue
        began = time.perf_counter()
        move = brain.step(canon, t)
        spent.append(time.perf_counter() - began)
        img = brain.draw(canon, move)
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
        writer.write(img)
    if writer is not None:
        writer.release()
    if spent:
        print(f"{len(spent)} frames, {np.mean(spent) * 1000:.0f} ms mean, {np.max(spent) * 1000:.0f} ms worst -> {out}")


class Helper:
    """The live loop, it reports progress through on_state and stops when the mouse moves."""

    def __init__(self, on_state: Callable[[dict], None] = print):
        self.on_state = on_state
        self.state = {"running": False, "status": "idle", "score": None, "best": 0, "runs": 0, "rate": None, "error": None}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _set(self, **changes) -> None:
        self.state.update(changes)
        self.on_state(dict(self.state))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        self._set(running=True, status="starting", error=None, score=None, rate=None)
        try:
            self._play()
        except Exception as error:  # the panel should say what broke instead of freezing
            self._set(error=str(error))
        finally:
            status = self.state["status"] if self.state["status"] in ("over", "no phone") else "stopped"
            self._set(running=False, status=status)

    def _frame(self, window: capture.Window, rect=None) -> tuple[np.ndarray, float, tuple[int, int], tuple[int, int, int, int]] | None:
        """Canonical frame, when it was taken, the grabbed size and the phone screen inside it."""
        t = time.monotonic()
        raw = capture.grab(window)
        if raw is None:
            return None
        rect = rect or screen.screen_rect(raw)
        canon = screen.normalize(raw, rect)
        return (canon, t, (raw.shape[1], raw.shape[0]), rect) if canon is not None else None

    def _play(self) -> None:
        window = capture.place_window()
        if window is None:
            self._set(status="no phone")
            return
        mouse = capture.Mouse()

        def touch(point: tuple[float, float], size, rect) -> tuple[float, float]:
            return capture.to_point(window, size, rect, *point)

        # get to the home screen, through the play button after a splat
        deadline = time.monotonic() + 8
        while not self._stop.is_set():
            got = self._frame(window)
            if got is None:
                self._set(status="no phone")
                return
            canon, _, size, rect = got
            view = V.read(canon)
            if view.home:
                break
            if view.over:
                mouse.tap(touch(PLAY, size, rect))
                time.sleep(1.0)
            elif time.monotonic() > deadline:
                self._set(error="Start from the home screen")
                return
            time.sleep(0.1)
        if self._stop.is_set():
            return

        brain = Brain(canon)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (ROOT / "captures" / "live").mkdir(parents=True, exist_ok=True)
        video = cv2.VideoWriter(str(ROOT / "captures" / "live" / f"{stamp}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                (screen.WIDTH, screen.HEIGHT))
        with open(ROOT / "captures" / "live" / f"{stamp}.txt", "w") as times:
            # the home frame goes first so a replay can learn the character from it
            video.write(canon)
            times.write(f"{time.monotonic():.4f}\n")
            try:
                self._run_one(window, mouse, touch, brain, size, rect, video, times)
            finally:
                video.release()

    def _run_one(self, window, mouse, touch, brain, size, rect, video, times) -> None:
        mouse.tap(touch(TAP, size, rect))
        began, busy, lost = time.monotonic(), time.monotonic() + bot.HOP, 0
        self._set(status="playing", score=0)
        while not self._stop.is_set():
            if mouse.moved():
                return
            # the phone screen does not move inside the window during a run
            got = self._frame(window, rect)
            if got is None:
                continue
            canon, t, size, rect = got
            video.write(canon)
            move = brain.step(canon, t)
            acted = move not in (None, "wait") and t >= busy
            times.write(f"{t:.4f} {move if acted else ''}\n")
            if brain.view.over:
                self._set(status="over", score=brain.best, best=max(self.state["best"], brain.best), runs=self.state["runs"] + 1)
                return
            lost = 0 if brain.near is not None else lost + 1
            if lost > 15:
                self._set(error="Lost sight of the character")
                return
            if not acted:
                continue
            point = touch(TAP, size, rect)
            if move == "up":
                mouse.tap(point)
            else:
                dx, dy = {"left": (-SWIPE, 0), "right": (SWIPE, 0), "down": (0, SWIPE)}[move]
                mouse.swipe(point, dx, dy)
            busy = time.monotonic() + bot.HOP
            elapsed = time.monotonic() - began
            self._set(score=brain.best, rate=brain.best / elapsed if elapsed > 1 else None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crossy-helper", description=__doc__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("ui", help="open the panel (default)")
    sub.add_parser("play", help="play one run without the panel")
    r = sub.add_parser("replay", help="run a recording through the helper and save what it saw")
    r.add_argument("video", type=Path)
    r.add_argument("--out", type=Path)
    r.add_argument("--fps", type=float, default=30)
    args = parser.parse_args(argv)
    if args.command == "replay":
        replay(args.video, args.out or args.video.with_name(args.video.stem + "_seen.mp4"), args.fps)
    elif args.command == "play":
        helper = Helper(lambda s: print(s["status"], s["score"], s["error"] or ""))
        helper.start()
        helper._thread.join()
    else:
        from panel import Panel

        Panel().run()


if __name__ == "__main__":
    main(sys.argv[1:])
