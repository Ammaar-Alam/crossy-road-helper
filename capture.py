"""Find, place, grab and touch the iPhone Mirroring window."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

import numpy as np

OWNER = "iPhone Mirroring"


@dataclass(frozen=True, slots=True)
class Window:
    id: int
    x: int
    y: int
    width: int
    height: int


def find_window() -> Window | None:
    import Quartz

    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    ) or []
    # menus and popovers share the owner, the phone itself is the biggest layer 0 window
    phones = [w for w in windows if w.get("kCGWindowOwnerName") == OWNER and w.get("kCGWindowLayer") == 0]
    if not phones:
        return None
    w = max(phones, key=lambda w: w["kCGWindowBounds"]["Width"] * w["kCGWindowBounds"]["Height"])
    b = w["kCGWindowBounds"]
    return Window(int(w["kCGWindowNumber"]), int(b["X"]), int(b["Y"]), int(b["Width"]), int(b["Height"]))


def place_window() -> Window | None:
    """Window > Move & Resize > Right, the big phone view with room for the panel on its left."""
    if find_window() is None:
        return None
    script = f"""
    tell application "System Events" to tell process "{OWNER}"
        set frontmost to true
        delay 0.3
        click menu item "Right" of menu 1 of menu item "Move & Resize" of menu 1 of menu bar item "Window" of menu bar 1
    end tell
    """
    subprocess.run(["osascript", "-e", script], capture_output=True, check=False)
    # the window drops out of the list while it animates to the new size
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        time.sleep(0.3)
        window = find_window()
        if window is not None:
            return window
    return None


def grab(window: Window) -> np.ndarray | None:
    """BGR frame of the window at one pixel per point, works even when covered."""
    import Quartz

    image = Quartz.CGWindowListCreateImage(
        Quartz.CGRectNull,
        Quartz.kCGWindowListOptionIncludingWindow,
        window.id,
        # retina detail costs twice the time and vision works at half size anyway
        Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageNominalResolution,
    )
    if image is None:
        return None
    width, height = Quartz.CGImageGetWidth(image), Quartz.CGImageGetHeight(image)
    stride = Quartz.CGImageGetBytesPerRow(image)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image))
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, stride // 4, 4)
    return np.ascontiguousarray(pixels[:, :width, :3])


def to_point(window: Window, size: tuple[int, int], rect: tuple[int, int, int, int], x: float, y: float) -> tuple[float, float]:
    """Canonical px to a screen point, size is the grabbed image and rect the phone screen inside it."""
    import screen

    rx, ry, rw, rh = rect
    return (window.x + (rx + x * rw / screen.WIDTH) * window.width / size[0],
            window.y + (ry + y * rh / screen.HEIGHT) * window.height / size[1])


class Mouse:
    """Taps and swipes as real mouse events, iPhone Mirroring turns them into touches."""

    HOLD = 0.04  # a finger lifted too soon is sometimes missed by the phone

    def __init__(self):
        # start and end of our last tap or swipe
        self.last: tuple[tuple[float, float], tuple[float, float]] | None = None

    def moved(self) -> bool:
        """Whether someone moved the mouse since our last event."""
        import Quartz

        if self.last is None:
            return False
        p = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        # the cursor can still be catching up anywhere along our last swipe
        (x0, y0), (x1, y1) = self.last
        dx, dy = x1 - x0, y1 - y0
        k = min(max(((p.x - x0) * dx + (p.y - y0) * dy) / max(dx * dx + dy * dy, 1e-9), 0.0), 1.0)
        return abs(p.x - (x0 + k * dx)) + abs(p.y - (y0 + k * dy)) > 8

    def _post(self, kind: int, point: tuple[float, float]) -> None:
        import Quartz

        event = Quartz.CGEventCreateMouseEvent(None, kind, point, Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def tap(self, point: tuple[float, float]) -> None:
        import Quartz

        self._post(Quartz.kCGEventMouseMoved, point)
        self._post(Quartz.kCGEventLeftMouseDown, point)
        time.sleep(self.HOLD)
        self._post(Quartz.kCGEventLeftMouseUp, point)
        self.last = point, point

    def swipe(self, point: tuple[float, float], dx: float, dy: float, steps: int = 8, gap: float = 0.012, hold: float = 0.012) -> None:
        """A drag the phone sees as a swipe, hold is how long the finger rests at the end before lifting."""
        import Quartz

        self._post(Quartz.kCGEventMouseMoved, point)
        self._post(Quartz.kCGEventLeftMouseDown, point)
        for i in range(1, steps + 1):
            time.sleep(gap)
            self._post(Quartz.kCGEventLeftMouseDragged, (point[0] + dx * i / steps, point[1] + dy * i / steps))
        end = (point[0] + dx, point[1] + dy)
        time.sleep(hold)
        self._post(Quartz.kCGEventLeftMouseUp, end)
        self.last = point, end
