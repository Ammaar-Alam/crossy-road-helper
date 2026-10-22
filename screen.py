"""Crop any capture of the phone to one canonical screen so every region is fixed."""

from __future__ import annotations

import cv2
import numpy as np

WIDTH = 664
HEIGHT = 1440


def screen_rect(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Phone screen as x, y, w, h inside a window or video frame.

    The screen is the widest run of bright columns and its bottom edge is the
    last bright row, the top is implied by the fixed iPhone aspect ratio.
    """
    bright = frame.max(axis=2) > 40
    cols = np.where(bright.mean(axis=0) > 0.5)[0]
    rows = np.where(bright.mean(axis=1) > 0.5)[0]
    if len(cols) < 100 or len(rows) < 100:
        return None
    x0, x1, y1 = int(cols.min()), int(cols.max()) + 1, int(rows.max()) + 1
    w = x1 - x0
    h = round(w * HEIGHT / WIDTH)
    return x0, max(0, y1 - h), w, min(h, y1)


def normalize(frame: np.ndarray, rect: tuple[int, int, int, int] | None = None) -> np.ndarray | None:
    rect = rect or screen_rect(frame)
    if rect is None:
        return None
    x, y, w, h = rect
    crop = frame[y : y + h, x : x + w]
    if crop.shape[1] != WIDTH or crop.shape[0] != HEIGHT:
        crop = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return crop
