"""Build a detection dataset from recorded runs and train the object detector.

  frames   sample canonical frames from captures/live/*.avi
  gemini   label sampled frames with Gemini boxes
  export   write a YOLO dataset, whole runs held out for validation
  train    fine tune a small YOLO model on it
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

import screen
import vision as V

ROOT = Path(__file__).parent
DATA = ROOT / "captures" / "dataset"
CLASSES = V.CLASSES
PROMPT = """This is a frame of the mobile game Crossy Road (isometric voxel art).
Find every object of these kinds and give a tight box around each:
- vehicle: any car, van, taxi, truck or bus on a road
- train: a train or train car on the rail tracks
- log: a floating wooden log on the river
- lily_pad: a floating lily pad on the river
- tree: a tree or bush standing on grass
- rock: a boulder or stone block standing on grass
- character: the player character the game is about, whatever it looks like (chicken, person, animal or anything else), usually in the lower middle of the screen; there is only one
Ignore coins, the score, buttons, banners, the title text and anything else.
Return a JSON list of {"label": kind, "box_2d": [ymin, xmin, ymax, xmax]} with coordinates normalized to 0-1000."""


def key() -> str:
    """API key from the environment or the repo's .env file."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        name, _, value = line.partition("=")
        if name.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and value.strip():
            return value.strip().strip("'\"")
    raise SystemExit("Put GEMINI_API_KEY=... in .env")


def frames(step: int) -> None:
    """Every step-th play frame of every lossless run, the home screen and splat screen left out."""
    out = DATA / "images"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for video in sorted((ROOT / "captures" / "live").glob("*.avi")):
        cap = cv2.VideoCapture(str(video))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % step:
                continue
            view = V.read(frame)
            if view.home or view.over:
                continue
            path = out / f"{video.stem}_{i:04d}.jpg"
            if not path.exists():
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                n += 1
    # the phone's own screen recording has the human's runs, other skins and worlds
    phone = ROOT / "captures" / "play1.mp4"
    cap = cv2.VideoCapture(str(phone))
    every = max(1, round(cap.get(cv2.CAP_PROP_FPS) / 2))
    i = 0
    while phone.exists():
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % every:
            continue
        canon = screen.normalize(frame)
        if canon is None or V.read(canon).home or V.read(canon).over:
            continue
        path = out / f"phone_{i:05d}.jpg"
        if not path.exists():
            cv2.imwrite(str(path), canon, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
    print(f"{n} new frames in {out}")


def gemini(model: str, workers: int, limit: int | None) -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key())
    out = DATA / "gemini"
    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted((DATA / "images").glob("*.jpg")) if not (out / f"{p.stem}.json").exists()]
    todo = todo[:limit] if limit else todo
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)

    def one(path: Path) -> str | None:
        try:
            reply = client.models.generate_content(
                model=model, config=config,
                contents=[types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg"), PROMPT])
            boxes = [b for b in json.loads(reply.text) if b.get("label") in CLASSES and len(b.get("box_2d", [])) == 4]
            (out / f"{path.stem}.json").write_text(json.dumps(boxes))
            return None
        except Exception as error:  # one bad reply should not stop the batch, it is retried next run
            return f"{path.name}: {error}"

    with ThreadPoolExecutor(workers) as pool:
        errors = [e for e in pool.map(one, todo) if e]
    print(f"labeled {len(todo) - len(errors)} of {len(todo)}")
    for error in errors[:5]:
        print(error)


def export(holdout: int) -> Path:
    """YOLO dataset from the Gemini boxes, the last few runs held out whole so validation is honest."""
    labeled = sorted((DATA / "gemini").glob("*.json"))
    runs = sorted({p.stem.rsplit("_", 1)[0] for p in labeled})
    # held out runs are recent live ones, the way the bot will see the game
    held = set([r for r in runs if r != "phone"][-holdout:])
    yolo = DATA / "yolo"
    # which runs are held out changes as more get labeled, so start clean
    shutil.rmtree(yolo, ignore_errors=True)
    for split in ("train", "val"):
        (yolo / split / "images").mkdir(parents=True, exist_ok=True)
        (yolo / split / "labels").mkdir(parents=True, exist_ok=True)
    for path in labeled:
        split = "val" if path.stem.rsplit("_", 1)[0] in held else "train"
        image = yolo / split / "images" / f"{path.stem}.jpg"
        if not image.exists():
            image.symlink_to(DATA / "images" / f"{path.stem}.jpg")
        lines = []
        for box in json.loads(path.read_text()):
            y0, x0, y1, x1 = (v / 1000 for v in box["box_2d"])
            if x1 > x0 and y1 > y0:
                lines.append(f"{CLASSES.index(box['label'])} {(x0 + x1) / 2:.5f} {(y0 + y1) / 2:.5f} {x1 - x0:.5f} {y1 - y0:.5f}")
        (yolo / split / "labels" / f"{path.stem}.txt").write_text("\n".join(lines))
    spec = yolo / "data.yaml"
    spec.write_text(f"path: {yolo}\ntrain: train/images\nval: val/images\nnames: {json.dumps(dict(enumerate(CLASSES)))}\n")
    print(f"{len(labeled)} frames, {len(runs) - len(held)} runs to train on, {len(held)} held out")
    return spec


def train(epochs: int, size: int) -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "captures" / "yolo"))
    from ultralytics import YOLO

    spec = DATA / "yolo" / "data.yaml"
    # flipping would mirror traffic directions and the lean of the lanes
    YOLO(ROOT / "captures" / "yolo11n.pt").train(data=str(spec), epochs=epochs, imgsz=size, rect=True, fliplr=0.0, mosaic=0.0,
                             project=str(ROOT / "captures" / "runs"), name="detect", exist_ok=True, device="mps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("frames").add_argument("--step", type=int, default=3)
    g = sub.add_parser("gemini")
    g.add_argument("--model", default="gemini-2.5-flash")
    g.add_argument("--workers", type=int, default=16)
    g.add_argument("--limit", type=int)
    sub.add_parser("export").add_argument("--holdout", type=int, default=4)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--size", type=int, default=960)
    args = parser.parse_args()
    random.seed(0)
    if args.command == "frames":
        frames(args.step)
    elif args.command == "gemini":
        gemini(args.model, args.workers, args.limit)
    elif args.command == "export":
        export(args.holdout)
    else:
        train(args.epochs, args.size)


if __name__ == "__main__":
    main()
