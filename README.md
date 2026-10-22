# Crossy Road Helper

Plays Crossy Road on your iPhone through iPhone Mirroring on a Mac. It watches the mirrored screen, predicts where every car and log will be, and hops as fast as it safely can.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Give your terminal **Screen Recording** (to see the phone) and **Accessibility** (to tap it) in System Settings > Privacy & Security, then open iPhone Mirroring with Crossy Road on its home screen.

## Use

```bash
./run.sh            # panel beside the phone, press Play
./run.sh play       # one run without the panel
./run.sh replay captures/live/<run>.mp4   # what the helper saw, lanes and moves drawn on top
.venv/bin/python -m unittest -v
```

While a run is going the helper owns the mouse. Move the mouse to stop it.

Each live run is saved under `captures/live/` with its frame times, so it can be replayed through the same code.

## How it works

- `capture.py` grabs the iPhone Mirroring window and sends taps and swipes as mouse events.
- `screen.py` crops any capture to one canonical phone screen.
- `vision.py` shears the screen so lanes run flat, finds the lane grid, tells grass, road, water and rails apart, and marks cars, logs, trees and trains per column. Columns are measured along the game's grid, so a forward hop keeps the same column.
- `bot.py` follows lanes as the camera scrolls, measures each lane's speed, predicts traffic and searches a few moves ahead for the furthest safe hop.
- `helper.py` runs the loop, `panel.py` and `panel.html` are the panel.

Tuned on the classic world. Other worlds recolor the ground and will need their own colors.
