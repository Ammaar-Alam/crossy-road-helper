"""Panel parked beside the phone that starts and stops the helper."""

import json
from pathlib import Path

import capture
from helper import Helper

ROOT = Path(__file__).parent
WIDTH, HEIGHT = 320, 780


class Panel:
    """Methods without an underscore are callable from the page."""

    def __init__(self):
        self._window = None
        self._helper = Helper(self._render)

    def start(self):
        self._helper.start()

    def stop(self):
        self._helper.stop()

    def record(self, on):
        self._helper.record(on)

    def _render(self, state: dict) -> None:
        if self._window is not None:
            self._window.evaluate_js(f"render({json.dumps(state)})")

    def run(self):
        import webview

        phone = capture.place_window()
        x, y = (phone.x - WIDTH - 12, phone.y) if phone else (80, 80)
        self._window = webview.create_window("Crossy Road", html=(ROOT / "panel.html").read_text(), x=max(0, x), y=y,
                                             width=WIDTH, height=HEIGHT, frameless=True, easy_drag=True,
                                             background_color="#2b2d34", js_api=self)
        self._window.events.closing += self._helper.stop
        webview.start(lambda: self._render(self._helper.state))
