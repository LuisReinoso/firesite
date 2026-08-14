#!/usr/bin/env python3
# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""Record the viewer's year playback as an MP4 and a GIF, for sharing.

Records the real page rather than recreating it in a video framework. A
recreation drifts from the tool the moment either changes, and the whole point
of the clip is to show what someone actually gets.

    python scripts/record_demo.py --serve docs --out docs/media

Needs playwright (`pip install playwright && playwright install chromium`) and
ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import socket
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def serve(directory: Path):
    port = free_port()
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()


def capture(url: str, frames_dir: Path, width: int, height: int,
            dataset: str, theme: str) -> int:
    from playwright.sync_api import sync_playwright

    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("*.png"):
        stale.unlink()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height},
                                device_scale_factor=2)
        page.goto(f"{url}/?data={dataset}", wait_until="networkidle")
        # Set the attribute and refresh the basemap directly. Clicking the toggle
        # as well would flip it straight back, which silently produced the wrong
        # theme for every recording.
        page.evaluate(
            "theme => { document.documentElement.setAttribute('data-theme', theme);"
            " applyTheme(); }",
            theme,
        )
        # Stop the autoplay so the recording controls the timeline itself and
        # every year gets exactly one frame.
        page.wait_for_timeout(2500)
        page.evaluate("stop(false)")
        page.wait_for_timeout(600)

        years = page.evaluate("years()")
        index = 0
        for year in years:
            page.evaluate(f"setCursor({year})")
            page.wait_for_timeout(260)  # let tiles and markers settle
            page.screenshot(path=str(frames_dir / f"{index:03d}.png"))
            index += 1
        # Hold on the full picture, which is the frame worth pausing on.
        page.evaluate("setCursor(null)")
        page.wait_for_timeout(700)
        for _ in range(12):
            page.screenshot(path=str(frames_dir / f"{index:03d}.png"))
            index += 1
        browser.close()
    return index


def encode(frames_dir: Path, out_dir: Path, fps: int) -> list[Path]:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is not on PATH")
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "firesite-demo.mp4"
    gif = out_dir / "firesite-demo.gif"
    pattern = str(frames_dir / "%03d.png")

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
         # yuv420p and even dimensions are what social platforms will actually play.
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p",
         "-c:v", "libx264", "-crf", "20", str(mp4)],
        check=True,
    )
    palette = frames_dir / "palette.png"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
         "-vf", "scale=900:-1:flags=lanczos,palettegen=stats_mode=diff", str(palette)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
         "-i", str(palette), "-lavfi",
         "scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
         str(gif)],
        check=True,
    )
    palette.unlink(missing_ok=True)
    return [mp4, gif]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--serve", default="docs", help="directory holding index.html")
    p.add_argument("--dataset", default="data/cotacachi.json")
    p.add_argument("--out", default="docs/media")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--theme", default="dark", choices=["light", "dark"])
    args = p.parse_args()

    root = Path(args.serve)
    if not (root / "index.html").exists():
        raise SystemExit(f"no index.html in {root}")

    frames = Path(args.out) / "frames"
    with serve(root) as url:
        time.sleep(0.4)
        count = capture(url, frames, args.width, args.height, args.dataset, args.theme)
    print(f"{count} frames", file=sys.stderr)

    written = encode(frames, Path(args.out), args.fps)
    for path in written:
        print(f"{path}  {path.stat().st_size / 1e6:.1f} MB")
    shutil.rmtree(frames, ignore_errors=True)


if __name__ == "__main__":
    main()
