# dev/record_demo.py
"""Record the demo video from the story page. Run from the project root with dev/serve.py already running:
    python dev/record_demo.py [--url http://127.0.0.1:8765/] [--live] [--shots 100] [--out dev/out]
By default the page loads the recorded take. With --live it submits a real job with --shots and waits for it
(lesson 3 circuits are slow: about 3 minutes at 100 shots).
Writes dev/out/demo.webm and dev/out/demo.mp4 (mp4 via ffmpeg).
"""
import argparse
import glob
import os
import shutil
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

VIEWPORT = {"width": 1280, "height": 800}

CAPTION_CSS = """
#demo-cap { position: fixed; left: 0; right: 0; bottom: 0; padding: 14px 24px; background: rgba(10,12,16,.92);
  color: #fff; font: 20px/1.4 system-ui, sans-serif; z-index: 9999; border-top: 2px solid #4f9cff; }
/* the story page has its controls along the bottom, so its captions sit on top */
#demo-cap.top { top: 0; bottom: auto; border-top: 0; border-bottom: 2px solid #4f9cff; }
"""

# Titles of SCRIPTS[3].steps in ui/storyText.js. The live page only exposes the step count (window.__story.steps),
# so the names live here and are checked against that count.
LESSON3_STEPS = ["Prepare", "Bell pair", "Bell measurement", "Checks", "Readout", "Result"]


def caption(page, text):
    page.evaluate("""(t) => { let c = document.getElementById('demo-cap');
        if (!c) { c = document.createElement('div'); c.id = 'demo-cap'; document.body.appendChild(c); }
        c.classList.toggle('top', !!window.__story);
        c.textContent = t; }""", text)


def wait_showing(page, timeout):
    page.wait_for_function("() => /^Showing /.test(document.getElementById('status').textContent)", timeout=timeout)
    page.wait_for_selector("#tabs", state="visible", timeout=30_000)


def play_through(page, label, timeout_s):
    """Press Play and follow the playhead step by step until the last step is reached."""
    n = page.evaluate("window.__story.steps")
    assert n == len(LESSON3_STEPS), f"story has {n} steps, recorder knows {len(LESSON3_STEPS)}"
    caption(page, f"{label} level: {LESSON3_STEPS[page.evaluate('window.__story.pos.step')]}")   # moves it to the top
    page.get_by_role("button", name="Play").click()
    shown = -1
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        s = page.evaluate("window.__story.pos.step")
        if s != shown:
            shown = s
            caption(page, f"{label} level: {LESSON3_STEPS[s]}")
        if s >= n - 1:
            return
        time.sleep(0.25)
    raise TimeoutError(f"{label} level did not reach the last step within {timeout_s} s (at step {shown})")


def ffmpegs():
    """ffmpeg builds to try, full builds first: Playwright's own cannot write mp4."""
    found = [shutil.which("ffmpeg")]
    for root in (os.environ.get("CONDA_PREFIX"), str(Path.home() / "anaconda3"), str(Path.home() / "miniconda3")):
        if root:
            found.append(os.path.join(root, "Library", "bin", "ffmpeg.exe"))
    found += glob.glob(os.path.join(os.environ.get("LOCALAPPDATA", ""), "ms-playwright", "ffmpeg-*", "ffmpeg-win64.exe"))
    return [f for f in found if f and os.path.exists(f)]


ENCODERS = [["-c:v", "libx264"], ["-c:v", "h264_mf", "-b:v", "4M"]]   # x264 where there is one, else Windows Media Foundation


def to_mp4(webm, mp4):
    for ff in ffmpegs():
        for enc in ENCODERS:
            r = subprocess.run([ff, "-y", "-i", str(webm), *enc, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                str(mp4)], capture_output=True)
            if r.returncode == 0:
                return True
    return False


def record(page, a):
    page.goto(a.url)
    page.wait_for_selector("#submit")
    page.add_style_tag(content=CAPTION_CSS)

    caption(page, "Logical teleportation on a tiny surface code. Three logical qubits, each a 3x3 patch of the "
                  "rotated surface code, distance 3.")
    time.sleep(4)

    if a.live:
        caption(page, f"Submitting one job with all circuits at {a.shots} shots per circuit. "
                      "The page polls the job and shows progress.")
        page.locator("#setup").evaluate("d => d.open = true")
        page.fill("#shots", str(a.shots))
        page.click("#submit")
        wait_showing(page, 900_000)
    else:
        caption(page, "Real run data throughout. This take is a local Aer stand-in with depolarising noise; "
                      "the page runs unchanged on Qollab and IonQ simulators.")
        page.click("#load-recorded")
        wait_showing(page, 60_000)
    time.sleep(3)

    caption(page, "27 physical qubits. Three patches: A carries the state, B and C share a logical Bell pair.")
    page.click("#tab-3")
    time.sleep(4)

    page.click("#story-open")
    page.wait_for_function("() => !!window.__story", timeout=30_000)
    play_through(page, "Logical", 90)
    time.sleep(3)   # the closing card

    page.evaluate("window.__story.setLevel('physical'); window.__story.setPos(0, 0)")
    play_through(page, "Physical", 120)
    time.sleep(3)

    page.keyboard.press("Escape")
    page.wait_for_function("() => !window.__story", timeout=10_000)
    caption(page, "Cards and charts: discarded, corrected, logical error after correction, and the same "
                  "experiment on bare qubits.")
    page.locator("#cards").evaluate("e => e.scrollIntoView({ behavior: 'smooth', block: 'center' })")
    time.sleep(2)

    caption(page, "Lesson 1 locates and corrects a single injected error; try it live on the page.")
    page.click("#tab-1")
    time.sleep(4)

    caption(page, "Team ZollerQu. This effort is supported by Qollab & IonQ.")
    time.sleep(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/")
    ap.add_argument("--live", action="store_true", help="submit a real job instead of loading the recorded take")
    ap.add_argument("--shots", type=int, default=100)
    ap.add_argument("--out", default="dev/out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        # reduced motion stays off, so the story plays at its own pace
        ctx = browser.new_context(viewport=VIEWPORT, color_scheme="dark", reduced_motion="no-preference",
                                  record_video_dir=str(out), record_video_size=VIEWPORT)
        page = ctx.new_page()
        video = page.video
        try:
            record(page, a)
        finally:
            ctx.close()
            webm = Path(video.path())
            browser.close()
    final = out / "demo.webm"
    shutil.move(str(webm), final)
    mp4 = to_mp4(final, out / "demo.mp4")
    print("wrote", final, "and demo.mp4" if mp4 else "(no ffmpeg with an h264 encoder found, webm only)")


if __name__ == "__main__":
    main()
