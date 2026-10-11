# dev/record_demo.py
"""Record the demo video of the placement page. Run from the project root with dev/serve.py already running:
    python dev/record_demo.py [--url http://127.0.0.1:8767/] [--out dev/out] [--mp4 docs/demo.mp4]
The script does one live run at the default m = 10 (sweep, then the Durr-Hoyer loop; several minutes), then loads the
recorded take. Writes dev/out/demo.webm and the mp4 (via ffmpeg).
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
"""

CAP_H = 110   # room kept free at the bottom of the viewport for the caption bar


def caption(page, text):
    page.evaluate("""(t) => { let c = document.getElementById('demo-cap');
        if (!c) { c = document.createElement('div'); c.id = 'demo-cap'; document.body.appendChild(c); }
        c.textContent = t; }""", text)


def center(page, sel):
    """Scroll the element to the middle of the area above the caption bar."""
    page.evaluate("""([s, h]) => { const r = document.querySelector(s).getBoundingClientRect();
        window.scrollBy({ top: r.top + r.height / 2 - (innerHeight - h) / 2, behavior: 'smooth' }); }""", [sel, CAP_H])
    time.sleep(1.2)


def both_fit(page, a, b):
    """True when one scroll position shows both elements in full above the caption bar."""
    return page.evaluate("""([a, b, h]) => { const x = document.querySelector(a).getBoundingClientRect(),
        y = document.querySelector(b).getBoundingClientRect();
        return Math.max(x.bottom, y.bottom) - Math.min(x.top, y.top) <= innerHeight - h - 10; }""", [a, b, CAP_H])


def show_both(page, a, b):
    """Scroll so a and b are both visible above the caption bar; False when they cannot fit together."""
    if not both_fit(page, a, b):
        return False
    page.evaluate("""([a, b, h]) => { const x = document.querySelector(a).getBoundingClientRect(),
        y = document.querySelector(b).getBoundingClientRect();
        const top = Math.min(x.top, y.top), bot = Math.max(x.bottom, y.bottom);
        window.scrollBy({ top: top + (bot - top) / 2 - (innerHeight - h) / 2, behavior: 'smooth' }); }""", [a, b, CAP_H])
    time.sleep(1.2)
    return True


def wait_refreshed(page):
    """After a control change: the debounced refresh starts, then ends with the N = ... status."""
    time.sleep(0.6)
    page.wait_for_function("() => /^N = \\d+ feasible/.test(document.getElementById('status').textContent)", timeout=120_000)
    page.wait_for_function("() => !document.getElementById('run').disabled", timeout=60_000)


def set_slider(page, sel, value):
    page.evaluate("""([s, v]) => { const e = document.querySelector(s); e.value = String(v);
        e.dispatchEvent(new Event('input', { bubbles: true })); e.dispatchEvent(new Event('change', { bubbles: true })); }""",
                  [sel, value])


def counters_line(page):
    return " / ".join(t for t in page.locator("#dh-counters").inner_text().splitlines() if t.strip())


def cards_line(page):
    """The cost card and the placements-scored card of ui/cards.js renderCards (the first two)."""
    cards = page.locator("#cards .card").evaluate_all(
        "els => els.map(e => [e.querySelector('.big').textContent, e.querySelector('.lbl').textContent])")
    cost, calls = cards[0], cards[1]
    return f"Result: {cost[0]} ({cost[1]}); {calls[0]} ({calls[1]})"


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
    page.wait_for_selector("#classical tbody tr", timeout=120_000)
    page.wait_for_function("() => !document.getElementById('run').disabled", timeout=60_000)
    page.add_style_tag(content=CAPTION_CSS)

    # a. the problem
    caption(page, "Where to put 4 stores in Champaign County, Illinois. 158 census block groups, real driving times. "
                  "The map shows the exact best placement; the table shows three classical methods.")
    center(page, "#map")
    time.sleep(5)

    # b. sliders
    caption(page, "Every weight is a slider; the classical answers and the map update instantly.")
    center(page, "#controls")
    time.sleep(1)
    set_slider(page, "#lam", 40)
    wait_refreshed(page)
    time.sleep(1.5)
    center(page, "#map")   # the controls and the map do not fit together: show the update on the map, then go back
    time.sleep(2)
    center(page, "#controls")
    set_slider(page, "#lam", 10)
    wait_refreshed(page)
    time.sleep(1)

    # c. live run: the theory-match sweep first
    caption(page, "Run quantum search: first a theory-match sweep, 25 Grover circuits at 1,000 shots. "
                  "Measured success probabilities against sin^2((2j+1) theta).")
    page.click("#run")
    page.wait_for_selector("#sweep path.theory", timeout=300_000)
    center(page, "#sweep")
    time.sleep(5)

    # d. Durr-Hoyer rounds
    caption(page, "Now Durr-Hoyer minimum finding: one measurement per round, the threshold tightens, "
                  "the map follows the current best placement.")
    together = show_both(page, "#map", "#dh-counters")
    time.sleep(3)
    deadline = time.time() + 600
    last_swap = time.time()
    on_map = True
    if not together:
        center(page, "#map")
    while page.locator("#dh-counters").get_attribute("data-done") != "true":
        if time.time() > deadline:
            raise TimeoutError("the Durr-Hoyer loop did not finish within 600 s")
        caption(page, "Round by round: " + counters_line(page))
        if not together and time.time() - last_swap > 4.5:   # the two do not fit on one screen: alternate
            on_map = not on_map
            center(page, "#map" if on_map else "#dh-counters")
            last_swap = time.time()
        time.sleep(0.5)

    # e. result cards
    caption(page, cards_line(page))
    center(page, "#cards")
    time.sleep(6)

    # f. the recorded take
    page.click("#load-recorded")
    page.wait_for_selector("#sweep path.theory", timeout=60_000)
    page.wait_for_selector("#dh-counters[data-done='true']", timeout=60_000)
    page.wait_for_selector("#scaling-fig", state="visible", timeout=60_000)
    page.wait_for_function("() => !document.getElementById('run').disabled", timeout=60_000)
    caption(page, "The shipped take at 12 sites (N = 495): exact minimum found after scoring 170 placements against 495. "
                  "Below: calls grow like sqrt N over 65 runs.")
    center(page, "#scaling-fig")
    time.sleep(7)

    # g. credits
    caption(page, "Team ZollerQu. Code and report: branch placement of github.com/Oussema-t/zollerqu-qollab. "
                  "This effort is supported by Qollab & IonQ.")
    time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8767/")
    ap.add_argument("--out", default="dev/out")
    ap.add_argument("--mp4", default="docs/demo.mp4")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport=VIEWPORT, color_scheme="dark", record_video_dir=str(out),
                                  record_video_size=VIEWPORT)
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
    mp4 = Path(a.mp4)
    mp4.parent.mkdir(parents=True, exist_ok=True)
    ok = to_mp4(final, mp4)
    print("wrote", final, f"and {mp4}" if ok else "(no ffmpeg with an h264 encoder found, webm only)")
    print(f"wall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
