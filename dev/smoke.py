"""Playwright smoke test of the page against the dev harness (start it first: python dev/serve.py 8766).

Run from the project root: python dev/smoke.py [url]. Screenshots go to dev/out/.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8766/"
OUT = Path(__file__).resolve().parent / "out"


def cards(page):
    return page.locator("#cards .card").evaluate_all(
        "els => els.map(e => e.querySelector('.big').textContent + ' | ' + e.querySelector('.lbl').textContent)")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # card labels carry non-ASCII signs
    OUT.mkdir(exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(URL)

        page.wait_for_selector("#classical tbody tr", timeout=120_000)
        page.wait_for_function("!document.getElementById('run').disabled", timeout=60_000)
        print("classical:", page.locator("#classical tbody").inner_text().replace("\n", " / "))
        page.screenshot(path=str(OUT / "01-classical.png"), full_page=True)

        page.click("#load-recorded")
        page.wait_for_selector("#sweep path.theory", timeout=60_000)
        page.wait_for_selector("#dh-counters[data-done='true']", timeout=60_000)
        page.wait_for_function("!document.getElementById('run').disabled", timeout=60_000)
        n_study = page.evaluate("fetch('data/recorded_local.json').then(r => r.json()).then(t => (t.study || []).length)")
        assert n_study > 0, "the take has no scaling study"
        assert page.locator("#scaling-fig").is_visible(), "scaling plot hidden after loading the take"
        assert page.locator("#scaling g.study circle.mean").count() == n_study, "scaling plot did not draw every study point"
        print("scaling:", page.locator("#scaling-cap").inner_text())
        print("scaling fractions:", page.locator("#scaling text.frac").evaluate_all("els => els.map(e => e.textContent)"))
        print("take status:", page.locator("#status").inner_text())
        print("take cards:", cards(page))
        page.screenshot(path=str(OUT / "02-take.png"), full_page=True)

        page.click("#run")
        page.wait_for_selector("#dh-counters[data-done='false']", timeout=10_000)
        # up to 120 rounds, each a short one-measurement job
        page.wait_for_selector("#dh-counters[data-done='true']", timeout=480_000)
        assert page.locator("#scaling-fig").is_visible(), "scaling plot gone after the live run"
        print("live cards:", cards(page))
        print("counters:", page.locator("#dh-counters").inner_text().replace("\n", " / "))
        print("status:", page.locator("#status").inner_text())
        print("banner:", page.locator("#banner").inner_text())
        page.screenshot(path=str(OUT / "03-quantum.png"), full_page=True)
        browser.close()

    if errors:
        print("console errors:", *errors, sep="\n  ")
        sys.exit(1)
    print("no console errors")


if __name__ == "__main__":
    main()
