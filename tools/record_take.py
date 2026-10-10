"""Generate data/recorded_aria1.json locally with Aer as a stand-in until the real Forte run.
Run from the project root: python tools/record_take.py
"""
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qec.aer_compat import simulate, stand_in_noise
from qec.lessons import build_all

SETTINGS = {"shots": 500, "inject": None}   # no injection: lesson 1 shows the noise floor


def main():
    circuits, meta = build_all(SETTINGS)
    start = time.time()
    counts = simulate(circuits, SETTINGS["shots"], noise_model=stand_in_noise(), seed=1234)
    print(f"simulated in {time.time() - start:.0f} s")
    take = {
        "source": "local-aer-stand-in",
        "backend": "Aer depolarizing stand-in (Forte run pending)",
        "date": datetime.date.today().isoformat(),
        "settings": SETTINGS, "meta": meta, "counts": counts,
    }
    out = Path(__file__).resolve().parents[1] / "data" / "recorded_aria1.json"
    out.write_text(json.dumps(take))
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
