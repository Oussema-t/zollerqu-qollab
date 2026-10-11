"""Build the Qollab upload bundle. Run from the project root: python tools/make_bundle.py
Writes dist/zollerqu-unlocked-qollab.zip with index.html, main.js, main.css, ui/*.js, qec/*.py (except aer_compat.py) and data/recorded_aria1.json.
Qollab runtime limits: at most 100 files, each at most 2 MB; index.html and main.js at the top level run as a JS project.
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "zollerqu-unlocked-qollab.zip"
MAX_FILES = 100
MAX_BYTES = 2 * 1024 * 1024


def collect():
    files = [ROOT / "index.html", ROOT / "main.js", ROOT / "main.css"]
    files += sorted((ROOT / "ui").glob("*.js"))
    # aer_compat.py imports qiskit_aer and is used only by dev/serve.py and tools/record_take.py; bridge.py never imports it
    files += sorted(p for p in (ROOT / "qec").glob("*.py") if "__pycache__" not in p.parts and p.name != "aer_compat.py")
    files += [ROOT / "data" / "recorded_aria1.json"]
    return files


def main():
    files = collect()
    missing = [str(p.relative_to(ROOT)) for p in files if not p.is_file()]
    if missing:
        sys.exit(f"missing files: {', '.join(missing)}")
    names = [p.relative_to(ROOT).as_posix() for p in files]
    for top in ("index.html", "main.js"):
        if top not in names:
            sys.exit(f"{top} must be at the top level of the bundle")
    if len(files) > MAX_FILES:
        sys.exit(f"{len(files)} files, Qollab allows at most {MAX_FILES}")
    big = [(n, p.stat().st_size) for n, p in zip(names, files) if p.stat().st_size > MAX_BYTES]
    if big:
        sys.exit("over the 2 MB per-file limit: " + ", ".join(f"{n} ({s} bytes)" for n, s in big))

    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for n, p in zip(names, files):
            z.write(p, n)

    with zipfile.ZipFile(OUT) as z:   # check what was written, not what was intended
        listed = z.namelist()
        assert "index.html" in listed and "main.js" in listed, "index.html and main.js must be at the zip's top level"
        assert len(listed) <= MAX_FILES
        assert all(i.file_size <= MAX_BYTES for i in z.infolist())
        total = 0
        for i in z.infolist():
            print(f"{i.file_size:>10}  {i.filename}")
            total += i.file_size
    print(f"{len(listed)} files, {total} bytes uncompressed, {OUT.stat().st_size} bytes zipped -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
