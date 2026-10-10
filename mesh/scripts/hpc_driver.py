"""HPC driver: load the notebook cells once, then fork parallel shards of Task 04 and Task 04b, merge, and finish.
usage: python hpc_driver.py <mode> <n_shards> [fingerprints-only]"""
import contextlib
import glob
import gzip
import io
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import nbformat

MODE, SHARDS = sys.argv[1], int(sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.makedirs("results", exist_ok=True)
T0 = time.time()


def log(message):
    print(f"[{time.time() - T0:7.1f} s] {message}", flush=True)


def load_namespace(mode):
    nb = nbformat.read(os.path.join(HERE, "mesh_idea_ZollerQu.ipynb"), as_version=4)
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in nb.cells:
            if cell.cell_type != "code" or cell.source.startswith("# TASK_01_IMPLEMENTATION_CELL_START"):
                continue
            if cell.source.startswith("# TASK_04"):
                break
            source = cell.source.replace('RUN_MODE = "small"', f'RUN_MODE = "{mode}"') if "RUN_SETTINGS_CELL_START" in cell.source else cell.source
            exec(source, ns)
    return ns


def run_cell(ns, path, label, quiet=True):
    """Execute a cell file in ns; a SystemExit (end of a shard) is normal. Returns the printed text."""
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer) if quiet else contextlib.nullcontext():
            exec(compile(open(path).read(), label, "exec"), ns)
    except SystemExit as stop:
        buffer.write(f"\n[{label}] {stop}\n")
    return buffer.getvalue()


def fork_shards(ns, path, label, shard_name, shards, extra=None):
    children = []
    for index in range(shards):
        pid = os.fork()
        if pid == 0:                                   # child: run one shard and leave
            try:
                ns[shard_name] = (index, shards)
                if extra:
                    ns.update(extra)
                text = run_cell(ns, path, f"{label}-shard{index}")
                open(f"{label}_shard{index}.log", "w").write(text)
            except BaseException as error:             # noqa: BLE001 - reported in the log
                open(f"{label}_shard{index}.log", "w").write(f"SHARD FAILED: {type(error).__name__}: {error}")
            finally:
                os._exit(0)
        children.append(pid)
    for pid in children:
        os.waitpid(pid, 0)


def merge(main_path, shard_pattern, entries_key):
    """Union of the shard checkpoints (and the existing main checkpoint if it was written by the same settings and code)."""
    parts = [json.load(gzip.open(path, "rt")) for path in sorted(glob.glob(shard_pattern))]
    if not parts:
        return 0, 0
    identity = lambda d: {k: v for k, v in d.items() if k != entries_key}
    merged = parts[0]
    if os.path.exists(main_path):
        base = json.load(gzip.open(main_path, "rt"))
        if identity(base) == identity(merged):
            base[entries_key].update(merged[entries_key])
            merged = base
    for part in parts[1:]:
        if identity(part) != identity(merged):
            raise RuntimeError("a shard was written by different settings or code")
        merged[entries_key].update(part[entries_key])
    with gzip.open(main_path, "wt") as handle:
        json.dump(merged, handle)
    return len(parts), len(merged[entries_key])


ns = load_namespace(MODE)
log(f"notebook cells loaded in {MODE} mode")

if len(sys.argv) > 3:                                  # fingerprints only (cross-platform check)
    fingerprints = {name: ns["mesh_cases"][name]["fingerprint"] for name in ns["BENCHMARK_CASES"]}
    exec(compile(open("task04_cell.py").read().split("# ------------------------------------------------- calibration of LR-QAOA ----")[0], "t4defs", "exec"), ns)
    instances = {f"{case}|12|7|{scenario}": ns["t4_build_instance"](ns["region_cases"][(case, 12, 7)], case, 12, 7, scenario)["fingerprint"]
                 for case in ns["BENCHMARK_CASES"] for scenario in ns["TARGET_SHARES"]}
    json.dump(dict(meshes=fingerprints, instances=instances), open("fingerprints.json", "w"), indent=1)
    log("fingerprints written")
    sys.exit(0)

# ---- Task 04: shards, merge, final run ----
fork_shards(ns, "task04_cell.py", "task04", "TASK04_SHARD", SHARDS)
log("Task 04 shards finished: " + "; ".join(open(f).read().strip().splitlines()[-1][:120] for f in sorted(glob.glob("task04_shard*.log"))))
print(merge(f"results/task04_checkpoint_{MODE}.json.gz", f"results/task04_checkpoint_{MODE}.shard*of{SHARDS}.json.gz", "instances"), flush=True)
text = run_cell(ns, "task04_cell.py", "task04-final")
open(f"task04_{MODE}_result.log", "w").write(text)
log(f"Task 04 final run done ({ns.get('t4_reused')} reused, {ns.get('t4_computed')} computed, failures {len(ns.get('task04_failures', []))})")

# ---- Task 04b: step-size calibration (threads), shards, merge, final run ----
ns["VARQITE_OVERRIDES"] = dict(threads=SHARDS)
ns["VARQITE_SHARD"] = "calibration"
text = run_cell(ns, "task04b_cell.py", "task04b-calibration")
open(f"task04b_{MODE}_calibration.log", "w").write(text)
log("varQITE calibration done")
ns.pop("VARQITE_SHARD")
fork_shards(ns, "task04b_cell.py", "task04b", "VARQITE_SHARD", SHARDS, extra=dict(VARQITE_OVERRIDES=dict(threads=1)))
log("varQITE shards finished: " + "; ".join(open(f).read().strip().splitlines()[-1][:120] for f in sorted(glob.glob("task04b_shard*.log"))))
print(merge(f"results/varqite_checkpoint_{MODE}.json.gz", f"results/varqite_checkpoint_{MODE}.shard*of{SHARDS}.json.gz", "entries"), flush=True)
ns.pop("VARQITE_SHARD", None)
ns["VARQITE_OVERRIDES"] = dict(threads=SHARDS)
text = run_cell(ns, "task04b_cell.py", "task04b-final")
open(f"task04b_{MODE}_result.log", "w").write(text)
log(f"Task 04b final run done ({len(ns.get('varqite_results', {}))} runs, failures {len(ns.get('vq_failures', []))})")
open(f"DONE_{MODE}", "w").write("done")
