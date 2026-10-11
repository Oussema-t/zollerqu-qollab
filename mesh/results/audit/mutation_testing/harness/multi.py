"""Load the notebook namespace ONCE, then run many configs sequentially, restoring the namespace bindings in between.
usage: python multi.py <mode> <tag> cfgfile1 cfgfile2 ...   (each cfg file defines CFG and optional pre(ns)/post(ns))"""
import sys, time, os, io, traceback, contextlib, copy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load_nb import load

mode, tag, cfgs = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(os.path.join(HERE, "wd_" + tag, "results"), exist_ok=True)
os.chdir(os.path.join(HERE, "wd_" + tag))          # the notebook's earlier cells write their plots here
t0 = time.time()
ns = load(mode, upto=7)
print(f"[load {mode}: {time.time()-t0:.1f} s]", flush=True)
ns0 = dict(ns)
run0 = copy.deepcopy(ns["RUN"])
base_code = open(os.path.join(HERE, "task04_cell.py")).read()

for cfgfile in cfgs:
    name = os.path.basename(cfgfile)[:-3]
    cfg_ns = {"np": np}
    exec(open(os.path.join(HERE, cfgfile)).read(), cfg_ns)
    CFG = cfg_ns["CFG"]
    code = base_code
    ok = True
    for old, new, count in CFG.get("mutations", []):
        if code.count(old) != count:
            print(f"##### {name}: MUTATION ANCHOR COUNT {code.count(old)} != {count}: {old!r}", flush=True); ok = False
        code = code.replace(old, new)
    if not ok: continue
    ns.clear(); ns.update(ns0)
    ns["RUN"].clear(); ns["RUN"].update(copy.deepcopy(run0))
    for k, v in CFG.get("run", {}).items():
        ns["RUN"][k] = v
    if "overrides" in CFG:
        ns["TASK04_OVERRIDES"] = CFG["overrides"]
    wd = os.path.join(HERE, CFG.get("workdir", "wd_" + name))
    os.makedirs(os.path.join(wd, "results"), exist_ok=True)
    os.chdir(wd)
    if "pre" in cfg_ns: cfg_ns["pre"](ns)
    lines = code.split("\n")
    buf = io.StringIO()
    status = "OK"
    t0 = time.time()
    print(f"##### {name}", flush=True)
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "task04_cell.py", "exec"), ns)
    except BaseException as e:
        status = f"CRASH {type(e).__name__}: {str(e)[:300]}"
        for fr in traceback.extract_tb(sys.exc_info()[2]):
            if fr.filename == "task04_cell.py":
                print(f"    cell line {fr.lineno} in {fr.name}: {lines[fr.lineno-1].strip()[:200]}")
            else:
                print(f"    {os.path.basename(fr.filename)}:{fr.lineno} in {fr.name}")
    plt.close("all")
    print(f"  [STATUS] {status}  [{time.time()-t0:.1f} s]", flush=True)
    open("stdout.txt", "w").write(buf.getvalue())
    if "t4_tests" in ns:
        for tname, result in ns["t4_tests"].items():
            if "FAIL" in str(result) or CFG.get("show_tests"): print(f"    TEST {tname}: {result}")
    if ns.get("task04_failures"):
        print("    task04_failures:", [str(f)[:300] for f in ns["task04_failures"]][:4], "n =", len(ns["task04_failures"]))
    if "post" in cfg_ns:
        try: cfg_ns["post"](ns)
        except Exception: traceback.print_exc()
    sys.stdout.flush()
print("ALL_DONE", flush=True)
