"""Generic reviewer runner. usage: python runner.py <config.py> ; config defines dict CFG.
Runs in its own working dir (CFG['workdir']) so results/ is separate."""
import sys, time, os, io, json, traceback, pickle, hashlib, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from load_nb import load

cfg_ns = {"np": np}
exec(open(sys.argv[1]).read(), cfg_ns)
CFG = cfg_ns["CFG"]
workdir = os.path.join(HERE, CFG.get("workdir", "wd_default"))
os.makedirs(os.path.join(workdir, "results"), exist_ok=True)
os.chdir(workdir)

code = open(os.path.join(HERE, "task04_cell.py")).read()
for old, new, count in CFG.get("mutations", []):
    assert code.count(old) == count, f"mutation anchor count {code.count(old)} != {count}: {old!r}"
    code = code.replace(old, new)

t0 = time.time()
ns = load(CFG.get("mode", "small"), upto=7)
print(f"[load {CFG.get('mode','small')}: {time.time()-t0:.1f} s]", flush=True)
for k, v in CFG.get("run", {}).items():
    ns["RUN"][k] = v
if "overrides" in CFG:
    ns["TASK04_OVERRIDES"] = CFG["overrides"]
if "pre" in cfg_ns:
    cfg_ns["pre"](ns)

def fingerprint(v):
    try:
        return hashlib.sha1(pickle.dumps(v, protocol=4)).hexdigest()
    except Exception:
        return "id:%d" % id(v)

snap = None
if CFG.get("nsdiff"):
    snap = {k: (id(v), fingerprint(v), type(v).__name__) for k, v in ns.items()}

lines = code.split("\n")
t0 = time.time()
status = "OK"
buf = io.StringIO()
try:
    if CFG.get("quiet", True):
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "task04_cell.py", "exec"), ns)
    else:
        exec(compile(code, "task04_cell.py", "exec"), ns)
except BaseException as e:
    status = f"CRASH {type(e).__name__}: {e}"
    tb = traceback.extract_tb(sys.exc_info()[2])
    print("=== TRACEBACK (cell frames) ===")
    for fr in tb:
        if fr.filename == "task04_cell.py":
            print(f"  cell line {fr.lineno} in {fr.name}: {lines[fr.lineno-1].strip()[:230]}")
        else:
            print(f"  {os.path.basename(fr.filename)}:{fr.lineno} in {fr.name}")
    print("  ->", status)
print(f"[STATUS] {status}  [{time.time()-t0:.1f} s]", flush=True)
out = buf.getvalue()
open("stdout.txt", "w").write(out)
if CFG.get("show_stdout"):
    print(out)
if "t4_tests" in ns:
    for name, result in ns["t4_tests"].items():
        print(f"  TEST {name}: {result}")
if "task04_failures" in ns:
    print("  failures:", ns["task04_failures"])
if snap is not None:
    changed = []
    for k, (i, fp, tn) in snap.items():
        if k == "__builtins__": continue
        if k not in ns: changed.append((k, "DELETED")); continue
        v = ns[k]
        if id(v) != i or fingerprint(v) != fp:
            changed.append((k, f"{tn} -> {type(v).__name__}; same_object={id(v)==i}; same_content={fingerprint(v)==fp}"))
    print("=== PRE-EXISTING NAMES CHANGED ===")
    for c in changed: print("  ", c)
    print("=== NEW NAMES ===")
    print("  ", sorted(k for k in ns if k not in snap))
if "post" in cfg_ns:
    try:
        cfg_ns["post"](ns)
    except Exception:
        traceback.print_exc()
