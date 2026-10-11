"""Reviewer harness: load the notebook namespace, run the Task 04 cell in small mode re-using MY full-run checkpoint, return ns."""
import sys, os, io, contextlib, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
def get_ns(workdir=os.path.join(HERE, "an"), quiet=True):
    os.makedirs(os.path.join(workdir, "results"), exist_ok=True)
    os.chdir(workdir)
    from load_nb import load
    ns = load("small", upto=7)
    code = open(os.path.join(HERE, "task04_cell.py")).read()
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "task04_cell", "exec"), ns)
    ns["_cell_stdout"] = buf.getvalue()
    print(f"[harness: cell executed in {time.time()-t0:.0f} s; computed {ns['t4_computed']}, reused {ns['t4_reused']}]", flush=True)
    return ns
