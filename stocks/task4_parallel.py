"""Parallel driver for Task 4: the same results as `python -m stocks.task4_report`, computed on several cores.

Every piece of task4_report.run_all is deterministic and independent given its inputs (fixed seeds), so the
pieces run in worker processes and are merged into the same structure, then saved and reported.
Offline tool for a many-core machine:   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m stocks.task4_parallel [workers]
"""
import multiprocessing as mp
import sys
import time

from stocks import ising, task3_report as t3, task4_report as t4, varqite

SETTINGS = t4.SETTINGS
_CACHE = {}


def _ctx():
    if "ctx" not in _CACHE:
        _CACHE["ctx"] = t3.load_context()
    return _CACHE["ctx"]


def _context(mode, balance):
    key = (mode, balance)
    if key not in _CACHE:
        _CACHE[key] = t4.Context(t3.size_setup(_ctx(), SETTINGS["n"]), mode, balance, SETTINGS["eta"])
    return _CACHE[key]


def _quiet(_):
    pass


def tune_job(args):
    mode, balance = args
    started = time.time()
    return t4.key_of(mode, balance), t4.tune_model(_context(mode, balance), SETTINGS, _quiet), time.time() - started


def extra_job(args):
    mode, name, model_result = args
    started = time.time()
    result = t4.attempt(name, t4.extra_tasks(_context(mode, False), model_result, SETTINGS)[name], _quiet)
    return mode, name, result, time.time() - started


def varqite_job(args):
    mode, kind, lr = args
    c = _context(mode, False)
    started = time.time()
    try:
        record = varqite.train_one(c.model, kind, lr, evaluate=c.evaluate, steps=SETTINGS["varqite_steps"])
    except Exception as err:
        record = {"kind": kind, "lr": lr, "error": f"{type(err).__name__}: {err}"}
    return mode, kind, lr, record, time.time() - started


def xy_job(models):
    contexts = {t4.key_of(m, b): _context(m, b) for m in ising.MODES for b in (False, True)}
    started = time.time()
    result = t4.attempt("XY mixer with Dicke start", lambda: {"modes": t4.xy_extra(contexts, models, SETTINGS)}, _quiet)
    return result, time.time() - started


def transfer_job(models):
    started = time.time()
    result = t4.attempt("parameter transfer check", lambda: t4.transfer_report(_ctx(), models, SETTINGS, _quiet), _quiet)
    return result, time.time() - started


def main(workers=None):
    started = time.time()
    workers = workers or mp.cpu_count()
    ctx = _ctx()
    setup = t3.size_setup(ctx, SETTINGS["n"])
    out = {"meta": t4.meta_block(setup, ctx["split"], ctx["dates"], SETTINGS)}
    out["meta"]["generated_by"] = f"python -m stocks.task4_parallel ({workers} workers)"
    print(f"phase 1: tuning 4 models on {workers} workers", flush=True)
    with mp.get_context("fork").Pool(workers) as pool:
        out["models"] = {}
        for key, result, seconds in pool.imap_unordered(tune_job, [(m, b) for m in ising.MODES for b in (False, True)]):
            out["models"][key] = result
            print(f"  {key}: {seconds:.0f} s", flush=True)
        contexts = {t4.key_of(m, b): _context(m, b) for m in ising.MODES for b in (False, True)}
        out["balance"] = t4.balance_diagnostics(contexts, out["models"], SETTINGS)
        out["circuits"] = t4.circuit_report(ctx, contexts, out["models"], SETTINGS)
        print("phase 2: extras, XY mixer and transfer check", flush=True)
        xy = pool.apply_async(xy_job, (out["models"],))
        transfer = pool.apply_async(transfer_job, (out["models"],))
        names = [n for n in t4.EXTRA_ORDER if n != "varqite_pauli"]
        extra_jobs = [pool.apply_async(extra_job, ((m, name, out["models"][t4.key_of(m, False)]),))
                      for m in ising.MODES for name in names]
        vq_jobs = [pool.apply_async(varqite_job, ((m, kind, lr),))
                   for m in ising.MODES for kind in SETTINGS["varqite_kinds"] for lr in SETTINGS["varqite_lrs"]]
        collected = {t4.key_of(m, False): {} for m in ising.MODES}
        for job in extra_jobs:
            mode, name, result, seconds = job.get()
            collected[t4.key_of(mode, False)][name] = result
            print(f"  extra {name} {mode}: {result['status']} in {seconds:.0f} s", flush=True)
        records = {m: [] for m in ising.MODES}
        for job in vq_jobs:
            mode, kind, lr, record, seconds = job.get()
            records[mode].append(record)
            print(f"  varqite {mode} ansatz {kind} lr {lr}: {'failed' if 'error' in record else 'ok'} in {seconds:.0f} s", flush=True)
        for m in ising.MODES:
            good = [r for r in records[m] if "error" not in r]
            c = contexts[t4.key_of(m, False)]
            merged = t4.attempt("varQITE (Pauli-term update)", lambda g=good, cc=c: t4.varqite_extra(cc, SETTINGS, runs=g), _quiet)
            merged["failed_runs"] = [r for r in records[m] if "error" in r]
            collected[t4.key_of(m, False)]["varqite_pauli"] = merged
        out["extras"] = {key: {name: value[name] for name in t4.EXTRA_ORDER if name in value} for key, value in collected.items()}
        out["extras"]["xy_mixer"], seconds = xy.get()
        print(f"  xy mixer: {out['extras']['xy_mixer']['status']} in {seconds:.0f} s", flush=True)
        out["extras"]["deferred"] = dict(t4.DEFERRED)
        out["transfer"], seconds = transfer.get()
        print(f"  transfer: {out['transfer']['status']} in {seconds:.0f} s", flush=True)
    out["meta"]["seconds"] = round(time.time() - started, 1)
    paths = t4.save(out)
    print("saved:", ", ".join(str(p) for p in paths), flush=True)
    return t4.print_report(t4.load(), with_tests=False)


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else None))
