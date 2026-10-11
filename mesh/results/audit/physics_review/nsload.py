"""Load the notebook namespace and exec ONLY definitions/config of the Task 04 cell (no calibration, no benchmark)."""
import ast, os, sys
import numpy as np
from load_nb import load

def load_defs(mode="small", overrides=None, mutate=None):
    os.makedirs("results", exist_ok=True)
    ns = load(mode, upto=7)
    if overrides:
        ns["TASK04_OVERRIDES"] = overrides
    src1 = open("task04_part1.py").read() + open("task04_part2.py").read()     # config + defs only
    src3 = open("task04_part3.py").read()
    if mutate:
        src1, src3 = mutate(src1), mutate(src3)
    exec(compile(src1, "task04_part12", "exec"), ns)
    tree = ast.parse(src3)
    keep = [node for node in tree.body if isinstance(node, ast.FunctionDef)
            or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "T4_WARM_FLAVOURS" for t in node.targets))]
    exec(compile(ast.Module(body=keep, type_ignores=[]), "task04_part3_defs", "exec"), ns)
    return ns
