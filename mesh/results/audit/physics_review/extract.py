"""Extract only the function/class definitions (and a few named constants) from task04_cell.py into a namespace."""
import ast, time, zlib
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def extract(extra=None, source_path="task04_cell.py", mutate=None):
    src = open(source_path).read()
    if mutate is not None:
        src = mutate(src)
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("T4_HADAMARD_CACHE", "T4_TOL", "T4_WARM_FLAVOURS") for t in node.targets):
            keep.append(node)
    mod = ast.Module(body=keep, type_ignores=[])
    ns = dict(np=np, minimize=minimize, QuantumCircuit=QuantumCircuit, Statevector=Statevector, time=time, zlib=zlib)
    if extra:
        ns.update(extra)
    exec(compile(mod, "task04_defs", "exec"), ns)
    return ns
