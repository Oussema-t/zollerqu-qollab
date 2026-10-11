import ast, builtins, nbformat, symtable, sys
cell = open("task04_cell.py").read()
tree = ast.parse(cell)

# 1. names assigned at top level of the Task 04 cell (incl. loop targets, with-as, imports, defs)
def targets(node, acc):
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            pass
    return acc
top_assigned = {}
class Top(ast.NodeVisitor):
    def visit_FunctionDef(self, node): top_assigned.setdefault(node.name, []).append(("def", node.lineno))
    def visit_ClassDef(self, node): top_assigned.setdefault(node.name, []).append(("class", node.lineno))
    def visit_Lambda(self, node): pass
    def visit_ListComp(self, node): pass
    def visit_DictComp(self, node): pass
    def visit_SetComp(self, node): pass
    def visit_GeneratorExp(self, node): pass
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store): top_assigned.setdefault(node.id, []).append(("assign", node.lineno))
    def visit_Import(self, node):
        for a in node.names: top_assigned.setdefault((a.asname or a.name).split(".")[0], []).append(("import", node.lineno))
    def visit_ImportFrom(self, node):
        for a in node.names: top_assigned.setdefault(a.asname or a.name, []).append(("import", node.lineno))
    def visit_ExceptHandler(self, node):
        if node.name: top_assigned.setdefault(node.name, []).append(("except", node.lineno))
        self.generic_visit(node)
Top().visit(tree)

# 2. earlier notebook cells: names defined at top level, and globals referenced by their functions
nb = nbformat.read("/Users/t/Downloads/zollerqu_mesh_workspace/mesh_idea_ZollerQu.ipynb", as_version=4)
cc = [c for c in nb.cells if c.cell_type == "code"][:7]
earlier_top = {}
earlier_func_globals = {}
def func_globals(src, label):
    st = symtable.symtable(src, label, "exec")
    out = {}
    def rec(tab, path):
        for child in tab.get_children():
            if child.get_type() in ("function",):
                g = {s.get_name() for s in child.get_symbols() if s.is_global() and s.is_referenced()}
                out[path + child.get_name()] = out.get(path + child.get_name(), set()) | g
            rec(child, path + child.get_name() + ".")
    rec(st, "")
    return out
for i, c in enumerate(cc):
    t = ast.parse(c.source)
    saved = dict(top_assigned); top_assigned.clear()
    Top().visit(t)
    for k, v in top_assigned.items(): earlier_top.setdefault(k, []).append(i)
    top_assigned.clear(); top_assigned.update(saved)
    for f, g in func_globals(c.source, f"cell{i}").items():
        earlier_func_globals[(i, f)] = g

t4_names = set(top_assigned)
overlap = sorted(n for n in t4_names if n in earlier_top)
print("Task-04 top-level names:", len(t4_names), "| also assigned at top level of earlier cells:", len(overlap))
print(overlap)
# which earlier functions read (as globals) a name that Task 04 re-binds?
bi = set(dir(builtins))
print("\nEarlier-cell functions that reference, as a GLOBAL, a name that the Task 04 cell re-binds:")
for (i, f), g in sorted(earlier_func_globals.items()):
    hit = sorted((g & t4_names) - bi)
    if hit: print(f"  cell {i} {f}: {hit}")
# 3. Task-04 functions: globals they reference that are loop/temporary variables of the cell (not defs/imports/CONSTANTS)
t4_fg = func_globals(cell, "task04")
defs = {k for k, v in top_assigned.items() if any(kind in ("def", "class", "import") for kind, _ in v)}
print("\nTask-04 functions referencing module-level names that are plain variables (not defs/imports):")
for f, g in sorted(t4_fg.items()):
    plain = sorted(n for n in g if n not in bi and n not in defs and (n in t4_names) )
    ext = sorted(n for n in g if n not in bi and n not in t4_names)
    print(f"  {f}: cell-vars={plain} | earlier-cell names={ext}")
