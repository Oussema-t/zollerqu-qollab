"""Figure of the nine benchmark objects (frozen seed-7 meshes) for the paper: report/fig_meshes.pdf.
Loads the notebook cells up to Task 01d (mesh definitions + frozen meshes) and draws every case on one page.
usage: python report/make_mesh_figure.py
"""
import contextlib
import io
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
MESH_COLOUR, HEAVY, COUPLING, CUT_GREY, SOURCE = "#3B75AF", "#CC79A7", "#009E73", "#D9D9D9", "#D55E00"


def load_mesh_cells():
    nb = nbformat.read(ROOT / "mesh_idea_ZollerQu.ipynb", as_version=4)
    ns = {}
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code" or cell.source.startswith("# TASK_01_IMPLEMENTATION_CELL_START"):
            continue
        if cell.source.startswith("# TASK_02_IMPLEMENTATION"):
            return ns
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cell.source, f"cell{index}", "exec"), ns)
    raise RuntimeError("Task 02 cell not found")


def draw_case(ax, name, spec, mesh):
    tri_xy = mesh["points"][mesh["triangles"]]
    ax.add_collection(PolyCollection(tri_xy, facecolors="none", edgecolors=MESH_COLOUR, linewidths=0.18, rasterized=True))
    for zone in spec.get("work_zones", []):
        x0, y0, x1, y1 = zone["box"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=HEAVY, alpha=0.25, hatch="xx", edgecolor=HEAVY, lw=0.6))
    for zone in spec.get("cost_zones", []):
        x0, y0, x1, y1 = zone["box"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=COUPLING, alpha=0.25, hatch="..", edgecolor=COUPLING, lw=0.6))
    for zone in spec.get("priority_zones", []):
        ax.add_patch(Circle(zone["center"], zone["radius"], fill=False, edgecolor="black", lw=1.6))
    for cut in spec["cutouts"]:
        ax.add_patch(Polygon(cut, closed=True, facecolor=CUT_GREY, edgecolor="#777777", hatch="///", lw=0.8))
    for loop in spec["loops"]:
        ax.add_patch(Polygon(loop, closed=True, fill=False, edgecolor="black", lw=1.0))
    for feature in spec["features"]:
        if feature["kind"] == "heat source":
            ax.scatter(*feature["center"], s=40 + 110 * feature["strength"], marker="*", c=SOURCE, edgecolors="black", linewidths=0.6, zorder=5)
            ax.annotate(f"{feature['strength']:.1f}", feature["center"], xytext=(5, -9), textcoords="offset points", fontsize=6.5,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    if name.startswith("4"):                                        # uniform pull at both ends (structural case)
        lo, hi = spec["hull"].min(axis=0), spec["hull"].max(axis=0)
        for y in np.linspace(lo[1] + 0.25, hi[1] - 0.25, 3):
            ax.add_patch(FancyArrowPatch((lo[0], y), (lo[0] - 0.22, y), arrowstyle="-|>", mutation_scale=7, color=COUPLING, lw=1))
            ax.add_patch(FancyArrowPatch((hi[0], y), (hi[0] + 0.22, y), arrowstyle="-|>", mutation_scale=7, color=COUPLING, lw=1))
    lo, hi = spec["hull"].min(axis=0), spec["hull"].max(axis=0)
    pad = 0.26 if name.startswith("4") else 0.04
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - 0.04, hi[1] + 0.04)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    work = float(np.sum(mesh["cell_work"]))
    ax.set_title(f"Case {name}\n{textwrap.fill(spec['title'], 30)}\n{len(mesh['triangles'])} triangles, total work {work:.0f}", fontsize=7.5)


if __name__ == "__main__":
    ns = load_mesh_cells()
    cases = ns["BENCHMARK_CASES"]
    fig, axes = plt.subplots(3, 3, figsize=(7.1, 7.0))
    for ax, name in zip(axes.flat, cases):
        draw_case(ax, name, ns["MESH_CASES"][name], ns["mesh_cases"][name])
    handles = [Line2D([], [], color=MESH_COLOUR, lw=0.8, label="mesh triangle edges"),
               Patch(facecolor=CUT_GREY, edgecolor="#777777", hatch="///", label="removed corner / hole"),
               Line2D([], [], marker="*", ls="none", markerfacecolor=SOURCE, markeredgecolor="black", ms=9, label="heat source (strength)"),
               Patch(facecolor=HEAVY, alpha=0.4, hatch="xx", edgecolor=HEAVY, label="heavy material: 3x work"),
               Patch(facecolor=COUPLING, alpha=0.4, hatch="..", edgecolor=COUPLING, label="strong coupling: borders cost 3x"),
               Line2D([], [], color="black", lw=1.6, marker="o", ms=8, markerfacecolor="none", ls="none", label="priority zone"),
               Line2D([], [], color=COUPLING, lw=1, marker=">", ms=4, label="uniform pull (case 4)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.3)
    out = ROOT / "report" / "fig_meshes.pdf"
    fig.savefig(out, dpi=300, metadata={"CreationDate": None})
    fig.savefig(ROOT / "results" / "mesh_objects_all_cases_seed7.png", dpi=160)
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB) with {len(cases)} cases; "
          f"{sum(len(ns['mesh_cases'][c]['triangles']) for c in cases)} triangles in total")
