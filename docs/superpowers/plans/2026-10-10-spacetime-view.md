# 3D Spacetime View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Every lesson panel shows the encoded circuit as an animated 3D spacetime block (patches in the plane, time up, gates as bars, data-driven top slice, logical worldsheets), rendered with three.js, which Qollab resolves by name.

**Architecture:** Builders in `qec/code412.py` and `qec/lessons.py` record phase names in `qc.metadata["phases"]` without changing gates. `qec/spacetime.py` turns a built circuit plus its metadata into layout JSON (qubit positions, time layers of ops, phase spans, logical sheets, injection). `qec/bridge.py` exposes `spacetime(index)`. `ui/spacetime.js` renders the scene and controls. `main.js` mounts it in `showLesson`. The dev harness import map adds `three` so the page runs locally.

**Tech stack:** Qiskit 2.4 (`qc.metadata` is a plain dict, verified), three.js 0.186.1 (Qollab runtime docs: "`three` and `three/addons/…` (three.js 0.186.1) ... can be imported by name"), jsdelivr for the harness (both URLs below return 200).

**Spec:** `docs/superpowers/specs/2026-10-09-logical-teleportation-tutorial-design.md`, addendum of Oct 10.

**Depends on:** `docs/superpowers/plans/2026-10-10-dev-harness-and-demo-video.md` Tasks 1 and 2 (server and shim), which must be built first with the import map of this plan's Task 0.

---

## Facts

| Fact | Source |
|---|---|
| Block b data qubits 4b+i, ax = 4n+2b, az = 4n+2b+1 | `qec/code412.py:14-23` |
| Local layout: q0 top-left, q1 top-right, q2 bottom-left, q3 bottom-right; Z0Z1 top edge, Z2Z3 bottom edge, XXXX face, Z_L = Z0Z2 left edge, X_L = X0X1 top edge | `qec/code412.py:7-9` |
| Encoder: h q2; cx q2,q3; cx q0,q1; cx q2,q0; cx q2,q1 | `qec/code412.py:34-41` |
| transversal_cx: four cx data(c,i) -> data(g,i) | `qec/code412.py:44-46` |
| extract_syndromes: h ax; cx ax -> q0,q2,q1,q3; h ax; cx q0..q3 -> az | `qec/code412.py:53-66` |
| finish: extraction per block, then rotations, then measure all | `qec/code412.py:90-96` |
| Lesson preps: L1 h on data(0,0) when X; L2 h data(0,0); L3 ry data(0,0), h data(1,0); L4 ry data(0,0), h+t data(1,0) | `qec/lessons.py:36,59,80-84,108-112` |
| Lesson CNOTs: L2 0->1; L3 1->2 then 0->1; L4 1->0 | `qec/lessons.py:62,85-86,113` |
| meta fields: lesson, name, kind, label, nblocks, bases, value_blocks, keep_if_zero, ideal_p1, index | `qec/lessons.py:11-17,133` |
| Analysis row has `blocks[bi] = {x_face, z_top, z_bottom, zzzz_anc}` and `bases` | `qec/analyze.py` (analyze_one), used in `ui/patches.js:38-40` |
| bridge caches circuits in `_LAST["circuits"]`, figures in `_LAST["figures"]`, cleared on build | `qec/bridge.py:10-17,28-33` |
| main.js showLesson renders patches, cards, charts, clears circuit image | `main.js:155-170` |
| Panel markup order: .grid (patches, text), #cards, #charts, #circuit-details | `index.html:38-49` |
| Palette: --x #e0a040, --z #4f9cff, --good #39c37a, --bad #ff5c5c, --focus #ffd166, --muted #a7adbb | `main.css:1-4` |
| Qollab JS project runs in a sandboxed iframe; `three` importable by name | runtime docs, fetched Oct 10 |
| `qc.metadata` is `{}` by default and survives `transpile` | verified in session, Qiskit 2.4.2 |

## Decisions

- Time axis is world +y (three.js default up). Plane (x, y) of the layout maps to world (x, 0, y).
- Patch pitch 2.2 plane units; layer spacing 0.22 world units.
- Dynamic `import('three')` inside `renderSpacetime`, so a failed import degrades to the text phase list instead of breaking `main.js`.
- Per-edge logical P(1) is not in the analysis output, so the top slice labels edges with the readout basis instead. (Deviation from the spec addendum sentence "Logical edge outcomes label the edges with P(1)"; noted here, spec sentence to be removed in Task 6.)
- Bare circuits get no spacetime view.
- Rendering uses a requestAnimationFrame loop while mounted; `dispose()` stops it. Pixel ratio capped at 2.

---

### Task 0: Harness import map for three

**Files:**
- Modify: `dev/serve.py` (PAGE template import map)
- Modify: `tests/test_dev_server.py` (`test_root_wraps_index_with_import_map`)

- [ ] **Step 1: Test** add to `test_root_wraps_index_with_import_map`:

```python
    assert '"three": "https://cdn.jsdelivr.net/npm/three@0.186.1/build/three.module.js"' in html
    assert '"three/addons/": "https://cdn.jsdelivr.net/npm/three@0.186.1/examples/jsm/"' in html
```

- [ ] **Step 2: Implement** in `dev/serve.py` replace the importmap line with:

```python
<script type="importmap">{"imports": {"qec.bridge": "/dev/bridge_shim.js", "three": "https://cdn.jsdelivr.net/npm/three@0.186.1/build/three.module.js", "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.186.1/examples/jsm/"}}</script>
```

- [ ] **Step 3:** `python -m pytest tests/test_dev_server.py -q` passes. Commit `Add three to the harness import map`.

### Task 1: Phase markers in the builders

**Files:**
- Modify: `qec/code412.py`
- Modify: `qec/lessons.py`
- Test: `tests/test_code412.py` (append)

- [ ] **Step 1: Failing test** append to `tests/test_code412.py`:

```python
def test_phases_are_recorded_without_changing_gates():
    from qec import code412 as c
    qc = c.new_circuit(2)
    c.mark(qc, "prepare A")
    qc.h(c.data(0, 0))
    c.encode(qc, 0)
    c.encode(qc, 1)
    c.transversal_cx(qc, 0, 1)
    c.finish(qc, 2, ["Z", "X"])
    names = [n for _, n in qc.metadata["phases"]]
    assert names == ["prepare A", "encode A", "encode B", "transversal CNOT A→B",
                     "syndrome extraction A", "syndrome extraction B", "readout rotation B (X)", "measure"]
    starts = [s for s, _ in qc.metadata["phases"]]
    assert starts == sorted(starts) and starts[0] == 0
    # gate count unchanged by marking: 1 + 5 + 5 + 4 + 10 + 10 + 4 + 12
    assert len(qc.data) == 51
```

- [ ] **Step 2:** run, expect AttributeError on `c.mark`.

- [ ] **Step 3: Implement** in `qec/code412.py`, after `n_qubits`:

```python
BLOCK_NAMES = "ABC"


def mark(qc, name):
    """Record the start of a named phase for the spacetime view. Gates are unchanged."""
    qc.metadata.setdefault("phases", []).append((len(qc.data), name))
```

and insert marks at the top of these functions:

```python
def encode(qc, b):
    """Encode the state on local q0 into the block. Not fault tolerant."""
    mark(qc, f"encode {BLOCK_NAMES[b]}")
    ...

def transversal_cx(qc, ctrl_b, targ_b):
    mark(qc, f"transversal CNOT {BLOCK_NAMES[ctrl_b]}→{BLOCK_NAMES[targ_b]}")
    ...

def inject(qc, b, pauli, i):
    mark(qc, f"inject {pauli} on q{i}")
    ...

def extract_syndromes(qc, nblocks, b):
    ax, az = anc_x(nblocks, b), anc_z(nblocks, b)
    q = [data(b, i) for i in range(4)]
    mark(qc, f"syndrome extraction {BLOCK_NAMES[b]}")
    ...

def rotate_block(qc, b, basis):
    if basis != "Z":
        mark(qc, f"readout rotation {BLOCK_NAMES[b]} ({basis})")
    q = [data(b, i) for i in range(4)]
    _rotate(qc, q, basis)

def finish(qc, nblocks, bases):
    """Syndrome extraction on every block, readout rotations, measure everything."""
    for b in range(nblocks):
        extract_syndromes(qc, nblocks, b)
    for b, basis in enumerate(bases):
        rotate_block(qc, b, basis)
    mark(qc, "measure")
    qc.measure(range(qc.num_qubits), range(qc.num_clbits))
```

In `qec/lessons.py` add a mark before each preparation on encoded circuits (bare circuits untouched):

```python
# _lesson1, inside `if prep:` for the encoded circuit:
            c.mark(qc, "prepare A")
            qc.h(c.data(0, 0))
# _lesson2 encoded:
        c.mark(qc, "prepare A")
        qc.h(c.data(0, 0))
# _lesson3 encoded:
        c.mark(qc, "prepare A")
        qc.ry(THETA, c.data(0, 0))
        c.encode(qc, 0)                 # A = psi
        c.mark(qc, "prepare B")
        qc.h(c.data(1, 0))
# _lesson4 encoded:
        c.mark(qc, "prepare A")
        qc.ry(THETA, c.data(0, 0))
        c.encode(qc, 0)
        c.mark(qc, "prepare B (magic state)")
        qc.h(c.data(1, 0))
        qc.t(c.data(1, 0))
```

- [ ] **Step 4:** `python -m pytest -q` all green (existing 48 + 1; the recorded take stays valid because gates are unchanged).
- [ ] **Step 5:** Commit `Record circuit phases for the spacetime view`.

### Task 2: Layout builder `qec/spacetime.py` and bridge entry

**Files:**
- Create: `qec/spacetime.py`
- Modify: `qec/bridge.py`
- Test: `tests/test_spacetime.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_spacetime.py
import json

from qec.lessons import build_all
from qec.spacetime import layout


def enc_layouts(inject=None):
    circuits, meta = build_all({"shots": 100, "inject": inject})
    return [(m, layout(qc, m)) for qc, m in zip(circuits, meta) if m["kind"] == "enc"]


def test_layers_are_disjoint_per_qubit_and_phases_tile_time():
    for m, L in enc_layouts({"pauli": "X", "qubit": 1}):
        T = L["T"]
        seen = set()
        for op in L["ops"]:
            for q in op["q"]:
                assert (op["t"], q) not in seen, (m["name"], op)
                seen.add((op["t"], q))
            assert 0 <= op["t"] <= T
        spans = L["phases"]
        assert spans[0]["t0"] == 0 and spans[-1]["t1"] == T and spans[-1]["name"] == "measure"
        for a, b in zip(spans, spans[1:]):
            assert b["t0"] == a["t1"] + 1, (m["name"], a, b)


def test_qubit_positions_and_blocks():
    m, L = [x for x in enc_layouts() if x[0]["name"] == "enc_Z" and x[0]["lesson"] == 3][0]
    assert len(L["qubits"]) == 18 and len(L["blocks"]) == 3
    b1 = L["blocks"][1]
    assert b1["name"] == "B" and b1["data"] == [4, 5, 6, 7] and b1["ax"] == 14 and b1["az"] == 15
    q5 = L["qubits"][5]
    assert q5["kind"] == "data" and q5["block"] == 1 and (q5["x"], q5["y"]) == (2.2 + 1, 0)
    assert L["qubits"][14]["kind"] == "ax" and (L["qubits"][14]["x"], L["qubits"][14]["y"]) == (2.2 + 0.5, 0.5)
    assert L["qubits"][15]["kind"] == "az" and L["qubits"][15]["y"] == 1.7


def test_transversal_cnot_is_four_parallel_bars():
    m, L = [x for x in enc_layouts() if x[0]["name"] == "enc_Z" and x[0]["lesson"] == 3][0]
    tv = [p for p in L["phases"] if p["name"].startswith("transversal")]
    assert [p["name"] for p in tv] == ["transversal CNOT B→C", "transversal CNOT A→B"]
    for p in tv:
        ops = [o for o in L["ops"] if o["phase"] == p["name"]]
        assert len(ops) == 4 and all(o["gate"] == "cx" and o["role"] == "transversal" for o in ops)
        assert len({o["t"] for o in ops}) == 1
    bc = [o for o in L["ops"] if o["phase"] == "transversal CNOT B→C"]
    assert sorted(o["q"] for o in bc) == [[4, 8], [5, 9], [6, 10], [7, 11]]


def test_injection_marker_sits_between_encode_and_extraction():
    m, L = [x for x in enc_layouts({"pauli": "X", "qubit": 1}) if x[0]["lesson"] == 1 and x[0]["name"] == "enc_Z"][0]
    inj = L["inject"]
    assert inj["q"] == 1 and inj["pauli"] == "X"
    enc = next(p for p in L["phases"] if p["name"] == "encode A")
    ext = next(p for p in L["phases"] if p["name"].startswith("syndrome"))
    assert enc["t1"] < inj["t"] < ext["t0"]
    assert inj["caught"] == ["z_top", "zzzz_anc"]


def test_no_injection_means_null():
    m, L = enc_layouts()[0]
    assert L["inject"] is None


def test_sheets_propagate_through_transversal_cnot():
    m, L = [x for x in enc_layouts() if x[0]["lesson"] == 2 and x[0]["name"] == "enc_ZZ"][0]
    t = next(o["t"] for o in L["ops"] if o["role"] == "transversal")
    base = [s for s in L["sheets"] if s["from"] is None]
    assert sorted((s["type"], s["block"]) for s in base) == [("X", 0), ("X", 1), ("Z", 0), ("Z", 1)]
    assert all(s["t0"] == 0 and s["t1"] == L["T"] for s in base)
    prop = [s for s in L["sheets"] if s["from"] is not None]
    assert sorted((s["type"], s["block"], s["from"], s["t0"]) for s in prop) == [("X", 1, 0, t), ("Z", 0, 1, t)]


def test_bridge_spacetime_round_trip():
    from qec import bridge
    bridge.build(json.dumps({"shots": 100, "inject": None}))
    L = json.loads(bridge.spacetime(4))
    assert L["T"] > 0 and L["ops"] and L["phases"] and L["blocks"]
    assert bridge.spacetime(4) == bridge.spacetime(4)   # cached
```

- [ ] **Step 2:** run, expect ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# qec/spacetime.py
"""Layout of an encoded circuit as a spacetime block for ui/spacetime.js.

Plane coordinates: block b sits at x = b * PITCH; local q0 (0,0), q1 (1,0), q2 (0,1), q3 (1,1);
the XXXX ancilla at the face centre (0.5, 0.5), the ZZZZ ancilla below the patch (0.5, 1.7).
Time: integer layers. Gates inside a phase are packed as early as possible; a new phase
starts above every layer used so far, so phases never overlap.
"""
from qec import code412 as c

PITCH = 2.2
LOCAL = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)}
CAUGHT = {  # which checks a single Pauli on local data qubit i flips
    "X": lambda i: [("z_top" if i in (0, 1) else "z_bottom"), "zzzz_anc"],
    "Z": lambda i: ["x_face"],
    "Y": lambda i: ["x_face", ("z_top" if i in (0, 1) else "z_bottom"), "zzzz_anc"],
}


def _role(phase):
    for key in ("transversal", "syndrome", "encode", "inject", "prepare", "readout", "measure"):
        if phase.startswith(key):
            return key
    return "other"


def _qubits(n):
    out = []
    for b in range(n):
        for i in range(4):
            x, y = LOCAL[i]
            out.append({"id": c.data(b, i), "kind": "data", "block": b, "local": i, "x": b * PITCH + x, "y": y})
    for b in range(n):
        out.append({"id": c.anc_x(n, b), "kind": "ax", "block": b, "local": None, "x": b * PITCH + 0.5, "y": 0.5})
        out.append({"id": c.anc_z(n, b), "kind": "az", "block": b, "local": None, "x": b * PITCH + 0.5, "y": 1.7})
    out.sort(key=lambda q: q["id"])
    return out


def layout(qc, meta):
    n = meta["nblocks"]
    marks = sorted(qc.metadata.get("phases", []))
    ops, phases = [], []
    last = {}            # qubit -> last layer used
    floor = 0            # first layer the current phase may use
    current = None
    mi = 0
    for k, ins in enumerate(qc.data):
        while mi < len(marks) and marks[mi][0] <= k:
            name = marks[mi][1]
            mi += 1
            if name != current:
                if current is not None:
                    phases[-1]["t1"] = max(last.values(), default=-1)
                floor = max(last.values(), default=-1) + 1
                current = name
                phases.append({"name": name, "t0": floor, "t1": floor})
        if current is None:
            current = "prepare"
            phases.append({"name": current, "t0": 0, "t1": 0})
        name = ins.operation.name
        if name == "barrier":
            continue
        qs = [qc.find_bit(q).index for q in ins.qubits]
        t = max([floor] + [last[q] + 1 for q in qs if q in last])
        for q in qs:
            last[q] = t
        ops.append({"t": t, "gate": name, "q": qs, "phase": current, "role": _role(current)})
    if phases:
        phases[-1]["t1"] = max(last.values(), default=0)
    T = phases[-1]["t1"] if phases else 0

    blocks = [{"name": c.BLOCK_NAMES[b], "data": [c.data(b, i) for i in range(4)],
               "ax": c.anc_x(n, b), "az": c.anc_z(n, b), "x0": b * PITCH, "y0": 0.0} for b in range(n)]

    inject = None
    for op in ops:
        if op["role"] == "inject":
            qid = op["q"][0]
            b, i = divmod(qid, 4)
            inject = {"q": qid, "t": op["t"], "pauli": op["gate"].upper(), "block": b, "caught": CAUGHT[op["gate"].upper()](i)}

    sheets = [{"type": typ, "block": b, "t0": 0, "t1": T, "from": None} for b in range(n) for typ in ("Z", "X")]
    for p in phases:
        if p["name"].startswith("transversal"):
            ctrl, targ = p["name"].split()[-1].split("→")
            cb, tb = c.BLOCK_NAMES.index(ctrl), c.BLOCK_NAMES.index(targ)
            sheets.append({"type": "X", "block": tb, "t0": p["t0"], "t1": T, "from": cb})
            sheets.append({"type": "Z", "block": cb, "t0": p["t0"], "t1": T, "from": tb})

    return {
        "lesson": meta["lesson"], "name": meta["name"], "bases": meta["bases"], "T": T,
        "pitch": PITCH, "qubits": _qubits(n), "blocks": blocks, "phases": phases, "ops": ops,
        "inject": inject, "sheets": sheets,
    }
```

Bridge (`qec/bridge.py`): add `"layouts": {}` to `_LAST`, clear it in `build`, and add:

```python
from qec.spacetime import layout

def spacetime(index):
    index = int(index)
    if index not in _LAST["layouts"]:
        _LAST["layouts"][index] = json.dumps(layout(_LAST["circuits"][index], _LAST["meta"][index]))
    return _LAST["layouts"][index]
```

Harness: in `dev/serve.py` `api()`, add `"spacetime"` to the tuple of plain JSON entry points. Shim: `export const spacetime = fn('spacetime');`.

- [ ] **Step 4:** `python -m pytest -q` green. Commit `Add spacetime layout builder and bridge entry`.

### Task 3: `ui/spacetime.js`

**Files:**
- Create: `ui/spacetime.js`

- [ ] **Step 1: Implement**

```js
// ui/spacetime.js
// The encoded circuit as a spacetime block: patches in the plane, time up, gates as bars,
// the measured syndrome on the top slice, logical operators as worldsheets. three.js is
// resolved by name on Qollab and by the dev harness import map.
const DT = 0.22;                 // world units per time layer
const C = {
  data: 0xe0a040, anc: 0x4f9cff, x: 0xe0a040, z: 0x4f9cff, bad: 0xff5c5c, focus: 0xffd166,
  encode: 0x9aa0ad, transversal: 0xf4f4f8, syndrome: 0x4f9cff, h: 0xdfe3ea, t: 0xb07cff, sdg: 0xb07cff, ry: 0x39c37a, measure: 0x6b7280,
};
const GATE_TEXT = { h: 'H', t: 'T', sdg: 'S†', ry: 'Ry', x: 'X', y: 'Y', z: 'Z', s: 'S' };

export async function renderSpacetime(root, L, row, opts = {}) {
  root.replaceChildren();
  const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let THREE, OrbitControls;
  try {
    THREE = await import('three');
    ({ OrbitControls } = await import('three/addons/controls/OrbitControls.js'));
  } catch (e) {
    root.appendChild(phaseList(L, null));
    const p = document.createElement('p');
    p.className = 'note';
    p.textContent = '3D view unavailable here (three.js did not load): ' + (e && e.message ? e.message : e);
    root.appendChild(p);
    return { dispose() {} };
  }

  // ----- DOM -----
  const wrap = document.createElement('div');
  wrap.className = 'st-wrap';
  const canvasBox = document.createElement('div');
  canvasBox.className = 'st-canvas';
  const bar = document.createElement('div');
  bar.className = 'st-bar';
  const play = document.createElement('button');
  play.type = 'button';
  play.textContent = 'Play';
  play.setAttribute('aria-label', 'Play the circuit through time');
  const slider = document.createElement('input');
  slider.type = 'range'; slider.min = '0'; slider.max = String(L.T); slider.step = '0.05'; slider.value = '0';
  slider.setAttribute('aria-label', 'Time step');
  const tLabel = document.createElement('span');
  tLabel.className = 'st-t';
  const sheetsToggle = document.createElement('label');
  sheetsToggle.className = 'st-check';
  const sheetsBox = document.createElement('input');
  sheetsBox.type = 'checkbox'; sheetsBox.checked = true;
  sheetsToggle.append(sheetsBox, ' logical worldsheets');
  const reset = document.createElement('button');
  reset.type = 'button'; reset.textContent = 'Reset view';
  bar.append(play, slider, tLabel, sheetsToggle, reset);
  const phases = phaseList(L, (t) => { setT(t); });
  wrap.append(canvasBox, bar, phases);
  root.appendChild(wrap);

  // ----- scene -----
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.domElement.setAttribute('role', 'img');
  renderer.domElement.setAttribute('aria-label', `Spacetime view of ${L.name}: ` + L.phases.map((p) => p.name).join(', '));
  canvasBox.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const H = L.T * DT;
  const xs = L.qubits.map((q) => q.x), ys = L.qubits.map((q) => q.y);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cz = (Math.min(...ys) + Math.max(...ys)) / 2;
  const span = Math.max(Math.max(...xs) - Math.min(...xs) + 1.5, H, 3);
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  const home = new THREE.Vector3(cx + span * 0.9, H * 0.75 + span * 0.5, cz + span * 1.1);
  camera.position.copy(home);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(cx, H / 2, cz);
  controls.enableDamping = true;
  controls.update();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(cx + 4, H + 6, cz + 5);
  scene.add(sun);

  const timed = [];   // objects with userData.t (visible when t <= now)
  const grown = [];   // objects with userData.t0/t1 (scaled along y)
  const topSlice = [];
  const sheetObjs = [];
  const mat = (color, o = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.05, ...o });
  const add = (obj, t) => { obj.userData.t = t; timed.push(obj); scene.add(obj); return obj; };
  const pos = (q) => L.qubits[q];

  // worldlines
  for (const q of L.qubits) {
    const g = new THREE.CylinderGeometry(0.03, 0.03, H, 10);
    g.translate(0, H / 2, 0);
    const m = new THREE.Mesh(g, mat(q.kind === 'data' ? C.data : C.anc, { transparent: true, opacity: q.kind === 'data' ? 0.95 : 0.7 }));
    m.position.set(q.x, 0, q.y);
    m.userData.t0 = 0; m.userData.t1 = L.T;
    grown.push(m); scene.add(m);
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), mat(q.kind === 'data' ? C.data : C.anc));
    dot.position.set(q.x, 0, q.y);
    scene.add(dot);
  }

  // patch tiles: ground (static) and top slice (data driven)
  const tiles = (y, bi, rates, basis, outline) => {
    const b = L.blocks[bi];
    const grp = new THREE.Group();
    const face = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat(C.x, { transparent: true, opacity: rates ? 0.15 + 0.85 * rates.x_face : 0.22, side: THREE.DoubleSide }));
    face.rotation.x = -Math.PI / 2; face.position.set(b.x0 + 0.5, y, b.y0 + 0.5);
    grp.add(face);
    const zRead = basis === 'Z';
    const cap = (thetaStart, zc, rate) => {
      const geo = new THREE.CircleGeometry(0.5, 32, thetaStart, Math.PI);
      const m = zRead || !rates
        ? new THREE.Mesh(geo, mat(C.z, { transparent: true, opacity: rates ? 0.15 + 0.85 * rate : 0.22, side: THREE.DoubleSide }))
        : new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineDashedMaterial({ color: 0x8a90a0, dashSize: 0.06, gapSize: 0.04 }));
      if (m.computeLineDistances) m.computeLineDistances();
      m.rotation.x = -Math.PI / 2; m.position.set(b.x0 + 0.5, y, zc);
      grp.add(m);
      return m;
    };
    const top = cap(0, b.y0, rates ? rates.z_top : 0);           // half disc on the far side of the top edge
    const bot = cap(Math.PI, b.y0 + 1, rates ? rates.z_bottom : 0);
    if (outline) {
      for (const [key, obj] of [['x_face', face], ['z_top', top], ['z_bottom', bot]]) {
        if (outline.includes(key)) {
          const e = new THREE.LineSegments(new THREE.EdgesGeometry(obj.geometry), new THREE.LineBasicMaterial({ color: C.bad, linewidth: 2 }));
          e.rotation.copy(obj.rotation); e.position.copy(obj.position); e.position.y += 0.005;
          grp.add(e);
        }
      }
    }
    // logical edges: Z_L left (q0-q2), X_L top (q0-q1)
    const edge = (len, axis, px, pz, color) => {
      const g = new THREE.BoxGeometry(axis === 'x' ? len : 0.05, 0.03, axis === 'z' ? len : 0.05);
      const m = new THREE.Mesh(g, mat(color, { emissive: color, emissiveIntensity: 0.25 }));
      m.position.set(px, y + 0.02, pz);
      grp.add(m);
    };
    edge(1, 'z', b.x0, b.y0 + 0.5, C.z);
    edge(1, 'x', b.x0 + 0.5, b.y0, C.x);
    if (rates) {
      const s = new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 12), mat(C.z, { transparent: true, opacity: 0.25 + 0.75 * rates.zzzz_anc, emissive: C.z, emissiveIntensity: rates.zzzz_anc }));
      s.position.set(b.x0 + 0.5, y, 1.7);
      grp.add(s);
      if (outline && outline.includes('zzzz_anc')) {
        const ring = new THREE.Mesh(new THREE.TorusGeometry(0.18, 0.015, 8, 32), mat(C.bad));
        ring.rotation.x = Math.PI / 2; ring.position.copy(s.position);
        grp.add(ring);
      }
      grp.add(sprite(THREE, `read in ${basis}`, 0.5, b.x0 + 0.5, y + 0.25, b.y0 + 1.35, '#e8eaf0'));
    }
    return grp;
  };
  L.blocks.forEach((b, bi) => scene.add(tiles(0, bi, null, L.bases[bi], null)));
  L.blocks.forEach((b, bi) => {
    const rates = (row && row.blocks && row.blocks[bi]) || { x_face: 0, z_top: 0, z_bottom: 0, zzzz_anc: 0 };
    const outline = L.inject && L.inject.block === bi ? L.inject.caught : null;
    const g = tiles(H, bi, rates, L.bases[bi], outline);
    g.userData.t = L.T; timed.push(g); topSlice.push(g); scene.add(g);
  });

  // gates
  for (const op of L.ops) {
    const y = op.t * DT;
    if (op.gate === 'measure') {
      const q = pos(op.q[0]);
      const m = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 0.04, 16), mat(C.measure));
      m.position.set(q.x, y, q.y);
      add(m, op.t);
    } else if (op.gate === 'cx') {
      const a = pos(op.q[0]), b = pos(op.q[1]);
      const color = op.role === 'transversal' ? C.transversal : op.role === 'syndrome' ? C.syndrome : C.encode;
      const from = new THREE.Vector3(a.x, y, a.y), to = new THREE.Vector3(b.x, y, b.y);
      const d = to.clone().sub(from), len = d.length();
      const bar = new THREE.Mesh(new THREE.CylinderGeometry(op.role === 'transversal' ? 0.035 : 0.022, op.role === 'transversal' ? 0.035 : 0.022, len, 8), mat(color, { emissive: color, emissiveIntensity: op.role === 'transversal' ? 0.35 : 0.1 }));
      bar.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
      bar.position.copy(from).add(to).multiplyScalar(0.5);
      add(bar, op.t);
      const ctrl = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 10), mat(color));
      ctrl.position.copy(from);
      add(ctrl, op.t);
      const targ = new THREE.Mesh(new THREE.TorusGeometry(0.085, 0.018, 8, 24), mat(color));
      targ.position.copy(to); targ.rotation.x = Math.PI / 2;
      add(targ, op.t);
    } else {
      const q = pos(op.q[0]);
      const isInject = op.role === 'inject';
      const color = isInject ? C.bad : (C[op.gate] || C.encode);
      const m = isInject
        ? new THREE.Mesh(new THREE.SphereGeometry(0.11, 16, 12), mat(C.bad, { emissive: C.bad, emissiveIntensity: 0.6 }))
        : new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.12, 0.16), mat(color));
      m.position.set(q.x, y, q.y);
      add(m, op.t);
      add(sprite(THREE, GATE_TEXT[op.gate] || op.gate, 0.32, q.x, y + 0.16, q.y, isInject ? '#ff5c5c' : '#e8eaf0'), op.t);
    }
  }

  // logical worldsheets
  for (const s of L.sheets) {
    const b = L.blocks[s.block];
    const color = s.type === 'Z' ? C.z : C.x;
    const h = (s.t1 - s.t0) * DT;
    const g = new THREE.PlaneGeometry(1, h);
    g.translate(0, h / 2, 0);
    const m = new THREE.Mesh(g, mat(color, { transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false }));
    if (s.type === 'Z') { m.rotation.y = Math.PI / 2; m.position.set(b.x0, s.t0 * DT, b.y0 + 0.5); }
    else { m.position.set(b.x0 + 0.5, s.t0 * DT, b.y0); }
    m.userData.t0 = s.t0; m.userData.t1 = s.t1;
    grown.push(m); sheetObjs.push(m); scene.add(m);
    if (s.from !== null) {   // horizontal bridge from the partner block at the CNOT step
      const f = L.blocks[s.from];
      const x0 = Math.min(f.x0, b.x0), x1 = Math.max(f.x0, b.x0);
      let bg;
      if (s.type === 'Z') { bg = new THREE.PlaneGeometry(x1 - x0, 1); bg.rotateX(-Math.PI / 2); }
      else { bg = new THREE.BoxGeometry(x1 - x0 + 1, 0.02, 0.06); }
      const bm = new THREE.Mesh(bg, mat(color, { transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false }));
      bm.position.set((x0 + x1) / 2 + (s.type === 'X' ? 0.5 : 0), s.t0 * DT, s.type === 'Z' ? b.y0 + 0.5 : b.y0);
      bm.userData.t = s.t0; timed.push(bm); sheetObjs.push(bm); scene.add(bm);
    }
  }

  // the "now" plane
  const nowGeo = new THREE.PlaneGeometry(Math.max(...xs) - Math.min(...xs) + 1.6, Math.max(...ys) - Math.min(...ys) + 1.6);
  const now = new THREE.Mesh(nowGeo, mat(0xffffff, { transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false }));
  now.rotation.x = -Math.PI / 2; now.position.set(cx, 0, cz);
  scene.add(now);

  // ----- time -----
  let tNow = 0;
  function setT(t) {
    tNow = Math.max(0, Math.min(L.T, t));
    slider.value = String(tNow);
    for (const o of timed) o.visible = o.userData.t <= tNow + 1e-6;
    for (const o of grown) {
      const f = Math.max(0, Math.min(1, (tNow - o.userData.t0) / Math.max(1e-6, o.userData.t1 - o.userData.t0)));
      o.scale.y = Math.max(1e-4, f); o.visible = f > 0;
    }
    for (const o of sheetObjs) o.visible = o.visible && sheetsBox.checked;
    now.position.y = tNow * DT;
    const ph = L.phases.find((p) => p.t0 <= Math.round(tNow) && Math.round(tNow) <= p.t1) || L.phases[L.phases.length - 1];
    tLabel.textContent = `t = ${Math.round(tNow)} / ${L.T}: ${ph.name}`;
    phases.querySelectorAll('li').forEach((li) => li.classList.toggle('on', li.dataset.name === ph.name));
  }
  let playing = false, lastTs = 0, raf = 0, disposed = false;
  const SPEED = L.T / 12;   // whole circuit in 12 s
  function frame(ts) {
    if (disposed) return;
    if (playing) {
      if (lastTs) setT(tNow + (ts - lastTs) / 1000 * SPEED);
      if (tNow >= L.T) { playing = false; play.textContent = 'Replay'; }
    }
    lastTs = ts;
    controls.update();
    renderer.render(scene, camera);
    raf = requestAnimationFrame(frame);
  }
  play.addEventListener('click', () => {
    if (tNow >= L.T) setT(0);
    playing = !playing;
    play.textContent = playing ? 'Pause' : 'Play';
  });
  slider.addEventListener('input', () => { playing = false; play.textContent = 'Play'; setT(parseFloat(slider.value)); });
  sheetsBox.addEventListener('change', () => setT(tNow));
  reset.addEventListener('click', () => { camera.position.copy(home); controls.target.set(cx, H / 2, cz); controls.update(); });

  function resize() {
    const w = Math.max(280, canvasBox.clientWidth);
    const h = Math.max(300, Math.min(520, Math.round(w * 0.56)));
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = '100%'; renderer.domElement.style.height = h + 'px';
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(canvasBox);
  resize();
  setT(reduce || opts.start === 'end' ? L.T : 0);
  if (!reduce && opts.autoplay !== false) { playing = true; play.textContent = 'Pause'; }
  raf = requestAnimationFrame(frame);

  return {
    setT, play: () => { if (!playing) play.click(); },
    dispose() {
      disposed = true; cancelAnimationFrame(raf); ro.disconnect(); controls.dispose();
      scene.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) { const ms = Array.isArray(o.material) ? o.material : [o.material]; ms.forEach((m) => { if (m.map) m.map.dispose(); m.dispose(); }); } });
      renderer.dispose();
    },
  };
}

function sprite(THREE, text, size, x, y, z, color) {
  const c = document.createElement('canvas');
  c.width = 128; c.height = 64;
  const ctx = c.getContext('2d');
  ctx.font = 'bold 40px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.lineWidth = 6; ctx.strokeStyle = 'rgba(15,17,21,0.9)'; ctx.strokeText(text, 64, 32);
  ctx.fillStyle = color; ctx.fillText(text, 64, 32);
  const tex = new THREE.CanvasTexture(c);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  s.scale.set(size, size / 2, 1);
  s.position.set(x, y, z);
  return s;
}

function phaseList(L, onJump) {
  const ol = document.createElement('ol');
  ol.className = 'st-phases';
  ol.setAttribute('aria-label', 'Circuit phases in time order');
  for (const p of L.phases) {
    const li = document.createElement('li');
    li.dataset.name = p.name;
    if (onJump) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = `${p.name} (t ${p.t0}–${p.t1})`;
      b.addEventListener('click', () => onJump(p.t1));
      li.appendChild(b);
    } else {
      li.textContent = `${p.name} (t ${p.t0}–${p.t1})`;
    }
    ol.appendChild(li);
  }
  return ol;
}
```

- [ ] **Step 2:** `node --check ui/spacetime.js`.
- [ ] **Step 3:** Commit `Add the 3D spacetime view module`.

### Task 4: Wire into the page

**Files:**
- Modify: `index.html` (panel: add `#spacetime` after `.grid`; rename circuit summary)
- Modify: `main.css` (styles)
- Modify: `main.js` (mount/dispose in showLesson)

- [ ] **Step 1: index.html** after the closing `</div>` of `.grid` (line 42) insert:

```html
    <section class="spacetime" id="spacetime" aria-label="Circuit in spacetime"></section>
```

and change the summary text at line 46 to `Raw Qiskit circuit (line drawing)`.

- [ ] **Step 2: main.css** append:

```css
.spacetime { margin-top: 16px; }
.st-canvas { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; background: radial-gradient(ellipse at 50% 20%, rgba(79,156,255,0.08), transparent 70%); touch-action: none; }
.st-canvas canvas { display: block; }
.st-bar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 8px; }
.st-bar input[type=range] { flex: 1 1 200px; accent-color: #1f6fe0; }
.st-t { font-size: 0.9rem; color: var(--muted); min-width: 14ch; }
.st-check { display: inline-flex; gap: 6px; align-items: center; color: var(--muted); font-size: 0.9rem; }
.st-phases { display: flex; flex-wrap: wrap; gap: 6px; list-style: none; padding: 0; margin: 8px 0 0; font-size: 0.85rem; }
.st-phases li button { padding: 4px 8px; font-size: 0.85rem; color: var(--muted); }
.st-phases li.on button { color: var(--fg); border-color: var(--fg); }
```

- [ ] **Step 3: main.js**
  - import: `import { renderSpacetime } from './ui/spacetime.js';`
  - state: add `spacetime: null, spacetimeToken: 0` to the `state` literal.
  - in `showLesson(n)`, after `renderCharts(...)`, add:

```js
  mountSpacetime(n, rows);
```

  - new function:

```js
async function mountSpacetime(lesson, rows) {
  const token = ++state.spacetimeToken;
  if (state.spacetime) { state.spacetime.dispose(); state.spacetime = null; }
  const root = $('spacetime');
  const row = rows.find((r) => r.kind === 'enc');
  if (!row) { root.replaceChildren(); return; }
  root.textContent = 'Building the spacetime view...';
  try {
    const L = JSON.parse(await bridge.spacetime.callPromising(row.index));
    if (token !== state.spacetimeToken) return;   // the tab changed meanwhile
    state.spacetime = await renderSpacetime(root, L, row);
    if (token !== state.spacetimeToken) { state.spacetime.dispose(); state.spacetime = null; }
  } catch (e) {
    if (token === state.spacetimeToken) root.textContent = 'Spacetime view unavailable: ' + errText(e);
  }
}
```

  - `loadRecorded` already rebuilds via `bridge.build`, so `spacetime(index)` has circuits. Nothing else changes.
- [ ] **Step 4:** `node --check main.js ui/spacetime.js`; `python -m pytest -q` still green.
- [ ] **Step 5:** Commit `Mount the spacetime view in every lesson`.

### Task 5: Browser smoke test on the harness

- [ ] Start `python dev/serve.py 8765`; open in the in-app browser; Load recorded take; confirm per lesson: canvas present, Play runs ~12 s, slider scrubs, phase buttons jump, worldsheets toggle, Reset view works, no console errors. Resize to 400 px wide: canvas shrinks, bar wraps.
- [ ] Screenshot each lesson at the end of the animation and mid-way through a transversal CNOT; judge: are bars readable, do sheets clutter, is the top slice legible. List fixes before making them if they change geometry decisions; cosmetic constant tweaks go straight in.
- [ ] Playwright check (`dev/check_spacetime.py`, scratch, not committed): open page, click Load recorded take, wait for `#spacetime canvas`, for each tab click and wait, assert no `pageerror`, save screenshots.

### Task 6: Docs

- [ ] README: in the lesson walkthrough mention the 3D spacetime view and its controls; credit Google Quantum AI, Nature 614, 676 (2023) for the figure convention and three.js (MIT) in Credits; the files list gains `ui/spacetime.js` and `qec/spacetime.py`.
- [ ] Spec addendum: remove the P(1) edge-label sentence (see Decisions).
- [ ] Commit `Document the spacetime view`.

### Task 7: Recorder and storyboard update (after the dev harness plan's Task 4 exists)

- [ ] In `dev/record_demo.py` STEPS: drop the `circuit` scene; for each lesson tab, after clicking the tab wait for `#spacetime canvas`, then hold 14 s while it plays (the view autoplays on mount), with the caption for the lesson; add a slow orbit by dragging the canvas 120 px horizontally over 3 s with `page.mouse` during lesson 3; move the "bare qubits" caption to lesson 3's cards; injection caption "fire in nearly every shot".
- [ ] Re-record, review mp4.

## Review

One fresh reviewer per task (diff + change summary), ten-minute cap, verifies against the Facts table. The final reviewer also opens the harness and checks the four lessons visually.
