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
    return { setT() {}, play() {}, dispose() {} };
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
  // Look along the planned diagonal, from far enough that the whole block (top slice included) fits the view.
  const radius = 0.5 * Math.hypot(Math.max(...xs) - Math.min(...xs) + 1.5, H + 0.6, Math.max(...ys) - Math.min(...ys) + 1.5);
  const home = new THREE.Vector3(span * 0.9, H * 0.25 + span * 0.5, span * 1.1).normalize()
    .multiplyScalar(0.8 * radius / Math.sin(THREE.MathUtils.degToRad(38 / 2))).add(new THREE.Vector3(cx, H / 2, cz));
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
      m.userData.src = geo;   // the outline below needs the disc, not its dashed edges
      m.rotation.x = -Math.PI / 2; m.position.set(b.x0 + 0.5, y, zc);
      grp.add(m);
      return m;
    };
    const top = cap(0, b.y0, rates ? rates.z_top : 0);           // half disc on the far side of the top edge
    const bot = cap(Math.PI, b.y0 + 1, rates ? rates.z_bottom : 0);
    if (outline) {
      for (const [key, obj] of [['x_face', face], ['z_top', top], ['z_bottom', bot]]) {
        if (outline.includes(key)) {
          const e = new THREE.LineSegments(new THREE.EdgesGeometry(obj.userData.src || obj.geometry), new THREE.LineBasicMaterial({ color: C.bad, linewidth: 2 }));
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
      renderer.forceContextLoss();   // free the WebGL context now; browsers cap live contexts
    },
  };
}

function sprite(THREE, text, size, x, y, z, color) {
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d');
  const font = 'bold 40px system-ui, sans-serif';
  ctx.font = font;
  c.width = Math.max(128, Math.ceil(ctx.measureText(text).width) + 16); c.height = 64;   // resizing resets the context
  ctx.font = font; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.lineWidth = 6; ctx.strokeStyle = 'rgba(15,17,21,0.9)'; ctx.strokeText(text, c.width / 2, 32);
  ctx.fillStyle = color; ctx.fillText(text, c.width / 2, 32);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  s.scale.set(size * c.width / 128, size / 2, 1);
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
