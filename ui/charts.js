const NS = 'http://www.w3.org/2000/svg';

function fmt(x) { return x === null || x === undefined ? 'n/a' : `${(100 * x).toFixed(1)}%`; }

function card(big, lbl, title) {
  const d = document.createElement('div');
  d.className = 'card';
  if (title) d.title = title;
  d.innerHTML = `<div class="big">${big}</div><div class="lbl">${lbl}</div>`;
  return d;
}

export function renderCards(root, rows) {
  root.replaceChildren();
  const encs = rows.filter((r) => r.kind === 'enc');
  const bares = rows.filter((r) => r.kind === 'bare');
  const mean = (xs) => { const v = xs.filter((x) => x !== null && x !== undefined); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
  root.appendChild(card(fmt(mean(encs.map((r) => r.flagged_fraction))), 'shots flagged by a check and discarded'));
  root.appendChild(card(fmt(mean(encs.map((r) => r.deviation_accepted))), 'logical error of accepted shots', 'absolute gap between measured and ideal P(1), averaged over the readouts shown'));
  root.appendChild(card(fmt(mean(bares.map((r) => r.deviation_all))), 'same experiment on bare physical qubits'));
  if (encs[0] && encs[0].kept_fraction !== null && encs[0].name.startsWith('enc') && rows[0].lesson === 4) {
    root.appendChild(card(fmt(mean(encs.map((r) => r.kept_fraction))), 'kept after post-selecting A = 0 (ideal 50%)'));
  }
}

function bars(title, items) {
  // items: [{label, value, ideal?}] values in [0,1]
  const wrap = document.createElement('div');
  wrap.className = 'chart';
  const h = document.createElement('h3');
  h.textContent = title;
  wrap.appendChild(h);
  const W = 320, rowH = 26, pad = 110, H = items.length * rowH + 24;
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('role', 'img');
  const t = document.createElementNS(NS, 'title');
  t.textContent = title + ': ' + items.map((i) => `${i.label} ${fmt(i.value)}`).join(', ');
  svg.appendChild(t);
  items.forEach((it, i) => {
    const y = 8 + i * rowH;
    const lab = document.createElementNS(NS, 'text');
    lab.setAttribute('x', pad - 6); lab.setAttribute('y', y + 14); lab.setAttribute('text-anchor', 'end');
    lab.setAttribute('font-size', 12); lab.setAttribute('fill', 'var(--fg)');
    lab.textContent = it.label;
    svg.appendChild(lab);
    const v = it.value === null || it.value === undefined ? 0 : it.value;
    const r = document.createElementNS(NS, 'rect');
    r.setAttribute('x', pad); r.setAttribute('y', y); r.setAttribute('height', 18);
    r.setAttribute('width', Math.max(1, (W - pad - 50) * v));
    r.setAttribute('fill', it.color || 'var(--z)');
    svg.appendChild(r);
    const val = document.createElementNS(NS, 'text');
    val.setAttribute('x', pad + (W - pad - 50) * v + 4); val.setAttribute('y', y + 14);
    val.setAttribute('font-size', 12); val.setAttribute('fill', 'var(--fg)');
    val.textContent = fmt(it.value);
    svg.appendChild(val);
    if (it.ideal !== undefined) {
      const x = pad + (W - pad - 50) * it.ideal;
      const m = document.createElementNS(NS, 'line');
      m.setAttribute('x1', x); m.setAttribute('x2', x); m.setAttribute('y1', y - 2); m.setAttribute('y2', y + 20);
      m.setAttribute('stroke', 'var(--good)'); m.setAttribute('stroke-width', 2);
      const mt = document.createElementNS(NS, 'title');
      mt.textContent = `ideal ${fmt(it.ideal)}`;
      m.appendChild(mt);
      svg.appendChild(m);
    }
  });
  wrap.appendChild(svg);
  return wrap;
}

export function renderCharts(root, rows) {
  root.replaceChildren();
  const labels = [...new Set(rows.map((r) => r.label))];
  labels.forEach((label) => {
    const enc = rows.find((r) => r.kind === 'enc' && r.label === label);
    const bare = rows.find((r) => r.kind === 'bare' && r.label === label);
    const items = [
      { label: 'accepted (encoded)', value: enc.p1_accepted, ideal: enc.ideal_p1, color: 'var(--z)' },
      { label: 'all shots (encoded)', value: enc.p1_all, ideal: enc.ideal_p1, color: 'var(--muted)' },
      { label: 'bare physical', value: bare ? bare.p1_all : null, ideal: enc.ideal_p1, color: 'var(--x)' },
    ];
    root.appendChild(bars(`P(logical = 1), readout ${label}. Green mark = ideal ${fmt(enc.ideal_p1)}`, items));
  });
}
