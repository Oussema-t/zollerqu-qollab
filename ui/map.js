// ui/map.js: Champaign County block groups, candidate sites and warehouses as a hand-drawn SVG.
// Equirectangular projection fitted to the block bounding box. Site i = bit i of a placement index.
const NS = 'http://www.w3.org/2000/svg';
const W = 640, PAD = 22, RMAX = 9;
export const SITE_COLORS = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)'];

function el(name, attrs, parent) {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

// Nearest open site per block by driving minutes; ties go to the lower site index, as in qopt/cost.py.
export function assign(data, chosen) {
  const open = [...chosen].sort((a, b) => a - b);
  return data.t_block_site.map((row) => {
    let best = open[0];
    for (const s of open) if (row[s] < row[best]) best = s;
    return best;
  });
}

// The warehouse with the smallest total minutes to the open sites (the one the cost model charges).
export function servingWarehouse(data, chosen) {
  let best = 0, bestSum = Infinity;
  data.t_wh_site.forEach((row, w) => {
    const sum = chosen.reduce((acc, s) => acc + row[s], 0);
    if (sum < bestSum) { bestSum = sum; best = w; }
  });
  return best;
}

export function renderMap(svgEl, data, { m, chosen = [], highlightWarehouse = null }) {
  const lat0 = data.blocks.reduce((a, b) => a + b.lat, 0) / data.blocks.length;
  const kx = Math.cos((lat0 * Math.PI) / 180);
  const xs = data.blocks.map((b) => b.lon * kx), ys = data.blocks.map((b) => b.lat);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const scale = (W - 2 * PAD) / (x1 - x0);
  const H = Math.round((y1 - y0) * scale + 2 * PAD);
  const px = (p) => PAD + (p.lon * kx - x0) * scale;
  const py = (p) => PAD + (y1 - p.lat) * scale;

  svgEl.replaceChildren();
  svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svgEl.setAttribute('role', 'img');
  svgEl.setAttribute('aria-label', chosen.length
    ? `Map of ${data.blocks.length} block groups with open sites ${chosen.join(', ')}, each block linked to its nearest open site.`
    : `Map of ${data.blocks.length} block groups and ${data.sites.length} candidate sites.`);

  const owner = chosen.length ? assign(data, chosen) : null;
  const colorOf = (s) => SITE_COLORS[chosen.indexOf(s) % SITE_COLORS.length];

  if (owner) {
    const g = el('g', { class: 'links' }, svgEl);
    data.blocks.forEach((b, i) => {
      const s = data.sites[owner[i]];
      el('line', { x1: px(b), y1: py(b), x2: px(s), y2: py(s), stroke: colorOf(owner[i]) }, g);
    });
  }

  const maxPop = Math.max(...data.blocks.map((b) => b.pop));
  const gb = el('g', { class: 'blocks' }, svgEl);
  data.blocks.forEach((b, i) => {
    const c = el('circle', { cx: px(b), cy: py(b), r: Math.max(1.2, RMAX * Math.sqrt(b.pop / maxPop)) }, gb);
    if (owner) c.style.fill = colorOf(owner[i]);   // style, so it wins over the stylesheet default
    el('title', {}, c).textContent = `Block group ${b.geoid}: ${b.pop.toLocaleString()} people`;
  });

  const gw = el('g', { class: 'warehouses' }, svgEl);
  data.warehouses.forEach((w, i) => {
    const x = px(w), y = py(w), r = 9;
    const t = el('polygon', { points: `${x},${y - r} ${x + r},${y + r * 0.75} ${x - r},${y + r * 0.75}`,
      class: i === highlightWarehouse ? 'wh on' : 'wh' }, gw);
    el('title', {}, t).textContent = `Warehouse ${i}` + (i === highlightWarehouse ? ' (serves this placement)' : '');
  });

  const gs = el('g', { class: 'sites' }, svgEl);
  data.sites.forEach((s, i) => {
    const x = px(s), y = py(s), h = 8;
    const on = chosen.includes(i);
    const g = el('g', { class: 'site' + (on ? ' on' : '') + (i >= m ? ' off' : '') }, gs);
    const r = el('rect', { x: x - h, y: y - h, width: 2 * h, height: 2 * h, rx: 2 }, g);
    if (on) r.style.fill = colorOf(i);
    el('text', { x, y: y + 4, 'text-anchor': 'middle' }, g).textContent = String(i);
    el('title', {}, g).textContent = `Site ${i} (bit ${i})` + (i >= m ? `: not a candidate at m = ${m}` : on ? ': open' : '');
  });
}
