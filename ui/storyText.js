// Per-lesson scripts for the story page: steps, captions, state expressions, logical symbols, cards.
// Any field may be a function of ctx (built by story.js): { rows, enc, bare, layout, settings, inj, nq }.
// inj = null or { pauli, q (local 0..8), caught: [check indices], cq (local qubit the decoder flips, or null), basis }.
// SVG strings (wire kets, labels above wires) mark the L subscript as "_L": '|0_L⟩'.

const pct = (x) => (x === null || x === undefined ? 'n/a' : `${(100 * x).toFixed(1)}%`);
const pts = (p, ideal) => (p === null || p === undefined ? 'n/a' : (Math.abs(p - ideal) * 100).toFixed(1));
const KL = (s) => `|${s}<sub>L</sub>⟩`;
// Patch letters in their hue; sb() is a coloured subscript ("BC" colours B and C separately).
const pl = (str) => str.replace(/\b[ABC]\b|(?<=[ABC])[ABC]|[ABC](?=[ABC])/g, (ch) => `<span class="q${ch.toLowerCase()}">${ch}</span>`);
const sb = (x) => `<sub>${pl(x)}</sub>`;
const ket = (s, sub) => `|${s}⟩` + (sub ? sb(sub) : '');
const mk = (x) => `<mark>${x}</mark>`;
const A = pl('A'), B = pl('B'), C = pl('C');

// Plaquettes in recipe order (check index k), per readout basis; the opposite type is not readable.
export const PLAQUETTES = {
  Z: [[1, 2, 4, 5], [3, 4, 6, 7], [0, 1], [7, 8]],
  X: [[0, 1, 3, 4], [4, 5, 7, 8], [2, 5], [3, 6]],
};
const plaqName = (basis, k) => `${basis}(${PLAQUETTES[basis][k].join(',')})`;
const andList = (xs) => (xs.length <= 1 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);

// Shared result cards: [big, label, tone]; tone 'grade' colours by distance from ideal.
function resultCards(ctx, withKept) {
  const { enc, bare } = ctx;
  const ideal = enc.ideal_p1;
  const cards = [
    [pct(enc.discard_fraction), 'discarded by checks', 'z'],
    [pct(enc.correct_fraction), 'corrected by the code', 'x'],
    [`${pts(enc.p1_corrected, ideal)} pts off ideal`, `coded, after correction · ${pct(enc.p1_corrected)} vs ideal ${pct(ideal)}`, 'grade', enc.p1_corrected],
    [`${bare ? pts(bare.p1_all, ideal) : 'n/a'} pts off ideal`, `bare qubits · ${bare ? pct(bare.p1_all) : 'n/a'}`, 'grade', bare ? bare.p1_all : null],
  ];
  if (withKept) cards.push([pct(enc.kept_fraction), 'kept, A read 0 (ideal 50%)', 'muted']);
  return cards;
}
const numbersLine = (ctx) => `Checks corrected ${pct(ctx.enc.correct_fraction)} of shots and discarded ${pct(ctx.enc.discard_fraction)}; `
  + `the result landed ${pts(ctx.enc.p1_corrected, ctx.enc.ideal_p1)} points from ideal, versus `
  + `${ctx.bare ? pts(ctx.bare.p1_all, ctx.enc.ideal_p1) : 'n/a'} for bare qubits.`;
const checksLine = (ctx) => `Ideal checks all read +1 and change nothing. In the real run, ${pct(ctx.enc.correct_fraction)} of shots had one flipped pattern and were corrected; ${pct(ctx.enc.discard_fraction)} had a pattern no single error makes and were discarded.`;

// ---------- physical captions (shared, with per-lesson overrides through SCRIPTS[l].phaseText) ----------
const ENCODE = "Sixteen gates spread q0's state over the patch: two CNOTs along the top row, H on q4 to q7, then ten CNOTs that fan out from those four.";
export function phaseText(lesson, name, ctx) {
  const own = SCRIPTS[lesson].phaseText;
  if (own && own[name]) return typeof own[name] === 'function' ? own[name](ctx) : own[name];
  return genericPhase(name, ctx);
}
function genericPhase(name, ctx) {
  if (name.startsWith('encode ')) return ENCODE;
  if (name.startsWith('transversal CNOT ')) {
    const [c, g] = name.slice(17).split('→');
    return `Nine CNOTs, each from a qubit of ${pl(c)} to the same qubit of ${pl(g)}. That is the whole logical CNOT.`;
  }
  if (name.startsWith('inject ')) {
    const p = name.split(' ')[1], q = name.split(' ').pop();
    return p === 'Z'
      ? `A Z error on ${q}, put there on purpose. It cannot change a Z measurement, so this readout will not see it.`
      : `An ${p} error on ${q}, put there on purpose. The qubit stays outlined red from here on.`;
  }
  if (name === 'no error') return 'Nothing happens here: no error was injected. Any error in this run comes from the device itself.';
  if (name.startsWith('readout rotation ')) {
    const P = pl(name.split(' ')[2]);
    return name.endsWith('(Y)')
      ? `S† then H on q0, H on q1 and q2 of ${P}: logical Y is then read from five qubits.`
      : `H on all nine qubits of ${P}, so that an ordinary Z measurement reads X.`;
  }
  if (name === 'measure') {
    return `All ${ctx.nq} qubits are measured. Each check of the readout basis is a parity of measured bits and glows with how often it fired. A red ring marks a qubit the decoder flipped.`;
  }
  if (name === 'logical readout') {
    return 'The logical value is a parity too: the left column (q0, q3, q6) when read in Z, the top row (q0, q1, q2) in X, taken after the correction.';
  }
  return '';
}
export function phaseMore(lesson, name) {
  const own = SCRIPTS[lesson].phaseMore;
  if (own && own[name]) return own[name];
  if (name.startsWith('encode ')) return ['one caveat', 'This encoder is not fault tolerant: one fault inside it can leave two errors in the patch, which distance 3 cannot fix.'];
  if (name.startsWith('transversal CNOT ')) return ['why it is safe', 'Each CNOT touches one qubit per patch, so one faulty gate leaves at most one error in each patch, and the checks can still fix it.'];
  if (name === 'measure') return ['why no helper qubits', 'The checks are read from the final measurement itself, so no helper qubits are needed. The price: errors are found at the end, not during the circuit.'];
  return null;
}

// ---------- lesson 1 ----------
const zl = (i) => [0, 3, 6].includes(i);
function l1Struck(ctx) {
  const { inj } = ctx;
  return inj && inj.pauli !== 'Z' && inj.caught.length > 0;
}
const L1 = {
  title: 'One patch, one error',
  intro: (ctx) => ({
    title: 'One patch, one error',
    subtitle: 'Nine physical qubits hold one logical qubit; a single error is found and fixed',
    line: `Patch ${A} starts in ${KL('0')}` + (ctx.inj ? `, then an ${ctx.inj.pauli} error strikes q${ctx.inj.q}` : '; no error is injected this time'),
  }),
  wires: [{ name: 'A', hue: 'qa', ket: '|0_L⟩' }],
  stateHeader: `Every ket here is a logical state of patch ${A}: nine physical qubits. ${KL('0')} is an equal superposition of 16 bit patterns, all even on every Z check.`,
  steps: [
    {
      title: 'Prepare', short: 'Prepare', phases: ['prepare', 'encode'],
      caption: `One logical qubit: nine physical qubits in a 3 by 3 grid. Encoding spreads |0⟩ over all nine, so no single qubit holds it. Eight checks watch the patch, four of each type.`,
      state: mk(`${KL('0')} = (|000000000⟩ + 15 more terms)/4`),
      stateNote: 'One patch, in the code space: every check would read +1.',
    },
    {
      title: (ctx) => (ctx.inj ? 'Error strikes' : 'No error injected'), short: (ctx) => (ctx.inj ? 'Error' : 'No error'),
      phases: ['inject'], virtual: 'no error',
      caption: (ctx) => {
        const { inj } = ctx;
        if (!inj) return 'No error is injected this time. Whatever the checks find below comes from the device itself: that is the noise floor.';
        if (inj.pauli === 'Z') return `A single Z error hits q${inj.q}. It leaves the code space, but a Z error cannot change a Z measurement, so this readout will never see it, and the logical Z is unharmed.`;
        return `A single ${inj.pauli} error hits q${inj.q}. The logical state is now outside the code space. Nothing in the hardware notices yet.`;
      },
      state: (ctx) => {
        const { inj } = ctx;
        if (!inj) return KL('0');
        const P = `${mk(`${inj.pauli}<sub>${inj.q}</sub>`)}${KL('0')}`;
        if (inj.pauli === 'Z') {
          const xs = PLAQUETTES.X.map((p, k) => (p.includes(inj.q) ? plaqName('X', k) : null)).filter(Boolean);
          return `${P}, outside the code space: ${andList(xs)} would read −1, but this run reads Z`;
        }
        const zs = PLAQUETTES.Z.map((p, k) => (p.includes(inj.q) ? plaqName('Z', k) : null)).filter(Boolean);
        return `${P}, outside the code space: ${zs.length > 1 ? 'checks' : 'check'} ${andList(zs)} ${zs.length > 1 ? 'read' : 'reads'} −1`;
      },
      stateNote: (ctx) => (ctx.inj ? 'The checks are not read yet: they come from the final measurement.' : 'Unchanged. Only the device’s own noise can flip a check now.'),
    },
    {
      title: 'Checks and correction', short: 'Checks', phases: ['readout rotation', 'measure'],
      caption: (ctx) => {
        const base = 'All nine qubits are measured, and each Z check is a parity of those bits.';
        if (!l1Struck(ctx)) return `${base} If one qubit flipped, the pattern points at it and the decoder flips it back. A pattern no single error makes is discarded.`;
        const { inj } = ctx;
        const fired = inj.caught.map((k) => plaqName('Z', k));
        const other = inj.cq !== null && inj.cq !== inj.q ? ` (q${inj.cq}, not q${inj.q}: the two together are an X check, so either fix works)` : '';
        return `${base} ${andList(fired)} ${fired.length > 1 ? 'read' : 'reads'} −1. Only one single bit flip makes that pattern, so the decoder flips q${inj.cq} back${other}.`;
      },
      more: ['two errors', 'Two errors in one patch either make a pattern no single error makes, and the shot is discarded, or point at the wrong qubit, and the fix completes a logical error. Distance 3 fixes one error, not two.'],
      state: (ctx) => {
        if (!l1Struck(ctx)) return KL('0');
        const { inj } = ctx;
        const syn = [0, 1, 2, 3].map((k) => (inj.caught.includes(k) ? 1 : 0)).join('');
        return `syndrome ${syn} → flip q${inj.cq}: ${mk(`X<sub>${inj.cq}</sub>X<sub>${inj.q}</sub>${KL('0')} = ${KL('0')}`)}`;
      },
      stateNote: (ctx) => (l1Struck(ctx)
        ? 'The flip is done on the recorded bits in post-processing; the hardware never acts on it.'
        : `Ideal checks read +1. In this run ${pct(ctx.enc.correct_fraction)} of shots were corrected and ${pct(ctx.enc.discard_fraction)} discarded: the device's own noise.`),
    },
    {
      title: 'Readout', short: 'Readout', phases: [], virtual: 'logical readout',
      caption: `${A} is read in Z. Logical Z is the parity of the left column, q0, q3 and q6, taken after the correction.`,
      state: `Z<sub>L</sub> = Z<sub>0</sub>Z<sub>3</sub>Z<sub>6</sub>: ${mk('v<sub>0</sub> ⊕ v<sub>3</sub> ⊕ v<sub>6</sub> = 0')} in the ideal run`,
      stateNote: (ctx) => {
        const { inj } = ctx;
        if (!l1Struck(ctx)) return 'Ideal outcome: 0 in every shot.';
        if (zl(inj.q)) return `q${inj.q} is in the left column: without the fix this parity would read 1, a logical error.`;
        return `q${inj.q} is not in the left column, so this parity would read 0 even unfixed; an error on q0, q3 or q6 would flip it, and the fix catches that too.`;
      },
    },
    {
      title: 'Result', short: 'Result', phases: [],
      caption: (ctx) => {
        const { enc, bare } = ctx;
        if (l1Struck(ctx)) return `With the error, ${pct(enc.correct_fraction)} of shots were corrected and ${pct(enc.discard_fraction)} discarded. The bare qubit got the same error and reads 1 in ${bare ? pct(bare.p1_all) : 'n/a'} of shots, with no warning.`;
        const blind = ctx.inj ? `The Z error never showed: a Z readout is blind to it, on the patch and the bare qubit alike. ` : '';
        return `${blind}This is the noise floor: ${pct(enc.discard_fraction)} of shots discarded, ${pct(enc.correct_fraction)} corrected, and the logical bit wrong in ${pct(enc.p1_corrected)} of the rest (ideal 0). The bare qubit: ${bare ? pct(bare.p1_all) : 'n/a'}.`;
      },
      state: (ctx) => `P(A reads 1) after correction: ${mk(pct(ctx.enc.p1_corrected))} (ideal 0)`,
      stateNote: (ctx) => `Bare qubit, same experiment: ${ctx.bare ? pct(ctx.bare.p1_all) : 'n/a'}.`,
    },
  ],
  labels: (ctx) => [['|0_L⟩'], [ctx.inj ? `${ctx.inj.pauli} on q${ctx.inj.q}` : '|0_L⟩'], [l1Struck(ctx) ? 'corrected' : '|0_L⟩'], ['measured'], ['measured']],
  symbols: (ctx) => [
    [{ t: 'box', w: 0, label: 'encode' }],
    [ctx.inj ? { t: 'error', w: 0, label: `${ctx.inj.pauli} on q${ctx.inj.q}` } : { t: 'note', w: 0, label: 'no error' }],
    [{ t: 'checks', w: 0 }, ...(l1Struck(ctx) ? [{ t: 'badge', w: 0, on: 'checks', text: '−1', tone: 'bad' }] : [])],
    [{ t: 'meter', w: 0, basis: 'Z' }, ...(l1Struck(ctx) ? [{ t: 'badge', w: 0, on: 'meter', text: 'fixed', tone: 'good' }] : [])],
    [],
  ],
  closing: (rows, ctx) => (l1Struck(ctx)
    ? [`One ${ctx.inj.pauli} error on q${ctx.inj.q} pushed the patch out of the code space, and its checks pointed straight at it.`,
      'The decoder flipped that bit back on the recorded data; no gate ever acted on a measurement.',
      numbersLine(ctx)]
    : ['Nine physical qubits hold one logical qubit; eight checks watch it.',
      'A single flipped qubit is located by its checks and fixed on the recorded bits.',
      numbersLine(ctx)]),
  cards: (rows, ctx) => resultCards(ctx, false),
};

// ---------- lesson 2 ----------
const BELL_AB = `(|00⟩ + |11⟩)${sb('AB')} / √2`;
const L2 = {
  title: 'Logical CNOT is nine physical CNOTs',
  intro: {
    title: 'A logical CNOT',
    subtitle: 'Nine physical CNOTs make one logical gate, and a logical Bell pair',
    line: `Two patches: ${A} starts in ${KL('+')}, ${B} in ${KL('0')}`,
  },
  wires: [{ name: 'A', hue: 'qa', ket: '|+_L⟩' }, { name: 'B', hue: 'qb', ket: '|0_L⟩' }],
  stateHeader: 'Every ket here is a logical state, i.e. a whole patch of nine qubits.',
  steps: [
    {
      title: 'Prepare', short: 'Prepare', phases: ['prepare', 'encode'],
      caption: `${A} is encoded as ${KL('+')}, ${B} as ${KL('0')}. Each is a patch of nine physical qubits in a 3 by 3 grid.`,
      state: mk(`${ket('+', 'A')} ${ket(0, 'B')}`), stateNote: 'Two independent patches.',
    },
    {
      title: 'Logical CNOT', short: 'CNOT', phases: ['transversal'],
      caption: `A logical CNOT from ${A} to ${B}. It is nine physical CNOTs, one between each pair of matching qubits, and nothing else. No decoding, no re-encoding: the gate is transversal.`,
      state: mk(BELL_AB), stateNote: `${A} and ${B} are a logical Bell pair.`,
    },
    {
      title: 'Checks', short: 'Checks', phases: ['readout rotation', 'measure'],
      caption: `Both patches are checked from their own measurement. A CNOT can copy an error from ${A} onto the same qubit of ${B}, never onto a neighbour, so each patch still holds at most one error and the decoder can fix it.`,
      state: BELL_AB, stateNote: checksLine,
    },
    {
      title: 'Readout', short: 'Readout', phases: [], virtual: 'logical readout',
      caption: 'Both are read in Z. The two logical bits must agree. The same circuit read in X must agree too; only entanglement does both.',
      state: `${mk(`|00⟩${sb('AB')} or |11⟩${sb('AB')}`)}, each with probability ½`,
      stateNote: 'A Z readout alone cannot tell a Bell pair from a copied coin flip; the XX readout in the lesson panel can.',
    },
    {
      title: 'Result', short: 'Result', phases: [],
      caption: (ctx) => {
        const xx = ctx.rows.find((r) => r.kind === 'enc' && r.name === 'enc_XX');
        return `The ZZ parity came out odd in ${pct(ctx.enc.p1_corrected)} of shots after correction (ideal 0)`
          + (xx ? `, the XX parity in ${pct(xx.p1_corrected)}` : '') + `. Bare qubits: ${ctx.bare ? pct(ctx.bare.p1_all) : 'n/a'}.`;
      },
      state: (ctx) => `P(A ⊕ B = 1) after correction: ${mk(pct(ctx.enc.p1_corrected))} (ideal 0)`,
      stateNote: (ctx) => `Bare qubits, same experiment: ${ctx.bare ? pct(ctx.bare.p1_all) : 'n/a'}.`,
    },
  ],
  labels: () => [['|+_L⟩', '|0_L⟩'], ['Bell pair with B', 'Bell pair with A'], ['Bell pair with B', 'Bell pair with A'], ['measured', 'measured'], ['measured', 'measured']],
  symbols: [
    [{ t: 'box', w: 0, label: 'encode' }, { t: 'box', w: 1, label: 'encode' }],
    [{ t: 'cnot', c: 0, g: 1, label: 'CNOT A→B' }],
    [{ t: 'checks', w: 0 }, { t: 'checks', w: 1 }],
    [{ t: 'meter', w: 0, basis: 'Z' }, { t: 'meter', w: 1, basis: 'Z' }],
    [{ t: 'post', on: 1, from: [0], label: 'A ⊕ B', sub: 'parity, in post-processing' }],
  ],
  closing: (rows, ctx) => [
    'A logical CNOT is nine physical CNOTs, qubit to matching qubit, and nothing else.',
    'Transversal: an error can copy to the same qubit of the other patch, but never spreads inside one.',
    numbersLine(ctx),
  ],
  cards: (rows, ctx) => resultCards(ctx, false),
};

// ---------- lesson 3 ----------
const BELL_BC = `(|00⟩ + |11⟩)${sb('BC')}`;
const AFTER_AB = `( α${ket(0, 'A')} ${BELL_BC} + β${ket(1, 'A')} (|10⟩ + |01⟩)${sb('BC')} ) / √2`;
const cf = (x) => mk(`(${x})${sb('C')}`);
const L3 = {
  title: 'Teleporting a logical qubit',
  intro: {
    title: 'Teleporting a logical qubit',
    subtitle: 'Across 27 physical qubits on a distance 3 surface code, corrected in post-processing',
    line: `Three patches: ${A} carries ψ, ${B} and ${C} are blank slates`,
  },
  wires: [{ name: 'A', hue: 'qa', ket: 'ψ' }, { name: 'B', hue: 'qb', ket: '|+_L⟩' }, { name: 'C', hue: 'qc', ket: '|0_L⟩' }],
  stateHeader: '|ψ⟩ = α|0⟩ + β|1⟩ with α ≈ 0.92, β ≈ 0.38. Every ket here is a logical state, i.e. a whole patch.',
  steps: [
    {
      title: 'Prepare', short: 'Prepare', phases: ['prepare', 'encode'],
      caption: `Three logical qubits. Each is a patch of nine physical qubits in a 3 by 3 grid. ${A} holds the state ψ, ${B} starts in ${KL('+')} and ${C} in ${KL('0')}. Encoding spreads each state over all nine qubits of its patch.`,
      state: `${ket('ψ', 'A')} ${ket('+', 'B')} ${ket(0, 'C')}`, stateNote: 'Three independent patches.',
    },
    {
      title: 'Bell pair', short: 'Bell pair', phases: ['transversal CNOT B→C'],
      caption: `A logical CNOT from ${B} to ${C} turns ${B} and ${C} into a logical Bell pair. One logical CNOT is just nine physical CNOTs, qubit to matching qubit, and nothing else. A gate like that is called transversal, and it is the easiest kind of logical gate there is.`,
      state: `${ket('ψ', 'A')} ⊗ ${mk(`${BELL_BC} / √2`)}`, stateNote: `${B} and ${C} are now one entangled pair; neither has a state of its own.`,
    },
    {
      title: 'Bell measurement', short: 'Bell meas.', phases: ['transversal CNOT A→B'],
      caption: `A second logical CNOT, from ${A} to ${B}. Together with reading ${A} in X and ${B} in Z (next), this is a Bell measurement of ${A} and ${B}. That is all teleportation needs.`,
      state: mk(AFTER_AB), stateNote: `${A}'s amplitudes α, β are now tied to the ${pl('BC')} pair. Nothing has been measured yet.`,
    },
    {
      title: 'Checks', short: 'Checks', phases: ['readout rotation', 'measure'],
      caption: 'Each patch is checked from its own measurement: the four plaquette parities of the basis it is read in. A single flipped pattern points at one qubit, which is corrected in post-processing; a pattern no single error makes is discarded.',
      more: ['distance 3', 'With nine qubits per patch the code has distance 3: one error per patch can be located and fixed, not just detected. Two errors cannot.'],
      state: AFTER_AB, stateNote: checksLine,
    },
    {
      title: 'Readout', short: 'Readout', phases: [], virtual: 'logical readout',
      caption: `Readout. ${A} is read in X, ${B} and ${C} in Z. The two outcomes from ${A} and ${B} tell us which Pauli correction ${C} needs. We apply it to the recorded bits afterwards, so the hardware never has to act on a measurement in real time.`,
      pre: `After H on ${A} (reading ${A} in X) and grouping by the outcomes a (from ${A}) and b (from ${B}):`,
      state: `½ [ |00⟩${sb('AB')} ${cf('α|0⟩ + β|1⟩')} + |01⟩${sb('AB')} ${cf('α|1⟩ + β|0⟩')}<br>`
        + `&nbsp;&nbsp;&nbsp;+ |10⟩${sb('AB')} ${cf('α|0⟩ − β|1⟩')} + |11⟩${sb('AB')} ${cf('α|1⟩ − β|0⟩')} ]`,
      table: { head: ['ab', `${C} holds`], rows: [['00', 'ψ'], ['01', 'X ψ'], ['10', 'Z ψ'], ['11', 'XZ ψ']] },
      stateNote: `Whatever a and b come out, ${C} holds ψ up to a Pauli we know. In this experiment the correction is a bit flip of ${C}'s recorded result: when ${C} is read in Z, flip it if b = 1; when read in X, flip it if a = 1.`,
    },
    {
      title: 'Result', short: 'Result', phases: [],
      caption: `ψ has arrived in ${C}. The numbers come from the actual run: how often ${C} read 1 compared with the ideal, how many shots the checks corrected or threw away, and how the same circuit did on bare physical qubits with no code at all.`,
      state: mk(ket('ψ', 'C')),
      stateNote: (ctx) => `After the frame correction, ${C} is ψ. Ideal P(${C} reads 1) = β² = ${pct(ctx.enc.ideal_p1)}. Measured: ${pct(ctx.enc.p1_corrected)} after correction.`,
    },
  ],
  labels: () => [
    ['ψ', '|+_L⟩', '|0_L⟩'], ['ψ', 'Bell pair with C', 'Bell pair with B'], ['ψ', 'Bell pair with C', 'Bell pair with B'],
    ['ψ', 'Bell pair with C', 'Bell pair with B'], ['measured', 'measured', 'ψ, up to a known Pauli'], ['measured', 'measured', 'ψ'],
  ],
  symbols: [
    [{ t: 'box', w: 0, label: 'encode' }, { t: 'box', w: 1, label: 'encode' }, { t: 'box', w: 2, label: 'encode' }],
    [{ t: 'cnot', c: 1, g: 2, label: 'CNOT B→C' }],
    [{ t: 'cnot', c: 0, g: 1, label: 'CNOT A→B' }],
    [{ t: 'checks', w: 0 }, { t: 'checks', w: 1 }, { t: 'checks', w: 2 }],
    [{ t: 'meter', w: 0, basis: 'X' }, { t: 'meter', w: 1, basis: 'Z' }, { t: 'meter', w: 2, basis: 'Z' }],
    [{ t: 'post', on: 2, from: [0, 1], label: 'Pauli frame', sub: 'in post-processing' }],
  ],
  closing: (rows, ctx) => [
    `A logical state moved from patch ${A} to patch ${C} by teleportation.`,
    'Every logical gate was built from a handful of physical gates: nine CNOTs per logical CNOT.',
    numbersLine(ctx),
  ],
  cards: (rows, ctx) => resultCards(ctx, false),
};

// ---------- lesson 4 ----------
const L4_AFTER = `( ${ket(0, 'B')} (α|0⟩ + β|1⟩)${sb('A')} + ω${ket(1, 'B')} (α|1⟩ + β|0⟩)${sb('A')} ) / √2`;
const L4 = {
  title: 'The T gate is a teleportation too',
  intro: {
    title: 'The T gate, teleported',
    subtitle: 'A magic state on a fresh patch carries the T gate onto the data',
    line: `Two patches: ${A} carries ψ, ${B} holds the magic state T|+⟩`,
  },
  wires: [{ name: 'A', hue: 'qa', ket: 'ψ' }, { name: 'B', hue: 'qb', ket: 'T|+_L⟩' }],
  stateHeader: '|ψ⟩ = α|0⟩ + β|1⟩ with α ≈ 0.92, β ≈ 0.38, and ω = e<sup>iπ/4</sup>. Every ket here is a logical state, i.e. a whole patch.',
  steps: [
    {
      title: 'Prepare', short: 'Prepare', phases: ['prepare', 'encode'],
      caption: `${A} carries ψ. ${B} is prepared as the magic state T|+⟩ before encoding: H then T on one qubit, then the encoder. The T gate is applied here, on a fresh patch, not on the data.`,
      state: `${ket('ψ', 'A')} ⊗ ${mk(`(|0⟩ + ω|1⟩)${sb('B')} / √2`)}`, stateNote: `${B} is the magic state T|+⟩, made before any data is touched.`,
    },
    {
      title: 'Logical CNOT B→A', short: 'CNOT', phases: ['transversal'],
      caption: `A transversal CNOT with the magic patch ${B} as control and the data patch ${A} as target: nine CNOTs from ${B}'s qubits to ${A}'s.`,
      state: mk(L4_AFTER), stateNote: `${A} and ${B} are now entangled. Nothing has been measured yet.`,
    },
    {
      title: 'Checks', short: 'Checks', phases: ['readout rotation', 'measure'],
      caption: 'Both patches are checked from their own measurement, and a single error in either patch is located and corrected as before.',
      state: L4_AFTER, stateNote: checksLine,
    },
    {
      title: 'Readout and post-selection', short: 'Readout', phases: [], virtual: 'logical readout',
      caption: `${A} is read in Z. When ${A} reads 0, ${B} holds Tψ. When ${A} reads 1, ${B} would need an S gate, which needs feed-forward, so here we keep only the ${A} = 0 shots. Half the shots survive.`,
      pre: `Grouping by ${A}'s outcome:`,
      state: `( ${ket(0, 'A')} ${mk(`(α|0⟩ + ωβ|1⟩)${sb('B')}`)} + ${ket(1, 'A')} (β|0⟩ + ωα|1⟩)${sb('B')} ) / √2`,
      stateNote: `The highlighted factor is Tψ, kept when ${A} reads 0. The ${A} = 1 branch is discarded.`,
    },
    {
      title: 'Result', short: 'Result', phases: [],
      caption: (ctx) => `${B} read in X gives 1 in 25% of kept shots ideally, and the same in Y; without the T gate X would give 14.6% and Y 50%. Measured: ${pct(ctx.enc.p1_corrected)}. Kept: ${pct(ctx.enc.kept_fraction)} (ideal 50%).`,
      state: mk(`T${ket('ψ', 'B')}`),
      stateNote: (ctx) => `Ideal P(${B} reads 1 in X) = 25%. Measured after correction: ${pct(ctx.enc.p1_corrected)}.`,
    },
  ],
  labels: () => [['ψ', 'T|+_L⟩'], ['entangled with B', 'entangled with A'], ['entangled with B', 'entangled with A'], ['measured', 'Tψ if A reads 0'], ['measured', 'Tψ']],
  symbols: [
    [{ t: 'box', w: 0, label: 'encode' }, { t: 'box', w: 1, label: 'encode' }],
    [{ t: 'cnot', c: 1, g: 0, label: 'CNOT B→A' }],
    [{ t: 'checks', w: 0 }, { t: 'checks', w: 1 }],
    [{ t: 'meter', w: 0, basis: 'Z' }, { t: 'meter', w: 1, basis: 'X' }],
    [{ t: 'post', on: 1, from: [0], label: 'keep A = 0', sub: 'post-selection' }],
  ],
  phaseText: {
    'prepare B (magic state)': `H then T on q0 of ${B}: the magic state T|+⟩, made on one qubit before the encoder spreads it.`,
  },
  phaseMore: {
    'prepare B (magic state)': ['why here', 'T is not transversal in this code. Applying it to one qubit before encoding is cheap but unprotected; real machines distil many noisy magic states into a clean one.'],
  },
  closing: (rows, ctx) => [
    'The T gate never touched the data: it was made on a fresh patch and teleported in.',
    `A transversal CNOT and a Z readout of ${A} did the teleporting; we kept the ${A} = 0 half (${pct(ctx.enc.kept_fraction)} of shots).`,
    numbersLine(ctx),
  ],
  cards: (rows, ctx) => resultCards(ctx, true),
};

// Prepare-phase captions that depend on the lesson.
L1.phaseText = {
  measure: (ctx) => {
    if (!l1Struck(ctx)) return genericPhase('measure', ctx);
    const { inj } = ctx, rates = (ctx.enc.blocks[0] || { checks: [] }).checks;
    const fired = inj.caught.map((k) => `${plaqName('Z', k)} (${pct(rates[k])})`);
    return `All nine qubits are measured. The checks that see q${inj.q}, ${andList(fired)}, are outlined red. The decoder flips q${inj.cq} back: the red ring.`;
  },
};
L2.phaseText = { 'prepare A': `H on q0 of ${A} gives |+⟩ to encode.` };
L3.phaseText = { 'prepare A': `A single Ry gate puts ψ on q0 of patch ${A}.`, 'prepare B': `H on q0 of ${B} gives |+⟩ to encode.` };
L4.phaseText['prepare A'] = `A single Ry gate puts ψ on q0 of patch ${A}.`;

export const SCRIPTS = { 1: L1, 2: L2, 3: L3, 4: L4 };
