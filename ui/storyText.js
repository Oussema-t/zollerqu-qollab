// Per-lesson scripts for the story page: steps, captions, state expressions, logical symbols, cards.
// Any field may be a function of ctx (built by story.js): { rows, enc, bare, layout, settings, inj, rings, nq }.
// rings = true when the physical picture draws at least one red ring (a qubit above ringCut in some block).
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
function resultCards(ctx) {
  const { enc, bare } = ctx;
  const ideal = enc.ideal_p1;
  const cards = [
    [pct(enc.discard_fraction), 'discarded by checks', 'z'],
    [pct(enc.correct_fraction), 'corrected by the code', 'x'],
    [`${pts(enc.p1_corrected, ideal)} pts off ideal`, `coded, after correction · ${pct(enc.p1_corrected)} vs ideal ${pct(ideal)}`, 'grade', enc.p1_corrected],
    [`${bare ? pts(bare.p1_all, ideal) : 'n/a'} pts off ideal`, `bare qubits · ${bare ? pct(bare.p1_all) : 'n/a'}`, 'grade', bare ? bare.p1_all : null],
  ];
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
    return `All ${ctx.nq} qubits are measured. Each check of the readout basis is a parity of measured bits and glows with how often it fired.${ctx.rings ? ' A red ring marks a qubit the decoder flipped.' : ''}`;
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
  cards: (rows, ctx) => resultCards(ctx),
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
  cards: (rows, ctx) => resultCards(ctx),
};

// ---------- lesson 3 ----------
const T_KET = `(${KL('0')} + ω${KL('1')}) / √2`;
const L3 = {
  title: 'A T state on a patch',
  intro: {
    title: 'A T state on a patch',
    subtitle: 'The one ingredient no transversal gate can make, prepared on one qubit and spread over nine',
    line: `Patch ${A} is prepared in T${KL('+')}`,
  },
  wires: [{ name: 'A', hue: 'qa', ket: 'T|+_L⟩' }],
  stateHeader: `ω = e<sup>iπ/4</sup>. Every ket here is a logical state of patch ${A}: nine physical qubits.`,
  steps: [
    {
      title: 'Prepare', short: 'Prepare', phases: ['prepare', 'encode'],
      caption: 'H puts q0 in |+⟩, T adds the π/4 phase; the sixteen encoder gates spread that one qubit over the patch.',
      state: mk(`T${KL('+')} = ${T_KET}`),
      stateNote: 'The T gate acted on one physical qubit before encoding. That step is not protected by the code.',
    },
    {
      title: 'Checks', short: 'Checks', phases: ['readout rotation', 'measure'],
      caption: 'The patch is checked from its own measurement. Read in X, the four X plaquettes are parities of the measured bits: a single flipped pattern points at one qubit, which is corrected in post-processing, and a pattern no single error makes is discarded. Read in Y, no check is readable.',
      state: T_KET, stateNote: checksLine,
    },
    {
      title: 'Readout', short: 'Readout', phases: [], virtual: 'logical readout',
      caption: `${A} is read in X and in Y: each should give 1 in 14.6 percent of shots.`,
      state: `⟨X<sub>L</sub>⟩ = ⟨Y<sub>L</sub>⟩ = cos(π/4), so ${mk('P(1) = sin²(π/8) ≈ 14.6%')} in both bases`,
      stateNote: `A plain ${KL('+')} would give 0% in X and 50% in Y. Both numbers together show the T phase.`,
    },
    {
      title: 'Result', short: 'Result', phases: [],
      caption: (ctx) => {
        const y = ctx.rows.find((r) => r.kind === 'enc' && r.name === 'enc_Y');
        return `Read in X, ${A} gave 1 in ${pct(ctx.enc.p1_corrected)} of shots after correction`
          + (y ? `, read in Y in ${pct(y.p1_corrected)}` : '') + ` (ideal 14.6% each). The bare qubit: ${ctx.bare ? pct(ctx.bare.p1_all) : 'n/a'}.`;
      },
      state: (ctx) => `P(${A} reads 1 in X) after correction: ${mk(pct(ctx.enc.p1_corrected))} (ideal ${pct(ctx.enc.ideal_p1)})`,
      stateNote: (ctx) => `Bare qubit, same experiment: ${ctx.bare ? pct(ctx.bare.p1_all) : 'n/a'}.`,
    },
  ],
  labels: () => [['T|+_L⟩'], ['T|+_L⟩'], ['measured'], ['measured']],
  symbols: (ctx) => [
    [{ t: 'box', w: 0, label: 'H, T, encode' }],
    [{ t: 'checks', w: 0 }],
    [{ t: 'meter', w: 0, basis: ctx.enc.bases[0] }],
    [],
  ],
  phaseText: {
    'prepare A': `H then T on q0 of patch ${A}: H gives |+⟩, T adds the phase e<sup>iπ/4</sup> to |1⟩.`,
  },
  phaseMore: {
    'prepare A': ['why here', 'T is not transversal in this code. Applying it to one qubit before encoding is cheap but unprotected; real machines distil many noisy T states into a clean one.'],
  },
  closing: (rows, ctx) => [
    `Patch ${A} holds the T state T${KL('+')}, the resource that lets a T gate be teleported onto data later.`,
    'It was made on a single qubit and spread over nine by the encoder; in X and in Y it reads 1 in 14.6% of shots ideally.',
    numbersLine(ctx),
  ],
  cards: (rows, ctx) => resultCards(ctx),
};

// ---------- lesson 4 ----------
const BELL_BC = `(|00⟩ + |11⟩)${sb('BC')}`;
const AFTER_AB = `( α${ket(0, 'A')} ${BELL_BC} + β${ket(1, 'A')} (|10⟩ + |01⟩)${sb('BC')} ) / √2`;
const cf = (x) => mk(`(${x})${sb('C')}`);
const L4 = {
  title: 'Teleporting the T state',
  intro: {
    title: 'Teleporting the T state',
    subtitle: 'Across 27 physical qubits on a distance 3 surface code, corrected in post-processing',
    line: `Three patches: ${A} carries the T state, ${B} and ${C} are blank slates`,
  },
  wires: [{ name: 'A', hue: 'qa', ket: 'T|+_L⟩' }, { name: 'B', hue: 'qb', ket: '|+_L⟩' }, { name: 'C', hue: 'qc', ket: '|0_L⟩' }],
  stateHeader: '|T⟩ = T|+⟩ = α|0⟩ + β|1⟩ with α = 1/√2, β = e<sup>iπ/4</sup>/√2. Every ket here is a logical state, i.e. a whole patch.',
  steps: [
    {
      title: 'Prepare', short: 'Prepare', phases: ['prepare', 'encode'],
      caption: `Three logical qubits. Each is a patch of nine physical qubits in a 3 by 3 grid. ${A} holds the T state T|+⟩ from lesson 3, ${B} starts in ${KL('+')} and ${C} in ${KL('0')}. Encoding spreads each state over all nine qubits of its patch.`,
      state: `${ket('T', 'A')} ${ket('+', 'B')} ${ket(0, 'C')}`, stateNote: 'Three independent patches.',
    },
    {
      title: 'Bell pair', short: 'Bell pair', phases: ['transversal CNOT B→C'],
      caption: `A logical CNOT from ${B} to ${C} turns ${B} and ${C} into a logical Bell pair. One logical CNOT is just nine physical CNOTs, qubit to matching qubit, and nothing else. A gate like that is called transversal, and it is the easiest kind of logical gate there is.`,
      state: `${ket('T', 'A')} ⊗ ${mk(`${BELL_BC} / √2`)}`, stateNote: `${B} and ${C} are now one entangled pair; neither has a state of its own.`,
    },
    {
      title: 'Bell measurement', short: 'Bell meas.', phases: ['transversal CNOT A→B'],
      caption: `A second logical CNOT, from ${A} to ${B}. Together with reading ${A} in X and ${B} in Z (next), this is a Bell measurement of ${A} and ${B}. That is all the T state needs to travel.`,
      state: mk(AFTER_AB), stateNote: `The T state's amplitudes α, β are now tied to the ${pl('BC')} pair. Nothing has been measured yet.`,
    },
    {
      title: 'Checks', short: 'Checks', phases: ['readout rotation', 'measure'],
      caption: 'Each patch is checked from its own measurement: the four plaquette parities of the basis it is read in. A single flipped pattern points at one qubit, which is corrected in post-processing; a pattern no single error makes is discarded. A patch read in Y has no readable checks.',
      more: ['distance 3', 'With nine qubits per patch the code has distance 3: one error per patch can be located and fixed, not just detected. Two errors cannot.'],
      state: AFTER_AB, stateNote: checksLine,
    },
    {
      title: 'Readout', short: 'Readout', phases: [], virtual: 'logical readout',
      caption: `Readout. ${A} is read in X, ${B} in Z, ${C} in X or in Y. The outcomes a (from ${A}) and b (from ${B}) tell us which Pauli correction ${C} needs. We apply it to the recorded bits afterwards: read in X, flip ${C} if a = 1; read in Y, flip it if a ⊕ b = 1.`,
      pre: `After H on ${A} (reading ${A} in X) and grouping by the outcomes a (from ${A}) and b (from ${B}):`,
      state: `½ [ |00⟩${sb('AB')} ${cf('α|0⟩ + β|1⟩')} + |01⟩${sb('AB')} ${cf('α|1⟩ + β|0⟩')}<br>`
        + `&nbsp;&nbsp;&nbsp;+ |10⟩${sb('AB')} ${cf('α|0⟩ − β|1⟩')} + |11⟩${sb('AB')} ${cf('α|1⟩ − β|0⟩')} ]`,
      table: { head: ['ab', `${C} holds`], rows: [['00', '|T⟩'], ['01', 'X|T⟩'], ['10', 'Z|T⟩'], ['11', 'XZ|T⟩']] },
      stateNote: `Whatever a and b come out, ${C} holds |T⟩ up to a Pauli we know. Z flips an X or a Y readout, X flips a Y readout: so read in X, flip ${C}'s recorded bit if a = 1; read in Y, flip it if a ⊕ b = 1.`,
    },
    {
      title: 'Result', short: 'Result', phases: [],
      caption: `The T state has arrived in ${C}. The numbers come from the actual run: how often ${C} read 1 compared with the ideal, how many shots the checks corrected or threw away, and how the same circuit did on bare physical qubits with no code at all.`,
      state: mk(ket('T', 'C')),
      stateNote: (ctx) => `After the frame correction, ${C} is |T⟩. Ideal P(${C} reads 1) = sin²(π/8) = ${pct(ctx.enc.ideal_p1)} in X and in Y. Measured: ${pct(ctx.enc.p1_corrected)} after correction.`,
    },
  ],
  labels: () => [
    ['T|+_L⟩', '|+_L⟩', '|0_L⟩'], ['T|+_L⟩', 'Bell pair with C', 'Bell pair with B'], ['T|+_L⟩', 'Bell pair with C', 'Bell pair with B'],
    ['T|+_L⟩', 'Bell pair with C', 'Bell pair with B'], ['measured', 'measured', 'T|+_L⟩, up to a known Pauli'], ['measured', 'measured', 'T|+_L⟩'],
  ],
  symbols: [
    [{ t: 'box', w: 0, label: 'encode' }, { t: 'box', w: 1, label: 'encode' }, { t: 'box', w: 2, label: 'encode' }],
    [{ t: 'cnot', c: 1, g: 2, label: 'CNOT B→C' }],
    [{ t: 'cnot', c: 0, g: 1, label: 'CNOT A→B' }],
    [{ t: 'checks', w: 0 }, { t: 'checks', w: 1 }, { t: 'checks', w: 2, sub: 'X; none in Y' }],
    [{ t: 'meter', w: 0, basis: 'X' }, { t: 'meter', w: 1, basis: 'Z' }, { t: 'meter', w: 2, basis: 'X/Y' }],
    [{ t: 'post', on: 2, from: [0, 1], label: 'Pauli frame', sub: 'in post-processing' }],
  ],
  phaseText: {
    'prepare A': `H then T on q0 of patch ${A} make the T state to encode.`,
    'prepare B': `H on q0 of ${B} gives |+⟩ to encode.`,
  },
  closing: (rows, ctx) => [
    `The T state moved from patch ${A} to patch ${C} by teleportation; the correction was applied to the recorded bits, never to the hardware.`,
    'Every logical gate was built from a handful of physical gates: nine CNOTs per logical CNOT.',
    numbersLine(ctx),
  ],
  cards: (rows, ctx) => resultCards(ctx),
};

// Prepare-phase captions that depend on the lesson.
L1.phaseText = {
  measure: (ctx) => {
    if (!l1Struck(ctx)) return genericPhase('measure', ctx);
    const { inj } = ctx, rates = (ctx.enc.blocks[0] || { checks: [] }).checks;
    const fired = inj.caught.map((k) => `${plaqName('Z', k)} (${pct(rates[k])})`);
    return `All nine qubits are measured. The checks that see q${inj.q}, ${andList(fired)}, are outlined red. The decoder flips q${inj.cq} back${ctx.rings ? ': the red ring' : ''}.`;
  },
};
L2.phaseText = { 'prepare A': `H on q0 of ${A} gives |+⟩ to encode.` };

export const SCRIPTS = { 1: L1, 2: L2, 3: L3, 4: L4 };
