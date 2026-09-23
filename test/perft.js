'use strict';
// Perft: count the leaf nodes of the move tree to a fixed depth. Any rule bug —
// a missing en-passant, a castling right that outlives its rook, a pinned piece
// allowed to move — changes the count, so matching published numbers is strong
// evidence the generator is exact.
//
// Two checks run here:
//   1. js/position.js (the engine's generator) against the published counts.
//   2. js/position.js against js/chess.js (the UI's generator) node for node.
// Two independent implementations agreeing is what keeps both honest.
//
// Usage: node test/perft.js [--deep]

import { Position } from '../js/position.js';
import { getLegalMoves, makeMove } from '../js/chess.js';

const DEEP = process.argv.includes('--deep');

// name, FEN, [perft(1), perft(2), …]
const SUITE = [
  ['startpos',
   'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
   [20, 400, 8902, 197281, 4865609, 119060324]],
  ['kiwipete',
   'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
   [48, 2039, 97862, 4085603, 193690690]],
  ['position 3',
   '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',
   [14, 191, 2812, 43238, 674624, 11030083]],
  ['position 4',
   'r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1',
   [6, 264, 9467, 422333, 15833292]],
  ['position 4 mirrored',
   'r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1',
   [6, 264, 9467, 422333, 15833292]],
  ['position 5',
   'rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8',
   [44, 1486, 62379, 2103487, 89941194]],
  ['position 6',
   'r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10',
   [46, 2079, 89890, 3894594, 164075551]],
];

// Node budget per position: perft blows up fast, and the point is correctness
// coverage across many rule corners rather than raw depth on one of them.
const BUDGET = DEEP ? 200_000_000 : 5_000_000;

function perft(pos, depth) {
  if (depth === 0) return 1;
  const n = pos.generate(pos.ply, false);
  const base = pos.ply * 256;
  const buf = pos.moveBuf;
  // Copy out: recursion reuses the buffer slot for deeper plies only, but the
  // current slot stays valid, so no copy is needed — read it directly.
  let nodes = 0;
  for (let i = 0; i < n; i++) {
    const m = buf[base + i];
    if (!pos.makeMove(m)) continue;
    nodes += depth === 1 ? 1 : perft(pos, depth - 1);
    pos.unmakeMove();
  }
  return nodes;
}

// Reference count through the UI's own rules module.
function perftRef(state, side, depth) {
  if (depth === 0) return 1;
  const moves = getLegalMoves(state, side);
  if (depth === 1) return moves.length;
  let nodes = 0;
  for (const mv of moves) {
    const next = makeMove(state, mv);
    nodes += perftRef(next, side === 'white' ? 'black' : 'white', depth - 1);
  }
  return nodes;
}

let failures = 0;
let checked = 0;

console.log(`Perft — engine generator vs published counts${DEEP ? ' (deep)' : ''}`);
console.log('='.repeat(72));

for (const [name, fen, expected] of SUITE) {
  process.stdout.write(`\n${name}\n  ${fen}\n`);
  for (let d = 1; d <= expected.length; d++) {
    if (expected[d - 1] > BUDGET) { console.log(`  depth ${d}: skipped (over node budget)`); break; }
    const pos = new Position().setFromFEN(fen);
    const t0 = Date.now();
    const got = perft(pos, d);
    const ms = Date.now() - t0;
    const ok = got === expected[d - 1];
    if (!ok) failures++;
    checked++;
    const nps = ms > 0 ? Math.round(got / (ms / 1000) / 1000) : 0;
    console.log(`  depth ${d}: ${String(got).padStart(11)} ${ok ? 'OK ' : `WRONG (want ${expected[d - 1]})`} ${String(ms).padStart(6)}ms  ${nps}k nps`);
  }
}

// ── Cross-check the two generators ─────────────────────────────────────────
console.log('\n' + '='.repeat(72));
console.log('Cross-check: js/position.js vs js/chess.js (independent generators)');
const CROSS_DEPTH = 4;
for (const [name, fen] of SUITE) {
  const pos = new Position().setFromFEN(fen);
  const { state, side } = pos.toUIState();
  for (let d = 1; d <= CROSS_DEPTH; d++) {
    const a = perft(new Position().setFromFEN(fen), d);
    const b = perftRef(state, side, d);
    checked++;
    const ok = a === b;
    if (!ok) failures++;
    console.log(`  ${name.padEnd(22)} depth ${d}: position.js ${String(a).padStart(9)}  chess.js ${String(b).padStart(9)}  ${ok ? 'MATCH' : 'MISMATCH'}`);
  }
}

console.log('\n' + '='.repeat(72));
console.log(failures === 0
  ? `All ${checked} perft checks passed.`
  : `${failures} of ${checked} perft checks FAILED.`);
process.exit(failures === 0 ? 0 : 1);
