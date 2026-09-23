'use strict';
// Tactical test suite.
//
// Mate positions are not checked against a hand-written answer key. A separate
// exhaustive minimax in this file works out the true shortest forced mate, and
// the engine is then required to find a mate of exactly that length and to play
// a move that really forces it. The reference search uses no transposition
// table, no pruning heuristics and no evaluation at all — only "is this
// checkmate" — so it cannot inherit a bug from the engine it is judging. That
// also means the suite stays honest if the positions are ever edited.
//
// Usage: node test/tactics.js [--ms 1000] [--nn]

import { Position } from '../js/position.js';
import { searchPosition, searchInfo, resetEngine, moveToString } from '../js/engine.js';
import { loadModelFromDisk, nnReady, fmt } from './harness.js';

const argMs = process.argv.indexOf('--ms');
const MS = argMs > 0 ? +process.argv[argMs + 1] : 1000;
const USE_NN = process.argv.includes('--nn');

// kind 'mate'    — reference solver finds the true distance; engine must match
// kind 'best'    — engine must play one of the listed moves
// kind 'score'   — engine's evaluation must be at least this many centipawns
const SUITE = [
  ['back-rank mate',        '6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1',                                      'mate'],
  ['queen mate',            '3k4/8/3K4/8/8/8/8/7Q w - - 0 1',                                         'mate'],
  ['rook ladder',           '7k/8/8/8/8/8/5R2/6RK w - - 0 1',                                         'mate'],
  ['bishop + rook net',     'r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1',                                  'mate'],
  ['knight fork mate',      'r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 0 1',      'mate'],
  ['two-rook ladder',       '7k/8/8/8/8/8/R7/1R5K w - - 0 1',                                         'mate'],
  ['scholar\'s mate',       'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1',     'mate'],

  ['WAC.001 deflection',    '2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1',            'best', ['g3g6']],
  ['win the hanging rook',  '4k3/8/8/8/8/8/4r3/4K2R w K - 0 1',                                       'best', ['e1e2']],
  ['knight forks rook',     '4k3/8/8/3r4/8/2N5/8/4K3 w - - 0 1',                                      'best', ['c3d5']],
  ['promote the passer',    '8/P6k/8/8/8/8/6K1/8 w - - 0 1',                                          'best', ['a7a8q']],
  ['take the loose rook',   '4k3/8/8/8/8/7r/8/4K2R w K - 0 1',                                        'best', ['h1h3']],

  ['K+P endgame is won',    '7k/5K2/6P1/8/8/8/8/8 w - - 0 1',                                         'score', 300],
  ['a rook up is winning',  '4k3/8/8/8/8/8/8/R3K3 w Q - 0 1',                                         'score', 300],
  ['bare kings are drawn',  '4k3/8/8/8/8/8/8/4K3 w - - 0 1',                                          'score', -10, 10],
];

// ── Independent mate solver ────────────────────────────────────────────────
// Exhaustive, no heuristics: "can the side to move force mate within N plies?"

function forcedMate(pos, plies) {
  if (plies <= 0) return false;
  const base = pos.ply * 256;
  const n = pos.generate(pos.ply, false);
  for (let i = 0; i < n; i++) {
    const m = pos.moveBuf[base + i];
    if (!pos.makeMove(m)) continue;
    const ok = allRepliesLose(pos, plies - 1);
    pos.unmakeMove();
    if (ok) return true;
  }
  return false;
}

// Opponent to move: true when they are mated now, or every legal reply still
// lets us force mate in the remaining plies.
function allRepliesLose(pos, plies) {
  const base = pos.ply * 256;
  const n = pos.generate(pos.ply, false);
  let anyLegal = false;
  for (let i = 0; i < n; i++) {
    const m = pos.moveBuf[base + i];
    if (!pos.makeMove(m)) continue;
    anyLegal = true;
    const ok = forcedMate(pos, plies - 1);
    pos.unmakeMove();
    if (!ok) return false;
  }
  if (!anyLegal) return pos.inCheck();     // checkmate, not stalemate
  return true;
}

/** Shortest forced mate in plies, or -1 if none within maxPlies. */
function shortestMate(fen, maxPlies = 5) {
  for (let plies = 1; plies <= maxPlies; plies += 2) {
    const pos = new Position().setFromFEN(fen);
    if (forcedMate(pos, plies)) return plies;
  }
  return -1;
}

/** Does playing `uci` still force mate in `plies - 1` for the opponent? */
function moveStillMates(fen, uci, plies) {
  const pos = new Position().setFromFEN(fen);
  const base = pos.ply * 256;
  const n = pos.generate(pos.ply, false);
  for (let i = 0; i < n; i++) {
    const m = pos.moveBuf[base + i];
    if (moveToString(m) !== uci) continue;
    if (!pos.makeMove(m)) return false;
    const ok = allRepliesLose(pos, plies - 1);
    pos.unmakeMove();
    return ok;
  }
  return false;
}

// ── Run ────────────────────────────────────────────────────────────────────
console.log(`Tactical suite — ${MS}ms per position, NN ${USE_NN ? 'on' : 'off'}`);
if (USE_NN) {
  loadModelFromDisk();
  console.log(`  neural net: ${nnReady() ? 'loaded' : 'NOT AVAILABLE'}`);
}
console.log('='.repeat(94));

let passed = 0;
let totalNodes = 0, totalMs = 0;

for (const entry of SUITE) {
  const [name, fen, kind] = entry;
  resetEngine();
  const pos = new Position().setFromFEN(fen);
  const t0 = Date.now();
  const m = searchPosition(pos, MS, USE_NN);
  const ms = Date.now() - t0;
  const got = moveToString(m);
  totalNodes += searchInfo.nodes;
  totalMs += ms;

  const score = searchInfo.score;
  const mateScore = Math.abs(score) > 31000;
  const enginePlies = mateScore ? 32000 - Math.abs(score) : -1;
  const shown = mateScore ? `#${Math.ceil(enginePlies / 2) * Math.sign(score)}` : (score / 100).toFixed(2);

  let ok, detail;
  if (kind === 'mate') {
    const truePlies = shortestMate(fen, 5);
    if (truePlies < 0) {
      ok = false; detail = 'reference solver found no mate within 3 moves — bad test position';
    } else {
      const rightLength = enginePlies === truePlies;
      const reallyMates = moveStillMates(fen, got, truePlies);
      ok = rightLength && reallyMates;
      detail = `true #${Math.ceil(truePlies / 2)}` +
               (rightLength ? '' : ` but engine claims ${shown}`) +
               (reallyMates ? '' : ' — played move does not force it');
    }
  } else if (kind === 'best') {
    ok = entry[3].includes(got);
    detail = `want ${entry[3].join('/')}`;
  } else {
    const lo = entry[3], hi = entry.length > 4 ? entry[4] : Infinity;
    ok = score >= lo && score <= hi;
    detail = hi === Infinity ? `want >= ${lo}cp` : `want ${lo}..${hi}cp`;
  }

  if (ok) passed++;
  console.log(
    `${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(24)} ${got.padEnd(6)} ${String(shown).padStart(7)}  ` +
    `d${String(searchInfo.depth).padStart(2)}/${String(searchInfo.seldepth).padStart(2)} ` +
    `${fmt(searchInfo.nodes).padStart(10)}n ${String(ms).padStart(5)}ms  ${detail}`
  );
  if (!ok) console.log(`      pv: ${searchInfo.pv.join(' ')}`);
}

console.log('='.repeat(94));
const nps = totalMs > 0 ? Math.round(totalNodes / (totalMs / 1000)) : 0;
console.log(`Passed ${passed}/${SUITE.length}   ${fmt(totalNodes)} nodes   ${fmt(nps)} nps`);
process.exit(passed === SUITE.length ? 0 : 1);
