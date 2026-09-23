'use strict';
// Head-to-head match: the real measure of whether the engine got stronger.
//
// Node counts and tactical puzzles are proxies. Playing games is not. Each pair
// of games starts from the same randomly generated opening with the colours
// swapped, so an unbalanced opening helps both sides exactly once and the
// result reflects play rather than luck of the draw.
//
// Usage:
//   node test/match.js --games 20 --ms 300                  new vs old engine
//   node test/match.js --games 20 --ms 300 --app            as shipped (NN on) vs old
//   node test/match.js --games 20 --ms 300 --mode nn        new+NN vs new without
//
// The Elo figure is the standard logistic conversion of the score rate, with a
// 95% confidence interval — with a few dozen games the interval is wide, and
// reporting it is the honest way to present the result.

import {
  initState, makeMove, getLegalMoves, getGameStatus, positionKey, opposite
} from '../js/chess.js';
import { searchBestMove as searchNew, resetEngine, zobristOf, setNNGain, setNNSplitEval } from '../js/engine.js';
import { searchBestMove as searchOld } from './legacy-engine.js';
import { loadModelFromDisk, nnReady, uiMoveToString } from './harness.js';

const arg = (name, dflt) => {
  const i = process.argv.indexOf('--' + name);
  return i > 0 ? process.argv[i + 1] : dflt;
};

const GAMES    = +arg('games', 20);
const MS       = +arg('ms', 300);
const MODE     = arg('mode', 'old');        // 'old' = vs legacy engine, 'nn' = NN on vs off
const GAIN     = +arg('gain', 1.0);         // strength of the network's correction in 'nn' mode
const SPLIT    = process.argv.includes('--split'); // keep the network out of pruning decisions
const MAX_PLIES = +arg('plies', 300);
const OPEN_PLIES = 4;

const nnLoaded = loadModelFromDisk();
console.log(`Match: ${GAMES} games, ${MS}ms per move, mode "${MODE}"`);
console.log(`Neural net: ${nnReady() ? 'loaded' : 'not available'}`);

// ── Players ────────────────────────────────────────────────────────────────
// Each player is a function (state, side, historyKeys) → chess.js move object.
let playerA, playerB, nameA, nameB;

if (MODE === 'nn') {
  if (!nnLoaded) { console.error('NN mode needs a loadable model.'); process.exit(2); }
  nameA = `new engine + NN (gain ${GAIN}${SPLIT ? ', split eval' : ''})`;
  nameB = 'new engine, hand eval only';
  playerA = (st, side, hist) => { setNNGain(GAIN); setNNSplitEval(SPLIT); return searchNew(st, MS, true, { history: hist }); };
  playerB = (st, side, hist) => { setNNSplitEval(false); return searchNew(st, MS, false, { history: hist }); };
} else {
  // Both sides play without the network here, deliberately. The current model
  // outputs a positional correction to be added to an evaluation, and the old
  // engine has no idea that is what it is — it would treat the correction as
  // the whole score. Running both without it isolates exactly what this
  // comparison is for: the search and the hand evaluation.
  // --app plays the new engine exactly as the app ships it: network on.
  const APP = process.argv.includes('--app');
  nameA = APP ? 'new engine (as shipped, NN on)' : 'new engine (hand eval)';
  nameB = 'old engine (hand eval)';
  playerA = (st, side, hist) => searchNew(st, MS, APP && nnLoaded, { history: hist });
  playerB = (st) => searchOld(st, MS, false);
}

// ── Seeded RNG so a match can be reproduced ────────────────────────────────
function makeRand(seed) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
}
const rand = makeRand(+arg('seed', 20260922));

// Random but legal opening, so the games are not all the same game.
function randomOpening() {
  for (let attempt = 0; attempt < 50; attempt++) {
    let st = initState();
    let side = 'white';
    let ok = true;
    for (let i = 0; i < OPEN_PLIES; i++) {
      const moves = getLegalMoves(st, side);
      if (moves.length === 0) { ok = false; break; }
      st = makeMove(st, moves[Math.floor(rand() * moves.length)]);
      side = opposite(side);
    }
    if (ok) return st;
  }
  return initState();
}

// ── One game ───────────────────────────────────────────────────────────────
// Returns 1 (white wins), 0 (draw), -1 (black wins) plus a reason.
function playGame(openingState, whitePlayer, blackPlayer) {
  let st = JSON.parse(JSON.stringify(openingState));
  let side = 'white';
  st._sideToMove = side;

  const counts = new Map();
  const keys = [];
  const record = () => {
    const k = positionKey(st, side);
    counts.set(k, (counts.get(k) || 0) + 1);
    const [lo, hi] = zobristOf(st, side);
    keys.push(lo, hi);
  };
  record();

  for (let ply = 0; ply < MAX_PLIES; ply++) {
    const rep = counts.get(positionKey(st, side)) || 1;
    const status = getGameStatus(st, side, rep);
    if (status.over) {
      if (status.result === 'white_wins') return { r: 1, why: status.reason };
      if (status.result === 'black_wins') return { r: -1, why: status.reason };
      return { r: 0, why: status.reason };
    }

    const player = side === 'white' ? whitePlayer : blackPlayer;
    st._sideToMove = side;
    const mv = player(st, side, keys);
    if (!mv) return { r: 0, why: 'no move returned' };

    st = makeMove(st, mv);
    side = opposite(side);
    st._sideToMove = side;
    record();
  }
  return { r: 0, why: 'move limit' };
}

// ── Run the match ──────────────────────────────────────────────────────────
let aWins = 0, bWins = 0, draws = 0;
const reasons = new Map();
const t0 = Date.now();

for (let pair = 0; pair < Math.ceil(GAMES / 2); pair++) {
  const opening = randomOpening();

  for (const aIsWhite of [true, false]) {
    if (aWins + bWins + draws >= GAMES) break;
    resetEngine();

    const res = playGame(
      opening,
      aIsWhite ? playerA : playerB,
      aIsWhite ? playerB : playerA
    );

    let outcome;
    if (res.r === 0) { draws++; outcome = 'draw'; }
    else if ((res.r === 1) === aIsWhite) { aWins++; outcome = `${nameA} wins`; }
    else { bWins++; outcome = `${nameB} wins`; }

    reasons.set(res.why, (reasons.get(res.why) || 0) + 1);
    const n = aWins + bWins + draws;
    console.log(
      `  game ${String(n).padStart(3)}: A plays ${aIsWhite ? 'White' : 'Black'} — ` +
      `${outcome.padEnd(30)} (${res.why})   running: +${aWins} =${draws} -${bWins}`
    );
  }
  if (aWins + bWins + draws >= GAMES) break;
}

// ── Result ─────────────────────────────────────────────────────────────────
const n = aWins + bWins + draws;
const score = aWins + draws / 2;
const rate = score / n;

// Elo difference from the score rate, and a 95% interval from the per-game
// variance. Draws count half, so the variance uses the three-outcome spread.
// A clean sweep has no finite Elo estimate; report a lower bound instead of
// printing "Infinity" as though it were a measurement.
const eloOf = (p) => -400 * Math.log10(1 / Math.min(0.9999, Math.max(0.0001, p)) - 1);
const elo = eloOf(rate);
const swept = rate >= 1 || rate <= 0;
const pw = aWins / n, pd = draws / n, pl = bWins / n;
const variance = pw * (1 - rate) ** 2 + pd * (0.5 - rate) ** 2 + pl * (0 - rate) ** 2;
const stderr = Math.sqrt(variance / n);
const lo = Math.max(0.0001, rate - 1.96 * stderr);
const hi = Math.min(0.9999, rate + 1.96 * stderr);
const eloLo = eloOf(lo);
const eloHi = eloOf(hi);

console.log('\n' + '='.repeat(72));
console.log(`${nameA}  vs  ${nameB}`);
console.log(`  ${aWins} wins, ${draws} draws, ${bWins} losses  out of ${n} games`);
console.log(`  score rate ${(rate * 100).toFixed(1)}%`);
console.log(swept
  ? `  Elo difference: ${rate >= 1 ? '>' : '<'} ${Math.abs(elo).toFixed(0)} (clean sweep — only a bound, not an estimate)`
  : `  Elo difference ${elo >= 0 ? '+' : ''}${elo.toFixed(0)}  (95% CI ${eloLo.toFixed(0)} … ${eloHi.toFixed(0)})`);
console.log(`  endings: ${[...reasons].map(([k, v]) => `${k} ×${v}`).join(', ')}`);
console.log(`  wall clock ${((Date.now() - t0) / 1000 / 60).toFixed(1)} min`);
