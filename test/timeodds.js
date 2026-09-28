'use strict';
// How much does thinking time matter? The engine plays itself, one side given
// `--mult` times the other's time. Separate engine instances, colours swapped
// every game, same random openings as the other harnesses.
//
// Usage: node test/timeodds.js --games 10 --ms 100 --mult 15 [--seed N] [--contempt cp]

import { initState, makeMove, getLegalMoves, getGameStatus, positionKey, opposite } from '../js/chess.js';
import { loadModelFromDisk } from './harness.js';

const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i > 0 ? process.argv[i + 1] : d; };
const GAMES = +arg('games', 10), MS = +arg('ms', 100), MULT = +arg('mult', 15), SEED = +arg('seed', 77);
const CONTEMPT = +arg('contempt', 0);   // draws count this much against the longer thinker

const fast = await import('../js/engine.js?fast');
const slow = await import('../js/engine.js?slow');
loadModelFromDisk();

let s = SEED >>> 0;
const rand = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
function opening() {
  let st = initState(), side = 'white';
  for (let i = 0; i < 4; i++) { const m = getLegalMoves(st, side); st = makeMove(st, m[Math.floor(rand() * m.length)]); side = opposite(side); }
  return st;
}

let w = 0, d = 0, l = 0, dFast = 0, nFast = 0, dSlow = 0, nSlow = 0, open = null;
for (let g = 0; g < GAMES; g++) {
  const slowWhite = g % 2 === 0;
  if (slowWhite) open = opening();
  fast.resetEngine(); slow.resetEngine();
  let st = JSON.parse(JSON.stringify(open)), side = 'white';
  const counts = new Map(), keys = [];
  const rec = () => { const k = positionKey(st, side); counts.set(k, (counts.get(k) || 0) + 1); keys.push(...fast.zobristOf(st, side)); };
  rec();
  let r = 0, why = 'move limit';
  for (let ply = 0; ply < 300; ply++) {
    const stat = getGameStatus(st, side, counts.get(positionKey(st, side)) || 1);
    if (stat.over) { why = stat.reason; r = stat.result === 'draw' ? 0 : stat.result === 'white_wins' ? 1 : -1; break; }
    st._sideToMove = side;
    const useSlow = (side === 'white') === slowWhite;
    const eng = useSlow ? slow : fast;
    const mv = eng.searchBestMove(st, useSlow ? MS * MULT : MS, true, { history: keys, contempt: useSlow ? CONTEMPT : 0 });
    if (useSlow) { dSlow += eng.searchInfo.depth; nSlow++; } else { dFast += eng.searchInfo.depth; nFast++; }
    st = makeMove(st, mv); side = opposite(side); st._sideToMove = side; rec();
  }
  const sc = r === 0 ? 0 : ((r === 1) === slowWhite ? 1 : -1);
  if (sc > 0) w++; else if (sc < 0) l++; else d++;
  console.log(`game ${g + 1}: ${MULT}x side as ${slowWhite ? 'White' : 'Black'} — ${sc > 0 ? 'wins' : sc < 0 ? 'LOSES' : 'draw'} (${why})   running +${w} =${d} -${l}`);
}
console.log(`\n${MULT}x time (contempt ${CONTEMPT}) vs 1x: ${w} wins, ${d} draws, ${l} losses   depth ${(dSlow / nSlow).toFixed(1)} vs ${(dFast / nFast).toFixed(1)}`);
