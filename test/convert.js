'use strict';
// Can the engine actually WIN a won game? Basic mates — K+Q, K+R, K+2B and
// friends against a lone king — need the defending king driven to the edge,
// which takes longer than any search horizon. The engine plays both sides and
// must deliver mate before the fifty-move rule or a repetition ends it.
//
// Before the mop-up term in js/evaluate.js, K+R v K from far apart and
// K+Q v K+R were drawn by the fifty-move rule at 80ms a move (phone speed).
//
// Usage: node test/convert.js [--ms 300]

import { Position } from '../js/position.js';
import { makeMove, getGameStatus, positionKey, opposite } from '../js/chess.js';
import { searchBestMove, resetEngine, zobristOf } from '../js/engine.js';
import { loadModelFromDisk } from './harness.js';

const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i > 0 ? process.argv[i + 1] : d; };
const MS = +arg('ms', 300);
loadModelFromDisk();

const CASES = [
  ['K+Q v K, king in centre',   '8/8/8/4k3/8/8/8/3QK3 w - - 0 1'],
  ['K+Q v K, far apart',        '7k/8/8/8/8/8/8/K2Q4 w - - 0 1'],
  ['K+R v K, king in centre',   '8/8/8/4k3/8/8/8/R3K3 w - - 0 1'],
  ['K+R v K, far apart',        '8/8/8/3k4/8/8/8/K6R w - - 0 1'],
  ['K+2B v K',                  '8/8/8/4k3/8/8/8/2B1KB2 w - - 0 1'],
  ['K+R+P v K+R (up a pawn+)',  '8/8/4k3/8/3PK3/8/8/R6r w - - 0 1'],
  // Known hard: the rook defends with endless checks and the winning technique
  // lies beyond a phone-speed search. Reported, but not required to pass.
  ['K+Q v K+R (known hard)',    '8/8/3rk3/8/8/3QK3/8/8 w - - 0 1', 'hard'],
];

let failures = 0;
for (const [name, fen, hard] of CASES) {
  resetEngine();
  const { state, side: s0 } = new Position().setFromFEN(fen).toUIState();
  let st = state, side = s0;
  const counts = new Map(), keys = [];
  const rec = () => { const k = positionKey(st, side); counts.set(k, (counts.get(k) || 0) + 1); keys.push(...zobristOf(st, side)); };
  rec();
  let moves = 0, end = 'move limit';
  for (let ply = 0; ply < 200; ply++) {
    const stat = getGameStatus(st, side, counts.get(positionKey(st, side)) || 1);
    if (stat.over) { end = stat.result === 'white_wins' ? 'MATE' : stat.reason; break; }
    st._sideToMove = side;
    const mv = searchBestMove(st, MS, true, { history: keys });
    st = makeMove(st, mv); side = opposite(side); st._sideToMove = side; rec();
    if (side === 'white') moves++;
  }
  const ok = end === 'MATE';
  if (!ok && !hard) failures++;
  console.log(`${ok ? 'PASS' : hard ? 'INFO' : 'FAIL'}  ${name.padEnd(28)} ${ok ? `mated in ${moves} moves` : `ended: ${end} after ${moves} moves`}`);
}
const required = CASES.filter(c => !c[2]).length;
console.log(failures ? `\n${failures} of ${required} required won endgames NOT won.`
                     : `\nAll ${required} required won endgames converted (known-hard cases reported above).`);
process.exit(failures ? 1 : 0);
