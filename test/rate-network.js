'use strict';
// Rate the network-only opponent in units of engine thinking time.
//
// The network plays exactly as the app ships it — opening book for moves 1–9,
// then looking --look moves ahead — against the full engine at one fixed
// thinking time per run. Games start from the normal position (the book
// supplies the variety) and colours alternate. Run it at several engine times
// and the time at which the two score even is the network's strength,
// expressed as "worth this much engine thinking".
//
// Usage: node test/rate-network.js --ms 40 --games 20 [--look 3] [--seed N]
// Prints one JSON summary line at the end for aggregation.

import fs from 'fs';
import { initState, makeMove, getGameStatus, positionKey, opposite } from '../js/chess.js';
import { loadModelFromDisk } from './harness.js';
import { setBook, bookMove } from '../js/book.js';

const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i > 0 ? process.argv[i + 1] : d; };
const MS = +arg('ms', 40), GAMES = +arg('games', 20), LOOK = +arg('look', 3), SEED = +arg('seed', 1);

// Separate engine instances for the two players, as in the app's two roles.
const engine  = await import('../js/engine.js?engine');
const network = await import('../js/engine.js?network');
loadModelFromDisk();
setBook(JSON.parse(fs.readFileSync(new URL('../model/book.json', import.meta.url), 'utf8')));

// Seeded choice among book moves, so a run can be reproduced.
let s = SEED >>> 0;
const rand = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };

let netScore = 0, w = 0, d = 0, l = 0;
let engDepth = 0, engMoves = 0, netMs = 0, netDepth = 0, netMoves = 0, bookMoves = 0;
const ends = {};

for (let g = 0; g < GAMES; g++) {
  const netWhite = g % 2 === 0;
  engine.resetEngine(); network.resetEngine();
  let st = initState(), side = 'white', ply = 0;
  const counts = new Map(), keys = [];
  const rec = () => { const k = positionKey(st, side); counts.set(k, (counts.get(k) || 0) + 1); keys.push(...engine.zobristOf(st, side)); };
  rec();
  let r = 0, why = 'move limit';
  for (; ply < 300; ply++) {
    const stat = getGameStatus(st, side, counts.get(positionKey(st, side)) || 1);
    if (stat.over) { why = stat.reason; r = stat.result === 'draw' ? 0 : stat.result === 'white_wins' ? 1 : -1; break; }
    st._sideToMove = side;
    let mv;
    if ((side === 'white') === netWhite) {
      mv = bookMove(st, side, ply, rand);
      if (mv) bookMoves++;
      else {
        const t0 = Date.now();
        mv = network.searchNetworkMove(st, { history: keys, plies: 2 * LOOK });
        netMs += Date.now() - t0; netDepth += network.searchInfo.depth; netMoves++;
      }
    } else {
      mv = engine.searchBestMove(st, MS, true, { history: keys });
      engDepth += engine.searchInfo.depth; engMoves++;
    }
    st = makeMove(st, mv); side = opposite(side); st._sideToMove = side; rec();
  }
  const sc = r === 0 ? 0.5 : ((r === 1) === netWhite ? 1 : 0);
  netScore += sc;
  if (sc === 1) w++; else if (sc === 0) l++; else d++;
  ends[why] = (ends[why] || 0) + 1;
  console.log(`game ${String(g + 1).padStart(2)}: network as ${netWhite ? 'White' : 'Black'} — ` +
              `${sc === 1 ? 'network wins' : sc === 0 ? 'engine wins' : 'draw'} (${why}, ${ply} plies)   ` +
              `network +${w} =${d} -${l}`);
}

const summary = {
  engineMs: MS, games: GAMES, look: LOOK, networkWins: w, draws: d, engineWins: l,
  networkScore: netScore / GAMES,
  engineDepth: +(engDepth / engMoves).toFixed(1),
  networkDepth: +(netDepth / netMoves).toFixed(1),
  networkMsPerMove: Math.round(netMs / netMoves),
  bookMovesPerGame: +(bookMoves / GAMES).toFixed(1),
  endings: ends,
};
console.log('\nnetwork (book + ' + LOOK + ' moves) vs engine at ' + MS + 'ms: ' +
            `${w} wins, ${d} draws, ${l} losses — network scores ${(100 * summary.networkScore).toFixed(1)}%`);
console.log('SUMMARY ' + JSON.stringify(summary));
