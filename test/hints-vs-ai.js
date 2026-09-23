'use strict';
// Does following Help mode's #1 suggestion beat the engine?
//
// Plays the app's exact procedure: the "human" side always plays hint #1 from
// searchTopMoves, the AI side plays searchBestMove. The two sides get separate
// engine instances — as in the app, where hints run in their own worker — so
// neither sees the other's transposition table.
//
// Usage: node test/hints-vs-ai.js --games 10 --ai 800 [--seed N]
//
// The hint's best-move search gets the same time as the AI's move, which is
// what the app does. (Before that was fixed, hints had a fixed 1.5s split
// three ways and re-sorted by score; this harness is how that was measured.)

import { initState, makeMove, getLegalMoves, getGameStatus, positionKey, opposite } from '../js/chess.js';
import { loadModelFromDisk, uiMoveToString } from './harness.js';

const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i > 0 ? process.argv[i + 1] : d; };
const GAMES = +arg('games', 10), AI_MS = +arg('ai', 800), HINT_MS = +arg('hint', AI_MS);
const SEED = +arg('seed', 1234);

// Two independent engine instances: a query string makes ESM load a fresh copy.
const aiEng   = await import('../js/engine.js?ai');
const hintEng = await import('../js/engine.js?hint');
loadModelFromDisk();

let s = SEED >>> 0;
const rand = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
function opening() {
  let st = initState(), side = 'white';
  for (let i = 0; i < 4; i++) { const m = getLegalMoves(st, side); st = makeMove(st, m[Math.floor(rand() * m.length)]); side = opposite(side); }
  return st;
}

let hw = 0, d = 0, aw = 0;
let hintTurns = 0, reordered = 0, depthHint = 0, depthAI = 0, aiTurns = 0;

for (let g = 0; g < GAMES; g++) {
  const humanWhite = g % 2 === 0;
  const st0 = g % 2 === 0 ? opening() : st0Prev;
  var st0Prev = st0;
  aiEng.resetEngine(); hintEng.resetEngine();
  let st = JSON.parse(JSON.stringify(st0)), side = 'white';
  const counts = new Map(), keys = [];
  const rec = () => { const k = positionKey(st, side); counts.set(k, (counts.get(k) || 0) + 1); keys.push(...aiEng.zobristOf(st, side)); };
  rec();
  let result = 'move limit', r = 0;
  for (let ply = 0; ply < 300; ply++) {
    const stat = getGameStatus(st, side, counts.get(positionKey(st, side)) || 1);
    if (stat.over) { result = stat.reason; r = stat.result === 'draw' ? 0 : (stat.result === 'white_wins' ? 1 : -1); break; }
    st._sideToMove = side;
    let mv;
    if ((side === 'white') === humanWhite) {
      const hints = hintEng.searchTopMoves(st, HINT_MS, 3, true, { history: keys });
      mv = hints[0].move;
      hintTurns++; depthHint += hints.find(h => h.pass === 0).depth;
      if (hints[0].pass !== 0) reordered++;   // must be 0 now that hints are not re-sorted
    } else {
      mv = aiEng.searchBestMove(st, AI_MS, true, { history: keys });
      aiTurns++; depthAI += aiEng.searchInfo.depth;
    }
    st = makeMove(st, mv); side = opposite(side); st._sideToMove = side; rec();
  }
  const humanScore = r === 0 ? 0 : ((r === 1) === humanWhite ? 1 : -1);
  if (humanScore > 0) hw++; else if (humanScore < 0) aw++; else d++;
  console.log(`game ${g + 1}: hint-follower as ${humanWhite ? 'White' : 'Black'} — ${humanScore > 0 ? 'WINS' : humanScore < 0 ? 'loses' : 'draw'} (${result})   running +${hw} =${d} -${aw}`);
}
console.log(`\nhint-follower: ${hw} wins, ${d} draws, ${aw} losses   (AI ${AI_MS}ms, hint best-move search ${HINT_MS}ms)`);
console.log(`avg depth of the hint's best-move search: ${(depthHint / hintTurns).toFixed(1)}   AI: ${(depthAI / aiTurns).toFixed(1)}`);
console.log(`hint #1 was NOT the full search's choice (sort reordered it): ${reordered}/${hintTurns} turns (${(100 * reordered / hintTurns).toFixed(0)}%)`);
