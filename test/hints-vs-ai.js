'use strict';
// Does following Help mode's #1 suggestion beat the engine?
//
// The "player" side always plays suggestion #1; the other side is the engine.
// Each side has its own engine instance — in the app they are separate
// workers — so neither sees the other's transposition table.
//
//   --mode new   the app today: a continuous analysis that keeps its memory
//                between moves, analyses for the player's think time, and
//                ponders during the engine's turn on the reply it expects
//   --mode old   the previous app: a fresh timed search per turn
//                (searchTopMoves), best move given the engine's think time
//
// The player "thinks" for as long as the engine does (--ai ms), so the two
// modes are compared at equal thinking time for the player. The pondering in
// "new" runs for exactly as long as the engine really spent on its move —
// what the two workers do side by side in the app.
//
// Usage: node test/hints-vs-ai.js --mode new --games 10 --ai 800 [--seed N]
//        [--player ms]     player think time (default: same as the engine)
//        [--hint-nn off]   measure the player's analysis without the network

import { initState, makeMove, getLegalMoves, getGameStatus, positionKey, opposite } from '../js/chess.js';
import { loadModelFromDisk } from './harness.js';

const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i > 0 ? process.argv[i + 1] : d; };
const GAMES = +arg('games', 10), AI_MS = +arg('ai', 800), SEED = +arg('seed', 1234);
const MODE = arg('mode', 'new');
const PLAYER_MS = +arg('player', AI_MS);                 // how long the player thinks
const HINT_NN = arg('hint-nn', 'on') !== 'off';          // network in the player's analysis
const STEP_MS = 150;

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

// Deepen an analysis in the worker's step size for `ms`.
function analyseFor(a, ms) {
  const t0 = Date.now();
  while (!a.done && Date.now() - t0 < ms) hintEng.analyseStep(a, Math.min(STEP_MS, ms - (Date.now() - t0)));
}

// What the worker does when the player moves (js/worker.js startPonder).
function ponderTarget(a, played, stAfter, keys) {
  let guess = a ? hintEng.expectedReply(a, played) : null;
  if (!guess) {
    const probe = hintEng.createAnalysis(stAfter, 1, HINT_NN, keys);
    const t0 = Date.now();
    while (!probe.done && probe.depth < 8 && Date.now() - t0 < 400) hintEng.analyseStep(probe, 100);
    if (!probe.lines[0]) return null;
    const next = makeMove(probe.state, probe.lines[0].move);
    next._sideToMove = opposite(probe.side);
    guess = { state: next, reply: probe.lines[0].move };
  }
  const side = guess.state._sideToMove;
  return hintEng.createAnalysis(guess.state, 3, HINT_NN, keys.concat(hintEng.zobristOf(guess.state, side)));
}

let hw = 0, d = 0, aw = 0;
let turns = 0, hits = 0, depthHint = 0, depthAI = 0, aiTurns = 0;
let prevOpening = null;

for (let g = 0; g < GAMES; g++) {
  const humanWhite = g % 2 === 0;
  const st0 = humanWhite ? (prevOpening = opening()) : prevOpening;
  aiEng.resetEngine(); hintEng.resetEngine();
  let st = JSON.parse(JSON.stringify(st0)), side = 'white';
  const counts = new Map(), keys = [];
  const rec = () => { const k = positionKey(st, side); counts.set(k, (counts.get(k) || 0) + 1); keys.push(...aiEng.zobristOf(st, side)); };
  rec();

  let analysis = null;       // the hint side's current analysis ("new" mode)
  let result = 'move limit', r = 0;

  for (let ply = 0; ply < 300; ply++) {
    const stat = getGameStatus(st, side, counts.get(positionKey(st, side)) || 1);
    if (stat.over) { result = stat.reason; r = stat.result === 'draw' ? 0 : (stat.result === 'white_wins' ? 1 : -1); break; }
    st._sideToMove = side;
    let mv;

    if ((side === 'white') === humanWhite) {
      turns++;
      if (MODE === 'old') {
        const hints = hintEng.searchTopMoves(st, PLAYER_MS, 3, HINT_NN, { history: keys });
        mv = hints[0].move;
        depthHint += hints[0].depth;
      } else {
        if (analysis && hintEng.analysisMatches(analysis, st)) hits++;
        else analysis = hintEng.createAnalysis(st, 3, HINT_NN, keys);
        analyseFor(analysis, PLAYER_MS);
        mv = analysis.lines[0].move;
        depthHint += analysis.depth;
      }
    } else {
      const t0 = Date.now();
      mv = aiEng.searchBestMove(st, AI_MS, true, { history: keys });
      const spent = Date.now() - t0;
      aiTurns++; depthAI += aiEng.searchInfo.depth;
      // Pondering happened in parallel with that search in the app.
      if (MODE === 'new' && analysis) analyseFor(analysis, spent);
    }

    const mover = side;
    const before = st;
    st = makeMove(st, mv); side = opposite(side); st._sideToMove = side; rec();

    // The player just moved: point the analysis at the expected reply.
    if (MODE === 'new' && (mover === 'white') === humanWhite) {
      analysis = getGameStatus(st, side, 1).over ? null : ponderTarget(analysis, mv, st, keys);
    }
    void before;
  }
  const humanScore = r === 0 ? 0 : ((r === 1) === humanWhite ? 1 : -1);
  if (humanScore > 0) hw++; else if (humanScore < 0) aw++; else d++;
  console.log(`game ${g + 1}: hint-follower as ${humanWhite ? 'White' : 'Black'} — ${humanScore > 0 ? 'WINS' : humanScore < 0 ? 'loses' : 'draw'} (${result})   running +${hw} =${d} -${aw}`);
}

console.log(`\n[${MODE}] hint-follower: ${hw} wins, ${d} draws, ${aw} losses   (engine ${AI_MS}ms, player thinks ${PLAYER_MS}ms, player NN ${HINT_NN ? 'on' : 'off'})`);
console.log(`avg depth when the player moved: ${(depthHint / turns).toFixed(1)}   engine: ${(depthAI / aiTurns).toFixed(1)}`);
if (MODE === 'new') console.log(`ponder hits (engine played the expected reply): ${hits}/${turns} (${(100 * hits / turns).toFixed(0)}%)`);
