'use strict';
// Search worker — runs the engine off the main thread so the board stays
// responsive and the spinner keeps animating while the AI thinks.
//
// The page runs two of these. One plays the engine's moves ('search'). The
// other serves Help mode ('analyse' / 'ponder' / 'stop'): a continuous
// analysis that keeps deepening for as long as it is wanted.
//
// This is only possible because inference is pure JavaScript (js/neural.js).
// A CDN-loaded TensorFlow.js could not be imported here, which is why earlier
// versions had to block the UI thread instead.

import {
  searchBestMove, resetEngine, searchInfo,
  createAnalysis, analyseStep, analysisMatches, expectedReply, zobristOf,
} from './engine.js';
import { makeMove } from './chess.js';
import { loadModel, isReady as nnIsReady } from './neural.js';
import { APP_VERSION } from './version.js';

// Announce which build this is before anything else, so the page can refuse a
// worker from a different release instead of waiting on it forever.
self.postMessage({ type: 'hello', version: APP_VERSION });

// Load the model once, up front. The main thread waits for the 'ready' message
// before enabling neural evaluation.
loadModel().then(() => {
  self.postMessage({ type: 'ready', nn: nnIsReady() });
});

// ── Help mode analysis ─────────────────────────────────────────────────────
//
// The analysis runs in short steps with a yield between them. A search is
// synchronous, so without the yields a new request could not even be read
// until the search finished; with them, the worker notices within one step
// and switches — keeping its transposition table, which the page used to lose
// by terminating the worker whenever a request went stale.
//
// Three phases, all on the same analysis machinery:
//   analyse  the player's turn: deepen, posting each new depth
//   ponder   the engine's turn: deepen the position after the engine's
//            expected reply, silently, so it is ready when the reply comes
//   stop     leave Help mode
const STEP_MS      = 150;     // work between yields; bounds switching latency
const MIN_SHOW     = 6;       // depths below this are noise, not worth showing
const CAP_MS       = 60000;   // stop analysing one position after this long

let job = null;               // { a, id, ponder, running }

// Zero-delay yield. setTimeout(0) is clamped to 4ms after a few nestings,
// which at 150ms steps would waste a few percent for nothing.
const channel = typeof MessageChannel !== 'undefined' ? new MessageChannel() : null;
function yieldToEvents() {
  if (!channel) return new Promise(r => setTimeout(r, 0));
  return new Promise(r => { channel.port1.onmessage = () => r(); channel.port2.postMessage(0); });
}

function post(j, done) {
  const a = j.a;
  if (j.ponder || a.depth < MIN_SHOW && !done && !a.done) return;
  self.postMessage({
    type: 'analysis', id: j.id,
    lines: a.lines.map(l => ({ move: l.move, score: l.score, mateIn: l.mateIn })),
    depth: a.depth, nodes: a.nodes, done: done || a.done,
  });
}

async function run(j) {
  if (j.running) return;
  j.running = true;
  try {
    while (job === j && !j.a.done && j.a.elapsed < CAP_MS) {
      if (analyseStep(j.a, STEP_MS)) post(j, false);
      await yieldToEvents();
    }
    if (job === j) post(j, true);
  } catch (err) {
    failed(j.id, err);
  } finally {
    j.running = false;
  }
}

// Tell the page, so it can fall back to analysing on its own thread rather
// than showing "Finding your best move" forever.
function failed(id, err) {
  if (job && job.id === id) job = null;
  self.postMessage({ type: 'analysis-error', id, message: String((err && err.message) || err) });
}

function startAnalysis(msg) {
  // Already on it — typically a ponder hit: the engine played the reply we
  // expected, so the analysis has been deepening this exact position all
  // through the engine's turn. Hand over what it has at once and carry on.
  if (job && analysisMatches(job.a, msg.state)) {
    job.id = msg.id;
    job.ponder = false;
    if (job.a.lines.length) post(job, job.a.done);
    run(job);
    return;
  }
  job = { a: createAnalysis(msg.state, msg.count, msg.useNN && nnIsReady(), msg.history),
          id: msg.id, ponder: false, running: false };
  run(job);
}

// The player has moved; the engine is thinking. Work out the reply the engine
// most likely plays and analyse the position after it.
function startPonder(msg) {
  const prev = job;
  job = null;                                  // stop whatever was running
  let guess = prev && prev.a ? expectedReply(prev.a, msg.played) : null;

  if (!guess) {
    // The player left the analysed lines. Find the engine's likely reply with
    // a short search of its own position, then ponder after that.
    const probe = createAnalysis(msg.state, 1, msg.useNN && nnIsReady(), msg.history);
    const t0 = Date.now();
    while (!probe.done && probe.depth < 8 && Date.now() - t0 < 400) analyseStep(probe, 100);
    const reply = probe.lines[0] && probe.lines[0].move;
    if (!reply) return;
    const next = makeMove(probe.state, reply);
    next._sideToMove = probe.side === 'white' ? 'black' : 'white';
    guess = { state: next, reply };
  }

  const side = guess.state._sideToMove;
  const history = (msg.history || []).concat(zobristOf(guess.state, side));
  job = { a: createAnalysis(guess.state, msg.count, msg.useNN && nnIsReady(), history),
          id: 0, ponder: true, running: false };
  run(job);
}

self.onmessage = (e) => {
  const msg = e.data;
  if (!msg) return;

  if (msg.type === 'reset')   { job = null; resetEngine(); return; }
  if (msg.type === 'stop')    { job = null; return; }
  if (msg.type === 'analyse') {
    try { startAnalysis(msg); } catch (err) { failed(msg.id, err); }
    return;
  }
  if (msg.type === 'ponder') {
    // A failed ponder costs nothing but the head start; the next 'analyse'
    // simply starts from scratch.
    try { startPonder(msg); } catch (_) { job = null; }
    return;
  }
  if (msg.type !== 'search') return;

  try {
    const move = searchBestMove(msg.state, msg.timeLimit, msg.useNN && nnIsReady(), {
      history: msg.history,
    });
    self.postMessage({
      type: 'result', id: msg.id, move,
      info: { depth: searchInfo.depth, score: searchInfo.score, nodes: searchInfo.nodes },
    });
  } catch (err) {
    self.postMessage({
      type: 'result', id: msg.id, move: null,
      error: String((err && err.message) || err),
    });
  }
};
