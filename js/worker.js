'use strict';
// Search worker — runs the engine off the main thread so the board stays
// responsive and the spinner keeps animating while the AI thinks.
//
// This is only possible because inference is pure JavaScript (js/neural.js).
// A CDN-loaded TensorFlow.js could not be imported here, which is why earlier
// versions had to block the UI thread instead.

import { searchBestMove, searchTopMoves, resetEngine, searchInfo } from './engine.js';
import { loadModel, isReady as nnIsReady } from './neural.js';

// Load the model once, up front. The main thread waits for the 'ready' message
// before enabling neural evaluation.
loadModel().then(() => {
  self.postMessage({ type: 'ready', nn: nnIsReady() });
});

self.onmessage = (e) => {
  const msg = e.data;
  if (!msg) return;

  if (msg.type === 'reset') { resetEngine(); return; }

  // Help mode: the player's best few moves. The page runs these in a second
  // worker so a hint search can never delay the engine's own move.
  if (msg.type === 'hints') {
    try {
      // Each suggestion is posted as soon as its search finishes, so the best
      // move appears without waiting for the alternatives.
      const hints = searchTopMoves(msg.state, msg.bestMs, msg.count, msg.useNN && nnIsReady(), {
        history: msg.history,
        restMs: msg.restMs,
        onFound: list => self.postMessage({ type: 'hints', id: msg.id, hints: list, done: false }),
      });
      self.postMessage({ type: 'hints', id: msg.id, hints, done: true });
    } catch (err) {
      self.postMessage({ type: 'hints', id: msg.id, hints: [], done: true,
                         error: String((err && err.message) || err) });
    }
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
