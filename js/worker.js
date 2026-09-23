'use strict';
// Search worker — runs the engine off the main thread so the board stays
// responsive and the spinner keeps animating while the AI thinks.
//
// This is only possible because inference is pure JavaScript (js/neural.js).
// A CDN-loaded TensorFlow.js could not be imported here, which is why earlier
// versions had to block the UI thread instead.

import { searchBestMove, resetEngine, searchInfo } from './engine.js';
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
