'use strict';
// Neural network wrapper using TensorFlow.js
// Works in both browser main thread and Web Workers

// boardToVector: convert 2D board to 768-element Float32Array
export function boardToVector(board) {
  const vec = new Float32Array(768);
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const piece = board[r][c];
      if (piece !== 0) {
        const sq = r * 8 + c;
        const idx = piece > 0 ? piece - 1 : 6 + (-piece) - 1;
        vec[idx * 64 + sq] = 1;
      }
    }
  }
  return vec;
}

// Normalize binary 0/1 vector to -1/+1 (matches MATLAB's mapminmax with [0,1] range)
function normalizeInput(vec) {
  return vec.map(v => v * 2 - 1);
}

// Neural network state
let _tf = null;
let _model = null;
let _normParams = null;
let _ready = false;
let _loadPromise = null;

export function isReady() {
  return _ready;
}

// Attempt to load TF.js and model. Safe to call multiple times.
export async function loadModel(tfLib) {
  if (_loadPromise) return _loadPromise;
  _loadPromise = _doLoad(tfLib);
  return _loadPromise;
}

async function _doLoad(tfLib) {
  try {
    _tf = tfLib || (typeof tf !== 'undefined' ? tf : null);
    if (!_tf) return;

    // Try to load normalization params
    try {
      const r = await fetch('model/normalization.json');
      if (r.ok) _normParams = await r.json();
    } catch (_) {}

    _model = await _tf.loadLayersModel('model/model.json');
    _ready = true;
  } catch (e) {
    // Model not found — NN disabled, engine falls back to material eval
    _ready = false;
  }
}

// Synchronous inference — returns centipawn score from White's perspective
// Returns null if model not ready
export function evaluate(board) {
  if (!_ready || !_model || !_tf) return null;

  return _tf.tidy(() => {
    const raw = boardToVector(board);
    const normalized = normalizeInput(raw);
    const input = _tf.tensor2d([normalized], [1, 768]);
    const rawOut = _model.predict(input).dataSync()[0];

    // Denormalize output if normalization params are available
    if (_normParams) {
      const { yMin, yMax } = _normParams;
      return (rawOut + 1) * (yMax - yMin) / 2 + yMin;
    }
    return rawOut;
  });
}
