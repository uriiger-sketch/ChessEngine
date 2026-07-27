'use strict';
// Pure-JavaScript neural-network inference — no TensorFlow.js, no CDN.
//
// Why not TF.js: the PWA must play offline, but a service worker cannot cache
// cross-origin CDN scripts, so a CDN-hosted TF.js can never be available
// offline. The trained model is a plain 5-layer MLP
// (768→256→128→64→32→1, ReLU hidden / linear output) — a handful of matrix
// multiplies, far faster run directly than through TF.js, whose per-call
// tensor allocation dominates for a model this small.
//
// Fast path: inputs are sparse 0/1 over piece-squares, so at most 32 of the 768
// entries are set and layer 1 is just "bias plus the rows of the kernel for the
// occupied squares" — ~32 row additions instead of 768×256 multiply-adds.
// train/train.js uses the identical 0/1 encoding.

const IN_DIM = 768;

let _ready       = false;
let _loadPromise = null;
let _lastError   = null;

let _l1W    = null;   // layer-1 kernel, row-major [768 × H1]
let _l1Bias = null;   // layer-1 bias
let _h1     = 0;      // layer-1 output width
let _rest   = [];     // layers 2..N: {W, b, inDim, outDim, relu}
let _yScale = 39;     // model output × _yScale → pawn units

let _bufA = null, _bufB = null;

export function isReady()   { return _ready; }
export function lastError() { return _lastError; }

// Piece code → one of 12 planes: White P..K = 0..5, Black p..k = 6..11
function planeIndex(piece) {
  return piece > 0 ? piece - 1 : 5 + (-piece);
}

// Raw 768-element encoding, kept for callers that want the input vector.
export function boardToVector(board) {
  const vec = new Float32Array(IN_DIM);
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p !== 0) vec[planeIndex(p) * 64 + r * 8 + c] = 1;
    }
  }
  return vec;
}

// ── Loading ────────────────────────────────────────────────────────────────
export async function loadModel() {
  if (_loadPromise) return _loadPromise;
  _loadPromise = _doLoad();
  return _loadPromise;
}

// Resolve model files relative to this module rather than the page, so the
// same code works on the main thread and inside a Web Worker.
function modelUrl(file) {
  try {
    return new URL('../model/' + file, import.meta.url).href;
  } catch (_) {
    return 'model/' + file;
  }
}

async function _doLoad() {
  try {
    const mRes = await fetch(modelUrl('model.json'));
    if (!mRes.ok) throw new Error(`model.json → HTTP ${mRes.status}`);
    const modelJson = await mRes.json();

    const group = modelJson.weightsManifest && modelJson.weightsManifest[0];
    if (!group) throw new Error('model.json has no weightsManifest');

    const binRes = await fetch(modelUrl(group.paths[0]));
    if (!binRes.ok) throw new Error(`${group.paths[0]} → HTTP ${binRes.status}`);
    const buf = await binRes.arrayBuffer();

    // weights.bin is the manifest tensors concatenated, all float32.
    const tensors = {};
    let off = 0;
    for (const spec of group.weights) {
      if (spec.dtype !== 'float32') throw new Error(`unsupported dtype ${spec.dtype}`);
      const n = spec.shape.reduce((a, b) => a * b, 1);
      if (off + n * 4 > buf.byteLength) throw new Error('weights.bin is truncated');
      // Copy rather than view: offset views into a shared buffer measurably
      // slow down the hot inference loops.
      tensors[spec.name] = new Float32Array(new Float32Array(buf, off, n));
      off += n * 4;
    }

    // Dense layer order + activations come from the topology.
    const dense = [];
    collectDense(modelJson.modelTopology, dense);
    if (dense.length === 0) throw new Error('no Dense layers in topology');

    const built = dense.map(d => {
      const W = tensors[`${d.name}/kernel`];
      const b = tensors[`${d.name}/bias`];
      if (!W || !b) throw new Error(`missing weights for ${d.name}`);
      return { W, b, outDim: b.length, inDim: W.length / b.length, relu: d.activation === 'relu' };
    });

    if (built[0].inDim !== IN_DIM) {
      throw new Error(`input dim ${built[0].inDim}, expected ${IN_DIM}`);
    }

    const l0 = built[0];
    _h1     = l0.outDim;
    _l1W    = l0.W;
    _l1Bias = l0.b;
    _rest   = built.slice(1);

    let widest = _h1;
    for (const l of _rest) widest = Math.max(widest, l.outDim);
    _bufA = new Float32Array(widest);
    _bufB = new Float32Array(widest);

    // Output denormalisation — training scaled labels by 1/yMax.
    try {
      const nRes = await fetch(modelUrl('normalization.json'));
      if (nRes.ok) {
        const n = await nRes.json();
        if (typeof n.yMax === 'number' && typeof n.yMin === 'number') {
          _yScale = (n.yMax - n.yMin) / 2;
        }
      }
    } catch (_) { /* keep default scale */ }

    _ready     = true;
    _lastError = null;
  } catch (e) {
    _ready     = false;
    _lastError = e.message || String(e);
  }
}

function collectDense(node, out) {
  if (Array.isArray(node)) { for (const v of node) collectDense(v, out); return; }
  if (!node || typeof node !== 'object') return;
  if (node.class_name === 'Dense' && node.config && node.config.name) {
    out.push({ name: node.config.name, activation: node.config.activation });
    return;                        // a Dense config contains no nested layers
  }
  for (const k in node) collectDense(node[k], out);
}

// ── Inference ──────────────────────────────────────────────────────────────
// Returns the score in PAWN units from White's perspective (+ = White better),
// or null when no model is loaded.
export function evaluate(board) {
  if (!_ready) return null;

  // Layer 1 via sparse row accumulation.
  let cur = _bufA, next = _bufB;
  cur.set(_l1Bias);
  for (let r = 0; r < 8; r++) {
    const rowArr = board[r];
    for (let c = 0; c < 8; c++) {
      const p = rowArr[c];
      if (p === 0) continue;
      const base = (planeIndex(p) * 64 + r * 8 + c) * _h1;
      for (let j = 0; j < _h1; j++) cur[j] += _l1W[base + j];
    }
  }
  for (let j = 0; j < _h1; j++) if (cur[j] < 0) cur[j] = 0;   // ReLU

  // Remaining dense layers. Input-major so ReLU zeros skip whole rows.
  let inDim = _h1;
  for (let li = 0; li < _rest.length; li++) {
    const L = _rest[li], W = L.W, outDim = L.outDim;
    for (let j = 0; j < outDim; j++) next[j] = L.b[j];
    for (let i = 0; i < inDim; i++) {
      const xi = cur[i];
      if (xi === 0) continue;
      const base = i * outDim;
      for (let j = 0; j < outDim; j++) next[j] += xi * W[base + j];
    }
    if (L.relu) for (let j = 0; j < outDim; j++) if (next[j] < 0) next[j] = 0;
    const t = cur; cur = next; next = t;
    inDim = outDim;
  }

  return cur[0] * _yScale;
}
