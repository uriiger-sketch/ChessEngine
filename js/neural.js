'use strict';
// Pure-JavaScript neural-network inference — no TensorFlow.js, no CDN.
//
// Why not TF.js: the PWA must play offline, but a service worker cannot cache
// cross-origin CDN scripts, so a CDN-hosted TF.js can never be available
// offline. The trained model is a plain MLP with a wide first layer and narrow
// ones after it — a handful of matrix multiplies, far faster run directly than
// through TF.js, whose per-call tensor allocation dominates for a model this
// small.
//
// Fast path: the input is sparse 0/1 over piece-squares, so at most ~32 of the
// 768 piece entries are set and the first layer is just "bias plus the kernel
// rows for the occupied squares" — ~32 row additions instead of 768×H
// multiply-adds. train/train.js uses the identical encoding.
//
// Two input layouts are supported, decided by the model's own input width:
//   768  piece-squares only (the original model)
//   773  piece-squares plus side-to-move and the four castling rights
// and three output conventions, read from model/normalization.json:
//   residual  the current model: a positional CORRECTION in pawn units, which
//             the engine adds to its own evaluation. Material is deliberately
//             not part of it — see train/train.js for why that matters.
//   logit     an absolute win-probability logit, scaled to pawn units
//   linear    an absolute score, raw output × yScale (the original model)
// so a freshly trained model and an older one both play.

const PIECE_DIM = 768;

// Extra feature offsets, used when the model is wide enough to have them.
const F_STM  = 768;   // 1 when White is to move
const F_WK   = 769, F_WQ = 770, F_BK = 771, F_BQ = 772;

let _ready       = false;
let _loadPromise = null;
let _lastError   = null;

let _l1W    = null;   // first-layer kernel, row-major [inDim × H1]
let _l1Bias = null;
let _h1     = 0;
let _inDim  = PIECE_DIM;
let _rest   = [];     // later layers: {W, b, inDim, outDim, relu}

// Output conversion
let _mode    = 'linear';
let _yScale  = 39;    // linear: output × _yScale = pawn units
let _cpScale = 175;   // logit/residual: output × _cpScale = centipawns
let _clampCp = 200;   // residual: how far the correction may ever move the score

let _bufA = null, _bufB = null;

export function isReady()   { return _ready; }
export function lastError() { return _lastError; }
export function inputDim()  { return _inDim; }
/** True when evaluate() returns a correction to add, not an absolute score. */
export function isResidual() { return _mode === 'residual'; }
/** Largest correction, in centipawns, a residual model is allowed to apply. */
export function clampCp()   { return _clampCp; }

// Piece code → one of 12 planes: White P..K = 0..5, Black p..k = 6..11
function planeIndex(piece) {
  return piece > 0 ? piece - 1 : 5 + (-piece);
}

// Raw dense encoding, for callers that want the input vector (the trainer's
// probes, mainly). `board` is a flat Int8Array(64).
export function boardToVector(board, stm, castling) {
  const vec = new Float32Array(_inDim || PIECE_DIM);
  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (p !== 0) vec[planeIndex(p) * 64 + sq] = 1;
  }
  if (vec.length > PIECE_DIM) {
    if (stm > 0) vec[F_STM] = 1;
    if (castling & 1) vec[F_WK] = 1;
    if (castling & 2) vec[F_WQ] = 1;
    if (castling & 4) vec[F_BK] = 1;
    if (castling & 8) vec[F_BQ] = 1;
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

    let normJson = null;
    try {
      const nRes = await fetch(modelUrl('normalization.json'));
      if (nRes.ok) normJson = await nRes.json();
    } catch (_) { /* optional */ }

    loadFromBuffers(modelJson, buf, normJson);
  } catch (e) {
    _ready     = false;
    _lastError = e.message || String(e);
  }
}

/**
 * Install a model from already-fetched data. The browser path uses fetch();
 * the Node test harnesses read the same files off disk and call this, so both
 * exercise exactly the same inference code.
 */
export function loadFromBuffers(modelJson, buf, normJson) {
  try {
    const group = modelJson.weightsManifest && modelJson.weightsManifest[0];
    if (!group) throw new Error('model.json has no weightsManifest');

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

    const dense = [];
    collectDense(modelJson.modelTopology, dense);
    if (dense.length === 0) throw new Error('no Dense layers in topology');

    const built = dense.map(d => {
      const W = tensors[`${d.name}/kernel`];
      const b = tensors[`${d.name}/bias`];
      if (!W || !b) throw new Error(`missing weights for ${d.name}`);
      return { W, b, outDim: b.length, inDim: W.length / b.length, relu: d.activation === 'relu' };
    });

    const l0 = built[0];
    if (l0.inDim !== PIECE_DIM && l0.inDim !== F_BQ + 1) {
      throw new Error(`input dim ${l0.inDim}, expected 768 or 773`);
    }
    _inDim  = l0.inDim;
    _h1     = l0.outDim;
    _l1W    = l0.W;
    _l1Bias = l0.b;
    _rest   = built.slice(1);

    let widest = _h1;
    for (const l of _rest) widest = Math.max(widest, l.outDim);
    _bufA = new Float32Array(widest);
    _bufB = new Float32Array(widest);

    if (normJson) {
      if (normJson.output === 'residual' || normJson.output === 'logit') {
        _mode = normJson.output;
        if (typeof normJson.cpScale === 'number' && isFinite(normJson.cpScale)) _cpScale = normJson.cpScale;
        if (typeof normJson.clampCp === 'number' && isFinite(normJson.clampCp)) _clampCp = normJson.clampCp;
      } else if (typeof normJson.yMax === 'number' && typeof normJson.yMin === 'number') {
        _mode = 'linear';
        _yScale = (normJson.yMax - normJson.yMin) / 2;
      }
    }

    _ready     = true;
    _lastError = null;
    return true;
  } catch (e) {
    _ready     = false;
    _lastError = e.message || String(e);
    return false;
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
/**
 * Evaluate a position in PAWN units from White's point of view (+ = White
 * better), or null when no model is loaded. For the current 'residual' model
 * this is a positional correction to add to an evaluation, not an evaluation;
 * isResidual() says which.
 *
 * @param {Int8Array|Array} board  flat 64-square board, or the UI's 8×8 array
 * @param {number} [stm]           1 White to move, -1 Black
 * @param {number} [castling]      bitmask wK=1, wQ=2, bK=4, bQ=8
 */
export function evaluate(board, stm, castling) {
  if (!_ready) return null;

  let cur = _bufA, next = _bufB;
  cur.set(_l1Bias);

  // Layer 1 by sparse row accumulation over the occupied squares.
  if (board.length === 64) {
    for (let sq = 0; sq < 64; sq++) {
      const p = board[sq];
      if (p === 0) continue;
      const base = (planeIndex(p) * 64 + sq) * _h1;
      for (let j = 0; j < _h1; j++) cur[j] += _l1W[base + j];
    }
  } else {
    // The UI still holds an 8×8 board for rendering.
    for (let r = 0; r < 8; r++) {
      const row = board[r];
      for (let c = 0; c < 8; c++) {
        const p = row[c];
        if (p === 0) continue;
        const base = (planeIndex(p) * 64 + r * 8 + c) * _h1;
        for (let j = 0; j < _h1; j++) cur[j] += _l1W[base + j];
      }
    }
  }

  if (_inDim > PIECE_DIM) {
    if (stm > 0) addRow(cur, F_STM);
    if (castling & 1) addRow(cur, F_WK);
    if (castling & 2) addRow(cur, F_WQ);
    if (castling & 4) addRow(cur, F_BK);
    if (castling & 8) addRow(cur, F_BQ);
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

  const raw = cur[0];
  if (_mode === 'residual') {
    // A positional correction, bounded so a confident network can never
    // outweigh a piece.
    const cp = Math.max(-_clampCp, Math.min(_clampCp, raw * _cpScale));
    return cp / 100;
  }
  if (_mode === 'logit') {
    // Win-probability logit → centipawns → pawn units. Clamped because the tail
    // of the logistic is meaningless precision and the search only needs to
    // know "winning", not "winning by 94 pawns".
    const cp = Math.max(-2000, Math.min(2000, raw * _cpScale));
    return cp / 100;
  }
  return raw * _yScale;
}

function addRow(acc, featureIndex) {
  const base = featureIndex * _h1;
  for (let j = 0; j < _h1; j++) acc[j] += _l1W[base + j];
}
