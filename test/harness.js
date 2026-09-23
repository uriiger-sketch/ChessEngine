'use strict';
// Shared helpers for the Node-side test harnesses.

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadFromBuffers, isReady, lastError, inputDim } from '../js/neural.js';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');

/**
 * Load model/ off disk into js/neural.js. The browser fetches these same files;
 * this just hands the bytes over directly, so the tests exercise the real
 * inference code rather than a stand-in.
 */
export function loadModelFromDisk(dir) {
  const d = dir || path.join(ROOT, 'model');
  try {
    const modelJson = JSON.parse(fs.readFileSync(path.join(d, 'model.json'), 'utf8'));
    const binName = modelJson.weightsManifest[0].paths[0];
    const bin = fs.readFileSync(path.join(d, binName));
    const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
    let norm = null;
    const np = path.join(d, 'normalization.json');
    if (fs.existsSync(np)) norm = JSON.parse(fs.readFileSync(np, 'utf8'));
    loadFromBuffers(modelJson, buf, norm);
    return isReady();
  } catch (e) {
    console.warn('  (model not loaded:', e.message + ')');
    return false;
  }
}

export { isReady as nnReady, lastError as nnError, inputDim as nnInputDim };

export function sqName(sq) {
  return String.fromCharCode(97 + (sq & 7)) + (8 - (sq >> 3));
}

/** UI move object → long algebraic, e.g. "e2e4" / "e7e8q". */
export function uiMoveToString(mv) {
  const s = (r, c) => String.fromCharCode(97 + c) + (8 - r);
  let out = s(mv.from[0], mv.from[1]) + s(mv.to[0], mv.to[1]);
  if (mv.promotion) out += 'nbrq'[Math.abs(mv.piece) - 2];
  return out;
}

export function fmt(n) { return n.toLocaleString('en-US'); }
