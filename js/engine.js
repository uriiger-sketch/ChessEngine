'use strict';
// Chess AI engine — ES module, runs on the main thread
// Implements iterative-deepening alpha-beta with transposition table and quiescence search

import {
  generateMoves, makeMove, getLegalMoves, isInCheck, evaluatePosition, opposite
} from './chess.js';

import { evaluate as nnEvaluate, isReady as nnIsReady } from './neural.js';

// ── Constants ──────────────────────────────────────────────────────────────
const MAX_DEPTH  = 12;
const MATE_SCORE = 5000;
const MV_VAL     = [0, 1, 3, 3, 5, 9, 10]; // abs(piece code) → value for MVV-LVA

// ── Transposition Table ────────────────────────────────────────────────────
const TT_SIZE  = 1 << 19; // 524288
const TT_MASK  = TT_SIZE - 1;
const ttKeyLo  = new Uint32Array(TT_SIZE);
const ttKeyHi  = new Uint32Array(TT_SIZE);
const ttScore  = new Int32Array(TT_SIZE);
const ttDepth  = new Uint8Array(TT_SIZE);
const ttFlag   = new Uint8Array(TT_SIZE); // 0=exact, 1=lower, 2=upper

// ── Zobrist Hashing ────────────────────────────────────────────────────────
const ZLO = new Uint32Array(769); // 768 piece-square + 1 side-to-move
const ZHI = new Uint32Array(769);

(function() {
  let s = 7;
  const next = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s; };
  for (let i = 0; i < 769; i++) { ZLO[i] = next(); ZHI[i] = next(); }
})();

function zIdx(piece, r, c) {
  const plane = piece > 0 ? piece - 1 : 6 + (-piece) - 1;
  return plane * 64 + r * 8 + c;
}

function computeZobrist(state) {
  let lo = 0, hi = 0;
  const board = state.board;
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p !== 0) { const i = zIdx(p, r, c); lo ^= ZLO[i]; hi ^= ZHI[i]; }
    }
  return [lo, hi];
}

function updateZobrist(lo, hi, mv, state) {
  const [fr, fc] = mv.from;
  const [tr, tc] = mv.to;
  const piece = mv.piece;
  // Determine original piece at from-square (before promotion changes the code)
  const origPiece = state.board[fr][fc];

  // Remove piece from source
  const iFrom = zIdx(origPiece, fr, fc);
  lo ^= ZLO[iFrom]; hi ^= ZHI[iFrom];

  // Remove captured piece
  if (mv.captured !== 0) {
    if (mv.enPassant) {
      // En passant: captured pawn is on the same row as the attacker
      const iCap = zIdx(mv.captured, fr, tc);
      lo ^= ZLO[iCap]; hi ^= ZHI[iCap];
    } else {
      const iCap = zIdx(mv.captured, tr, tc);
      lo ^= ZLO[iCap]; hi ^= ZHI[iCap];
    }
  }

  // Place piece at destination
  const iTo = zIdx(piece, tr, tc);
  lo ^= ZLO[iTo]; hi ^= ZHI[iTo];

  // Castling rook movements
  if (mv.castle) {
    if (tr === 7 && tc === 6) { lo ^= ZLO[zIdx(4,7,7)] ^ ZLO[zIdx(4,7,5)]; hi ^= ZHI[zIdx(4,7,7)] ^ ZHI[zIdx(4,7,5)]; }
    if (tr === 7 && tc === 2) { lo ^= ZLO[zIdx(4,7,0)] ^ ZLO[zIdx(4,7,3)]; hi ^= ZHI[zIdx(4,7,0)] ^ ZHI[zIdx(4,7,3)]; }
    if (tr === 0 && tc === 6) { lo ^= ZLO[zIdx(-4,0,7)] ^ ZLO[zIdx(-4,0,5)]; hi ^= ZHI[zIdx(-4,0,7)] ^ ZHI[zIdx(-4,0,5)]; }
    if (tr === 0 && tc === 2) { lo ^= ZLO[zIdx(-4,0,0)] ^ ZLO[zIdx(-4,0,3)]; hi ^= ZHI[zIdx(-4,0,0)] ^ ZHI[zIdx(-4,0,3)]; }
  }

  // Side-to-move flip
  lo ^= ZLO[768]; hi ^= ZHI[768];

  return [lo, hi];
}

// ── Move Ordering ──────────────────────────────────────────────────────────
function sortMoves(moves) {
  const n = moves.length;
  const scores = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const mv = moves[i];
    if (mv.captured !== 0) scores[i] = 10 * MV_VAL[Math.abs(mv.captured)] - MV_VAL[Math.abs(mv.piece)];
  }
  // Insertion sort (small arrays, captures first)
  for (let i = 1; i < n; i++) {
    const s = scores[i], m = moves[i];
    let j = i - 1;
    while (j >= 0 && scores[j] < s) { scores[j+1] = scores[j]; moves[j+1] = moves[j]; j--; }
    scores[j+1] = s; moves[j+1] = m;
  }
  return moves;
}

// ── Evaluation ─────────────────────────────────────────────────────────────
const GAMMA_NET = 0.95;

function evalPosition(state, useNN) {
  let val = evaluatePosition(state); // material+positional, Black=positive

  if (useNN && nnIsReady()) {
    const nnScore = nnEvaluate(state.board); // White perspective
    if (nnScore !== null) val -= GAMMA_NET * nnScore;
  }
  return val;
}

// ── Quiescence Search ──────────────────────────────────────────────────────
function quiescence(state, lo, hi, alpha, beta, side, useNN) {
  const standPat = evalPosition(state, useNN);
  const maximize = side === 'black';

  if (maximize) {
    if (standPat >= beta) return standPat;
    if (standPat > alpha) alpha = standPat;
  } else {
    if (standPat <= alpha) return standPat;
    if (standPat < beta) beta = standPat;
  }

  const moves = generateMoves(state, side)
    .filter(mv => mv.captured !== 0 || mv.enPassant);
  sortMoves(moves);

  let best = standPat;
  for (const mv of moves) {
    const child = makeMove(state, mv);
    if (isInCheck(child, side)) continue;
    const [nlo, nhi] = updateZobrist(lo, hi, mv, state);
    const score = quiescence(child, nlo, nhi, alpha, beta, opposite(side), useNN);
    if (maximize) {
      if (score > best) best = score;
      if (score > alpha) alpha = score;
    } else {
      if (score < best) best = score;
      if (score < beta) beta = score;
    }
    if (alpha >= beta) break;
  }
  return best;
}

// ── Alpha-Beta ─────────────────────────────────────────────────────────────
function search(state, lo, hi, depth, alpha, beta, side, ply, useNN) {
  const maximize = side === 'black';
  const idx = (lo >>> 0) & TT_MASK;

  // TT lookup
  if (ttKeyLo[idx] === lo && ttKeyHi[idx] === hi && ttDepth[idx] >= depth) {
    const ts = ttScore[idx], tf = ttFlag[idx];
    if (tf === 0) return [null, ts];
    if (tf === 1 && ts >= beta) return [null, ts];
    if (tf === 2 && ts <= alpha) return [null, ts];
  }

  if (depth === 0) return [null, quiescence(state, lo, hi, alpha, beta, side, useNN)];

  const legal = getLegalMoves(state, side);
  if (legal.length === 0) {
    if (isInCheck(state, side)) {
      return [null, maximize ? (-MATE_SCORE + ply) : (MATE_SCORE - ply)];
    }
    return [null, 0]; // stalemate
  }

  sortMoves(legal);

  let bestMove = null;
  let bestScore = maximize ? -Infinity : Infinity;
  const origAlpha = alpha, origBeta = beta;

  for (const mv of legal) {
    const child = makeMove(state, mv);
    const [nlo, nhi] = updateZobrist(lo, hi, mv, state);
    const [, score] = search(child, nlo, nhi, depth - 1, alpha, beta, opposite(side), ply + 1, useNN);

    if (maximize) {
      if (score > bestScore) { bestScore = score; bestMove = mv; }
      if (score > alpha) alpha = score;
    } else {
      if (score < bestScore) { bestScore = score; bestMove = mv; }
      if (score < beta) beta = score;
    }
    if (alpha >= beta) break;
  }

  // Store in TT
  ttKeyLo[idx] = lo; ttKeyHi[idx] = hi;
  ttScore[idx] = bestScore; ttDepth[idx] = depth;
  ttFlag[idx] = bestScore <= origAlpha ? 2 : bestScore >= origBeta ? 1 : 0;

  return [bestMove, bestScore];
}

// ── Iterative Deepening ────────────────────────────────────────────────────
export function searchBestMove(state, timeLimit, useNN) {
  const t0 = Date.now();
  const side = state._sideToMove || 'black';
  const [lo, hi] = computeZobrist(state);
  let bestMove = null;

  for (let depth = 1; depth <= MAX_DEPTH; depth++) {
    if (Date.now() - t0 >= timeLimit) break;
    const [mv, score] = search(state, lo, hi, depth, -Infinity, Infinity, side, 0, useNN);
    if (mv) bestMove = mv;
    if (Math.abs(score) >= MATE_SCORE - 100) break;
  }
  return bestMove;
}
