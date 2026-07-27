'use strict';
// Chess AI engine — iterative deepening alpha-beta with:
//   • Abort-on-timeout (only completes full iterations)
//   • Transposition table with move storage
//   • Killer moves (2 per ply)
//   • History heuristic
//   • Null-move pruning (R=3 deep, R=2 shallow)
//   • Late-move reduction (LMR)
//   • Aspiration windows in iterative deepening

import {
  generateMoves, makeMove, getLegalMoves, isInCheck,
  evaluatePosition, opposite
} from './chess.js';

import { evaluate as nnEvaluate, isReady as nnIsReady } from './neural.js';

// ── Constants ──────────────────────────────────────────────────────────────
const MAX_DEPTH  = 14;
const MATE_SCORE = 5000;
// Share of the leaf evaluation taken from the neural net vs the hand-crafted
// evaluation. The hand eval keeps tactical material accounting exact; the net
// contributes the positional judgement learned from master games.
const NN_WEIGHT  = 0.4;
// Piece values for MVV-LVA and SEE (indexed by abs piece code)
const MV_VAL = [0, 100, 320, 330, 500, 900, 20000];

// ── Transposition Table ────────────────────────────────────────────────────
const TT_SIZE = 1 << 20; // 1 048 576 entries (~16 MB)
const TT_MASK = TT_SIZE - 1;
const ttKeyLo  = new Uint32Array(TT_SIZE);
const ttKeyHi  = new Uint32Array(TT_SIZE);
const ttScore  = new Int32Array(TT_SIZE);
const ttDepth  = new Uint8Array(TT_SIZE);
const ttFlag   = new Uint8Array(TT_SIZE); // 0=exact 1=lower 2=upper
const ttMoves  = new Array(TT_SIZE).fill(null); // best move per entry

// ── Zobrist Keys ───────────────────────────────────────────────────────────
const ZLO = new Uint32Array(769); // 768 piece-square + 1 side-to-move
const ZHI = new Uint32Array(769);
(function() {
  let s = 7;
  const next = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s; };
  for (let i = 0; i < 769; i++) { ZLO[i] = next(); ZHI[i] = next(); }
})();

function zIdx(piece, r, c) {
  return (piece > 0 ? piece - 1 : 6 + (-piece) - 1) * 64 + r * 8 + c;
}

function computeZobrist(state) {
  let lo = 0, hi = 0;
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) {
      const p = state.board[r][c];
      if (p) { const i = zIdx(p,r,c); lo ^= ZLO[i]; hi ^= ZHI[i]; }
    }
  return [lo, hi];
}

function updateZobrist(lo, hi, mv, board) {
  const [fr,fc] = mv.from, [tr,tc] = mv.to;
  const orig = board[fr][fc]; // original piece at source (before promotion)

  const iFrom = zIdx(orig, fr, fc);
  lo ^= ZLO[iFrom]; hi ^= ZHI[iFrom];

  if (mv.captured) {
    const capR = mv.enPassant ? fr : tr;
    const capC = tc;
    const iCap = zIdx(mv.captured, capR, capC);
    lo ^= ZLO[iCap]; hi ^= ZHI[iCap];
  }

  const iTo = zIdx(mv.piece, tr, tc);
  lo ^= ZLO[iTo]; hi ^= ZHI[iTo];

  if (mv.castle) {
    if (tr===7&&tc===6){lo^=ZLO[zIdx(4,7,7)]^ZLO[zIdx(4,7,5)];hi^=ZHI[zIdx(4,7,7)]^ZHI[zIdx(4,7,5)];}
    if (tr===7&&tc===2){lo^=ZLO[zIdx(4,7,0)]^ZLO[zIdx(4,7,3)];hi^=ZHI[zIdx(4,7,0)]^ZHI[zIdx(4,7,3)];}
    if (tr===0&&tc===6){lo^=ZLO[zIdx(-4,0,7)]^ZLO[zIdx(-4,0,5)];hi^=ZHI[zIdx(-4,0,7)]^ZHI[zIdx(-4,0,5)];}
    if (tr===0&&tc===2){lo^=ZLO[zIdx(-4,0,0)]^ZLO[zIdx(-4,0,3)];hi^=ZHI[zIdx(-4,0,0)]^ZHI[zIdx(-4,0,3)];}
  }

  lo ^= ZLO[768]; hi ^= ZHI[768]; // flip side-to-move
  return [lo, hi];
}

// ── Search Heuristic Tables ────────────────────────────────────────────────
const MAX_PLY = 64;
// Killer moves: [ply][0|1]
const killers = Array.from({length: MAX_PLY}, () => [null, null]);
// History: keyed by (piece+6)*64 + to_square → higher = better quiet move
const historyTable = new Int32Array(13 * 64);

function histIdx(mv) { return (mv.piece + 6) * 64 + mv.to[0] * 8 + mv.to[1]; }

function addKiller(mv, ply) {
  if (ply >= MAX_PLY) return;
  const k = killers[ply];
  if (!mvEq(k[0], mv)) { k[1] = k[0]; k[0] = mv; }
}

function addHistory(mv, depth) {
  const i = histIdx(mv);
  historyTable[i] += depth * depth;
  if (historyTable[i] > 32000) {
    for (let j = 0; j < historyTable.length; j++) historyTable[j] >>= 2;
  }
}

function mvEq(a, b) {
  return a && b &&
    a.from[0] === b.from[0] && a.from[1] === b.from[1] &&
    a.to[0]   === b.to[0]   && a.to[1]   === b.to[1]   &&
    a.piece   === b.piece;
}

// ── Move Scoring & Sorting ─────────────────────────────────────────────────
function scoreMove(mv, ply, ttMv) {
  if (mvEq(mv, ttMv))       return 10_000_000; // TT move first
  if (mv.captured || mv.enPassant) {
    const vic = MV_VAL[Math.abs(mv.captured || 1)];
    const att = MV_VAL[Math.abs(mv.piece)];
    return 1_000_000 + vic * 10 - att;          // MVV-LVA for captures
  }
  if (mv.promotion)          return 900_000;
  const k = killers[ply < MAX_PLY ? ply : 0];
  if (mvEq(mv, k[0]))       return 800_000;
  if (mvEq(mv, k[1]))       return 700_000;
  return historyTable[histIdx(mv)];              // history heuristic
}

function sortMoves(moves, ply, ttMv) {
  const n = moves.length;
  const sc = new Int32Array(n);
  for (let i = 0; i < n; i++) sc[i] = scoreMove(moves[i], ply, ttMv);
  // Insertion sort — fast for small arrays typical in chess
  for (let i = 1; i < n; i++) {
    const s = sc[i], m = moves[i];
    let j = i - 1;
    while (j >= 0 && sc[j] < s) { sc[j+1] = sc[j]; moves[j+1] = moves[j]; j--; }
    sc[j+1] = s; moves[j+1] = m;
  }
}

// ── Evaluation ─────────────────────────────────────────────────────────────
// evaluatePosition() is Black-positive centipawns; the net returns White-positive
// pawn units, so it needs both a sign flip and a ×100 scale conversion. The two
// are combined as a weighted blend rather than a sum — adding them would count
// material twice, and previously the missing ×100 left the net contributing at
// most ~37cp, i.e. effectively nothing.
// Network inference costs ~50µs versus ~5µs for the hand evaluation, so leaf
// scores are memoised by Zobrist key. Searches revisit positions constantly,
// which makes this cache worth far more than micro-optimising the forward pass.
const EV_SIZE = 1 << 18;
const EV_MASK = EV_SIZE - 1;
const evKeyLo = new Int32Array(EV_SIZE);
const evKeyHi = new Int32Array(EV_SIZE);
const evVal   = new Float32Array(EV_SIZE);
const evUsed  = new Uint8Array(EV_SIZE);

function evalPosition(state, useNN, lo, hi) {
  const useCache = lo !== undefined;
  let slot = 0;
  if (useCache) {
    slot = lo & EV_MASK;
    if (evUsed[slot] && evKeyLo[slot] === (lo | 0) && evKeyHi[slot] === (hi | 0)) {
      return evVal[slot];
    }
  }

  const hand = evaluatePosition(state);
  let val = hand;
  if (useNN && nnIsReady()) {
    const s = nnEvaluate(state.board);
    if (s !== null) val = (1 - NN_WEIGHT) * hand + NN_WEIGHT * (-s * 100);
  }

  if (useCache) {
    evKeyLo[slot] = lo | 0; evKeyHi[slot] = hi | 0;
    evVal[slot] = val; evUsed[slot] = 1;
  }
  return val;
}

// ── Endgame detection (suppress null-move in low-material positions) ───────
function isEndgame(board) {
  let major = 0;
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) {
      const p = Math.abs(board[r][c]);
      if (p >= 2 && p <= 5) major++;
    }
  return major <= 4;
}

// ── Quiescence Search ──────────────────────────────────────────────────────
function quiescence(state, lo, hi, alpha, beta, side, useNN, qDepth) {
  if (aborted) return 0;

  const maximize = side === 'black';
  const standPat = evalPosition(state, useNN, lo, hi);

  if (maximize) {
    if (standPat >= beta) return beta;
    if (standPat > alpha) alpha = standPat;
  } else {
    if (standPat <= alpha) return alpha;
    if (standPat < beta) beta = standPat;
  }

  // Delta pruning: skip if no capture can possibly improve
  const DELTA = 1000;
  if (maximize && standPat < alpha - DELTA) return alpha;
  if (!maximize && standPat > beta + DELTA)  return beta;

  // Limit quiescence depth to avoid explosion
  if (qDepth <= 0) return standPat;

  const caps = generateMoves(state, side).filter(mv => mv.captured || mv.enPassant);
  sortMoves(caps, 0, null);

  let best = standPat;
  for (const mv of caps) {
    if (aborted) break;
    const child = makeMove(state, mv);
    if (isInCheck(child, side)) continue;
    const [nlo, nhi] = updateZobrist(lo, hi, mv, state.board);
    const score = quiescence(child, nlo, nhi, alpha, beta, opposite(side), useNN, qDepth - 1);

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

// ── Alpha-Beta Search ──────────────────────────────────────────────────────
let aborted      = false;
let startTime    = 0;
let timeLimitMs  = 5000;
let nodeCount    = 0;

function search(state, lo, hi, depth, alpha, beta, side, ply, useNN, isNullMove) {
  // Periodic time check. Checked every 512 nodes rather than 2048: a neural
  // leaf evaluation costs ~50µs, so a coarser interval can overshoot the
  // budget noticeably before the abort is noticed.
  if ((++nodeCount & 511) === 0 && Date.now() - startTime >= timeLimitMs) {
    aborted = true;
  }
  if (aborted) return [null, 0];

  const maximize = side === 'black';
  const idx = (lo >>> 0) & TT_MASK;
  let ttMv = null;

  // Transposition table probe
  if (ttKeyLo[idx] === lo && ttKeyHi[idx] === hi) {
    ttMv = ttMoves[idx];
    if (ttDepth[idx] >= depth) {
      const ts = ttScore[idx], tf = ttFlag[idx];
      if (tf === 0) return [ttMv, ts];
      if (tf === 1 && ts >= beta)  return [ttMv, ts];
      if (tf === 2 && ts <= alpha) return [ttMv, ts];
    }
  }

  // Leaf node → quiescence
  if (depth <= 0) return [null, quiescence(state, lo, hi, alpha, beta, side, useNN, 6)];

  const inCheck = isInCheck(state, side);

  // Check extension: don't reduce depth when in check
  if (inCheck) depth++;

  // Null-move pruning (skip if: in check, null move already done, endgame, shallow depth)
  if (!isNullMove && !inCheck && depth >= 3 && !isEndgame(state.board)) {
    const R = depth >= 6 ? 3 : 2;
    const nullState = { ...state, enPassantTarget: null, _sideToMove: opposite(side) };
    const [nlo, nhi] = [lo ^ ZLO[768], hi ^ ZHI[768]];
    const [, ns] = search(nullState, nlo, nhi, depth - R - 1, alpha, beta, opposite(side), ply + 1, false, true);
    if (!aborted) {
      if (maximize && ns >= beta) return [null, beta];
      if (!maximize && ns <= alpha) return [null, alpha];
    }
  }

  const legal = getLegalMoves(state, side);

  if (legal.length === 0) {
    return [null, inCheck
      ? (maximize ? -MATE_SCORE + ply : MATE_SCORE - ply)
      : 0];
  }

  sortMoves(legal, ply, ttMv);

  let bestMove = null;
  let bestScore = maximize ? -Infinity : Infinity;
  const origAlpha = alpha, origBeta = beta;

  for (let i = 0; i < legal.length; i++) {
    if (aborted) break;
    const mv = legal[i];
    const child = makeMove(state, mv);
    const [nlo, nhi] = updateZobrist(lo, hi, mv, state.board);
    const isCapOrPromo = mv.captured || mv.enPassant || mv.promotion;
    const givesCheck = isInCheck(child, opposite(side));

    let score;

    // Late Move Reduction: quiet, non-checking, non-first moves at depth >= 3
    const doLMR = depth >= 3 && i >= 3 && !isCapOrPromo && !givesCheck && !inCheck;
    if (doLMR) {
      // Reduction amount scales with move index and depth
      const R = 1 + (i >= 6 ? 1 : 0) + (depth >= 8 && i >= 12 ? 1 : 0);
      // Reduced search with a null window
      let nullAlpha = maximize ? alpha     : beta - 1;
      let nullBeta  = maximize ? alpha + 1 : beta;
      const [, lmrScore] = search(child, nlo, nhi, depth - 1 - R, nullAlpha, nullBeta, opposite(side), ply + 1, useNN, false);

      if (aborted) break;

      // If LMR score improves alpha (and we didn't prune), re-search at full depth
      const needFull = maximize ? lmrScore > alpha : lmrScore < beta;
      if (needFull) {
        const [, full] = search(child, nlo, nhi, depth - 1, alpha, beta, opposite(side), ply + 1, useNN, false);
        score = full;
      } else {
        score = lmrScore;
      }
    } else {
      const [, s] = search(child, nlo, nhi, depth - 1, alpha, beta, opposite(side), ply + 1, useNN, false);
      score = s;
    }

    if (aborted) break;

    if (maximize) {
      if (score > bestScore) { bestScore = score; bestMove = mv; }
      if (score > alpha) alpha = score;
    } else {
      if (score < bestScore) { bestScore = score; bestMove = mv; }
      if (score < beta) beta = score;
    }

    if (alpha >= beta) {
      // Beta cutoff — record in heuristic tables
      if (!isCapOrPromo) {
        addKiller(mv, ply);
        addHistory(mv, depth);
      }
      break;
    }
  }

  // Store result in TT
  if (!aborted) {
    ttKeyLo[idx] = lo; ttKeyHi[idx] = hi;
    ttScore[idx] = bestScore; ttDepth[idx] = Math.min(depth, 255);
    ttMoves[idx] = bestMove;
    ttFlag[idx] = bestScore <= origAlpha ? 2 : bestScore >= origBeta ? 1 : 0;
  }

  return [bestMove, bestScore];
}

// ── Iterative Deepening ────────────────────────────────────────────────────
export function searchBestMove(state, timeLimit, useNN) {
  startTime   = Date.now();
  timeLimitMs = timeLimit;
  aborted     = false;
  nodeCount   = 0;

  // Reset per-search heuristics (keep history across moves — it's global knowledge)
  for (let i = 0; i < MAX_PLY; i++) { killers[i][0] = null; killers[i][1] = null; }

  const side = state._sideToMove || 'black';
  const [lo, hi] = computeZobrist(state);
  let bestMove = null;
  let prevScore = 0;

  // Aspiration window parameters
  const ASP_DELTA = 50;

  for (let depth = 1; depth <= MAX_DEPTH; depth++) {
    aborted = false;

    // Aspiration windows: start with narrow window, widen on fail
    let alpha, beta;
    if (depth <= 4) {
      // Full window for first few depths (too shallow for aspiration)
      alpha = -Infinity; beta = Infinity;
    } else {
      alpha = prevScore - ASP_DELTA;
      beta  = prevScore + ASP_DELTA;
    }

    let mv, score;
    let aspirationFailed = false;

    while (true) {
      aborted = false;
      [mv, score] = search(state, lo, hi, depth, alpha, beta, side, 0, useNN, false);

      if (aborted) { aspirationFailed = true; break; }

      if (score <= alpha) {
        // Fail low — widen window down
        alpha = Math.max(alpha - ASP_DELTA * 4, -Infinity);
      } else if (score >= beta) {
        // Fail high — widen window up
        beta = Math.min(beta + ASP_DELTA * 4, Infinity);
      } else {
        break; // Within window — search is reliable
      }
    }

    if (!aspirationFailed && mv) {
      bestMove = mv;
      prevScore = score;
    }

    // Stop if we've found a forced mate
    if (Math.abs(score) >= MATE_SCORE - 100) break;

    // Time management: don't start the next iteration if we've used ≥ 55% of the
    // budget, since each deeper iteration typically costs 3–5× the previous one.
    const elapsed = Date.now() - startTime;
    if (elapsed >= timeLimitMs * 0.55) break;
  }

  return bestMove;
}
