'use strict';
// Search — negamax with alpha-beta, principal variation search, and the usual
// modern pruning set.
//
// What changed from the previous engine, and why it matters:
//
//  • Negamax instead of an explicit min/max split. The old code carried a
//    `maximize` flag through every branch and duplicated each comparison for
//    the two cases, which is where sign bugs live. Here one side's score is
//    just the negation of the other's.
//  • Make/unmake on a single board (js/position.js) instead of cloning a 2-D
//    array per node. This is the difference between roughly 100k and several
//    million nodes per second.
//  • The transposition key now covers castling rights and en passant. It did
//    not before, so positions that differed only in those respects shared an
//    entry and returned each other's scores.
//  • Mate scores are stored relative to the node, not the root, so a mate found
//    through the table reports the right distance.
//  • Draws by repetition, the fifty-move rule and insufficient material are
//    recognised inside the search. Previously the engine could not see a
//    perpetual coming and would walk into or throw away a drawn line.
//  • Quiescence answers checks instead of standing pat on them, and throws out
//    losing captures with a static exchange evaluation.

import {
  Position, MAX_PLY,
  mvFrom, mvTo, mvPromo, mvPiece, mvCapType, mvIsCap, mvIsQuiet,
  matchUIMove, NO_MOVE
} from './position.js';
import { evaluate as handEvaluate } from './evaluate.js';
import { getLegalMoves } from './chess.js';
import { evaluate as nnEvaluate, isReady as nnIsReady, isResidual as nnIsResidual } from './neural.js';

// ── Constants ──────────────────────────────────────────────────────────────
const MAX_DEPTH   = 64;
const MATE        = 32000;
const MATE_BOUND  = MATE - 1024;   // scores above this are mates
const INF         = 32767;

// ── The neural network's role, and what it is actually worth ───────────────
//
// The network is trained only on the repository's master PGN files (see
// train/train.js) and outputs a bounded positional CORRECTION, which is added
// to the hand evaluation rather than averaged with it. Averaging would drag
// material toward whatever the network says; adding a clamped correction
// cannot, so material always survives intact.
//
// Then it was measured, by self-play at equal time — network on versus the same
// engine with it off, 20-40 games per setting, colours swapped each pair:
//
//     gain 1.00, network also used for pruning   11.3%   -359 Elo
//     gain 1.00, pruning on hand eval only       15.0%   -301 Elo
//     gain 0.60, pruning on hand eval only       27.5%   -168 Elo
//     gain 0.40, network also used for pruning   35.0%   -108 Elo
//     gain 0.35, pruning on hand eval only       40.8%    -65 Elo  (60 games)
//     gain 0.00  (network off)                      —        baseline
//
// The trend is monotone: the less the network is applied, the stronger the
// engine plays. So the network does NOT improve playing strength here, and
// saying otherwise would be wishful. Two things explain it. The label is a game
// result, which at the level of a single position is mostly noise — the network
// learns a smooth prior, and inside a search that prior is usually less
// informed than the search's own deeper look. And inference is not free: it
// roughly halves the node rate, so it has to earn its keep before it breaks
// even, which it does not.
//
// The network is nonetheless always on: that is a product decision, taken
// with these numbers in view, to keep the engine one that learned from the
// masters. The constants below are the least costly configuration measured,
// and at them the engine still beats its predecessor decisively (see
// test/match.js). NN_SPLIT_EVAL keeps the correction out of the pruning margins
// (reverse futility, razoring, null move), which are tuned to the hand
// evaluation's scale and were the single most damaging place to inject noise —
// worth about 130 Elo on its own at gain 1.0.
//
// The natural next step, if this is revisited, is not a better value network
// but a policy one: use the master games to ORDER moves rather than to score
// positions. Bad ordering only costs a little speed, where a bad score changes
// which move gets played.
let NN_GAIN = 0.35;
let NN_SPLIT_EVAL = true;

/** Tuning hooks for test/match.js, so these numbers stay measurable. */
export function setNNGain(g)      { NN_GAIN = g; evOk.fill(0); evCachedWithNN = null; }
export function getNNGain()       { return NN_GAIN; }
export function setNNSplitEval(v) { NN_SPLIT_EVAL = !!v; evOk.fill(0); evCachedWithNN = null; }

// Fallback blend weight, used only for an older absolute-scoring model.
const NN_WEIGHT = 0.35;

const MVV = [0, 100, 320, 330, 500, 900, 0];

// ── Transposition table ────────────────────────────────────────────────────
const TT_BITS = 20;
const TT_SIZE = 1 << TT_BITS;
const TT_MASK = TT_SIZE - 1;
const ttKey   = new Int32Array(TT_SIZE);   // keyHi, used to verify the slot
const ttMove  = new Int32Array(TT_SIZE);
const ttScore = new Int32Array(TT_SIZE);
const ttDepth = new Int8Array(TT_SIZE);
const ttFlag  = new Uint8Array(TT_SIZE);   // 0 empty, 1 exact, 2 lower, 3 upper
const ttAge   = new Uint8Array(TT_SIZE);
let   ttGen   = 0;

function ttClear() {
  ttKey.fill(0); ttMove.fill(0); ttScore.fill(0);
  ttDepth.fill(0); ttFlag.fill(0); ttAge.fill(0);
}

// ── Evaluation cache ───────────────────────────────────────────────────────
// A neural forward pass costs far more than the hand evaluation, and searches
// revisit the same positions constantly, so leaf scores are memoised.
const EV_BITS = 18;
const EV_SIZE = 1 << EV_BITS;
const EV_MASK = EV_SIZE - 1;
const evKey = new Int32Array(EV_SIZE);
const evVal = new Int32Array(EV_SIZE);
const evOk  = new Uint8Array(EV_SIZE);

// Cached scores are only valid for the network setting they were computed
// under. The app always runs with it on, but the test harnesses switch it, so
// rather than widen the key the cache is dropped when the setting changes.
let evCachedWithNN = null;
function evalCacheFor(useNN) {
  if (evCachedWithNN !== useNN) { evOk.fill(0); evCachedWithNN = useNN; }
}

// ── Heuristic tables ───────────────────────────────────────────────────────
const killers     = new Int32Array(MAX_PLY * 2);
const historyTbl  = new Int32Array(2 * 64 * 64);
const counterTbl  = new Int32Array(2 * 7 * 64);
const moveScores  = new Int32Array(MAX_PLY * 256);

const pvLength = new Int32Array(MAX_PLY);
const pvTable  = new Int32Array(MAX_PLY * MAX_PLY);

// Reduction table: deeper searches and later moves get cut harder.
const LMR = new Int32Array(64 * 64);
(function buildLMR() {
  for (let d = 1; d < 64; d++)
    for (let m = 1; m < 64; m++)
      LMR[d * 64 + m] = Math.max(0, (0.75 + Math.log(d) * Math.log(m) / 2.25) | 0);
})();

// ── Search state ───────────────────────────────────────────────────────────
let aborted    = false;
let startTime  = 0;
let hardLimit  = 0;
let nodes      = 0;
let useNNFlag  = false;
let seldepth   = 0;

const pos = new Position();

// ── Leaf evaluation ────────────────────────────────────────────────────────
// `forPruning` asks for the hand evaluation alone. The pruning margins
// (reverse futility, razoring, null move) are tuned against the hand
// evaluation's scale and are very sensitive to noise in it — a wrong decision
// there discards a whole subtree, where a wrong leaf score only misvalues one.
function evalLeaf(p, forPruning) {
  const slot = p.keyLo & EV_MASK;
  if (!forPruning && evOk[slot] && evKey[slot] === p.keyHi) return evVal[slot];

  let v = handEvaluate(p);
  if (forPruning) return v;

  if (useNNFlag && NN_GAIN !== 0 && nnIsReady()) {
    const s = nnEvaluate(p.board, p.stm, p.castling);
    if (s !== null) {
      // The net reports pawn units from White's point of view; the search works
      // in centipawns from the side to move's.
      const nnCp = s * 100 * (p.stm > 0 ? 1 : -1);
      v = nnIsResidual()
        ? (v + NN_GAIN * nnCp) | 0
        : ((1 - NN_WEIGHT) * v + NN_WEIGHT * nnCp) | 0;
    }
  }

  evKey[slot] = p.keyHi; evVal[slot] = v; evOk[slot] = 1;
  return v;
}

// ── Move ordering ──────────────────────────────────────────────────────────
function scoreMoves(p, n, ttMv, ply, prevMove) {
  const base = p.ply * 256;
  const buf = p.moveBuf;
  const side = p.stm > 0 ? 0 : 1;
  const k0 = killers[ply * 2], k1 = killers[ply * 2 + 1];

  let counter = NO_MOVE;
  if (prevMove !== NO_MOVE) {
    counter = counterTbl[(side * 7 + mvPiece(prevMove)) * 64 + mvTo(prevMove)];
  }

  for (let i = 0; i < n; i++) {
    const m = buf[base + i];
    let s;
    if (m === ttMv) {
      s = 1 << 30;
    } else if (mvIsCap(m) || mvPromo(m)) {
      const gain = MVV[mvCapType(m)] * 16 - MVV[mvPiece(m)] + (mvPromo(m) ? MVV[mvPromo(m)] : 0);
      // Losing captures are searched after the quiet moves, not before them.
      s = p.see(m) >= 0 ? (1 << 28) + gain : -(1 << 28) + gain;
    } else if (m === k0) {
      s = (1 << 27);
    } else if (m === k1) {
      s = (1 << 27) - 1;
    } else if (m === counter) {
      s = (1 << 26);
    } else {
      s = historyTbl[(side * 64 + mvFrom(m)) * 64 + mvTo(m)];
    }
    moveScores[base + i] = s;
  }
}

// Selection sort one move at a time: most nodes cut off after a handful of
// moves, so sorting the whole list up front would be wasted work.
function pickMove(p, n, i) {
  const base = p.ply * 256;
  const buf = p.moveBuf;
  let best = i;
  for (let j = i + 1; j < n; j++) if (moveScores[base + j] > moveScores[base + best]) best = j;
  if (best !== i) {
    const tm = buf[base + i]; buf[base + i] = buf[base + best]; buf[base + best] = tm;
    const ts = moveScores[base + i]; moveScores[base + i] = moveScores[base + best]; moveScores[base + best] = ts;
  }
  return buf[base + i];
}

function recordQuiet(p, m, depth, ply, prevMove) {
  const side = p.stm > 0 ? 0 : 1;
  const k = ply * 2;
  if (killers[k] !== m) { killers[k + 1] = killers[k]; killers[k] = m; }

  const idx = (side * 64 + mvFrom(m)) * 64 + mvTo(m);
  historyTbl[idx] += depth * depth;
  if (historyTbl[idx] > 1 << 20) {
    for (let i = 0; i < historyTbl.length; i++) historyTbl[i] >>= 4;
  }

  if (prevMove !== NO_MOVE) {
    counterTbl[(side * 7 + mvPiece(prevMove)) * 64 + mvTo(prevMove)] = m;
  }
}

// ── Quiescence ─────────────────────────────────────────────────────────────
// Only searching captures leaves the engine blind when it is in check, so check
// positions here generate every legal reply instead of standing pat.
//
// That is limited to the first few quiescence plies. A position where one side
// can check forever otherwise expands without bound — the ply ceiling and the
// clock would stop it eventually, but not before a single node had swallowed
// the whole move's thinking time.
const Q_CHECK_PLIES = 3;

function quiescence(p, alpha, beta, ply, qPly = 0) {
  if ((++nodes & 2047) === 0 && Date.now() >= hardLimit) aborted = true;
  if (aborted) return 0;
  if (ply > seldepth) seldepth = ply;

  if (p.isRepetition() || p.isFiftyMove() || p.insufficientMaterial()) return 0;
  if (p.ply >= MAX_PLY - 4) return evalLeaf(p);

  const inCheck = p.inCheck() && qPly < Q_CHECK_PLIES;
  let best = -INF;

  if (!inCheck) {
    best = evalLeaf(p);
    if (best >= beta) return best;
    if (best > alpha) alpha = best;
  }

  const n = p.generate(p.ply, !inCheck);
  const base = p.ply * 256;
  scoreMoves(p, n, NO_MOVE, ply, NO_MOVE);

  let legal = 0;
  for (let i = 0; i < n; i++) {
    const m = pickMove(p, n, i);

    if (!inCheck) {
      // Delta pruning: even winning this piece outright would not reach alpha.
      if (!mvPromo(m) && best + MVV[mvCapType(m)] + 200 < alpha) continue;
      // Throw away captures that lose material on the exchange.
      if (p.see(m) < 0) continue;
    }

    if (!p.makeMove(m)) continue;
    legal++;
    const score = -quiescence(p, -beta, -alpha, ply + 1, qPly + 1);
    p.unmakeMove();

    if (aborted) return 0;
    if (score > best) {
      best = score;
      if (score > alpha) alpha = score;
      if (alpha >= beta) break;
    }
  }

  if (inCheck && legal === 0) return -MATE + ply;   // checkmate
  return best;
}

// ── Main search ────────────────────────────────────────────────────────────
function negamax(p, depth, alpha, beta, ply, isPV, canNull, prevMove) {
  if ((++nodes & 2047) === 0 && Date.now() >= hardLimit) aborted = true;
  if (aborted) return 0;

  pvLength[ply] = ply;

  if (ply > 0) {
    if (p.isRepetition() || p.isFiftyMove() || p.insufficientMaterial()) return 0;
    if (p.ply >= MAX_PLY - 4) return evalLeaf(p);

    // Mate-distance pruning: a mate already found closer to the root cannot be
    // beaten by anything down here.
    if (alpha < -MATE + ply) alpha = -MATE + ply;
    if (beta > MATE - ply - 1) beta = MATE - ply - 1;
    if (alpha >= beta) return alpha;
  }

  if (depth <= 0) return quiescence(p, alpha, beta, ply);

  const slot = p.keyLo & TT_MASK;
  let ttMv = NO_MOVE;
  if (ttFlag[slot] !== 0 && ttKey[slot] === p.keyHi) {
    ttMv = ttMove[slot];
    if (!isPV && ttDepth[slot] >= depth) {
      let s = ttScore[slot];
      // Mate scores are stored relative to this node; put the distance back.
      if (s > MATE_BOUND) s -= ply;
      else if (s < -MATE_BOUND) s += ply;
      const f = ttFlag[slot];
      if (f === 1) return s;
      if (f === 2 && s >= beta) return s;
      if (f === 3 && s <= alpha) return s;
    }
  }

  const inCheck = p.inCheck();
  if (inCheck) depth++;                       // check extension

  const staticEval = inCheck ? -INF : evalLeaf(p, NN_SPLIT_EVAL);

  if (!isPV && !inCheck && Math.abs(beta) < MATE_BOUND) {
    // Reverse futility: we are so far ahead that even giving up material would
    // hold beta, so there is nothing to prove here.
    if (depth <= 7 && staticEval - 80 * depth >= beta) return staticEval;

    // Razoring: so far behind that only a tactic saves it — let quiescence look.
    if (depth <= 3 && staticEval + 300 * depth < alpha) {
      const q = quiescence(p, alpha, beta, ply);
      if (q < alpha) return q;
    }

    // Null move: hand the opponent a free move; if we are still winning, this
    // node is not worth a full search. Skipped without pieces, where zugzwang
    // makes the assumption false.
    if (canNull && depth >= 3 && staticEval >= beta && p.hasNonPawnMaterial()) {
      // Clamped so the reduced search keeps at least one ply: an unclamped
      // reduction at shallow depth drops straight into quiescence, which tells
      // us almost nothing and prunes on almost nothing.
      let R = 3 + ((depth / 4) | 0) + Math.min(3, ((staticEval - beta) / 200) | 0);
      if (R > depth - 1) R = depth - 1;
      p.makeNull();
      const s = -negamax(p, depth - R - 1, -beta, -beta + 1, ply + 1, false, false, NO_MOVE);
      p.unmakeNull();
      if (aborted) return 0;
      if (s >= beta) {
        if (s >= MATE_BOUND) return beta;     // don't trust mates found via a null move
        if (depth < 10) return s;
        // Deep nodes get a verification search with null move disabled.
        const v = negamax(p, depth - R - 1, beta - 1, beta, ply, false, false, prevMove);
        if (aborted) return 0;
        if (v >= beta) return s;
      }
    }
  }

  // Internal iterative deepening: with no table move, a shallow search is a
  // cheaper way to find one than searching the moves in a bad order.
  if (ttMv === NO_MOVE && depth >= 5 && isPV) {
    negamax(p, depth - 3, alpha, beta, ply, isPV, false, prevMove);
    if (aborted) return 0;
    if (ttFlag[slot] !== 0 && ttKey[slot] === p.keyHi) ttMv = ttMove[slot];
  }

  const n = p.generate(p.ply, false);
  scoreMoves(p, n, ttMv, ply, prevMove);

  const origAlpha = alpha;
  let bestScore = -INF;
  let bestMove  = NO_MOVE;
  let moveCount = 0;

  const futile = !isPV && !inCheck && depth <= 6 &&
                 staticEval + 120 + 130 * depth <= alpha && Math.abs(alpha) < MATE_BOUND;

  for (let i = 0; i < n; i++) {
    const m = pickMove(p, n, i);
    if (ply === 0 && rootExclude !== null && rootExclude.includes(m)) continue;
    const quiet = mvIsQuiet(m);

    // Skip obviously losing captures in shallow, non-PV nodes.
    if (!isPV && !inCheck && depth <= 5 && moveCount > 0 && !quiet &&
        bestScore > -MATE_BOUND && p.see(m) < -50 * depth) continue;

    if (!p.makeMove(m)) continue;
    moveCount++;

    const givesCheck = p.inCheck();

    if (quiet && moveCount > 1 && bestScore > -MATE_BOUND && !isPV && !inCheck && !givesCheck) {
      // Late move pruning: this late in a well-ordered list, quiet moves at
      // shallow depth essentially never turn out to be best.
      if (depth <= 6 && moveCount > 4 + depth * depth) { p.unmakeMove(); continue; }
      if (futile) { p.unmakeMove(); continue; }
    }

    let score;
    if (moveCount === 1) {
      score = -negamax(p, depth - 1, -beta, -alpha, ply + 1, isPV, true, m);
    } else {
      let r = 0;
      if (depth >= 3 && moveCount > 2 && quiet) {
        r = LMR[Math.min(depth, 63) * 64 + Math.min(moveCount, 63)];
        if (isPV) r--;
        if (givesCheck) r--;
        const side = p.stm > 0 ? 1 : 0;   // mover's side (stm already flipped)
        if (historyTbl[(side * 64 + mvFrom(m)) * 64 + mvTo(m)] > 8000) r--;
        if (r < 0) r = 0;
        if (r > depth - 2) r = depth - 2;
      }

      // Zero-window probe, re-searched wider only when it beats alpha.
      score = -negamax(p, depth - 1 - r, -alpha - 1, -alpha, ply + 1, false, true, m);
      if (!aborted && score > alpha && r > 0) {
        score = -negamax(p, depth - 1, -alpha - 1, -alpha, ply + 1, false, true, m);
      }
      if (!aborted && score > alpha && score < beta) {
        score = -negamax(p, depth - 1, -beta, -alpha, ply + 1, true, true, m);
      }
    }

    p.unmakeMove();
    if (aborted) return 0;

    if (score > bestScore) {
      bestScore = score;
      bestMove = m;
      if (score > alpha) {
        alpha = score;
        // Record the principal variation for this node.
        pvTable[ply * MAX_PLY + ply] = m;
        for (let j = ply + 1; j < pvLength[ply + 1]; j++) {
          pvTable[ply * MAX_PLY + j] = pvTable[(ply + 1) * MAX_PLY + j];
        }
        pvLength[ply] = pvLength[ply + 1];

        if (alpha >= beta) {
          if (quiet) recordQuiet(p, m, depth, ply, prevMove);
          break;
        }
      }
    }
  }

  if (moveCount === 0) return inCheck ? -MATE + ply : 0;   // mate or stalemate

  // Store, preferring deeper entries but always replacing stale generations.
  // A root searched with moves excluded has a score for a restricted move list,
  // which is not the position's value, so it is kept out of the table.
  const restricted = ply === 0 && rootExclude !== null;
  if (!aborted && !restricted && (ttFlag[slot] === 0 || ttAge[slot] !== ttGen || ttDepth[slot] <= depth)) {
    let s = bestScore;
    if (s > MATE_BOUND) s += ply;
    else if (s < -MATE_BOUND) s -= ply;
    ttKey[slot] = p.keyHi;
    ttMove[slot] = bestMove;
    ttScore[slot] = s;
    ttDepth[slot] = Math.min(depth, 127);
    ttFlag[slot] = bestScore <= origAlpha ? 3 : bestScore >= beta ? 2 : 1;
    ttAge[slot] = ttGen;
  }

  return bestScore;
}

// ── Iterative deepening ────────────────────────────────────────────────────
export const searchInfo = { depth: 0, seldepth: 0, nodes: 0, score: 0, pv: [], timeMs: 0 };

// Moves the root must skip. Used to find the second- and third-best moves for
// hints: search, exclude what was found, search again.
let rootExclude = null;

function beginSearch(timeLimit, useNN) {
  startTime = Date.now();
  hardLimit = startTime + Math.max(30, timeLimit * 0.95);
  aborted = false;
  nodes = 0;
  seldepth = 0;
  useNNFlag = !!useNN;
  evalCacheFor(useNNFlag);
  ttGen = (ttGen + 1) & 255;
  killers.fill(0);
  // History is aged rather than cleared: move quality carries over between
  // moves, but old data should not outweigh what this search learns.
  for (let i = 0; i < historyTbl.length; i++) historyTbl[i] >>= 1;
}

// One iterative-deepening search from the root, up to the current hard limit.
// Only completed iterations count; the answer never comes from a search that
// was cut off partway through.
function iterate(p, softLimit) {
  pvLength.fill(0);
  pvTable[0] = NO_MOVE;

  let best = NO_MOVE, bestScore = 0, done = 0;
  for (let depth = 1; depth <= MAX_DEPTH; depth++) {
    let score;
    // Aspiration windows: assume the score moved little since the last
    // iteration and re-search wider only when that assumption fails.
    if (depth <= 4) {
      score = negamax(p, depth, -INF, INF, 0, true, true, NO_MOVE);
    } else {
      let delta = 25;
      let alpha = Math.max(-INF, bestScore - delta);
      let beta  = Math.min(INF, bestScore + delta);
      while (true) {
        score = negamax(p, depth, alpha, beta, 0, true, true, NO_MOVE);
        if (aborted) break;
        if (score <= alpha)      { beta = (alpha + beta) >> 1; alpha = Math.max(-INF, alpha - delta); }
        else if (score >= beta)  { beta = Math.min(INF, beta + delta); }
        else break;
        delta += delta >> 1;
        if (delta > 1200) { alpha = -INF; beta = INF; }
      }
    }

    if (aborted) break;
    if (pvLength[0] === 0) break;          // terminal root — no move to make

    best = pvTable[0];
    bestScore = score;
    done = depth;
    searchInfo.depth = depth;
    searchInfo.seldepth = seldepth;
    searchInfo.nodes = nodes;
    searchInfo.score = score;
    searchInfo.pv = readPV();
    searchInfo.timeMs = Date.now() - startTime;

    if (Math.abs(score) > MATE_BOUND) break;          // forced mate found
    if (Date.now() >= softLimit) break;               // no time for another pass
  }

  // If even depth 1 was cut short, the table's move is better than nothing —
  // unless moves are being excluded, in which case it may be one of them.
  if (best === NO_MOVE && !rootExclude) {
    const slot = p.keyLo & TT_MASK;
    if (ttFlag[slot] !== 0 && ttKey[slot] === p.keyHi) best = ttMove[slot];
  }
  return { move: best, score: bestScore, depth: done };
}

function loadRoot(state, side, opts) {
  pos.setFromState(state, side);
  if (opts && opts.history && opts.history.length) pos.setHistory(opts.history);
  else pos.setHistory([pos.keyLo, pos.keyHi]);
}

/** Mate distance in moves (signed, from the mover's view), or null. */
function mateIn(score) {
  if (Math.abs(score) <= MATE_BOUND) return null;
  const moves = Math.ceil((MATE - Math.abs(score)) / 2);
  return score > 0 ? moves : -moves;
}

/**
 * Pick a move for the side to move.
 *
 * @param {object} state      chess.js-style game state
 * @param {number} timeLimit  milliseconds of thinking time
 * @param {boolean} useNN     blend the neural network into leaf scores
 * @param {object} [opts]     { history: number[] } zobrist keys of earlier game
 *                            positions, as [lo, hi, lo, hi, …]
 * @returns {object|null}     a chess.js move object, or null if there is none
 */
export function searchBestMove(state, timeLimit, useNN, opts) {
  const side = state._sideToMove || state.sideToMove || 'white';
  const uiMoves = getLegalMoves(state, side);
  if (uiMoves.length === 0) return null;

  beginSearch(timeLimit, useNN);
  loadRoot(state, side, opts);

  // Nothing to think about with a single legal reply.
  if (uiMoves.length === 1) {
    searchInfo.depth = 0; searchInfo.nodes = 0; searchInfo.score = 0;
    searchInfo.pv = []; searchInfo.timeMs = 0; searchInfo.seldepth = 0;
    return uiMoves[0];
  }

  const r = iterate(pos, startTime + timeLimit * 0.5);
  if (r.move === NO_MOVE) return uiMoves[0];
  searchInfo.depth = r.depth;
  return matchUIMove(r.move, uiMoves) || uiMoves[0];
}

/**
 * The best `count` moves for the side to move, for hints.
 *
 * Found by exclusion: search, remember the best move, search again with it
 * barred at the root, and so on. The first pass sees every move, so its answer
 * IS the engine's best move and always stays first. The later passes only
 * answer "best among the rest", and their order is kept as found: re-sorting
 * the list by score once let a later pass's move — one the full search had
 * already weighed and rejected — jump to first place on the strength of a
 * shallower search's number.
 *
 * `bestMs` is the first pass's time; callers give it the same budget the
 * engine gets for its own move, so the suggestion is exactly as deep as the
 * move the player is facing. Each later pass gets `opts.restMs`.
 * `opts.onFound(list)` is called after every pass, so the best move can be
 * shown before the alternatives are ready. Scores are centipawns from the
 * mover's view.
 *
 * @returns {{move: object, score: number, mateIn: number|null, pass: number, depth: number}[]}
 */
export function searchTopMoves(state, bestMs, count, useNN, opts = {}) {
  const side = state._sideToMove || state.sideToMove || 'white';
  const uiMoves = getLegalMoves(state, side);
  const want = Math.min(count, uiMoves.length);
  if (want === 0) return [];

  loadRoot(state, side, opts);
  const restMs = opts.restMs || Math.max(250, bestMs * 0.25);
  const found = [];
  const excluded = [];

  try {
    for (let i = 0; i < want; i++) {
      const ms = i === 0 ? bestMs : restMs;
      beginSearch(ms, useNN);
      rootExclude = excluded.length ? excluded : null;
      const r = iterate(pos, startTime + ms * 0.5);
      if (r.move === NO_MOVE) break;
      const ui = matchUIMove(r.move, uiMoves);
      if (!ui) break;
      found.push({ move: ui, score: r.score, mateIn: mateIn(r.score), pass: i, depth: r.depth });
      excluded.push(r.move);
      if (opts.onFound) opts.onFound(found.slice());
    }
  } finally {
    rootExclude = null;
  }
  return found;
}

function readPV() {
  const out = [];
  for (let i = 0; i < pvLength[0] && i < 32; i++) {
    const m = pvTable[i];
    if (m === NO_MOVE) break;
    out.push(moveToString(m));
  }
  return out;
}

export function moveToString(m) {
  const f = mvFrom(m), t = mvTo(m);
  const s = (sq) => String.fromCharCode(97 + (sq & 7)) + (8 - (sq >> 3));
  return s(f) + s(t) + (mvPromo(m) ? 'nbrq'[mvPromo(m) - 2] : '');
}

// Scratch position used only for hashing and one-off evaluation, kept apart
// from the search's own.
const keyScratch = new Position();

/**
 * Static evaluation of a game position in centipawns, White-positive — the same
 * leaf score the search uses, so the evaluation bar and the engine cannot
 * disagree about what a position is worth.
 */
export function staticEvalOf(state, side, useNN) {
  keyScratch.setFromState(state, side);
  const prev = useNNFlag;
  useNNFlag = !!useNN;
  evalCacheFor(useNNFlag);
  const v = evalLeaf(keyScratch);
  useNNFlag = prev;
  evalCacheFor(prev);
  return keyScratch.stm > 0 ? v : -v;
}

/**
 * Zobrist key of a game position, as [lo, hi]. Callers record one of these per
 * played position and hand the list back through searchBestMove's `history`,
 * which is what lets the search see a repetition coming.
 */
export function zobristOf(state, side) {
  keyScratch.setFromState(state, side);
  return [keyScratch.keyLo, keyScratch.keyHi];
}

/** Wipe all learned state — used between games so nothing leaks across. */
export function resetEngine() {
  ttClear();
  evOk.fill(0);
  evCachedWithNN = null;
  historyTbl.fill(0);
  counterTbl.fill(0);
  killers.fill(0);
  ttGen = 0;
}

/** Search a Position directly. Used by the test harnesses. */
export function searchPosition(position, timeLimit, useNN) {
  beginSearch(timeLimit, useNN);
  hardLimit = startTime + timeLimit;

  const p = position;
  if (p.histN === 0) p.setHistory([p.keyLo, p.keyHi]);

  // A mated or stalemated root has nothing to return.
  const n0 = p.generate(p.ply, false);
  let anyLegal = false;
  for (let i = 0; i < n0; i++) {
    if (p.makeMove(p.moveBuf[p.ply * 256 + i])) { p.unmakeMove(); anyLegal = true; break; }
  }
  if (!anyLegal) {
    searchInfo.depth = 0; searchInfo.nodes = 0; searchInfo.pv = [];
    searchInfo.score = p.inCheck() ? -MATE : 0;
    return NO_MOVE;
  }

  return iterate(p, startTime + timeLimit * 0.5).move;
}
