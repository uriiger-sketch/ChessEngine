'use strict';
// Hand-crafted evaluation, tapered between a middlegame and an endgame score.
//
// The old evaluation had one set of piece-square tables and used them at every
// stage of the game. That is a real playing weakness, most visibly for the
// king: the middlegame table drives it into the corner, which is exactly wrong
// once the queens come off and the king becomes an attacking piece. Here every
// term has a middlegame and an endgame value, and the two are blended by how
// much material is left, so the same position is judged differently at move 10
// and move 60.
//
// Scores are accumulated White-positive and flipped at the end, because the
// search is negamax and wants "good for the side to move".
//
// Piece-square tables and base piece values are the PeSTO set — the standard
// published tuning that most modern amateur engines start from. They are laid
// out rank 8 first, which is already this project's square order, so a White
// piece reads table[sq] and a Black piece reads table[sq ^ 56].

const MG_VAL = [0, 82, 337, 365, 477, 1025, 0];
const EG_VAL = [0, 94, 281, 297, 512,  936, 0];

// Phase weights: 24 at the start of the game, 0 in a bare king endgame.
const PHASE_INC = [0, 0, 1, 1, 2, 4, 0];
const TOTAL_PHASE = 24;

const MG_PAWN = [
    0,   0,   0,   0,   0,   0,  0,   0,
   98, 134,  61,  95,  68, 126, 34, -11,
   -6,   7,  26,  31,  65,  56, 25, -20,
  -14,  13,   6,  21,  23,  12, 17, -23,
  -27,  -2,  -5,  12,  17,   6, 10, -25,
  -26,  -4,  -4, -10,   3,   3, 33, -12,
  -35,  -1, -20, -23, -15,  24, 38, -22,
    0,   0,   0,   0,   0,   0,  0,   0,
];
const EG_PAWN = [
    0,   0,   0,   0,   0,   0,   0,   0,
  178, 173, 158, 134, 147, 132, 165, 187,
   94, 100,  85,  67,  56,  53,  82,  84,
   32,  24,  13,   5,  -2,   4,  17,  17,
   13,   9,  -3,  -7,  -7,  -8,   3,  -1,
    4,   7,  -6,   1,   0,  -5,  -1,  -8,
   13,   8,   8,  10,  13,   0,   2,  -7,
    0,   0,   0,   0,   0,   0,   0,   0,
];
const MG_KNIGHT = [
  -167, -89, -34, -49,  61, -97, -15, -107,
   -73, -41,  72,  36,  23,  62,   7,  -17,
   -47,  60,  37,  65,  84, 129,  73,   44,
    -9,  17,  19,  53,  37,  69,  18,   22,
   -13,   4,  16,  13,  28,  19,  21,   -8,
   -23,  -9,  12,  10,  19,  17,  25,  -16,
   -29, -53, -12,  -3,  -1,  18, -14,  -19,
  -105, -21, -58, -33, -17, -28, -19,  -23,
];
const EG_KNIGHT = [
  -58, -38, -13, -28, -31, -27, -63, -99,
  -25,  -8, -25,  -2,  -9, -25, -24, -52,
  -24, -20,  10,   9,  -1,  -9, -19, -41,
  -17,   3,  22,  22,  22,  11,   8, -18,
  -18,  -6,  16,  25,  16,  17,   4, -18,
  -23,  -3,  -1,  15,  10,  -3, -20, -22,
  -42, -20, -10,  -5,  -2, -20, -23, -44,
  -29, -51, -23, -15, -22, -18, -50, -64,
];
const MG_BISHOP = [
  -29,   4, -82, -37, -25, -42,   7,  -8,
  -26,  16, -18, -13,  30,  59,  18, -47,
  -16,  37,  43,  40,  35,  50,  37,  -2,
   -4,   5,  19,  50,  37,  37,   7,  -2,
   -6,  13,  13,  26,  34,  12,  10,   4,
    0,  15,  15,  15,  14,  27,  18,  10,
    4,  15,  16,   0,   7,  21,  33,   1,
  -33,  -3, -14, -21, -13, -12, -39, -21,
];
const EG_BISHOP = [
  -14, -21, -11,  -8,  -7,  -9, -17, -24,
   -8,  -4,   7, -12,  -3, -13,  -4, -14,
    2,  -8,   0,  -1,  -2,   6,   0,   4,
   -3,   9,  12,   9,  14,  10,   3,   2,
   -6,   3,  13,  19,   7,  10,  -3,  -9,
  -12,  -3,   8,  10,  13,   3,  -7, -15,
  -14, -18,  -7,  -1,   4,  -9, -15, -27,
  -23,  -9, -23,  -5,  -9, -16,  -5, -17,
];
const MG_ROOK = [
   32,  42,  32,  51,  63,   9,  31,  43,
   27,  32,  58,  62,  80,  67,  26,  44,
   -5,  19,  26,  36,  17,  45,  61,  16,
  -24, -11,   7,  26,  24,  35,  -8, -20,
  -36, -26, -12,  -1,   9,  -7,   6, -23,
  -45, -25, -16, -17,   3,   0,  -5, -33,
  -44, -16, -20,  -9,  -1,  11,  -6, -71,
  -19, -13,   1,  17,  16,   7, -37, -26,
];
const EG_ROOK = [
   13,  10,  18,  15,  12,  12,   8,   5,
   11,  13,  13,  11,  -3,   3,   8,   3,
    7,   7,   7,   5,   4,  -3,  -5,  -3,
    4,   3,  13,   1,   2,   1,  -1,   2,
    3,   5,   8,   4,  -5,  -6,  -8, -11,
   -4,   0,  -5,  -1,  -7, -12,  -8, -16,
   -6,  -6,   0,   2,  -9,  -9, -11,  -3,
   -9,   2,   3,  -1,  -5, -13,   4, -20,
];
const MG_QUEEN = [
  -28,   0,  29,  12,  59,  44,  43,  45,
  -24, -39,  -5,   1, -16,  57,  28,  54,
  -13, -17,   7,   8,  29,  56,  47,  57,
  -27, -27, -16, -16,  -1,  17,  -2,   1,
   -9, -26,  -9, -10,  -2,  -4,   3,  -3,
  -14,   2, -11,  -2,  -5,   2,  14,   5,
  -35,  -8,  11,   2,   8,  15,  -3,   1,
   -1, -18,  -9,  10, -15, -25, -31, -50,
];
const EG_QUEEN = [
   -9,  22,  22,  27,  27,  19,  10,  20,
  -17,  20,  32,  41,  58,  25,  30,   0,
  -20,   6,   9,  49,  47,  35,  19,   9,
    3,  22,  24,  45,  57,  40,  57,  36,
  -18,  28,  19,  47,  31,  34,  39,  23,
  -16, -27,  15,   6,   9,  17,  10,   5,
  -22, -23, -30, -16, -16, -23, -36, -32,
  -33, -28, -22, -43,  -5, -32, -20, -41,
];
const MG_KING = [
  -65,  23,  16, -15, -56, -34,   2,  13,
   29,  -1, -20,  -7,  -8,  -4, -38, -29,
   -9,  24,   2, -16, -20,   6,  22, -22,
  -17, -20, -12, -27, -30, -25, -14, -36,
  -49,  -1, -27, -39, -46, -44, -33, -51,
  -14, -14, -22, -46, -44, -30, -15, -27,
    1,   7,  -8, -64, -43, -16,   9,   8,
  -15,  36,  12, -54,   8, -28,  24,  14,
];
const EG_KING = [
  -74, -35, -18, -18, -11,  15,   4, -17,
  -12,  17,  14,  17,  17,  38,  23,  11,
   10,  17,  23,  15,  20,  45,  44,  13,
   -8,  22,  24,  27,  26,  33,  26,   3,
  -18,  -4,  21,  24,  27,  23,   9, -11,
  -19,  -3,  11,  21,  23,  16,   7,  -9,
  -27, -11,   4,  13,  14,   4,  -5, -17,
  -53, -34, -21, -11, -28, -14, -24, -43,
];

const MG_PST = [null, MG_PAWN, MG_KNIGHT, MG_BISHOP, MG_ROOK, MG_QUEEN, MG_KING];
const EG_PST = [null, EG_PAWN, EG_KNIGHT, EG_BISHOP, EG_ROOK, EG_QUEEN, EG_KING];

// ── Positional term weights ────────────────────────────────────────────────
// Indexed by the pawn's relative rank (0 = own back rank, 6 = one step from
// promoting). A passed pawn is worth far more in an endgame than with a full
// board still on, which is exactly what the taper is for.
const PASSED_MG = [0,  5, 12, 20,  35,  60, 100, 0];
const PASSED_EG = [0, 15, 25, 45,  80, 130, 200, 0];

const ISOLATED_MG = -14, ISOLATED_EG = -18;
const DOUBLED_MG  = -10, DOUBLED_EG  = -22;
const BACKWARD_MG =  -8, BACKWARD_EG = -10;
const BISHOP_PAIR_MG = 28, BISHOP_PAIR_EG = 48;
const ROOK_OPEN_MG = 32, ROOK_OPEN_EG = 12;
const ROOK_SEMI_MG = 14, ROOK_SEMI_EG =  8;
const ROOK_7TH_MG  = 22, ROOK_7TH_EG  = 32;
const TEMPO = 12;

// Mobility is counted as "safe squares this piece can reach", and weighted per
// piece type — a rook with no squares is nearly worthless, a knight cares less.
const MOB_MG = [0, 0, 4, 4, 3, 1, 0];
const MOB_EG = [0, 0, 4, 5, 5, 4, 0];

// King danger accumulates a weight per enemy piece attacking the king zone and
// is then squared, so three attackers are far worse than two.
//
// It takes at least MIN_KING_ATTACKERS of them to count for anything. A single
// piece with a line into the king zone is not an attack, and scoring it as one
// is badly wrong: without this gate a black queen sitting on its own starting
// square, seeing d2 down a half-open d-file, cost White 78cp — enough that the
// evaluation preferred giving up the d-pawn to being a pawn up.
const KING_ATT_WEIGHT = [0, 0, 18, 16, 28, 50, 0];
const MIN_KING_ATTACKERS = 2;
const KING_DANGER_DIV = 40;
const SHIELD_MISSING_MG = -18;

const DIRS_R = [[-1,0],[1,0],[0,1],[0,-1]];
const DIRS_B = [[-1,1],[-1,-1],[1,1],[1,-1]];
const KNIGHT_D = [[1,2],[1,-2],[-1,2],[-1,-2],[2,1],[2,-1],[-2,1],[-2,-1]];

// Precomputed step tables so the evaluation walks indices, not coordinates.
const N_TBL = [], K_ZONE = [], SLIDE_R = [], SLIDE_B = [];
(function build() {
  const ok = (r, c) => r >= 0 && r < 8 && c >= 0 && c < 8;
  for (let sq = 0; sq < 64; sq++) {
    const r = sq >> 3, c = sq & 7;
    const n = [];
    for (const [dr, dc] of KNIGHT_D) if (ok(r + dr, c + dc)) n.push((r + dr) * 8 + c + dc);
    N_TBL.push(Int8Array.from(n));

    // King zone: the king's square plus every square it could step to.
    const z = [sq];
    for (let dr = -1; dr <= 1; dr++)
      for (let dc = -1; dc <= 1; dc++)
        if ((dr || dc) && ok(r + dr, c + dc)) z.push((r + dr) * 8 + c + dc);
    K_ZONE.push(Int8Array.from(z));

    const rr = [], bb = [];
    for (const [dr, dc] of DIRS_R) {
      const line = []; let nr = r + dr, nc = c + dc;
      while (ok(nr, nc)) { line.push(nr * 8 + nc); nr += dr; nc += dc; }
      rr.push(Int8Array.from(line));
    }
    for (const [dr, dc] of DIRS_B) {
      const line = []; let nr = r + dr, nc = c + dc;
      while (ok(nr, nc)) { line.push(nr * 8 + nc); nr += dr; nc += dc; }
      bb.push(Int8Array.from(line));
    }
    SLIDE_R.push(rr); SLIDE_B.push(bb);
  }
})();

// Scratch state, reused so evaluation allocates nothing.
const wPawnCnt = new Int8Array(8), bPawnCnt = new Int8Array(8);
const wMostAdv = new Int8Array(8), bMostAdv = new Int8Array(8);  // rows
const wRearmost = new Int8Array(8), bRearmost = new Int8Array(8);
const pawnAtkW = new Uint8Array(64), pawnAtkB = new Uint8Array(64);
// King zones as square masks rather than lists: mobility already visits every
// reachable square, and a lookup there has to stay a single array read.
const zoneW = new Uint8Array(64), zoneB = new Uint8Array(64);

/**
 * Static evaluation in centipawns, from the side-to-move's point of view.
 * @param {import('./position.js').Position} pos
 */
export function evaluate(pos) {
  const b = pos.board;

  let mg = 0, eg = 0, phase = 0;
  let wBishops = 0, bBishops = 0;
  let wNonPawn = 0, bNonPawn = 0;
  let wPawns = 0, bPawns = 0;

  wPawnCnt.fill(0); bPawnCnt.fill(0);
  wMostAdv.fill(8); bMostAdv.fill(-1);
  wRearmost.fill(-1); bRearmost.fill(8);
  pawnAtkW.fill(0); pawnAtkB.fill(0);

  // ── Pass 1: material, piece-square tables, pawn skeleton ────────────────
  for (let sq = 0; sq < 64; sq++) {
    const p = b[sq];
    if (p === 0) continue;
    const white = p > 0;
    const pt = white ? p : -p;
    const tsq = white ? sq : sq ^ 56;

    if (white) { mg += MG_VAL[pt] + MG_PST[pt][tsq]; eg += EG_VAL[pt] + EG_PST[pt][tsq]; }
    else       { mg -= MG_VAL[pt] + MG_PST[pt][tsq]; eg -= EG_VAL[pt] + EG_PST[pt][tsq]; }
    phase += PHASE_INC[pt];

    const r = sq >> 3, c = sq & 7;
    if (pt === 1) {
      if (white) {
        wPawns++; wPawnCnt[c]++;
        if (r < wMostAdv[c]) wMostAdv[c] = r;
        if (r > wRearmost[c]) wRearmost[c] = r;
        if (c > 0 && r > 0) pawnAtkW[(r - 1) * 8 + c - 1] = 1;
        if (c < 7 && r > 0) pawnAtkW[(r - 1) * 8 + c + 1] = 1;
      } else {
        bPawns++; bPawnCnt[c]++;
        if (r > bMostAdv[c]) bMostAdv[c] = r;
        if (r < bRearmost[c]) bRearmost[c] = r;
        if (c > 0 && r < 7) pawnAtkB[(r + 1) * 8 + c - 1] = 1;
        if (c < 7 && r < 7) pawnAtkB[(r + 1) * 8 + c + 1] = 1;
      }
    } else if (pt === 3) {
      if (white) wBishops++; else bBishops++;
    }
    if (pt >= 2 && pt <= 5) { if (white) wNonPawn += MG_VAL[pt]; else bNonPawn += MG_VAL[pt]; }
  }

  if (wBishops >= 2) { mg += BISHOP_PAIR_MG; eg += BISHOP_PAIR_EG; }
  if (bBishops >= 2) { mg -= BISHOP_PAIR_MG; eg -= BISHOP_PAIR_EG; }

  // ── Pass 2: pawn structure ──────────────────────────────────────────────
  for (let c = 0; c < 8; c++) {
    if (wPawnCnt[c] > 1) { mg += (wPawnCnt[c] - 1) * DOUBLED_MG; eg += (wPawnCnt[c] - 1) * DOUBLED_EG; }
    if (bPawnCnt[c] > 1) { mg -= (bPawnCnt[c] - 1) * DOUBLED_MG; eg -= (bPawnCnt[c] - 1) * DOUBLED_EG; }

    const wAdj = (c > 0 && wPawnCnt[c - 1]) || (c < 7 && wPawnCnt[c + 1]);
    const bAdj = (c > 0 && bPawnCnt[c - 1]) || (c < 7 && bPawnCnt[c + 1]);
    if (wPawnCnt[c] && !wAdj) { mg += ISOLATED_MG; eg += ISOLATED_EG; }
    if (bPawnCnt[c] && !bAdj) { mg -= ISOLATED_MG; eg -= ISOLATED_EG; }

    // Backward: the rearmost pawn on the file is behind both neighbours and so
    // cannot be defended by them.
    if (wPawnCnt[c] && wAdj) {
      const left  = c > 0 ? wRearmost[c - 1] : -1;
      const right = c < 7 ? wRearmost[c + 1] : -1;
      const support = Math.max(left, right);           // larger row = further back
      if (support !== -1 && wRearmost[c] > support) { mg += BACKWARD_MG; eg += BACKWARD_EG; }
    }
    if (bPawnCnt[c] && bAdj) {
      const left  = c > 0 ? bRearmost[c - 1] : 8;
      const right = c < 7 ? bRearmost[c + 1] : 8;
      const support = Math.min(left, right);
      if (support !== 8 && bRearmost[c] < support) { mg -= BACKWARD_MG; eg -= BACKWARD_EG; }
    }

    // Passed pawns — only the most advanced pawn on a file can be passed.
    if (wPawnCnt[c]) {
      const r = wMostAdv[c];
      let passed = true;
      for (let f = Math.max(0, c - 1); f <= Math.min(7, c + 1); f++) {
        if (bPawnCnt[f] && bMostAdv[f] >= 0) {
          // any black pawn strictly ahead of (above) our pawn stops it
          if (bRearmost[f] < r) { passed = false; break; }
        }
      }
      if (passed) {
        const rel = 7 - r - 1;                        // 0 on rank 2 … 5 on rank 7
        const i = Math.max(0, Math.min(6, rel + 1));
        mg += PASSED_MG[i]; eg += PASSED_EG[i];
      }
    }
    if (bPawnCnt[c]) {
      const r = bMostAdv[c];
      let passed = true;
      for (let f = Math.max(0, c - 1); f <= Math.min(7, c + 1); f++) {
        if (wPawnCnt[f] && wRearmost[f] > r) { passed = false; break; }
      }
      if (passed) {
        const rel = r - 1;
        const i = Math.max(0, Math.min(6, rel + 1));
        mg -= PASSED_MG[i]; eg -= PASSED_EG[i];
      }
    }
  }

  // ── Pass 3: mobility, rook placement, king attacks ──────────────────────
  const wkSq = pos.kingSq[0], bkSq = pos.kingSq[1];
  let wKingDanger = 0, bKingDanger = 0;
  let wAttackers = 0, bAttackers = 0;

  zoneW.fill(0); zoneB.fill(0);
  const zw = K_ZONE[wkSq], zb = K_ZONE[bkSq];
  for (let i = 0; i < zw.length; i++) zoneW[zw[i]] = 1;
  for (let i = 0; i < zb.length; i++) zoneB[zb[i]] = 1;

  for (let sq = 0; sq < 64; sq++) {
    const p = b[sq];
    if (p === 0) continue;
    const white = p > 0;
    const pt = white ? p : -p;
    if (pt === 1 || pt === 6) continue;

    const unsafe = white ? pawnAtkB : pawnAtkW;
    const enemyKingZone = white ? zoneB : zoneW;
    let mob = 0, zoneHits = 0;

    if (pt === 2) {
      const tbl = N_TBL[sq];
      for (let i = 0; i < tbl.length; i++) {
        const to = tbl[i];
        const t = b[to];
        if (t !== 0 && (t > 0) === white) continue;
        if (!unsafe[to]) mob++;
        zoneHits += enemyKingZone[to];
      }
    } else {
      const useR = pt === 4 || pt === 5;
      const useB = pt === 3 || pt === 5;
      if (useR) {
        const rays = SLIDE_R[sq];
        for (let d = 0; d < 4; d++) {
          const ray = rays[d];
          for (let i = 0; i < ray.length; i++) {
            const to = ray[i], t = b[to];
            if (t === 0 || (t > 0) !== white) {
              if (!unsafe[to]) mob++;
              zoneHits += enemyKingZone[to];
            }
            if (t !== 0) break;
          }
        }
      }
      if (useB) {
        const rays = SLIDE_B[sq];
        for (let d = 0; d < 4; d++) {
          const ray = rays[d];
          for (let i = 0; i < ray.length; i++) {
            const to = ray[i], t = b[to];
            if (t === 0 || (t > 0) !== white) {
              if (!unsafe[to]) mob++;
              zoneHits += enemyKingZone[to];
            }
            if (t !== 0) break;
          }
        }
      }
    }

    const mMg = mob * MOB_MG[pt], mEg = mob * MOB_EG[pt];
    if (white) { mg += mMg; eg += mEg; if (zoneHits) { bKingDanger += KING_ATT_WEIGHT[pt]; bAttackers++; } }
    else       { mg -= mMg; eg -= mEg; if (zoneHits) { wKingDanger += KING_ATT_WEIGHT[pt]; wAttackers++; } }

    if (pt === 4) {
      const c = sq & 7, r = sq >> 3;
      const ownP = white ? wPawnCnt[c] : bPawnCnt[c];
      const oppP = white ? bPawnCnt[c] : wPawnCnt[c];
      let rMg = 0, rEg = 0;
      if (!ownP && !oppP)   { rMg += ROOK_OPEN_MG; rEg += ROOK_OPEN_EG; }
      else if (!ownP)       { rMg += ROOK_SEMI_MG; rEg += ROOK_SEMI_EG; }
      // A rook on the opponent's second rank cuts off their king and eats pawns.
      if ((white && r === 1) || (!white && r === 6)) { rMg += ROOK_7TH_MG; rEg += ROOK_7TH_EG; }
      if (white) { mg += rMg; eg += rEg; } else { mg -= rMg; eg -= rEg; }
    }
  }

  // ── King shelter ────────────────────────────────────────────────────────
  // Only a middlegame concern; in the endgame the king wants to be active, and
  // the endgame king table already says so.
  mg += shelter(b, wkSq, true, wPawnCnt, wMostAdv);
  mg -= shelter(b, bkSq, false, bPawnCnt, bMostAdv);

  if (wAttackers >= MIN_KING_ATTACKERS) mg -= (wKingDanger * wKingDanger) / KING_DANGER_DIV;
  if (bAttackers >= MIN_KING_ATTACKERS) mg += (bKingDanger * bKingDanger) / KING_DANGER_DIV;

  // ── Taper ───────────────────────────────────────────────────────────────
  const ph = Math.min(phase, TOTAL_PHASE);
  let score = (mg * ph + eg * (TOTAL_PHASE - ph)) / TOTAL_PHASE;

  // Endgames where the stronger side has no pawns are far more drawish than the
  // material count suggests: a lone extra knight or bishop cannot mate at all.
  if (score > 0 && wPawns === 0 && wNonPawn - bNonPawn < MG_VAL[4]) score /= 4;
  if (score < 0 && bPawns === 0 && bNonPawn - wNonPawn < MG_VAL[4]) score /= 4;

  score += pos.stm > 0 ? TEMPO : -TEMPO;
  score = score | 0;                       // the search works in whole centipawns
  return pos.stm > 0 ? score : -score;
}

// Penalty for missing pawn cover on the king's file and its neighbours.
function shelter(b, kSq, white, pawnCnt, mostAdv) {
  const kr = kSq >> 3, kc = kSq & 7;
  // Only bother once the king has actually tucked away on its own two ranks.
  if (white ? kr < 5 : kr > 2) return 0;
  let s = 0;
  for (let c = Math.max(0, kc - 1); c <= Math.min(7, kc + 1); c++) {
    if (!pawnCnt[c]) { s += SHIELD_MISSING_MG; continue; }
    const r = mostAdv[c];
    const dist = white ? kr - r : r - kr;
    if (dist > 3) s += SHIELD_MISSING_MG / 2;       // pawn has run too far up
  }
  return s;
}

/** Game phase in [0,1]: 1 = full material, 0 = bare kings. */
export function gamePhase(pos) {
  let phase = 0;
  for (let sq = 0; sq < 64; sq++) {
    const p = pos.board[sq];
    if (p === 0) continue;
    phase += PHASE_INC[p > 0 ? p : -p];
  }
  return Math.min(phase, TOTAL_PHASE) / TOTAL_PHASE;
}

export { MG_VAL as PIECE_MG_VALUES };
