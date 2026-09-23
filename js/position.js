'use strict';
// Fast mutable position — the representation the search runs on.
//
// js/chess.js stays the rules authority for the UI: it owns the 2-D board the
// DOM renders and the move objects the UI consumes. The search needs something
// different. It visits millions of nodes, and chess.js allocates a fresh 2-D
// board (nine arrays) for every single one of them, which dominated the old
// engine's runtime.
//
// Here the board is one flat Int8Array, moves are packed into int32s, and
// make/unmake works against an undo stack — a whole search runs without
// allocating. test/perft.js cross-checks this generator against chess.js on the
// standard perft suite, which is what keeps two move generators honest.
//
// Square index: sq = row * 8 + col, row 0 = rank 8 (Black's back rank), so
// sq 0 = a8 and sq 63 = h1 — the same orientation chess.js uses.
// Piece codes also match chess.js: +1..+6 = White pawn..king, negative = Black.

// ── Move encoding ──────────────────────────────────────────────────────────
// bits  0..5   from square
// bits  6..11  to square
// bits 12..14  promotion piece type (0 = none, else 2..5)
// bits 15..18  flags
// bits 19..22  moving piece type (1..6)
// bits 23..26  captured piece type (0..6)
export const FLAG_CAP    = 1 << 15;
export const FLAG_EP     = 1 << 16;
export const FLAG_CASTLE = 1 << 17;
export const FLAG_DBL    = 1 << 18;

export const NO_MOVE = 0;

export function mkMove(from, to, pieceType, capType, promo, flags) {
  return from | (to << 6) | (promo << 12) | flags |
         (pieceType << 19) | (capType << 23);
}
export function mvFrom(m)    { return m & 63; }
export function mvTo(m)      { return (m >>> 6) & 63; }
export function mvPromo(m)   { return (m >>> 12) & 7; }
export function mvPiece(m)   { return (m >>> 19) & 15; }
export function mvCapType(m) { return (m >>> 23) & 15; }
export function mvIsCap(m)   { return (m & FLAG_CAP) !== 0; }
export function mvIsEP(m)    { return (m & FLAG_EP) !== 0; }
export function mvIsCastle(m){ return (m & FLAG_CASTLE) !== 0; }
// "Quiet" for pruning purposes: neither wins material nor promotes.
export function mvIsQuiet(m) { return (m & (FLAG_CAP | FLAG_EP)) === 0 && ((m >>> 12) & 7) === 0; }

// ── Castling rights bits ───────────────────────────────────────────────────
export const CR_WK = 1, CR_WQ = 2, CR_BK = 4, CR_BQ = 8;

// Squares whose occupancy changing invalidates castling rights.
const CASTLE_MASK = new Uint8Array(64).fill(15);
CASTLE_MASK[56] = 15 & ~CR_WQ;            // a1
CASTLE_MASK[63] = 15 & ~CR_WK;            // h1
CASTLE_MASK[60] = 15 & ~(CR_WK | CR_WQ);  // e1
CASTLE_MASK[0]  = 15 & ~CR_BQ;            // a8
CASTLE_MASK[7]  = 15 & ~CR_BK;            // h8
CASTLE_MASK[4]  = 15 & ~(CR_BK | CR_BQ);  // e8

// ── Precomputed geometry ───────────────────────────────────────────────────
const DIRS = [[-1,0],[1,0],[0,1],[0,-1],[-1,1],[-1,-1],[1,1],[1,-1]];
// DIRS 0..3 are rook lines, 4..7 bishop lines.

const RAYS = [];        // RAYS[sq][dir] = Int8Array of squares along that ray
const KNIGHT_MOVES = []; // Int8Array per square
const KING_MOVES   = []; // Int8Array per square
const PAWN_FROM    = [[], []]; // [0]=white attackers, [1]=black: squares a pawn attacks sq from

(function buildTables() {
  const ok = (r, c) => r >= 0 && r < 8 && c >= 0 && c < 8;
  for (let sq = 0; sq < 64; sq++) {
    const r = sq >> 3, c = sq & 7;

    const rays = [];
    for (const [dr, dc] of DIRS) {
      const line = [];
      let nr = r + dr, nc = c + dc;
      while (ok(nr, nc)) { line.push(nr * 8 + nc); nr += dr; nc += dc; }
      rays.push(Int8Array.from(line));
    }
    RAYS.push(rays);

    const kn = [];
    for (const [dr, dc] of [[1,2],[1,-2],[-1,2],[-1,-2],[2,1],[2,-1],[-2,1],[-2,-1]]) {
      if (ok(r + dr, c + dc)) kn.push((r + dr) * 8 + c + dc);
    }
    KNIGHT_MOVES.push(Int8Array.from(kn));

    const kg = [];
    for (const [dr, dc] of DIRS) if (ok(r + dr, c + dc)) kg.push((r + dr) * 8 + c + dc);
    KING_MOVES.push(Int8Array.from(kg));

    // A White pawn moves toward row 0, so one attacking sq sits at row+1.
    const wf = [], bf = [];
    for (const dc of [-1, 1]) {
      if (ok(r + 1, c + dc)) wf.push((r + 1) * 8 + c + dc);
      if (ok(r - 1, c + dc)) bf.push((r - 1) * 8 + c + dc);
    }
    PAWN_FROM[0].push(Int8Array.from(wf));
    PAWN_FROM[1].push(Int8Array.from(bf));
  }
})();

// ── Zobrist keys ───────────────────────────────────────────────────────────
// 64-bit keys held as two 32-bit halves. Castling rights, the en-passant file
// and the side to move are all hashed: without them positions that differ only
// in those respects collide in the transposition table, which the previous
// engine did and which silently corrupts scores.
const Z_PIECE_LO = new Uint32Array(12 * 64);
const Z_PIECE_HI = new Uint32Array(12 * 64);
const Z_CASTLE_LO = new Uint32Array(16);
const Z_CASTLE_HI = new Uint32Array(16);
const Z_EP_LO = new Uint32Array(8);
const Z_EP_HI = new Uint32Array(8);
let Z_STM_LO = 0, Z_STM_HI = 0;

(function buildZobrist() {
  let s = 0x9e3779b9;
  const next = () => {
    // xorshift32 — fast and well-distributed enough for hashing
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;  s >>>= 0;
    return s;
  };
  for (let i = 0; i < 12 * 64; i++) { Z_PIECE_LO[i] = next(); Z_PIECE_HI[i] = next(); }
  const base = [];
  for (let i = 0; i < 4; i++) base.push([next(), next()]);
  for (let m = 0; m < 16; m++) {
    let lo = 0, hi = 0;
    for (let i = 0; i < 4; i++) if (m & (1 << i)) { lo ^= base[i][0]; hi ^= base[i][1]; }
    Z_CASTLE_LO[m] = lo >>> 0; Z_CASTLE_HI[m] = hi >>> 0;
  }
  for (let i = 0; i < 8; i++) { Z_EP_LO[i] = next(); Z_EP_HI[i] = next(); }
  Z_STM_LO = next(); Z_STM_HI = next();
})();

// piece code (±1..±6) → 0..11 plane index
export function pieceIndex(p) { return p > 0 ? p - 1 : 5 + (-p); }

// ── Undo record layout ─────────────────────────────────────────────────────
const U_STRIDE = 7;
const U_MOVE = 0, U_CAP = 1, U_CASTLE = 2, U_EP = 3, U_HALF = 4, U_LO = 5, U_HI = 6;

export const MAX_PLY = 128;
const MAX_MOVES = 256;
const HIST_CAP  = 2048;

// Hoisted out of the generator: array literals inside the hot move loop would
// allocate on every piece of every node.
const PROMO_ORDER = Object.freeze([5, 4, 3, 2]);
const PAWN_CAP_DC = Object.freeze([-1, 1]);

export class Position {
  constructor() {
    this.board    = new Int8Array(64);
    this.stm      = 1;          // 1 = White to move, -1 = Black
    this.castling = 0;
    this.ep       = -1;         // en-passant target square, -1 = none
    this.halfmove = 0;
    this.keyLo    = 0;
    this.keyHi    = 0;
    this.kingSq   = new Int8Array(2);  // [0] = White king, [1] = Black king
    this.ply      = 0;                 // plies played since the search root

    this.undo = new Int32Array(MAX_PLY * U_STRIDE);
    // Repetition history: keys of every position reached, including the ones
    // that happened in the real game before the search started.
    this.histLo = new Int32Array(HIST_CAP);
    this.histHi = new Int32Array(HIST_CAP);
    this.histN  = 0;
    this.rootHistN = 0;

    this.moveBuf = new Int32Array(MAX_PLY * MAX_MOVES);
  }

  // ── Setup ────────────────────────────────────────────────────────────────
  // Load from the UI's chess.js-style state object.
  setFromState(st, sideToMove) {
    const b = this.board;
    for (let r = 0; r < 8; r++)
      for (let c = 0; c < 8; c++) b[r * 8 + c] = st.board[r][c] | 0;

    this.stm = sideToMove === 'white' ? 1 : -1;
    this.castling = (st.wKc ? CR_WK : 0) | (st.wQc ? CR_WQ : 0) |
                    (st.bKc ? CR_BK : 0) | (st.bQc ? CR_BQ : 0);
    this.ep = st.enPassantTarget ? st.enPassantTarget[0] * 8 + st.enPassantTarget[1] : -1;
    // Only keep an en-passant square when a capture is actually available: it
    // keeps transposition entries and repetition keys from splitting on a
    // detail that cannot affect play.
    if (this.ep >= 0 && !this._epRelevant(this.ep, this.stm)) this.ep = -1;
    this.halfmove = st.halfmoveClock | 0;
    this.ply = 0;
    this.histN = 0;
    this.rootHistN = 0;
    this._locateKings();
    this._rehash();
    return this;
  }

  // FEN is only used by the test harnesses (perft, tactical suites), but it
  // lives here so those tests drive exactly the code the engine runs.
  setFromFEN(fen) {
    const parts = fen.trim().split(/\s+/);
    this.board.fill(0);
    let sq = 0;
    for (const ch of parts[0]) {
      if (ch === '/') continue;
      if (ch >= '1' && ch <= '8') { sq += +ch; continue; }
      const t = { p: 1, n: 2, b: 3, r: 4, q: 5, k: 6 }[ch.toLowerCase()];
      this.board[sq++] = ch === ch.toUpperCase() ? t : -t;
    }
    this.stm = parts[1] === 'b' ? -1 : 1;
    const cr = parts[2] || '-';
    this.castling = (cr.includes('K') ? CR_WK : 0) | (cr.includes('Q') ? CR_WQ : 0) |
                    (cr.includes('k') ? CR_BK : 0) | (cr.includes('q') ? CR_BQ : 0);
    this.ep = -1;
    if (parts[3] && parts[3] !== '-') {
      const f = parts[3].charCodeAt(0) - 97;
      const r = 8 - +parts[3][1];
      const e = r * 8 + f;
      if (this._epRelevant(e, this.stm)) this.ep = e;
    }
    this.halfmove = parts[4] ? +parts[4] : 0;
    this.ply = 0; this.histN = 0; this.rootHistN = 0;
    this._locateKings();
    this._rehash();
    return this;
  }

  // Convert back to the chess.js state shape, for cross-checking the two move
  // generators against each other.
  toUIState() {
    const board = [];
    for (let r = 0; r < 8; r++) {
      const row = [];
      for (let c = 0; c < 8; c++) row.push(this.board[r * 8 + c]);
      board.push(row);
    }
    return {
      state: {
        board,
        wKc: !!(this.castling & CR_WK), wQc: !!(this.castling & CR_WQ),
        bKc: !!(this.castling & CR_BK), bQc: !!(this.castling & CR_BQ),
        lastMove: null, selected: null,
        enPassantTarget: this.ep >= 0 ? [this.ep >> 3, this.ep & 7] : null,
        halfmoveClock: this.halfmove
      },
      side: this.stm > 0 ? 'white' : 'black'
    };
  }

  // Seed the repetition history with keys from earlier in the real game so the
  // search knows when a line would repeat a position already played.
  setHistory(keys) {
    this.histN = 0;
    for (let i = 0; i + 1 < keys.length; i += 2) {
      this.histLo[this.histN] = keys[i] | 0;
      this.histHi[this.histN] = keys[i + 1] | 0;
      this.histN++;
      if (this.histN >= HIST_CAP - MAX_PLY - 2) break;
    }
    this.rootHistN = this.histN;
  }

  clone() {
    const p = new Position();
    p.board.set(this.board);
    p.stm = this.stm; p.castling = this.castling; p.ep = this.ep;
    p.halfmove = this.halfmove; p.keyLo = this.keyLo; p.keyHi = this.keyHi;
    p.kingSq.set(this.kingSq);
    p.histLo.set(this.histLo); p.histHi.set(this.histHi);
    p.histN = this.histN; p.rootHistN = this.rootHistN;
    return p;
  }

  _locateKings() {
    for (let sq = 0; sq < 64; sq++) {
      if (this.board[sq] === 6) this.kingSq[0] = sq;
      else if (this.board[sq] === -6) this.kingSq[1] = sq;
    }
  }

  _rehash() {
    let lo = 0, hi = 0;
    for (let sq = 0; sq < 64; sq++) {
      const p = this.board[sq];
      if (p !== 0) { const i = pieceIndex(p) * 64 + sq; lo ^= Z_PIECE_LO[i]; hi ^= Z_PIECE_HI[i]; }
    }
    lo ^= Z_CASTLE_LO[this.castling]; hi ^= Z_CASTLE_HI[this.castling];
    if (this.ep >= 0) { lo ^= Z_EP_LO[this.ep & 7]; hi ^= Z_EP_HI[this.ep & 7]; }
    if (this.stm < 0) { lo ^= Z_STM_LO; hi ^= Z_STM_HI; }
    this.keyLo = lo | 0; this.keyHi = hi | 0;
  }

  // Would a pawn of `bySide` actually be able to capture onto the ep square?
  _epRelevant(epSq, bySide) {
    const capturerRow = bySide > 0 ? (epSq >> 3) + 1 : (epSq >> 3) - 1;
    if (capturerRow < 0 || capturerRow > 7) return false;
    const c = epSq & 7;
    const want = bySide;                       // a pawn of the capturing side
    if (c > 0 && this.board[capturerRow * 8 + c - 1] === want) return true;
    if (c < 7 && this.board[capturerRow * 8 + c + 1] === want) return true;
    return false;
  }

  // ── Attacks ──────────────────────────────────────────────────────────────
  isAttacked(sq, bySign) {
    const b = this.board;

    const pf = PAWN_FROM[bySign > 0 ? 0 : 1][sq];
    for (let i = 0; i < pf.length; i++) if (b[pf[i]] === bySign) return true;

    const kn = KNIGHT_MOVES[sq];
    const wantN = bySign * 2;
    for (let i = 0; i < kn.length; i++) if (b[kn[i]] === wantN) return true;

    const kg = KING_MOVES[sq];
    const wantK = bySign * 6;
    for (let i = 0; i < kg.length; i++) if (b[kg[i]] === wantK) return true;

    const rays = RAYS[sq];
    const wantR = bySign * 4, wantQ = bySign * 5, wantB = bySign * 3;
    for (let d = 0; d < 4; d++) {
      const ray = rays[d];
      for (let i = 0; i < ray.length; i++) {
        const p = b[ray[i]];
        if (p !== 0) { if (p === wantR || p === wantQ) return true; break; }
      }
    }
    for (let d = 4; d < 8; d++) {
      const ray = rays[d];
      for (let i = 0; i < ray.length; i++) {
        const p = b[ray[i]];
        if (p !== 0) { if (p === wantB || p === wantQ) return true; break; }
      }
    }
    return false;
  }

  inCheck(side) {
    const s = side === undefined ? this.stm : side;
    return this.isAttacked(this.kingSq[s > 0 ? 0 : 1], -s);
  }

  // ── Move generation ──────────────────────────────────────────────────────
  // Writes pseudo-legal moves into this.moveBuf at the slot for `ply` and
  // returns how many were written. `capturesOnly` also keeps promotions, since
  // quiescence needs those.
  generate(ply, capturesOnly) {
    const b = this.board;
    const us = this.stm, them = -us;
    const buf = this.moveBuf;
    const base = ply * MAX_MOVES;
    let n = 0;

    const pawnPromoRow = us > 0 ? 0 : 7;
    const pawnStartRow = us > 0 ? 6 : 1;
    const fwd = us > 0 ? -8 : 8;

    for (let sq = 0; sq < 64; sq++) {
      const p = b[sq];
      if (p === 0 || (p > 0) !== (us > 0)) continue;
      const pt = p > 0 ? p : -p;

      if (pt === 1) {
        const r = sq >> 3, c = sq & 7;
        const one = sq + fwd;
        if (b[one] === 0) {
          const promoRow = (one >> 3) === pawnPromoRow;
          if (promoRow) {
            for (let k = 0; k < 4; k++) buf[base + n++] = mkMove(sq, one, 1, 0, PROMO_ORDER[k], 0);
          } else if (!capturesOnly) {
            buf[base + n++] = mkMove(sq, one, 1, 0, 0, 0);
            if (r === pawnStartRow) {
              const two = one + fwd;
              if (b[two] === 0) buf[base + n++] = mkMove(sq, two, 1, 0, 0, FLAG_DBL);
            }
          }
        }
        for (let k = 0; k < 2; k++) {
          const dc = PAWN_CAP_DC[k];
          const nc = c + dc;
          if (nc < 0 || nc > 7) continue;
          const to = one + dc;
          const t = b[to];
          if (t !== 0 && (t > 0) !== (us > 0)) {
            const ct = t > 0 ? t : -t;
            if ((to >> 3) === pawnPromoRow) {
              for (let q = 0; q < 4; q++) buf[base + n++] = mkMove(sq, to, 1, ct, PROMO_ORDER[q], FLAG_CAP);
            } else {
              buf[base + n++] = mkMove(sq, to, 1, ct, 0, FLAG_CAP);
            }
          } else if (to === this.ep && t === 0) {
            buf[base + n++] = mkMove(sq, to, 1, 1, 0, FLAG_CAP | FLAG_EP);
          }
        }

      } else if (pt === 2 || pt === 6) {
        const tbl = pt === 2 ? KNIGHT_MOVES[sq] : KING_MOVES[sq];
        for (let i = 0; i < tbl.length; i++) {
          const to = tbl[i];
          const t = b[to];
          if (t !== 0 && (t > 0) === (us > 0)) continue;
          if (t === 0) { if (!capturesOnly) buf[base + n++] = mkMove(sq, to, pt, 0, 0, 0); }
          else buf[base + n++] = mkMove(sq, to, pt, t > 0 ? t : -t, 0, FLAG_CAP);
        }

      } else {
        const d0 = pt === 3 ? 4 : 0;
        const d1 = pt === 4 ? 4 : 8;
        const rays = RAYS[sq];
        for (let d = d0; d < d1; d++) {
          const ray = rays[d];
          for (let i = 0; i < ray.length; i++) {
            const to = ray[i];
            const t = b[to];
            if (t === 0) { if (!capturesOnly) buf[base + n++] = mkMove(sq, to, pt, 0, 0, 0); continue; }
            if ((t > 0) !== (us > 0)) buf[base + n++] = mkMove(sq, to, pt, t > 0 ? t : -t, 0, FLAG_CAP);
            break;
          }
        }
      }
    }

    // Castling — legality (king not passing through attacked squares) is
    // checked here rather than after the move, since make/unmake only verifies
    // that the mover's king is not left en prise.
    if (!capturesOnly) {
      if (us > 0) {
        if ((this.castling & CR_WK) && b[61] === 0 && b[62] === 0 && b[63] === 4 &&
            !this.isAttacked(60, them) && !this.isAttacked(61, them) && !this.isAttacked(62, them))
          buf[base + n++] = mkMove(60, 62, 6, 0, 0, FLAG_CASTLE);
        if ((this.castling & CR_WQ) && b[59] === 0 && b[58] === 0 && b[57] === 0 && b[56] === 4 &&
            !this.isAttacked(60, them) && !this.isAttacked(59, them) && !this.isAttacked(58, them))
          buf[base + n++] = mkMove(60, 58, 6, 0, 0, FLAG_CASTLE);
      } else {
        if ((this.castling & CR_BK) && b[5] === 0 && b[6] === 0 && b[7] === -4 &&
            !this.isAttacked(4, them) && !this.isAttacked(5, them) && !this.isAttacked(6, them))
          buf[base + n++] = mkMove(4, 6, 6, 0, 0, FLAG_CASTLE);
        if ((this.castling & CR_BQ) && b[3] === 0 && b[2] === 0 && b[1] === 0 && b[0] === -4 &&
            !this.isAttacked(4, them) && !this.isAttacked(3, them) && !this.isAttacked(2, them))
          buf[base + n++] = mkMove(4, 2, 6, 0, 0, FLAG_CASTLE);
      }
    }

    return n;
  }

  // ── Make / unmake ────────────────────────────────────────────────────────
  // Returns false and leaves the position untouched if the move would leave the
  // mover in check (generation is pseudo-legal).
  makeMove(m) {
    const b = this.board;
    const us = this.stm;
    const from = m & 63, to = (m >>> 6) & 63;
    const promo = (m >>> 12) & 7;
    const piece = b[from];
    const pt = piece > 0 ? piece : -piece;

    const u = this.ply * U_STRIDE;
    const undo = this.undo;
    undo[u + U_MOVE]   = m;
    undo[u + U_CASTLE] = this.castling;
    undo[u + U_EP]     = this.ep;
    undo[u + U_HALF]   = this.halfmove;
    undo[u + U_LO]     = this.keyLo;
    undo[u + U_HI]     = this.keyHi;

    let lo = this.keyLo, hi = this.keyHi;

    // Retire the old castling / ep contributions before anything changes.
    lo ^= Z_CASTLE_LO[this.castling]; hi ^= Z_CASTLE_HI[this.castling];
    if (this.ep >= 0) { lo ^= Z_EP_LO[this.ep & 7]; hi ^= Z_EP_HI[this.ep & 7]; }

    // Remove the captured piece.
    let capSq = -1, capPiece = 0;
    if (m & FLAG_EP) {
      capSq = (from & ~7) | (to & 7);          // same rank as the mover, file of the target
      capPiece = b[capSq];
    } else if (b[to] !== 0) {
      capSq = to; capPiece = b[to];
    }
    undo[u + U_CAP] = capPiece;
    if (capPiece !== 0) {
      const ci = pieceIndex(capPiece) * 64 + capSq;
      lo ^= Z_PIECE_LO[ci]; hi ^= Z_PIECE_HI[ci];
      b[capSq] = 0;
    }

    // Move the piece (applying promotion).
    const fi = pieceIndex(piece) * 64 + from;
    lo ^= Z_PIECE_LO[fi]; hi ^= Z_PIECE_HI[fi];
    const landed = promo ? (us > 0 ? promo : -promo) : piece;
    const ti = pieceIndex(landed) * 64 + to;
    lo ^= Z_PIECE_LO[ti]; hi ^= Z_PIECE_HI[ti];
    b[from] = 0;
    b[to] = landed;

    if (pt === 6) this.kingSq[us > 0 ? 0 : 1] = to;

    // Castling moves the rook too.
    if (m & FLAG_CASTLE) {
      let rf, rt;
      if (to === 62)      { rf = 63; rt = 61; }
      else if (to === 58) { rf = 56; rt = 59; }
      else if (to === 6)  { rf = 7;  rt = 5;  }
      else                { rf = 0;  rt = 3;  }
      const rook = b[rf];
      b[rf] = 0; b[rt] = rook;
      const ri = pieceIndex(rook) * 64;
      lo ^= Z_PIECE_LO[ri + rf] ^ Z_PIECE_LO[ri + rt];
      hi ^= Z_PIECE_HI[ri + rf] ^ Z_PIECE_HI[ri + rt];
    }

    this.castling &= CASTLE_MASK[from] & CASTLE_MASK[to];

    // En passant is only recorded when the capture is genuinely available.
    this.ep = -1;
    if (m & FLAG_DBL) {
      const epSq = (from + to) >> 1;
      if (this._epRelevant(epSq, -us)) this.ep = epSq;
    }

    this.halfmove = (pt === 1 || capPiece !== 0) ? 0 : this.halfmove + 1;

    lo ^= Z_CASTLE_LO[this.castling]; hi ^= Z_CASTLE_HI[this.castling];
    if (this.ep >= 0) { lo ^= Z_EP_LO[this.ep & 7]; hi ^= Z_EP_HI[this.ep & 7]; }
    lo ^= Z_STM_LO; hi ^= Z_STM_HI;

    this.keyLo = lo | 0; this.keyHi = hi | 0;
    this.stm = -us;
    this.ply++;

    // Reject the move if it left our own king attacked.
    if (this.isAttacked(this.kingSq[us > 0 ? 0 : 1], -us)) {
      this.unmakeMove();
      return false;
    }

    this.histLo[this.histN] = this.keyLo;
    this.histHi[this.histN] = this.keyHi;
    this.histN++;
    return true;
  }

  unmakeMove() {
    this.ply--;
    const u = this.ply * U_STRIDE;
    const undo = this.undo;
    const m = undo[u + U_MOVE];
    const b = this.board;

    const from = m & 63, to = (m >>> 6) & 63;
    const promo = (m >>> 12) & 7;

    this.stm = -this.stm;
    const us = this.stm;

    const landed = b[to];
    b[to] = 0;
    b[from] = promo ? (us > 0 ? 1 : -1) : landed;

    if ((landed > 0 ? landed : -landed) === 6) this.kingSq[us > 0 ? 0 : 1] = from;

    if (m & FLAG_CASTLE) {
      let rf, rt;
      if (to === 62)      { rf = 63; rt = 61; }
      else if (to === 58) { rf = 56; rt = 59; }
      else if (to === 6)  { rf = 7;  rt = 5;  }
      else                { rf = 0;  rt = 3;  }
      b[rf] = b[rt]; b[rt] = 0;
    }

    const capPiece = undo[u + U_CAP];
    if (capPiece !== 0) {
      const capSq = (m & FLAG_EP) ? ((from & ~7) | (to & 7)) : to;
      b[capSq] = capPiece;
    }

    this.castling = undo[u + U_CASTLE];
    this.ep       = undo[u + U_EP];
    this.halfmove = undo[u + U_HALF];
    this.keyLo    = undo[u + U_LO];
    this.keyHi    = undo[u + U_HI];
    if (this.histN > this.rootHistN) this.histN--;
  }

  // Treat the current position as a fresh root, dropping the undo stack and the
  // repetition history. Replaying a whole game move by move would otherwise run
  // `ply` past MAX_PLY, overflowing the undo and move buffers; the trainer
  // never takes a move back, so it commits after each one.
  commit() {
    this.ply = 0;
    this.histN = 0;
    this.rootHistN = 0;
  }

  // A null move hands the turn over without moving — used by null-move pruning.
  makeNull() {
    const u = this.ply * U_STRIDE;
    const undo = this.undo;
    undo[u + U_MOVE]   = NO_MOVE;
    undo[u + U_CAP]    = 0;
    undo[u + U_CASTLE] = this.castling;
    undo[u + U_EP]     = this.ep;
    undo[u + U_HALF]   = this.halfmove;
    undo[u + U_LO]     = this.keyLo;
    undo[u + U_HI]     = this.keyHi;

    let lo = this.keyLo, hi = this.keyHi;
    if (this.ep >= 0) { lo ^= Z_EP_LO[this.ep & 7]; hi ^= Z_EP_HI[this.ep & 7]; }
    lo ^= Z_STM_LO; hi ^= Z_STM_HI;
    this.keyLo = lo | 0; this.keyHi = hi | 0;

    this.ep = -1;
    this.stm = -this.stm;
    this.halfmove++;
    this.ply++;
    // Deliberately not pushed to the repetition history: a null move does not
    // occur in a real game, and recording it would invent repetitions.
  }

  unmakeNull() {
    this.ply--;
    const u = this.ply * U_STRIDE;
    const undo = this.undo;
    this.stm      = -this.stm;
    this.castling = undo[u + U_CASTLE];
    this.ep       = undo[u + U_EP];
    this.halfmove = undo[u + U_HALF];
    this.keyLo    = undo[u + U_LO];
    this.keyHi    = undo[u + U_HI];
  }

  // ── Draw detection ───────────────────────────────────────────────────────
  // One earlier occurrence is enough inside the tree: a line that can repeat
  // once can repeat twice, so treating it as drawn keeps the search from
  // chasing or fearing phantom progress.
  isRepetition() {
    const lo = this.keyLo, hi = this.keyHi;
    const stop = Math.max(0, this.histN - 1 - this.halfmove);
    for (let i = this.histN - 3; i >= stop; i -= 2) {
      if (this.histLo[i] === lo && this.histHi[i] === hi) return true;
    }
    return false;
  }

  isFiftyMove() { return this.halfmove >= 100; }

  // K v K, K+minor v K, and K+B v K+B on same-coloured squares.
  insufficientMaterial() {
    let minors = 0, bishops = 0, lightB = 0, darkB = 0;
    for (let sq = 0; sq < 64; sq++) {
      const p = this.board[sq];
      if (p === 0) continue;
      const pt = p > 0 ? p : -p;
      if (pt === 6) continue;
      if (pt === 1 || pt === 4 || pt === 5) return false;
      minors++;
      if (pt === 3) { bishops++; if (((sq >> 3) + (sq & 7)) % 2 === 0) lightB++; else darkB++; }
    }
    if (minors <= 1) return true;
    if (minors === 2 && bishops === 2 && (lightB === 2 || darkB === 2)) return true;
    return false;
  }

  hasNonPawnMaterial(side) {
    const s = side === undefined ? this.stm : side;
    for (let sq = 0; sq < 64; sq++) {
      const p = this.board[sq];
      if (p === 0 || (p > 0) !== (s > 0)) continue;
      const pt = p > 0 ? p : -p;
      if (pt >= 2 && pt <= 5) return true;
    }
    return false;
  }

  // ── Static exchange evaluation ───────────────────────────────────────────
  // Plays out the capture sequence on one square with the cheapest available
  // attacker each time, and returns the net material in centipawns. Used to
  // order captures and to throw away losing ones instead of searching them.
  see(m) {
    const to = (m >>> 6) & 63;
    const from = m & 63;
    const b = this.board;
    const gain = SEE_GAIN;

    // Occupancy is tracked by blanking squares as pieces are consumed and
    // restoring them at the end — cheaper than copying the board, and it lets
    // x-rays behind a consumed piece join the exchange.
    const removed = SEE_REMOVED;
    let nRemoved = 0;

    if (m & FLAG_EP) {
      const epCapSq = (from & ~7) | (to & 7);
      removed[nRemoved++] = epCapSq; removed[nRemoved++] = b[epCapSq];
      b[epCapSq] = 0;
    }

    const promo = (m >>> 12) & 7;
    const capType = (m >>> 23) & 15;
    gain[0] = (m & FLAG_EP) ? SEE_VAL[1] : SEE_VAL[capType];
    if (promo) gain[0] += SEE_VAL[promo] - SEE_VAL[1];

    const mover = b[from];
    // Value of whatever now stands on the square and is itself capturable.
    let onSquare = promo ? SEE_VAL[promo] : SEE_VAL[mover > 0 ? mover : -mover];
    removed[nRemoved++] = from; removed[nRemoved++] = mover;
    b[from] = 0;

    let side = -this.stm;
    let d = 0;

    while (true) {
      const nextSq = this._leastValuableAttacker(to, side);
      if (nextSq < 0) break;
      d++;
      gain[d] = onSquare - gain[d - 1];
      // Neither side would enter the exchange from here on.
      if (Math.max(-gain[d - 1], gain[d]) < 0) break;
      const np = b[nextSq];
      onSquare = SEE_VAL[np > 0 ? np : -np];
      removed[nRemoved++] = nextSq; removed[nRemoved++] = np;
      b[nextSq] = 0;
      side = -side;
      if (d >= 30) break;
    }

    for (let i = nRemoved - 2; i >= 0; i -= 2) b[removed[i]] = removed[i + 1];

    while (d > 0) { gain[d - 1] = -Math.max(-gain[d - 1], gain[d]); d--; }
    return gain[0];
  }

  // Cheapest piece of `side` that attacks `sq`, with the current (partially
  // blanked) occupancy — so x-rays behind a consumed piece are picked up.
  _leastValuableAttacker(sq, side) {
    const b = this.board;

    const pf = PAWN_FROM[side > 0 ? 0 : 1][sq];
    for (let i = 0; i < pf.length; i++) if (b[pf[i]] === side) return pf[i];

    const kn = KNIGHT_MOVES[sq], wantN = side * 2;
    for (let i = 0; i < kn.length; i++) if (b[kn[i]] === wantN) return kn[i];

    const rays = RAYS[sq];
    const wantB = side * 3, wantR = side * 4, wantQ = side * 5;
    let queenSq = -1;
    for (let d = 4; d < 8; d++) {
      const ray = rays[d];
      for (let i = 0; i < ray.length; i++) {
        const p = b[ray[i]];
        if (p === 0) continue;
        if (p === wantB) return ray[i];
        if (p === wantQ && queenSq < 0) queenSq = ray[i];
        break;
      }
    }
    for (let d = 0; d < 4; d++) {
      const ray = rays[d];
      for (let i = 0; i < ray.length; i++) {
        const p = b[ray[i]];
        if (p === 0) continue;
        if (p === wantR) return ray[i];
        if (p === wantQ && queenSq < 0) queenSq = ray[i];
        break;
      }
    }
    if (queenSq >= 0) return queenSq;

    const kg = KING_MOVES[sq], wantK = side * 6;
    for (let i = 0; i < kg.length; i++) if (b[kg[i]] === wantK) return kg[i];

    return -1;
  }
}

const SEE_VAL     = [0, 100, 320, 330, 500, 900, 20000];
const SEE_GAIN    = new Int32Array(32);
const SEE_REMOVED = new Int32Array(64);

// ── Interop with js/chess.js move objects ──────────────────────────────────
// The UI speaks in {from:[r,c], to:[r,c], piece, captured, …} objects. Rather
// than reconstructing one — and risking a field the UI depends on being subtly
// wrong — match the engine's move against the list chess.js itself generated.
export function matchUIMove(m, uiMoves) {
  const from = mvFrom(m), to = mvTo(m), promo = mvPromo(m);
  const fr = from >> 3, fc = from & 7, tr = to >> 3, tc = to & 7;
  for (const mv of uiMoves) {
    if (mv.from[0] !== fr || mv.from[1] !== fc) continue;
    if (mv.to[0] !== tr || mv.to[1] !== tc) continue;
    if (promo) {
      const want = Math.abs(mv.piece);
      if (!mv.promotion || want !== promo) continue;
    } else if (mv.promotion) {
      continue;
    }
    return mv;
  }
  return null;
}
