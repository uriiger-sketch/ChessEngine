'use strict';
// ChessNN Training Script — improved with game-result labels
// Labels blend material balance (early game) with actual game outcome (late game),
// giving the NN genuine positional understanding beyond piece counting.
//
// Usage: cd train && npm install && node train.js

const tf   = require('@tensorflow/tfjs-node');
const fs   = require('fs');
const path = require('path');

const PGN_DIR   = path.join(__dirname, '..');
const MODEL_DIR = path.join(__dirname, '..', 'model');
const PGN_FILES = [
  '1Carlsen.pgn', '2Caruana.pgn', '3Fischer.pgn',
  '4Capablanca.pgn', '5Kasparov.pgn', '6Nakamura.pgn', '7Tal.pgn'
];

// ── Hyperparameters ────────────────────────────────────────────────────────
// Label design (this is the part that decides whether the net is any good):
//
//   label = materialBalance + RESULT_BONUS × gameResult × phaseRamp
//
// Material is the backbone: it is exactly computable and therefore perfectly
// learnable, which anchors the network in reality. The game result is layered
// on top only as a small positional nudge — it encodes "masters who reached
// this kind of position went on to win", which is real signal, but at the
// level of a single position the eventual result is mostly noise. Scaling the
// result term to a whole-game-deciding magnitude makes that noise swamp the
// material signal and the network learns nothing useful, so RESULT_BONUS is
// deliberately kept to roughly a piece-fragment of advantage.
const RESULT_BONUS  = 1.5;   // pawn-units of positional credit for winning
const Y_SCALE       = 10;    // labels normalised by ±10 pawns
const Y_CLAMP       = 10;    // clamp labels to the normalised range

const EPOCHS        = 8;
const BATCH_SIZE    = 512;
const LEARNING_RATE = 0.001;
const VAL_FRACTION  = 0.02;  // held-out set to measure real generalisation
const SAMPLE_RATE   = 0.35;  // fraction of positions kept per game
const SKIP_PLY      = 8;     // ignore opening book plies (identical across games)
const MAX_POSITIONS = 1_400_000;
const MAX_PIECES    = 32;    // slots per position in the sparse store

// ── Chess Board Logic ──────────────────────────────────────────────────────

const SIMPLE_VALS = [0, 1, 3, 3, 5, 9, 0];

function initState() {
  return {
    board: [
      [-4,-2,-3,-5,-6,-3,-2,-4],
      [-1,-1,-1,-1,-1,-1,-1,-1],
      [ 0, 0, 0, 0, 0, 0, 0, 0],
      [ 0, 0, 0, 0, 0, 0, 0, 0],
      [ 0, 0, 0, 0, 0, 0, 0, 0],
      [ 0, 0, 0, 0, 0, 0, 0, 0],
      [ 1, 1, 1, 1, 1, 1, 1, 1],
      [ 4, 2, 3, 5, 6, 3, 2, 4]
    ],
    wKc: true, wQc: true, bKc: true, bQc: true,
    enPassantTarget: null, lastMove: null
  };
}

function cloneState(st) {
  return {
    board: st.board.map(r => [...r]),
    wKc: st.wKc, wQc: st.wQc, bKc: st.bKc, bQc: st.bQc,
    enPassantTarget: st.enPassantTarget ? [...st.enPassantTarget] : null,
    lastMove: st.lastMove
  };
}

function inBounds(r, c) { return r >= 0 && r < 8 && c >= 0 && c < 8; }

function squareAttacked(board, r, c, attackerSide) {
  const aSign = attackerSide === 'white' ? 1 : -1;
  const pr = r + aSign;
  if (inBounds(pr, c-1) && board[pr][c-1] === aSign) return true;
  if (inBounds(pr, c+1) && board[pr][c+1] === aSign) return true;
  for (const [dr,dc] of [[1,2],[1,-2],[-1,2],[-1,-2],[2,1],[2,-1],[-2,1],[-2,-1]]) {
    const nr=r+dr, nc=c+dc;
    if (inBounds(nr,nc) && board[nr][nc] === aSign*2) return true;
  }
  for (const [dr,dc] of [[1,0],[-1,0],[0,1],[0,-1]]) {
    let nr=r+dr, nc=c+dc;
    while (inBounds(nr,nc)) {
      if (board[nr][nc] !== 0) { if (board[nr][nc]===aSign*4||board[nr][nc]===aSign*5) return true; break; }
      nr+=dr; nc+=dc;
    }
  }
  for (const [dr,dc] of [[1,1],[1,-1],[-1,1],[-1,-1]]) {
    let nr=r+dr, nc=c+dc;
    while (inBounds(nr,nc)) {
      if (board[nr][nc] !== 0) { if (board[nr][nc]===aSign*3||board[nr][nc]===aSign*5) return true; break; }
      nr+=dr; nc+=dc;
    }
  }
  for (const [dr,dc] of [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]) {
    const nr=r+dr, nc=c+dc;
    if (inBounds(nr,nc) && board[nr][nc]===aSign*6) return true;
  }
  return false;
}

function findKing(board, side) {
  const code = side === 'white' ? 6 : -6;
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) if (board[r][c]===code) return [r,c];
  return null;
}

function generateMoves(st, side) {
  const moves = [];
  const board = st.board;
  const fs = side === 'white' ? 1 : -1;
  const es = -fs;

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p === 0 || Math.sign(p) !== fs) continue;
      const pt = Math.abs(p);

      if (pt === 1) {
        const fwd = fs === 1 ? -1 : 1;
        const sr = fs === 1 ? 6 : 1;
        const er = fs === 1 ? 3 : 4;
        const pr = fs === 1 ? 0 : 7;
        const nr = r + fwd;
        if (inBounds(nr, c) && board[nr][c] === 0) {
          addPawnMove(moves, r, c, nr, c, p, 0, fs, pr);
          if (r === sr && board[r+2*fwd][c] === 0) moves.push({from:[r,c],to:[r+2*fwd,c],piece:p,captured:0});
        }
        for (const dc of [-1, 1]) {
          const cc = c + dc;
          if (!inBounds(nr, cc)) continue;
          if (Math.sign(board[nr][cc]) === es) addPawnMove(moves, r, c, nr, cc, p, board[nr][cc], fs, pr);
          else if (st.enPassantTarget && r === er && nr === st.enPassantTarget[0] && cc === st.enPassantTarget[1])
            moves.push({from:[r,c],to:[nr,cc],piece:p,captured:fs===1?-1:1,enPassant:true});
        }
      } else if (pt === 2) {
        for (const [dr,dc] of [[1,2],[1,-2],[-1,2],[-1,-2],[2,1],[2,-1],[-2,1],[-2,-1]]) {
          const nr=r+dr, nc=c+dc;
          if (inBounds(nr,nc) && Math.sign(board[nr][nc]) !== fs)
            moves.push({from:[r,c],to:[nr,nc],piece:p,captured:board[nr][nc]});
        }
      } else if (pt >= 3 && pt <= 5) {
        const dirs = pt===3?[[1,1],[1,-1],[-1,1],[-1,-1]]:pt===4?[[1,0],[-1,0],[0,1],[0,-1]]:[[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]];
        for (const [dr,dc] of dirs) {
          let nr=r+dr, nc=c+dc;
          while (inBounds(nr,nc)) {
            if (board[nr][nc]===0) moves.push({from:[r,c],to:[nr,nc],piece:p,captured:0});
            else { if (Math.sign(board[nr][nc])===es) moves.push({from:[r,c],to:[nr,nc],piece:p,captured:board[nr][nc]}); break; }
            nr+=dr; nc+=dc;
          }
        }
      } else if (pt === 6) {
        for (const [dr,dc] of [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]) {
          const nr=r+dr, nc=c+dc;
          if (inBounds(nr,nc) && Math.sign(board[nr][nc]) !== fs) moves.push({from:[r,c],to:[nr,nc],piece:p,captured:board[nr][nc]});
        }
        if (side==='white'&&r===7&&c===4) {
          if (st.wKc&&board[7][5]===0&&board[7][6]===0&&board[7][7]===4&&!squareAttacked(board,7,4,'black')&&!squareAttacked(board,7,5,'black')&&!squareAttacked(board,7,6,'black'))
            moves.push({from:[7,4],to:[7,6],piece:6,captured:0,castle:'K'});
          if (st.wQc&&board[7][3]===0&&board[7][2]===0&&board[7][1]===0&&board[7][0]===4&&!squareAttacked(board,7,4,'black')&&!squareAttacked(board,7,3,'black')&&!squareAttacked(board,7,2,'black'))
            moves.push({from:[7,4],to:[7,2],piece:6,captured:0,castle:'Q'});
        }
        if (side==='black'&&r===0&&c===4) {
          if (st.bKc&&board[0][5]===0&&board[0][6]===0&&board[0][7]===-4&&!squareAttacked(board,0,4,'white')&&!squareAttacked(board,0,5,'white')&&!squareAttacked(board,0,6,'white'))
            moves.push({from:[0,4],to:[0,6],piece:-6,captured:0,castle:'K'});
          if (st.bQc&&board[0][3]===0&&board[0][2]===0&&board[0][1]===0&&board[0][0]===-4&&!squareAttacked(board,0,4,'white')&&!squareAttacked(board,0,3,'white')&&!squareAttacked(board,0,2,'white'))
            moves.push({from:[0,4],to:[0,2],piece:-6,captured:0,castle:'Q'});
        }
      }
    }
  }
  return moves;
}

function addPawnMove(moves, fr, fc, tr, tc, piece, captured, fs, promoRow) {
  if (tr === promoRow) {
    for (const code of [5,4,3,2]) moves.push({from:[fr,fc],to:[tr,tc],piece:fs*code,captured,promotion:true});
  } else {
    moves.push({from:[fr,fc],to:[tr,tc],piece,captured});
  }
}

function makeMove(st, mv) {
  const next = cloneState(st);
  const board = next.board;
  const [fr,fc] = mv.from, [tr,tc] = mv.to;
  board[tr][tc] = mv.piece;
  board[fr][fc] = 0;
  if (mv.enPassant) board[fr][tc] = 0;
  if (mv.castle) {
    if (tr===7&&tc===6){board[7][5]=4;board[7][7]=0;}
    if (tr===7&&tc===2){board[7][3]=4;board[7][0]=0;}
    if (tr===0&&tc===6){board[0][5]=-4;board[0][7]=0;}
    if (tr===0&&tc===2){board[0][3]=-4;board[0][0]=0;}
  }
  if (Math.abs(mv.piece)===6){if(mv.piece>0){next.wKc=false;next.wQc=false;}else{next.bKc=false;next.bQc=false;}}
  if (fr===7&&fc===7) next.wKc=false;
  if (fr===7&&fc===0) next.wQc=false;
  if (fr===0&&fc===7) next.bKc=false;
  if (fr===0&&fc===0) next.bQc=false;
  next.enPassantTarget = (Math.abs(mv.piece)===1&&Math.abs(tr-fr)===2) ? [(fr+tr)/2, tc] : null;
  next.lastMove = mv;
  return next;
}

function isLegal(st, mv, side) {
  const next = makeMove(st, mv);
  const kp = findKing(next.board, side);
  if (!kp) return false;
  return !squareAttacked(next.board, kp[0], kp[1], side==='white'?'black':'white');
}

function getLegalMoves(st, side) {
  return generateMoves(st, side).filter(mv => isLegal(st, mv, side));
}

// ── PGN Parsing ────────────────────────────────────────────────────────────

function splitGames(text) {
  const normalized = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  return normalized.split(/\n\n+(?=\[)/g)
    .map(g => g.trim())
    .filter(g => g.length > 0 && g.includes('.'));
}

// Extract game result from PGN headers: returns +1 (white), -1 (black), 0 (draw), null (unknown)
function parseResult(gameText) {
  const m = gameText.match(/\[Result\s+"([^"]+)"\]/);
  if (!m) return null;
  if (m[1] === '1-0')         return  1;
  if (m[1] === '0-1')         return -1;
  if (m[1].includes('1/2'))   return  0;
  return null;
}

function sanToMove(san, state, side) {
  const clean = san.replace(/[+#!?]/g, '').trim();
  if (!clean) return null;

  if (clean === 'O-O' || clean === '0-0') {
    const r = side === 'white' ? 7 : 0;
    return {from:[r,4],to:[r,6],piece:state.board[r][4],captured:0,castle:'K'};
  }
  if (clean === 'O-O-O' || clean === '0-0-0') {
    const r = side === 'white' ? 7 : 0;
    return {from:[r,4],to:[r,2],piece:state.board[r][4],captured:0,castle:'Q'};
  }

  let s = clean;
  let promoPiece = null;
  if (s.includes('=')) { const pp = s.split('='); promoPiece = pp[1][0]; s = pp[0]; }

  const pMap = {N:2,B:3,R:4,Q:5,K:6};
  let pieceType = 1;
  if (pMap[s[0]]) { pieceType = pMap[s[0]]; s = s.slice(1); }
  s = s.replace('x','');

  if (s.length < 2) return null;
  const tc = s[s.length-2].charCodeAt(0) - 97;
  const tr = 8 - parseInt(s[s.length-1], 10);
  if (tc < 0 || tc > 7 || tr < 0 || tr > 7) return null;
  s = s.slice(0, -2);

  let disFile = -1, disRank = -1;
  for (const ch of s) {
    if (ch >= 'a' && ch <= 'h') disFile = ch.charCodeAt(0) - 97;
    else if (ch >= '1' && ch <= '8') disRank = 8 - parseInt(ch, 10);
  }

  const fs = side === 'white' ? 1 : -1;
  for (const mv of getLegalMoves(state, side)) {
    const [fr, fc] = mv.from;
    if (Math.abs(mv.piece) !== pieceType) continue;
    if (mv.to[0] !== tr || mv.to[1] !== tc) continue;
    if (disFile !== -1 && fc !== disFile) continue;
    if (disRank !== -1 && fr !== disRank) continue;
    if (promoPiece && mv.promotion) {
      const pm = {Q:5,R:4,B:3,N:2};
      if (mv.piece !== fs * pm[promoPiece]) continue;
    } else if (mv.promotion && !promoPiece) {
      if (Math.abs(mv.piece) !== 5) continue;
    }
    return mv;
  }
  return null;
}

// Material balance from White's perspective in pawn units
function simpleEval(board) {
  let score = 0;
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p > 0) score += SIMPLE_VALS[p];
      else if (p < 0) score -= SIMPLE_VALS[-p];
    }
  return score;
}

function boardToVector(board) {
  const vec = new Float32Array(768);
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p !== 0) {
        const sq = r * 8 + c;
        const idx = p > 0 ? p - 1 : 6 + (-p) - 1;
        vec[idx * 64 + sq] = 1;
      }
    }
  return vec;
}

// ── Sparse Position Store ──────────────────────────────────────────────────
// A dense 768-float row per position would need ~4 GB for a million positions.
// Only ≤32 squares are ever occupied, so each position is stored as its list
// of set input indices and expanded to a dense row per batch.

class PositionStore {
  constructor(capacity) {
    this.capacity = capacity;
    this.idx    = new Uint16Array(capacity * MAX_PIECES);
    this.counts = new Uint8Array(capacity);
    this.labels = new Float32Array(capacity);
    this.n = 0;
  }

  add(board, label) {
    if (this.n >= this.capacity) return false;
    const base = this.n * MAX_PIECES;
    let k = 0;
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const p = board[r][c];
        if (p === 0 || k >= MAX_PIECES) continue;
        const plane = p > 0 ? p - 1 : 5 + (-p);   // W P..K = 0..5, B p..k = 6..11
        this.idx[base + k++] = plane * 64 + r * 8 + c;
      }
    }
    this.counts[this.n] = k;
    this.labels[this.n] = label;
    this.n++;
    return true;
  }
}

// Colour-mirror an input index: flip the board vertically and swap piece
// colours. Applied with the label negated, this teaches the network that chess
// is colour-symmetric — without it the net happily scores the symmetric start
// position as a large advantage for one side.
function mirrorIndex(i) {
  const plane = (i / 64) | 0;
  const sq    = i % 64;
  const r     = (sq / 8) | 0;
  const c     = sq % 8;
  const mPlane = plane < 6 ? plane + 6 : plane - 6;
  return mPlane * 64 + (7 - r) * 8 + c;
}

// Expand a range of stored positions into a dense batch.
//
// Inputs stay as sparse 0/1 rather than being rescaled to ±1. That matters a
// lot: with ±1 the 736 empty squares become a constant −1 background that
// dominates every gradient, and the network fails to learn even exact material
// (measured: 2.52 pawns RMSE with ±1 vs 0.61 with 0/1 on the same task).
// js/neural.js uses the identical encoding at runtime.
function buildBatch(store, order, from, to, mirror) {
  const rows = to - from;
  const xs = new Float32Array(rows * 768);
  const ys = new Float32Array(rows);
  for (let b = 0; b < rows; b++) {
    const p    = order[from + b];
    const base = p * MAX_PIECES;
    const cnt  = store.counts[p];
    const flip = mirror && (b & 1) === 0;      // mirror half of every batch
    const off  = b * 768;
    for (let k = 0; k < cnt; k++) {
      const i = store.idx[base + k];
      xs[off + (flip ? mirrorIndex(i) : i)] = 1;
    }
    ys[b] = (flip ? -store.labels[p] : store.labels[p]) / Y_SCALE;
  }
  return { xs, ys, rows };
}

function shuffleInPlace(arr, rand) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
  }
}

// Deterministic RNG so training runs are reproducible.
function makeRand(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// ── Model Definition ───────────────────────────────────────────────────────

function createModel() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 256, activation: 'relu', inputShape: [768] }));
  m.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  m.add(tf.layers.dense({ units:  64, activation: 'relu' }));
  m.add(tf.layers.dense({ units:  32, activation: 'relu' }));
  m.add(tf.layers.dense({ units:   1, activation: 'linear' }));
  m.compile({ optimizer: tf.train.adam(LEARNING_RATE), loss: 'meanSquaredError' });
  return m;
}

// ── Sanity Probes ──────────────────────────────────────────────────────────
// A good evaluation net must score the symmetric start position near zero and
// must move decisively in the right direction when material is removed.

function probeBoards() {
  const start = initState().board;
  const clone = b => b.map(r => [...r]);
  const noBQ = clone(start); noBQ[0][3] = 0;
  const noWQ = clone(start); noWQ[7][3] = 0;
  const noBR = clone(start); noBR[0][0] = 0; noBR[0][7] = 0;
  const noWN = clone(start); noWN[7][1] = 0; noWN[7][6] = 0;
  return [
    ['start position (expect ~0.0)',      start, 0],
    ['black queen removed (expect ~+9)',  noBQ,  9],
    ['white queen removed (expect ~-9)',  noWQ, -9],
    ['both black rooks gone (expect ~+10)', noBR, 10],
    ['both white knights gone (expect ~-6)', noWN, -6],
  ];
}

function runProbes(model) {
  const probes = probeBoards();
  const xs = new Float32Array(probes.length * 768);
  probes.forEach(([, board], bi) => {
    for (let r = 0; r < 8; r++)
      for (let c = 0; c < 8; c++) {
        const p = board[r][c];
        if (p !== 0) {
          const plane = p > 0 ? p - 1 : 5 + (-p);
          xs[bi * 768 + plane * 64 + r * 8 + c] = 1;
        }
      }
  });
  const t = tf.tensor2d(xs, [probes.length, 768]);
  const out = model.predict(t);
  const vals = out.dataSync();
  t.dispose(); out.dispose();

  console.log('\nSanity probes (pawn units, + = White better):');
  let allGood = true;
  probes.forEach(([label, , expected], i) => {
    const got = vals[i] * Y_SCALE;
    const ok = Math.abs(got - expected) <= Math.max(2.5, Math.abs(expected) * 0.45);
    if (!ok) allGood = false;
    console.log(`  ${ok ? 'OK  ' : 'BAD '} ${label.padEnd(40)} → ${got >= 0 ? '+' : ''}${got.toFixed(2)}`);
  });
  return allGood;
}

// ── Data Collection ────────────────────────────────────────────────────────

function collectPositions() {
  const store = new PositionStore(MAX_POSITIONS);
  const rand = makeRand(12345);
  let totalGames = 0, parsedGames = 0;
  let wWins = 0, bWins = 0, draws = 0;

  for (const file of PGN_FILES) {
    const filePath = path.join(PGN_DIR, file);
    if (!fs.existsSync(filePath)) { console.warn(`  Skipping ${file} (not found)`); continue; }

    process.stdout.write(`  ${file.padEnd(18)}`);
    const games = splitGames(fs.readFileSync(filePath, 'utf8'));
    const before = store.n;

    for (const gameText of games) {
      totalGames++;
      const result = parseResult(gameText);   // +1 White, −1 Black, 0 draw, null unknown
      if (result === 1) wWins++; else if (result === -1) bWins++; else if (result === 0) draws++;

      const moveText = gameText
        .replace(/\[[^\]]*\]/g, '')
        .replace(/\{[^}]*\}/g, '')
        .replace(/\([^)]*\)/g, '')
        .replace(/\d+\.\.\./g, '')
        .replace(/\d+\./g, ' ')
        .trim();

      const tokens = moveText.split(/\s+/)
        .filter(t => t && !/^(1-0|0-1|1\/2-1\/2|\*)$/.test(t));
      if (tokens.length < SKIP_PLY + 4) continue;

      let state = initState();
      let side  = 'white';
      let ply   = 0;
      let ok    = true;

      for (const token of tokens) {
        const san = token.replace(/[+#!?]+$/g, '');
        if (!san) continue;
        const mv = sanToMove(san, state, side);
        if (!mv) { ok = false; break; }
        state = makeMove(state, mv);
        ply++;
        side = side === 'white' ? 'black' : 'white';

        if (ply <= SKIP_PLY) continue;
        if (rand() >= SAMPLE_RATE) continue;

        const material = simpleEval(state.board);   // White-positive, pawn units
        // Positional credit ramps in as the game progresses: near the start the
        // outcome says little, by move ~40 it reflects a genuine advantage.
        const ramp  = Math.min(1, (ply - SKIP_PLY) / 60);
        const bonus = result === null ? 0 : RESULT_BONUS * result * ramp;
        const label = Math.max(-Y_CLAMP, Math.min(Y_CLAMP, material + bonus));

        if (!store.add(state.board, label)) break;
      }
      if (ok) parsedGames++;
      if (store.n >= store.capacity) break;
    }
    console.log(`${(store.n - before).toLocaleString()} positions`);
    if (store.n >= store.capacity) { console.log('  (position cap reached)'); break; }
  }

  return { store, totalGames, parsedGames, wWins, bWins, draws };
}

// ── Training ───────────────────────────────────────────────────────────────

async function main() {
  console.log('ChessNN Training');
  console.log('================');
  console.log(`Label      : material + ${RESULT_BONUS} × result × phaseRamp  (clamped ±${Y_CLAMP}, scaled by ${Y_SCALE})`);
  console.log(`Sampling   : ${SAMPLE_RATE * 100}% of plies after ply ${SKIP_PLY}`);
  console.log(`Training   : ${EPOCHS} epochs, batch ${BATCH_SIZE}, lr ${LEARNING_RATE}`);
  console.log(`Augment    : colour-mirroring (half of every batch)`);
  console.log(`PGN files  : ${PGN_FILES.length}\n`);

  if (!fs.existsSync(MODEL_DIR)) fs.mkdirSync(MODEL_DIR, { recursive: true });

  console.log('Reading games...');
  const t0 = Date.now();
  const { store, totalGames, parsedGames, wWins, bWins, draws } = collectPositions();
  console.log(`\n  Games seen      : ${totalGames.toLocaleString()} (fully parsed: ${parsedGames.toLocaleString()})`);
  console.log(`  Results         : White ${wWins}, Draw ${draws}, Black ${bWins}`);
  console.log(`  Positions stored: ${store.n.toLocaleString()}`);
  console.log(`  Collection time : ${((Date.now() - t0) / 1000).toFixed(0)}s`);

  if (store.n < 1000) { console.error('Not enough positions — aborting.'); process.exit(1); }

  // Global shuffle. Streaming batches in file order would let the last author
  // in the list dominate the final weights.
  const rand  = makeRand(999);
  const order = new Uint32Array(store.n);
  for (let i = 0; i < store.n; i++) order[i] = i;
  shuffleInPlace(order, rand);

  const valCount   = Math.max(2000, Math.floor(store.n * VAL_FRACTION));
  const valOrder   = order.slice(0, valCount);
  const trainOrder = order.slice(valCount);
  console.log(`  Train / val     : ${trainOrder.length.toLocaleString()} / ${valOrder.length.toLocaleString()}`);

  const model = createModel();
  model.summary();

  // Baseline: how well does "always predict the dataset mean" do? Any useful
  // model must beat this comfortably.
  let mean = 0;
  for (let i = 0; i < trainOrder.length; i++) mean += store.labels[trainOrder[i]] / Y_SCALE;
  mean /= trainOrder.length;
  let baseline = 0;
  for (let i = 0; i < valOrder.length; i++) {
    const d = store.labels[valOrder[i]] / Y_SCALE - mean;
    baseline += d * d;
  }
  baseline /= valOrder.length;
  console.log(`\nBaseline val MSE (predict-the-mean): ${baseline.toFixed(5)}\n`);

  const val = buildBatch(store, valOrder, 0, valOrder.length, false);
  const valXs = tf.tensor2d(val.xs, [val.rows, 768]);
  const valYs = tf.tensor2d(val.ys, [val.rows, 1]);

  const CHUNK = 50_000;   // positions expanded to dense form at a time
  const tTrain = Date.now();

  for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    shuffleInPlace(trainOrder, rand);
    let seen = 0, lossSum = 0, lossN = 0;

    for (let start = 0; start < trainOrder.length; start += CHUNK) {
      const end = Math.min(start + CHUNK, trainOrder.length);
      const { xs, ys, rows } = buildBatch(store, trainOrder, start, end, true);
      const xt = tf.tensor2d(xs, [rows, 768]);
      const yt = tf.tensor2d(ys, [rows, 1]);

      const h = await model.fit(xt, yt, {
        epochs: 1, batchSize: BATCH_SIZE, shuffle: true, verbose: 0
      });
      lossSum += h.history.loss[0]; lossN++;
      seen += rows;
      xt.dispose(); yt.dispose();

      process.stdout.write(`\r  epoch ${epoch}/${EPOCHS}  ${seen.toLocaleString()}/${trainOrder.length.toLocaleString()}  loss ${(lossSum / lossN).toFixed(5)}   `);
    }

    const ev = model.evaluate(valXs, valYs);
    const valLoss = (Array.isArray(ev) ? ev[0] : ev).dataSync()[0];
    if (Array.isArray(ev)) ev.forEach(t => t.dispose()); else ev.dispose();
    console.log(`\r  epoch ${epoch}/${EPOCHS}  train ${(lossSum / lossN).toFixed(5)}  val ${valLoss.toFixed(5)}  (baseline ${baseline.toFixed(5)})        `);
  }

  console.log(`\nTraining time: ${((Date.now() - tTrain) / 1000 / 60).toFixed(1)} min`);

  const passed = runProbes(model);

  valXs.dispose(); valYs.dispose();

  console.log(`\nSaving model to ${MODEL_DIR}...`);
  await model.save(`file://${MODEL_DIR}`);
  fs.writeFileSync(
    path.join(MODEL_DIR, 'normalization.json'),
    JSON.stringify({ xMin: 0, xMax: 1, yMin: -Y_SCALE, yMax: Y_SCALE }, null, 2)
  );

  console.log(passed
    ? '\nDone — probes passed, model saved.'
    : '\nDone — model saved, but some probes look off (see above).');
  console.log('Serve the project root to play:  python3 -m http.server 8080');
}

main().catch(err => { console.error(err); process.exit(1); });
