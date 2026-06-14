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

const BATCH_SIZE    = 5000;
const BATCH_EPOCHS  = 5;   // more epochs per batch for deeper learning
const LEARNING_RATE = 0.001;
const Y_MAX = 39;          // label clamp in pawn-units
const Y_MIN = -39;
const SAMPLE_RATE   = 0.5; // sample 50% of positions (was 25%)
const SKIP_PLY      = 10;  // ignore first 10 half-moves (too similar across games)

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

// ── Model Definition ───────────────────────────────────────────────────────

function createModel() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 256, activation: 'relu', inputShape: [768] }));
  m.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  m.add(tf.layers.dense({ units:  64, activation: 'relu' }));
  m.add(tf.layers.dense({ units:  32, activation: 'relu' }));
  m.add(tf.layers.dense({ units:   1, activation: 'linear' }));
  m.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: 'meanSquaredError'
  });
  return m;
}

// ── Training Loop ──────────────────────────────────────────────────────────

async function trainBatch(model, xArr, yArr) {
  // Inputs: [0,1] → [-1,+1]; labels: pawn-units → [-1,+1]
  const xNorm = xArr.map(v => v * 2 - 1);
  const yNorm = yArr.map(v => v / Y_MAX);

  const xs = tf.tensor2d(xNorm, [yArr.length, 768]);
  const ys = tf.tensor2d(yNorm, [yArr.length, 1]);
  const h  = await model.fit(xs, ys, {
    epochs: BATCH_EPOCHS,
    batchSize: 256,
    shuffle: true,
    verbose: 0
  });
  xs.dispose(); ys.dispose();
  return h.history.loss[h.history.loss.length - 1];
}

async function main() {
  console.log('ChessNN Training Script (with game-result labels)');
  console.log('==================================================');
  console.log(`Batch size: ${BATCH_SIZE}, Epochs/batch: ${BATCH_EPOCHS}`);
  console.log(`Sample rate: ${SAMPLE_RATE * 100}%, Skip first ${SKIP_PLY} ply`);
  console.log(`PGN files: ${PGN_FILES.join(', ')}\n`);

  if (!fs.existsSync(MODEL_DIR)) fs.mkdirSync(MODEL_DIR, { recursive: true });

  const model = createModel();
  model.summary();

  let xBuf = new Float32Array(BATCH_SIZE * 768);
  let yBuf = new Float32Array(BATCH_SIZE);
  let nInBatch = 0;
  let totalPositions = 0;
  let totalGames = 0;
  let batchCount = 0;
  let lastLoss = null;
  let wonGames = 0, lostGames = 0, drawnGames = 0;

  async function flushBatch() {
    if (nInBatch === 0) return;
    batchCount++;
    const xSlice = Array.from(xBuf.slice(0, nInBatch * 768));
    const ySlice = Array.from(yBuf.slice(0, nInBatch));
    lastLoss = await trainBatch(model, xSlice, ySlice);
    nInBatch = 0;
    console.log(`  Batch ${batchCount}: loss=${lastLoss.toFixed(4)}, total=${totalPositions.toLocaleString()}`);
  }

  for (const file of PGN_FILES) {
    const filePath = path.join(PGN_DIR, file);
    if (!fs.existsSync(filePath)) { console.warn(`  Skipping ${file} (not found)`); continue; }

    console.log(`\nProcessing ${file}...`);
    const text = fs.readFileSync(filePath, 'utf8');
    const games = splitGames(text);
    console.log(`  Found ${games.length} games`);

    let filePositions = 0;

    for (const gameText of games) {
      // ── Parse game result for label blending ────────────────────────────
      const result = parseResult(gameText); // +1, -1, 0, or null
      if (result === 1) wonGames++;
      else if (result === -1) lostGames++;
      else if (result === 0) drawnGames++;

      const resultScore = result !== null ? result * Y_MAX : null;

      // ── Strip PGN headers and comments ──────────────────────────────────
      const moveText = gameText
        .replace(/\[[^\]]*\]/g, '')
        .replace(/\{[^}]*\}/g, '')
        .replace(/\([^)]*\)/g, '')
        .replace(/\d+\.\.\./g, '')
        .replace(/\d+\./g, ' ')
        .trim();

      const tokens = moveText.split(/\s+/)
        .filter(t => t && !/^(1-0|0-1|1\/2-1\/2|\*)$/.test(t));

      let state = initState();
      let side  = 'white';
      let ok    = true;
      let ply   = 0;

      for (const token of tokens) {
        if (!ok) break;
        const san = token.replace(/[+#!?]+$/g, '');
        if (!san) continue;

        const mv = sanToMove(san, state, side);
        if (!mv) { ok = false; break; }
        state = makeMove(state, mv);
        ply++;
        side = side === 'white' ? 'black' : 'white';

        // Skip first SKIP_PLY half-moves (opening book positions all look the same)
        if (ply <= SKIP_PLY) continue;

        if (Math.random() >= SAMPLE_RATE) continue;

        const matScore = simpleEval(state.board); // White-perspective, pawn units

        let label;
        if (resultScore !== null) {
          // Blend: ramp result weight from 20% at ply 10 → 80% at ply 80+
          const resultWeight = Math.min(0.8, (ply - SKIP_PLY) / 70 * 0.8 + 0.2);
          label = resultWeight * resultScore + (1 - resultWeight) * matScore;
        } else {
          label = matScore;
        }
        label = Math.max(Y_MIN, Math.min(Y_MAX, label));

        const vec = boardToVector(state.board);
        xBuf.set(vec, nInBatch * 768);
        yBuf[nInBatch] = label;
        nInBatch++;
        totalPositions++;
        filePositions++;

        if (nInBatch === BATCH_SIZE) await flushBatch();
      }
      totalGames++;
    }
    console.log(`  Collected ${filePositions.toLocaleString()} positions from this file`);
  }

  await flushBatch();

  console.log(`\nTraining complete!`);
  console.log(`  Games: ${totalGames.toLocaleString()} (W:${wonGames} D:${drawnGames} B:${lostGames})`);
  console.log(`  Total positions: ${totalPositions.toLocaleString()}`);
  console.log(`  Total batches:   ${batchCount}`);
  if (lastLoss !== null) console.log(`  Final loss:      ${lastLoss.toFixed(4)}`);

  console.log(`\nSaving model to ${MODEL_DIR}...`);
  await model.save(`file://${MODEL_DIR}`);

  const normParams = { xMin: 0, xMax: 1, yMin: Y_MIN, yMax: Y_MAX };
  fs.writeFileSync(path.join(MODEL_DIR, 'normalization.json'), JSON.stringify(normParams, null, 2));

  console.log('Done! Model saved.');
  console.log('Serve the project root with any HTTP server to play:');
  console.log('  cd .. && python3 -m http.server 8080');
}

main().catch(err => { console.error(err); process.exit(1); });
