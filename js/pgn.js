'use strict';
// PGN parser — ES module, used by both the browser and the training script wrapper

import { initState, makeMove, generateMoves, filterLegal, opposite } from './chess.js';

export function splitGames(text) {
  const normalized = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  return normalized.split(/\n\n+(?=\[)/g)
    .map(g => g.trim())
    .filter(g => g.length > 0 && g.includes('.'));
}

export function parseGame(gameText) {
  // Returns array of {vec: Float32Array(768), score: number}
  // Replays the game and samples 1-in-4 positions
  const samples = [];

  let text = gameText
    .replace(/\[[^\]]*\]/g, '')  // remove header tags
    .replace(/\{[^}]*\}/g, '')   // remove comments
    .replace(/\([^)]*\)/g, '')   // remove variations
    .replace(/\d+\.\.\./g, '')   // remove black move numbers
    .replace(/\d+\./g, ' ')      // remove move numbers
    .trim();

  const tokens = text.split(/\s+/).filter(t => t && !/^(1-0|0-1|1\/2-1\/2|\*)$/.test(t));

  let state = initState();
  let side = 'white';

  for (const token of tokens) {
    const san = token.replace(/[+#!?]+$/g, '');
    if (!san) continue;

    const mv = sanToMove(san, state, side);
    if (!mv) continue;

    state = makeMove(state, mv);

    if (Math.random() < 0.25) {
      samples.push({
        vec: boardToVector(state.board),
        score: simpleEval(state.board)
      });
    }
    side = opposite(side);
  }

  if (samples.length === 0 && state) {
    samples.push({
      vec: boardToVector(state.board),
      score: simpleEval(state.board)
    });
  }

  return samples;
}

function boardToVector(board) {
  const vec = new Float32Array(768);
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const piece = board[r][c];
      if (piece !== 0) {
        const sq = r * 8 + c;
        const idx = piece > 0 ? piece - 1 : 6 + (-piece) - 1;
        vec[idx * 64 + sq] = 1;
      }
    }
  }
  return vec;
}

function simpleEval(board) {
  const vals = [0, 1, 3, 3, 5, 9, 0];
  let score = 0;
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p > 0) score += vals[p];
      else if (p < 0) score -= vals[-p];
    }
  return score;
}

export function sanToMove(san, state, side) {
  const clean = san.replace(/[+#!?]/g, '').trim();
  if (!clean) return null;

  // Castling
  if (clean === 'O-O' || clean === '0-0') {
    const r = side === 'white' ? 7 : 0;
    return { from: [r, 4], to: [r, 6], piece: state.board[r][4], captured: 0, castle: 'K' };
  }
  if (clean === 'O-O-O' || clean === '0-0-0') {
    const r = side === 'white' ? 7 : 0;
    return { from: [r, 4], to: [r, 2], piece: state.board[r][4], captured: 0, castle: 'Q' };
  }

  let s = clean;
  let promoPiece = null;

  // Promotion: e8=Q
  if (s.includes('=')) {
    const parts = s.split('=');
    promoPiece = parts[1][0];
    s = parts[0];
  }

  // Piece type
  const pieceChars = { N: 2, B: 3, R: 4, Q: 5, K: 6 };
  let pieceType = 1; // default pawn
  if (pieceChars[s[0]]) {
    pieceType = pieceChars[s[0]];
    s = s.slice(1);
  }

  // Remove 'x' (capture indicator)
  s = s.replace('x', '');

  if (s.length < 2) return null;

  // Destination square (last two chars)
  const destFile = s[s.length - 2];
  const destRank = s[s.length - 1];
  if (destFile < 'a' || destFile > 'h') return null;
  const tc = destFile.charCodeAt(0) - 'a'.charCodeAt(0);
  const tr = 8 - parseInt(destRank, 10);
  s = s.slice(0, -2);

  // Disambiguation
  let disFile = -1, disRank = -1;
  for (const ch of s) {
    if (ch >= 'a' && ch <= 'h') disFile = ch.charCodeAt(0) - 'a'.charCodeAt(0);
    else if (ch >= '1' && ch <= '8') disRank = 8 - parseInt(ch, 10);
  }

  const friendSign = side === 'white' ? 1 : -1;
  const allMoves = filterLegal(generateMoves(state, side), state, side);

  for (const mv of allMoves) {
    const [fr, fc] = mv.from;
    if (Math.abs(mv.piece) !== pieceType) continue;
    if (mv.to[0] !== tr || mv.to[1] !== tc) continue;
    if (disFile !== -1 && fc !== disFile) continue;
    if (disRank !== -1 && fr !== disRank) continue;

    if (promoPiece && mv.promotion) {
      const promoMap = { Q: 5, R: 4, B: 3, N: 2 };
      const wantedCode = promoMap[promoPiece] * friendSign;
      if (mv.piece !== wantedCode) continue;
    } else if (mv.promotion && !promoPiece) {
      // Default: pick queen promotion
      if (Math.abs(mv.piece) !== 5) continue;
    }

    return mv;
  }
  return null;
}
