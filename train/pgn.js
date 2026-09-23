'use strict';
// PGN reading for the trainer: split a file into games, read the result header,
// and replay the moves.
//
// The rules come from js/position.js — the same generator the engine plays
// with, verified against published perft counts. The trainer used to carry its
// own second copy of the chess rules, which meant the network could be trained
// on positions the engine would never actually reach.

import { Position, mvFrom, mvTo, mvPromo, mvPiece, mvIsCastle } from '../js/position.js';

/** Split a PGN file into individual game texts. */
export function splitGames(text) {
  const normalized = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  return normalized.split(/\n\n+(?=\[)/g)
    .map(g => g.trim())
    .filter(g => g.length > 0 && g.includes('.'));
}

/** +1 White won, −1 Black won, 0 draw, null unknown. */
export function parseResult(gameText) {
  const m = gameText.match(/\[Result\s+"([^"]+)"\]/);
  if (!m) return null;
  if (m[1] === '1-0')       return  1;
  if (m[1] === '0-1')       return -1;
  if (m[1].includes('1/2')) return  0;
  return null;
}

/** Strip headers, comments and variations; return the bare move tokens. */
export function moveTokens(gameText) {
  const moveText = gameText
    .replace(/\[[^\]]*\]/g, '')
    .replace(/\{[^}]*\}/g, '')
    .replace(/;[^\n]*/g, '')
    .replace(/\([^)]*\)/g, '')
    .replace(/\$\d+/g, '')
    .replace(/\d+\.\.\./g, ' ')
    .replace(/\d+\./g, ' ')
    .trim();

  return moveText.split(/\s+/)
    .filter(t => t && !/^(1-0|0-1|1\/2-1\/2|\*)$/.test(t));
}

const PIECE_LETTER = { N: 2, B: 3, R: 4, Q: 5, K: 6 };
const PROMO_LETTER = { Q: 5, R: 4, B: 3, N: 2 };

/**
 * Resolve one SAN token against a position. Returns the packed move, or 0 when
 * the token does not correspond to exactly one legal move.
 */
export function sanToMove(pos, san) {
  let s = san.replace(/[+#!?]+$/g, '').replace(/[!?]/g, '').trim();
  if (!s) return 0;

  const base = pos.ply * 256;
  const n = pos.generate(pos.ply, false);

  // Castling is written as a whole, not as a king move to a square.
  if (s === 'O-O' || s === '0-0') {
    return findLegal(pos, base, n, m => mvIsCastle(m) && (mvTo(m) & 7) === 6);
  }
  if (s === 'O-O-O' || s === '0-0-0') {
    return findLegal(pos, base, n, m => mvIsCastle(m) && (mvTo(m) & 7) === 2);
  }

  let promo = 0;
  const eq = s.indexOf('=');
  if (eq >= 0) { promo = PROMO_LETTER[s[eq + 1]] || 5; s = s.slice(0, eq); }

  let pieceType = 1;
  if (PIECE_LETTER[s[0]]) { pieceType = PIECE_LETTER[s[0]]; s = s.slice(1); }
  s = s.replace('x', '');

  if (s.length < 2) return 0;
  const tc = s.charCodeAt(s.length - 2) - 97;
  const tr = 8 - +s[s.length - 1];
  if (!(tc >= 0 && tc < 8 && tr >= 0 && tr < 8)) return 0;
  const to = tr * 8 + tc;
  s = s.slice(0, -2);

  // Whatever is left disambiguates the origin: a file, a rank, or both.
  let disFile = -1, disRank = -1;
  for (const ch of s) {
    if (ch >= 'a' && ch <= 'h') disFile = ch.charCodeAt(0) - 97;
    else if (ch >= '1' && ch <= '8') disRank = 8 - +ch;
  }

  // Some sources omit "=Q" on a promotion; a pawn reaching the last rank has to
  // promote to something, and a queen is what was meant.
  const wantPromo = promo || (pieceType === 1 && (tr === 0 || tr === 7) ? 5 : 0);

  return findLegal(pos, base, n, m =>
    mvPiece(m) === pieceType &&
    mvTo(m) === to &&
    mvPromo(m) === wantPromo &&
    (disFile < 0 || (mvFrom(m) & 7) === disFile) &&
    (disRank < 0 || (mvFrom(m) >> 3) === disRank)
  );
}

// Return the single legal move matching `pred`; 0 if none or if it is ambiguous.
function findLegal(pos, base, n, pred) {
  let found = 0;
  for (let i = 0; i < n; i++) {
    const m = pos.moveBuf[base + i];
    if (!pred(m)) continue;
    if (!pos.makeMove(m)) continue;
    pos.unmakeMove();
    if (found) return 0;          // ambiguous — refuse rather than guess
    found = m;
  }
  return found;
}

/** Material balance in pawn units, White-positive. */
const SIMPLE = [0, 1, 3, 3, 5, 9, 0];
export function material(board) {
  let s = 0;
  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (p > 0) s += SIMPLE[p];
    else if (p < 0) s -= SIMPLE[-p];
  }
  return s;
}

export { Position };
