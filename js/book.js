'use strict';
// Opening book: what the network-only opponent plays in the first 9 moves, and
// the opening names shown during a game. Built by train/book.js from the
// master games in the repository plus the main lines of opening literature.

import { getLegalMoves } from './chess.js';
import { zobristOf } from './engine.js';

let book = null;
let loading = null;

export function bookReady() { return !!book; }

/** Fetch model/book.json (relative to this module, so it works in workers). */
export function loadBook() {
  if (loading) return loading;
  loading = fetch(new URL('../model/book.json', import.meta.url))
    .then(r => (r.ok ? r.json() : null))
    .then(j => { if (j && j.pos) book = j; })
    .catch(() => {});
  return loading;
}

/** Install an already-read book (the Node tests read it off disk). */
export function setBook(json) { book = json && json.pos ? json : null; }

function keyOf(state, side) {
  const [lo, hi] = zobristOf(state, side);
  return (lo >>> 0).toString(36) + ':' + (hi >>> 0).toString(36);
}

/** The opening name for a position, if it is a named book position. */
export function openingName(state, side) {
  if (!book) return null;
  return book.names[keyOf(state, side)] || null;
}

/**
 * A book move for the position, or null when it is not in the book (or the
 * game is past move 9). Chosen at random, weighted by how often masters played
 * each move and how it scored for them, among moves that were a real choice —
 * so the opponent varies its openings instead of repeating one line forever,
 * without ever picking a move that one player tried once and lost with.
 */
export function bookMove(state, side, gamePly, rand = Math.random) {
  if (!book || gamePly >= book.maxPly) return null;
  const entry = book.pos[keyOf(state, side)];
  if (!entry) return null;

  const total = entry.reduce((t, e) => t + e[1], 0);
  const choices = entry.filter(([, n, score]) =>
    n >= Math.max(2, total * 0.06) && (n < 8 || score >= 35));
  if (!choices.length) return null;

  const weight = ([, n, score]) => n * (0.3 + score / 100);
  let r = rand() * choices.reduce((t, c) => t + weight(c), 0);
  let pick = choices[choices.length - 1];
  for (const c of choices) { r -= weight(c); if (r <= 0) { pick = c; break; } }

  const u = pick[0];
  const fc = u.charCodeAt(0) - 97, fr = 8 - +u[1];
  const tc = u.charCodeAt(2) - 97, tr = 8 - +u[3];
  const promo = u[4] ? { n: 2, b: 3, r: 4, q: 5 }[u[4]] : 0;
  return getLegalMoves(state, side).find(m =>
    m.from[0] === fr && m.from[1] === fc && m.to[0] === tr && m.to[1] === tc &&
    (!promo || Math.abs(m.piece) === promo)) || null;
}
