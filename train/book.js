'use strict';
// Build the opening book (model/book.json) for the first 9 moves.
//
// Two sources, both offline, nothing downloaded:
//   • the master games in the repository's seven PGN files — for every
//     position in the first 9 moves, which moves were played, how often, and
//     how they scored for the side that played them;
//   • the main lines of opening literature in train/openings.js, so the
//     standard systems are covered even where these seven players' games are
//     thin, and so positions can be named ("Sicilian Defence, Najdorf").
//
// Positions are keyed by the engine's own Zobrist hash (js/position.js), which
// covers side to move, castling rights and en passant — so transpositions into
// a book position are recognised however they were reached.
//
// Usage: cd train && node book.js

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { Position, mvFrom, mvTo, mvPromo } from '../js/position.js';
import { splitGames, parseResult, moveTokens, sanToMove } from './pgn.js';
import { OPENINGS } from './openings.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const OUT  = path.join(ROOT, 'model', 'book.json');
const PGN_FILES = ['1Carlsen.pgn', '2Caruana.pgn', '3Fischer.pgn', '4Capablanca.pgn',
                   '5Kasparov.pgn', '6Nakamura.pgn', '7Tal.pgn'];

const MAX_PLY      = 18;   // moves 1–9 for both sides
const MIN_GAMES    = 3;    // a position must have been reached this often by masters…
const MIN_MOVE     = 2;    // …and a move played this often, to be kept
const CURATED_BIAS = 6;    // literature moves count as this many extra games

export const bookKey = (lo, hi) => (lo >>> 0).toString(36) + ':' + (hi >>> 0).toString(36);

function uci(m) {
  const s = sq => String.fromCharCode(97 + (sq & 7)) + (8 - (sq >> 3));
  return s(mvFrom(m)) + s(mvTo(m)) + (mvPromo(m) ? 'nbrq'[mvPromo(m) - 2] : '');
}

const START = {
  board: [[-4,-2,-3,-5,-6,-3,-2,-4],[-1,-1,-1,-1,-1,-1,-1,-1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1],[4,2,3,5,6,3,2,4]],
  wKc: true, wQc: true, bKc: true, bQc: true, enPassantTarget: null, halfmoveClock: 0,
};

// key → { total, moves: Map(uci → {n, score, curated}) }
const table = new Map();
function note(key, move, score, weight, curated) {
  let e = table.get(key);
  if (!e) { e = { total: 0, curated: false, moves: new Map() }; table.set(key, e); }
  let mv = e.moves.get(move);
  if (!mv) { mv = { n: 0, score: 0, curated: false }; e.moves.set(move, mv); }
  mv.n += weight; mv.score += score * weight; e.total += weight;
  if (curated) { mv.curated = true; e.curated = true; }
}

// ── Master games ───────────────────────────────────────────────────────────
let games = 0;
const pos = new Position();
for (const file of PGN_FILES) {
  const text = fs.readFileSync(path.join(ROOT, file), 'utf8');
  for (const g of splitGames(text)) {
    const result = parseResult(g);
    if (result === null) continue;
    pos.setFromState(START, 'white');
    let ply = 0;
    for (const tok of moveTokens(g)) {
      if (ply >= MAX_PLY) break;
      const m = sanToMove(pos, tok);
      if (!m) break;
      // Score for the side that played the move: 1 win, ½ draw, 0 loss.
      const mover = pos.stm;
      const score = result === 0 ? 0.5 : (result === mover ? 1 : 0);
      note(bookKey(pos.keyLo, pos.keyHi), uci(m), score, 1, false);
      pos.makeMove(m); pos.commit(); ply++;
    }
    games++;
  }
}

// ── Literature lines ───────────────────────────────────────────────────────
const namesAt = new Map();          // key → [names of lines through it]
for (const [name, line] of OPENINGS) {
  pos.setFromState(START, 'white');
  const toks = line.split(/\s+/);
  if (toks.length > MAX_PLY) throw new Error(`${name}: longer than ${MAX_PLY} plies`);
  toks.forEach((tok, i) => {
    const m = sanToMove(pos, tok);
    if (!m) throw new Error(`${name}: "${tok}" (ply ${i + 1}) is not legal here`);
    note(bookKey(pos.keyLo, pos.keyHi), uci(m), 0.5, CURATED_BIAS, true);
    pos.makeMove(m); pos.commit();
    const k = bookKey(pos.keyLo, pos.keyHi);
    if (!namesAt.has(k)) namesAt.set(k, []);
    namesAt.get(k).push(name);
  });
}

// A position on one line takes that line's name; one shared by several lines
// of the same family takes the family name; anything shared across families
// ("1.e4 e5" belongs to a dozen) stays unnamed.
const names = {};
for (const [k, list] of namesAt) {
  const uniq = [...new Set(list)];
  if (uniq.length === 1) { names[k] = uniq[0]; continue; }
  const fams = [...new Set(uniq.map(n => n.split(',')[0]))];
  if (fams.length === 1) names[k] = fams[0];
}

// ── Prune and write ────────────────────────────────────────────────────────
const out = {};
let kept = 0, moves = 0;
for (const [k, e] of table) {
  if (!e.curated && e.total < MIN_GAMES) continue;
  const list = [];
  for (const [u, mv] of e.moves) {
    if (!mv.curated && mv.n < MIN_MOVE) continue;
    list.push([u, Math.round(mv.n), Math.round(100 * mv.score / mv.n)]);
  }
  if (!list.length) continue;
  list.sort((a, b) => b[1] - a[1]);
  out[k] = list;
  kept++; moves += list.length;
}

const json = { v: 1, maxPly: MAX_PLY, source: { games, lines: OPENINGS.length, files: PGN_FILES },
               pos: out, names };
fs.writeFileSync(OUT, JSON.stringify(json));
console.log(`Opening book: ${kept.toLocaleString()} positions, ${moves.toLocaleString()} moves, ` +
            `${Object.keys(names).length} named positions`);
console.log(`From ${games.toLocaleString()} master games and ${OPENINGS.length} literature lines`);
console.log(`Wrote ${OUT} (${(fs.statSync(OUT).size / 1024).toFixed(0)} KB)`);
