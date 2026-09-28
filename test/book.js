'use strict';
// The opening book, checked through the app's own rules (js/chess.js).
//
//   • every literature line in train/openings.js is recognised at every step
//     and carries its name at the end — which proves the book builder and the
//     app hash positions the same way;
//   • random walks through the book, as the network-only opponent would play
//     it, only ever produce legal moves and stop by move 9.
//
// Usage: node test/book.js   (after `npm run book`)

import fs from 'fs';
import { initState, makeMove, getLegalMoves, opposite } from '../js/chess.js';
import { setBook, bookMove, openingName } from '../js/book.js';
import { Position } from '../js/position.js';
import { sanToMove } from '../train/pgn.js';
import { OPENINGS } from '../train/openings.js';

const book = JSON.parse(fs.readFileSync(new URL('../model/book.json', import.meta.url), 'utf8'));
setBook(book);

let failures = 0;
const check = (name, ok, detail) => {
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};

// Play a SAN line through chess.js (the UI's rules), returning each state.
function replay(line) {
  let st = initState(), side = 'white';
  const p = new Position().setFromState(st, 'white');
  const states = [];
  for (const tok of line.split(/\s+/)) {
    st._sideToMove = side;
    states.push({ st, side });
    const m = sanToMove(p, tok);
    p.makeMove(m); p.commit();
    const f = m & 63, t = (m >>> 6) & 63, promo = (m >>> 12) & 7;
    const mv = getLegalMoves(st, side).find(x =>
      x.from[0] === f >> 3 && x.from[1] === (f & 7) && x.to[0] === t >> 3 && x.to[1] === (t & 7) &&
      (!promo || Math.abs(x.piece) === promo));
    st = makeMove(st, mv); side = opposite(side);
  }
  st._sideToMove = side;
  return { states, end: { st, side } };
}

let named = 0;
for (const [name, line] of OPENINGS) {
  const { end } = replay(line);
  if (openingName(end.st, end.side) === name) named++;
  else console.log(`      ${name}: ends named "${openingName(end.st, end.side)}"`);
}
check('every literature line ends on its own name', named === OPENINGS.length, `${named}/${OPENINGS.length}`);

// Random walks through the book.
let illegal = 0, overran = 0, depth = 0;
for (let g = 0; g < 400; g++) {
  let st = initState(), side = 'white', ply = 0;
  while (true) {
    st._sideToMove = side;
    const m = bookMove(st, side, ply);
    if (!m) break;
    const same = x => x.from[0] === m.from[0] && x.from[1] === m.from[1] &&
                      x.to[0] === m.to[0] && x.to[1] === m.to[1] && x.piece === m.piece;
    if (!getLegalMoves(st, side).some(same)) { illegal++; break; }
    st = makeMove(st, m); side = opposite(side); ply++;
    if (ply > book.maxPly) { overran++; break; }
  }
  depth += ply;
}
check('book moves are always legal', illegal === 0, `${illegal} illegal in 400 walks`);
check('the book stops by move 9', overran === 0);
check('games usually stay in book for most of 9 moves', depth / 400 >= 10,
      `average ${(depth / 400).toFixed(1)} of ${book.maxPly} half-moves`);

// Variety: the first move should not always be the same.
const firsts = new Set();
for (let i = 0; i < 60; i++) { const st = initState(); st._sideToMove = 'white'; firsts.add(JSON.stringify(bookMove(st, 'white', 0).to)); }
check('the network varies its first move', firsts.size >= 2, `${firsts.size} different first moves`);

console.log('\n' + '='.repeat(60));
console.log(failures === 0 ? 'All book checks passed.' : `${failures} book check(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
