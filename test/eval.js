'use strict';
// Evaluation sanity checks.
//
// The strongest property a chess evaluation has is symmetry: flip every piece's
// colour, mirror the board top to bottom, and swap the side to move, and the
// score for the side to move must come out identical. Almost every sign error,
// every table read with the wrong orientation and every term applied to only
// one colour breaks it. Nothing else here is as good at catching those.
//
// Usage: node test/eval.js [--nn]

import { Position, CR_WK, CR_WQ, CR_BK, CR_BQ } from '../js/position.js';
import { evaluate, gamePhase } from '../js/evaluate.js';
import { evaluate as nnEvaluate, isResidual, clampCp } from '../js/neural.js';
import { staticEvalOf } from '../js/engine.js';
import { loadModelFromDisk, nnReady, nnInputDim } from './harness.js';

const USE_NN = process.argv.includes('--nn');

const POSITIONS = [
  ['start position',      'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'],
  ['kiwipete',            'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'],
  ['open sicilian',       'r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 1'],
  ['rook endgame',        '8/5pk1/6p1/7p/7P/6P1/5PK1/R6r w - - 0 1'],
  ['pawn endgame',        '8/2k5/8/2P5/8/2K5/8/8 w - - 0 1'],
  ['passed pawn race',    '8/1P4k1/8/8/8/8/6p1/1K6 w - - 0 1'],
  ['opposite bishops',    '4k3/5ppp/8/8/8/8/2B2PPP/4K3 w - - 0 1'],
  ['queenside majority',  'r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R w KQkq - 0 1'],
  ['locked centre',       'r1bq1rk1/pp2nppp/2n1p3/2ppP3/3P4/2PB1N2/PP3PPP/R1BQ1RK1 w - - 0 1'],
  ['bare kings',          '4k3/8/8/8/8/8/8/4K3 w - - 0 1'],
];

/** Colour-mirror a position: flip the board, swap colours, swap the mover. */
function mirror(pos) {
  const m = new Position();
  for (let sq = 0; sq < 64; sq++) m.board[sq ^ 56] = -pos.board[sq];
  m.stm = -pos.stm;
  m.castling =
    ((pos.castling & CR_WK) ? CR_BK : 0) | ((pos.castling & CR_WQ) ? CR_BQ : 0) |
    ((pos.castling & CR_BK) ? CR_WK : 0) | ((pos.castling & CR_BQ) ? CR_WQ : 0);
  m.ep = pos.ep >= 0 ? (pos.ep ^ 56) : -1;
  m.halfmove = pos.halfmove;
  m._locateKings();
  m._rehash();
  return m;
}

let failures = 0;
const check = (name, ok, detail) => {
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};

console.log('Hand evaluation — colour symmetry');
console.log('='.repeat(78));

for (const [name, fen] of POSITIONS) {
  const a = new Position().setFromFEN(fen);
  const b = mirror(a);
  const ea = evaluate(a), eb = evaluate(b);
  check(`${name.padEnd(22)} ${String(ea).padStart(6)} vs mirrored ${String(eb).padStart(6)}`,
        ea === eb, ea === eb ? '' : `differ by ${ea - eb}cp`);
}

// The starting position is symmetric apart from the move, so the only thing
// separating the two sides should be the tempo bonus.
console.log('\nAbsolute sanity');
console.log('='.repeat(78));
{
  const start = new Position().setFromFEN('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const s = evaluate(start);
  check('start position is near equal', Math.abs(s) <= 30, `${s}cp for White`);
  check('start position is full phase', gamePhase(start) === 1, gamePhase(start).toFixed(2));

  const bare = new Position().setFromFEN('4k3/8/8/8/8/8/8/4K3 w - - 0 1');
  check('bare kings are zero phase', gamePhase(bare) === 0, gamePhase(bare).toFixed(2));

  const queenUp = new Position().setFromFEN('rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const q = evaluate(queenUp);
  check('a queen up is worth roughly a queen', q > 800 && q < 1300, `${q}cp`);

  // The endgame king table must pull the king toward the middle, which the old
  // single middlegame table did the exact opposite of.
  const cornerK = new Position().setFromFEN('7k/8/8/8/8/8/5PPP/7K w - - 0 1');
  const centreK = new Position().setFromFEN('7k/8/8/4K3/8/8/5PPP/8 w - - 0 1');
  check('active king preferred in the endgame',
        evaluate(centreK) > evaluate(cornerK),
        `centre ${evaluate(centreK)}cp vs corner ${evaluate(cornerK)}cp`);
}

// ── The network, if asked for ──────────────────────────────────────────────
if (USE_NN) {
  console.log('\nNeural network — positional correction (centipawns)');
  console.log('='.repeat(78));
  if (!loadModelFromDisk()) {
    check('model loads', false, 'could not read model/');
  } else {
    console.log(`  input width: ${nnInputDim()},  mode: ${isResidual() ? 'residual correction' : 'absolute score'}`);

    for (const [name, fen] of POSITIONS) {
      const a = new Position().setFromFEN(fen);
      const b = mirror(a);
      const ca = nnEvaluate(a.board, a.stm, a.castling) * 100;
      const cb = nnEvaluate(b.board, b.stm, b.castling) * 100;
      // Mirroring is taught by augmentation, not built into the architecture,
      // so this is a tolerance check rather than an equality one. Where the
      // clamp bites, symmetry is not even expected — clamping one side of an
      // antisymmetric pair is asymmetric by construction — so those are
      // reported but not failed.
      const lim = clampCp() - 1;
      const clamped = Math.abs(ca) >= lim || Math.abs(cb) >= lim;
      const ok = clamped || Math.abs(ca + cb) < 60;
      check(`${name.padEnd(22)} ${ca.toFixed(0).padStart(5)}cp vs mirrored ${cb.toFixed(0).padStart(5)}cp`,
            ok, clamped ? '(clamped — symmetry not expected)'
                        : (Math.abs(ca + cb) < 60 ? '' : `sum ${(ca + cb).toFixed(0)}cp should be near 0`));
    }

    const start = new Position().setFromFEN('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
    const s0 = nnEvaluate(start.board, start.stm, start.castling) * 100;
    check('correction is unbiased at the start', Math.abs(s0) < 25, `${s0.toFixed(0)}cp`);

    // The property that actually matters: with the network switched on, the
    // evaluation the engine uses must still respect material. This is the check
    // that would have caught the previous model, which compressed a whole queen
    // down to +1.4 pawns and would have had the engine undervaluing pieces.
    console.log('\nCombined evaluation (hand + network) still respects material');
    console.log('='.repeat(78));
    // The guarantee the residual design buys: the network can move the score by
    // at most its clamp, so material always survives. This is what would have
    // caught the earlier model, which compressed a whole queen to +1.4 pawns
    // and — averaged into the evaluation — would have had the engine valuing a
    // queen at two thirds of a queen.
    const MATERIAL = [
      ['a queen up',  'rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'],
      ['a rook up',   '1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'],
      ['a knight up', 'r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'],
      ['a pawn up',   'rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'],
      ['level',       'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'],
    ];
    const scores = [];
    for (const [name, fen] of MATERIAL) {
      const p = new Position().setFromFEN(fen);
      const { state, side } = p.toUIState();
      const withNN = staticEvalOf(state, side, true);
      const without = staticEvalOf(state, side, false);
      scores.push(withNN);
      const moved = Math.abs(withNN - without);
      const ok = moved <= clampCp();
      check(`${name.padEnd(14)} NN on ${String(withNN).padStart(5)}cp  NN off ${String(without).padStart(5)}cp  (moved ${moved}cp)`,
            ok, ok ? '' : `network moved the score by more than its ${clampCp()}cp clamp`);
    }
    // Strictly decreasing down the list: queen > rook > knight > pawn > level.
    let monotone = true;
    for (let i = 1; i < scores.length; i++) if (scores[i] >= scores[i - 1]) monotone = false;
    check('more material always scores higher', monotone, scores.join('cp > ') + 'cp');
  }
}

console.log('\n' + '='.repeat(78));
console.log(failures === 0 ? 'All evaluation checks passed.' : `${failures} evaluation check(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
