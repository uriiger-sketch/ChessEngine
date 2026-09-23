'use strict';
// Train the evaluation network.
//
// The network starts empty — randomly initialised weights — and every number it
// ever sees comes from the seven PGN files of master games in this repository.
// Nothing is downloaded, no pretrained weights are imported, and no other
// engine labels the positions. `createModel()` below is the whole starting
// point, and `collectPositions()` is the whole data source.
//
// ── What the network is asked to predict ──────────────────────────────────
//
// Not the evaluation. The positional *correction* to it.
//
// Two earlier designs failed, and both failures were instructive. Training on a
// pawn-unit label that was mostly material taught the network to reproduce the
// material count the hand evaluation already does exactly — no new knowledge.
// Training it to predict an absolute win probability failed differently and
// worse: master games are drawish, so every target sits near 0.5, and the
// network learned to output a compressed range. Measured on the previous run,
// a whole queen up scored +1.4 pawns. Blending that into the evaluation would
// have *shrunk* material — a queen up reading 635cp instead of 900cp — and
// quietly distorted every exchange the engine considered.
//
// So material is not the network's job. It is handed to the network for free:
//
//   predicted win probability = sigmoid( material / MATERIAL_SCALE  +  L )
//                                        \_______ fixed _______/     \_ learned
//
// The material term is a fixed offset the network cannot change, and the only
// thing it learns is L, the correction — "given the material, what else about
// this position decided how the game went?". That is precisely the knowledge a
// hand-written evaluation lacks and the master games contain, and because
// material never passes through the network, no amount of noise in L can
// distort it. The engine adds L (clamped) to its own evaluation rather than
// averaging with it.
//
// The target the correction is fitted against blends the material-implied
// probability with how the game actually ended:
//
//   y = (1 − w) · sigmoid(material / MATERIAL_SCALE)  +  w · (result + 1) / 2
//
// with w ramping up through the game, because at move 12 the eventual result
// says very little and by move 50 it says a great deal. The baseline printed
// during training is the loss at L = 0 — pure material, no network — so the
// training log answers directly whether the network adds anything at all.
//
// MATERIAL_SCALE is set so one logit is about 175 centipawns, the conventional
// relation between evaluation and expected score. That fixes the units: the
// network's output converts to centipawns with the same constant.
//
// ── Inputs ────────────────────────────────────────────────────────────────
//
// 768 piece-square indicators, plus side-to-move and the four castling rights
// (773 total). The old model had no idea whose turn it was or whether anyone
// could still castle, which are not small omissions.
//
// Inputs stay 0/1 rather than ±1. With ±1 the ~736 empty squares become a
// constant −1 background that dominates every gradient and the network fails to
// learn even exact material. js/neural.js uses the identical encoding, and its
// sparse fast path depends on it.
//
// Usage: cd train && npm install && node train.js

import tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { Position } from '../js/position.js';
import { splitGames, parseResult, moveTokens, sanToMove, material } from './pgn.js';

const HERE      = path.dirname(fileURLToPath(import.meta.url));
const PGN_DIR   = path.join(HERE, '..');
const MODEL_DIR = path.join(HERE, '..', 'model');
const PGN_FILES = [
  '1Carlsen.pgn', '2Caruana.pgn', '3Fischer.pgn',
  '4Capablanca.pgn', '5Kasparov.pgn', '6Nakamura.pgn', '7Tal.pgn'
];

// ── Hyperparameters ────────────────────────────────────────────────────────
// One logit ≈ 175 centipawns ≈ 1.75 pawns, the usual evaluation/score relation.
const MATERIAL_SCALE = 1.75;
const CP_PER_LOGIT   = 100 * MATERIAL_SCALE;
// How far the learned correction is ever allowed to move the evaluation. A
// positional insight is worth a pawn or two; it is never worth a queen, and
// this is what guarantees the network cannot overturn material.
const CLAMP_CP       = 200;
const W_MIN = 0.20, W_MAX = 0.65;  // weight given to the game result
const W_RAMP_PLIES = 110;      // plies over which that weight ramps up

const EPOCHS        = 18;
const BATCH_SIZE    = 1024;
const LEARNING_RATE = 1e-3;
const VAL_FRACTION  = 0.02;
const SAMPLE_RATE   = 0.55;
const SKIP_PLY      = 10;      // opening book plies are identical across games
const MAX_POSITIONS = 3_000_000;
const MAX_PIECES    = 32;
const PATIENCE      = 3;       // epochs without improvement before stopping

const IN_DIM  = 773;
const F_STM   = 768;
const F_CR    = 769;           // wK, wQ, bK, bQ at 769..772

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// ── Sparse position store ──────────────────────────────────────────────────
// A dense 773-float row per position would need ~9 GB at this many positions.
// At most 32 squares are ever occupied, so each position is kept as its list of
// set indices and expanded to dense rows one batch at a time.
class PositionStore {
  constructor(capacity) {
    this.capacity = capacity;
    this.idx    = new Uint16Array(capacity * MAX_PIECES);
    this.counts = new Uint8Array(capacity);
    this.extra  = new Uint8Array(capacity);   // bit0 = White to move, bits1-4 = castling
    this.labels = new Float32Array(capacity);
    this.mat    = new Float32Array(capacity); // kept for calibrating the output scale
    this.n = 0;
  }

  add(board, stm, castling, label, mat) {
    if (this.n >= this.capacity) return false;
    const base = this.n * MAX_PIECES;
    let k = 0;
    for (let sq = 0; sq < 64; sq++) {
      const p = board[sq];
      if (p === 0 || k >= MAX_PIECES) continue;
      const plane = p > 0 ? p - 1 : 5 + (-p);   // W P..K = 0..5, B p..k = 6..11
      this.idx[base + k++] = plane * 64 + sq;
    }
    this.counts[this.n] = k;
    this.extra[this.n]  = (stm > 0 ? 1 : 0) | ((castling & 15) << 1);
    this.labels[this.n] = label;
    this.mat[this.n]    = mat;
    this.n++;
    return true;
  }
}

// Colour-mirror an input index: flip the board vertically and swap colours.
// Applied together with y → 1 − y, this teaches the network that chess is
// colour-symmetric. Without it the net happily calls the symmetric starting
// position a large advantage for one side.
function mirrorIndex(i) {
  const plane = (i / 64) | 0;
  const sq    = i % 64;
  const mPlane = plane < 6 ? plane + 6 : plane - 6;
  return mPlane * 64 + (sq ^ 56);
}

// Castling rights follow the colour swap: white's rights become black's.
function mirrorCastling(cr) {
  return ((cr & 1) << 2) | ((cr & 2) << 2) | ((cr & 4) >> 2) | ((cr & 8) >> 2);
}

// Expands stored positions into dense rows, plus the fixed material offset that
// rides alongside each one into the loss.
function buildBatch(store, order, from, to, mirror) {
  const rows = to - from;
  const xs = new Float32Array(rows * IN_DIM);
  const ys = new Float32Array(rows);
  const offs = new Float32Array(rows);
  for (let b = 0; b < rows; b++) {
    const p    = order[from + b];
    const base = p * MAX_PIECES;
    const cnt  = store.counts[p];
    const flip = mirror && (b & 1) === 0;      // mirror half of every batch
    const off  = b * IN_DIM;

    for (let k = 0; k < cnt; k++) {
      const i = store.idx[base + k];
      xs[off + (flip ? mirrorIndex(i) : i)] = 1;
    }

    const extra = store.extra[p];
    let stmWhite = (extra & 1) === 1;
    let cr = (extra >> 1) & 15;
    if (flip) { stmWhite = !stmWhite; cr = mirrorCastling(cr); }
    if (stmWhite) xs[off + F_STM] = 1;
    if (cr & 1) xs[off + F_CR] = 1;
    if (cr & 2) xs[off + F_CR + 1] = 1;
    if (cr & 4) xs[off + F_CR + 2] = 1;
    if (cr & 8) xs[off + F_CR + 3] = 1;

    ys[b] = flip ? 1 - store.labels[p] : store.labels[p];
    const mat = flip ? -store.mat[p] : store.mat[p];
    offs[b] = mat / MATERIAL_SCALE;
  }
  return { xs, ys, offs, rows };
}

function shuffleInPlace(arr, rand) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
  }
}

// Deterministic RNG so a training run can be reproduced exactly.
function makeRand(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// ── Model ──────────────────────────────────────────────────────────────────
// A wide first layer over the sparse piece-square inputs and narrow layers
// after it. That shape is what makes the pure-JS inference in js/neural.js
// fast: only the ~32 occupied rows of the first kernel are ever touched, and
// everything downstream is tiny.
function createModel() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 384, activation: 'relu', inputShape: [IN_DIM] }));
  m.add(tf.layers.dense({ units:  32, activation: 'relu' }));
  m.add(tf.layers.dense({ units:  32, activation: 'relu' }));
  // Linear output: the learned correction L, in logits. Not compiled with a
  // loss — the material offset has to be added to the output before the loss
  // sees it, which model.fit() has no way to express, so training runs through
  // an explicit optimizer loop below.
  m.add(tf.layers.dense({ units: 1, activation: 'linear' }));
  return m;
}

// One pass over a chunk, updating the weights. The loss is sigmoid cross
// entropy on (material offset + correction), so gradients only ever flow into
// the correction.
function trainChunk(model, optimizer, xt, yt, ot, rows) {
  let lossSum = 0, batches = 0;
  for (let i = 0; i < rows; i += BATCH_SIZE) {
    const len = Math.min(BATCH_SIZE, rows - i);
    tf.tidy(() => {
      const xb = xt.slice([i, 0], [len, IN_DIM]);
      const yb = yt.slice([i, 0], [len, 1]);
      const ob = ot.slice([i, 0], [len, 1]);
      const lossT = optimizer.minimize(
        () => tf.losses.sigmoidCrossEntropy(yb, model.apply(xb, { training: true }).add(ob)),
        true
      );
      lossSum += lossT.dataSync()[0];
      batches++;
    });
  }
  return lossSum / Math.max(1, batches);
}

// Mean loss over a set, optionally with the correction forced to zero — which
// is exactly "material only, no network", the baseline worth beating.
function datasetLoss(model, xt, yt, ot, rows, zeroCorrection) {
  let sum = 0, seen = 0;
  for (let i = 0; i < rows; i += 4096) {
    const len = Math.min(4096, rows - i);
    tf.tidy(() => {
      const yb = yt.slice([i, 0], [len, 1]);
      const ob = ot.slice([i, 0], [len, 1]);
      let logits;
      if (zeroCorrection) {
        logits = ob;
      } else {
        const xb = xt.slice([i, 0], [len, IN_DIM]);
        logits = model.predict(xb).add(ob);
      }
      sum += tf.losses.sigmoidCrossEntropy(yb, logits).dataSync()[0] * len;
      seen += len;
    });
  }
  return sum / seen;
}

// ── Data collection ────────────────────────────────────────────────────────
function collectPositions() {
  const store = new PositionStore(MAX_POSITIONS);
  const rand  = makeRand(12345);
  const pos   = new Position();

  let totalGames = 0, parsedGames = 0;
  let wWins = 0, bWins = 0, draws = 0;
  let skippedNoisy = 0;

  // Positions of one game are buffered and only committed once the whole game
  // has replayed cleanly. A game that fails to parse halfway through has a
  // wrong board from that point on, and half-wrong data is worse than none.
  const pendBoard = [];
  const pendMeta  = [];

  for (const file of PGN_FILES) {
    const filePath = path.join(PGN_DIR, file);
    if (!fs.existsSync(filePath)) { console.warn(`  Skipping ${file} (not found)`); continue; }

    process.stdout.write(`  ${file.padEnd(18)}`);
    const games = splitGames(fs.readFileSync(filePath, 'utf8'));
    const before = store.n;
    let fileParsed = 0;

    for (const gameText of games) {
      totalGames++;
      const result = parseResult(gameText);
      if (result === null) continue;              // unfinished game: no outcome signal
      if (result === 1) wWins++; else if (result === -1) bWins++; else draws++;

      const tokens = moveTokens(gameText);
      if (tokens.length < SKIP_PLY + 6) continue;

      pos.setFromState(startState(), 'white');
      pendBoard.length = 0; pendMeta.length = 0;

      let ok = true;
      let ply = 0;

      for (const token of tokens) {
        const mv = sanToMove(pos, token);
        if (!mv) { ok = false; break; }

        // Decide about the position BEFORE the move, because the move itself
        // tells us whether this is a quiet position worth learning from.
        if (ply >= SKIP_PLY && rand() < SAMPLE_RATE) {
          const noisy = (mv & (1 << 15)) !== 0 || ((mv >>> 12) & 7) !== 0 || pos.inCheck();
          if (noisy) {
            // A static evaluation cannot judge a position in the middle of an
            // exchange or a check. Training on them teaches the net noise.
            skippedNoisy++;
          } else {
            const mat = material(pos.board);
            const w   = Math.min(W_MAX, W_MIN + (ply - SKIP_PLY) / W_RAMP_PLIES * (W_MAX - W_MIN));
            const y   = (1 - w) * sigmoid(mat / MATERIAL_SCALE) + w * (result + 1) / 2;
            pendBoard.push(Int8Array.from(pos.board));
            pendMeta.push(pos.stm, pos.castling, y, mat);
          }
        }

        if (!pos.makeMove(mv)) { ok = false; break; }
        pos.commit();          // replaying, never taking back — keep ply at 0
        ply++;
      }

      if (!ok) continue;
      parsedGames++; fileParsed++;
      for (let i = 0; i < pendBoard.length; i++) {
        if (!store.add(pendBoard[i], pendMeta[i * 4], pendMeta[i * 4 + 1],
                       pendMeta[i * 4 + 2], pendMeta[i * 4 + 3])) break;
      }
      if (store.n >= store.capacity) break;
    }

    console.log(`${(store.n - before).toLocaleString().padStart(9)} positions from ${fileParsed.toLocaleString()} games`);
    if (store.n >= store.capacity) { console.log('  (position cap reached)'); break; }
  }

  return { store, totalGames, parsedGames, wWins, bWins, draws, skippedNoisy };
}

function startState() {
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
    enPassantTarget: null, halfmoveClock: 0
  };
}

// ── Sanity probes ──────────────────────────────────────────────────────────
// An evaluation network that cannot score the symmetric starting position near
// zero, or that does not move decisively when a queen disappears, is broken no
// matter what its loss curve looks like.
function probeBoards() {
  const start = new Position().setFromState(startState(), 'white');
  const clone = (mutate) => {
    const p = new Position().setFromState(startState(), 'white');
    mutate(p);
    return p;
  };
  return [
    ['start position',              start,                                        0,   1.0],
    ['Black queen removed',         clone(p => p.board[3] = 0),                   9,   3.0],
    ['White queen removed',         clone(p => p.board[59] = 0),                 -9,   3.0],
    ['both Black rooks removed',    clone(p => { p.board[0] = 0; p.board[7] = 0; }),  10, 3.5],
    ['both White knights removed',  clone(p => { p.board[57] = 0; p.board[62] = 0; }), -6, 3.0],
    ['Black down a rook and pawn',  clone(p => { p.board[0] = 0; p.board[8] = 0; }),   6, 3.0],
  ];
}

function runProbes(model) {
  const probes = probeBoards();
  const xs = new Float32Array(probes.length * IN_DIM);
  probes.forEach(([, p], bi) => {
    const off = bi * IN_DIM;
    for (let sq = 0; sq < 64; sq++) {
      const v = p.board[sq];
      if (v === 0) continue;
      const plane = v > 0 ? v - 1 : 5 + (-v);
      xs[off + plane * 64 + sq] = 1;
    }
    if (p.stm > 0) xs[off + F_STM] = 1;
    for (let i = 0; i < 4; i++) if (p.castling & (1 << i)) xs[off + F_CR + i] = 1;
  });

  const t = tf.tensor2d(xs, [probes.length, IN_DIM]);
  const out = model.predict(t);
  const logits = out.dataSync();
  t.dispose(); out.dispose();

  // The network outputs a correction, so the checks are about the correction
  // behaving itself — staying small, staying bounded, and being antisymmetric
  // between a position and its colour mirror. Material correctness is not
  // tested here because material never goes through the network at all.
  console.log('\nSanity probes — learned correction only (centipawns):');
  let allGood = true;
  probes.forEach(([label], i) => {
    const cp = logits[i] * CP_PER_LOGIT;
    const ok = Math.abs(cp) <= CLAMP_CP * 1.5;
    if (!ok) allGood = false;
    console.log(`  ${ok ? 'OK  ' : 'BAD '} ${label.padEnd(30)} → ${cp >= 0 ? '+' : ''}${cp.toFixed(0)}cp` +
                (Math.abs(cp) > CLAMP_CP ? `  (will clamp to ±${CLAMP_CP})` : ''));
  });

  // The starting position is colour-symmetric, so its correction must be ~0 or
  // the network has learned a side bias the mirroring was supposed to remove.
  const startCp = logits[0] * CP_PER_LOGIT;
  const startOk = Math.abs(startCp) < 25;
  if (!startOk) allGood = false;
  console.log(`  ${startOk ? 'OK  ' : 'BAD '} start position is unbiased      → ${startCp.toFixed(0)}cp (want |x| < 25)`);
  return allGood;
}

// ── Training ───────────────────────────────────────────────────────────────
async function main() {
  console.log('ChessNN training — empty network, master games only');
  console.log('='.repeat(72));
  console.log(`Label    : (1-w)·sigmoid(material/${MATERIAL_SCALE}) + w·result,  w ramps ${W_MIN} → ${W_MAX}`);
  console.log(`Inputs   : ${IN_DIM} (768 piece-square + side-to-move + 4 castling rights)`);
  console.log(`Sampling : ${(SAMPLE_RATE * 100).toFixed(0)}% of quiet plies after ply ${SKIP_PLY}`);
  console.log(`Training : up to ${EPOCHS} epochs, batch ${BATCH_SIZE}, lr ${LEARNING_RATE}, early stop after ${PATIENCE}`);
  console.log(`Augment  : colour mirroring on half of every batch`);
  console.log(`Sources  : ${PGN_FILES.join(', ')}\n`);

  if (!fs.existsSync(MODEL_DIR)) fs.mkdirSync(MODEL_DIR, { recursive: true });

  console.log('Reading games...');
  const t0 = Date.now();
  const { store, totalGames, parsedGames, wWins, bWins, draws, skippedNoisy } = collectPositions();
  console.log(`\n  Games seen       : ${totalGames.toLocaleString()} (replayed in full: ${parsedGames.toLocaleString()})`);
  console.log(`  Results          : White ${wWins.toLocaleString()}, Draw ${draws.toLocaleString()}, Black ${bWins.toLocaleString()}`);
  console.log(`  Positions stored : ${store.n.toLocaleString()}`);
  console.log(`  Skipped as noisy : ${skippedNoisy.toLocaleString()} (captures, promotions, checks)`);
  console.log(`  Collection time  : ${((Date.now() - t0) / 1000).toFixed(0)}s`);

  if (store.n < 10000) { console.error('Not enough positions — aborting.'); process.exit(1); }

  // Global shuffle: streaming in file order would let the last author in the
  // list dominate the final weights.
  const rand  = makeRand(999);
  const order = new Uint32Array(store.n);
  for (let i = 0; i < store.n; i++) order[i] = i;
  shuffleInPlace(order, rand);

  // Capped as well as floored: the validation set is expanded to dense rows all
  // at once, and 2% of three million positions would be a 180 MB tensor.
  const valCount   = Math.min(50000, Math.max(20000, Math.floor(store.n * VAL_FRACTION)));
  const valOrder   = order.slice(0, valCount);
  const trainOrder = order.slice(valCount);
  console.log(`  Train / val      : ${trainOrder.length.toLocaleString()} / ${valOrder.length.toLocaleString()}`);

  const model = createModel();
  model.summary();

  const val = buildBatch(store, valOrder, 0, valOrder.length, false);
  const valXs = tf.tensor2d(val.xs, [val.rows, IN_DIM]);
  const valYs = tf.tensor2d(val.ys, [val.rows, 1]);
  const valOs = tf.tensor2d(val.offs, [val.rows, 1]);

  // The baseline that matters: material alone, correction forced to zero. If
  // training does not beat this, the network has learned nothing the engine did
  // not already know, and the honest thing is to say so.
  const baseline = datasetLoss(model, valXs, valYs, valOs, val.rows, true);
  console.log(`\nBaseline val loss (material only, no network): ${baseline.toFixed(5)}\n`);

  const optimizer = tf.train.adam(LEARNING_RATE);
  const CHUNK = 25_000;
  const tTrain = Date.now();

  let bestVal = Infinity, bestWeights = null, sinceImproved = 0;
  let lr = LEARNING_RATE;

  for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    shuffleInPlace(trainOrder, rand);
    let seen = 0, lossSum = 0, lossN = 0;

    for (let start = 0; start < trainOrder.length; start += CHUNK) {
      const end = Math.min(start + CHUNK, trainOrder.length);
      const { xs, ys, offs, rows } = buildBatch(store, trainOrder, start, end, true);
      const xt = tf.tensor2d(xs, [rows, IN_DIM]);
      const yt = tf.tensor2d(ys, [rows, 1]);
      const ot = tf.tensor2d(offs, [rows, 1]);

      lossSum += trainChunk(model, optimizer, xt, yt, ot, rows);
      lossN++;
      seen += rows;
      xt.dispose(); yt.dispose(); ot.dispose();

      process.stdout.write(`\r  epoch ${epoch}/${EPOCHS}  ${seen.toLocaleString()}/${trainOrder.length.toLocaleString()}  loss ${(lossSum / lossN).toFixed(5)}   `);
    }

    const valLoss = datasetLoss(model, valXs, valYs, valOs, val.rows, false);
    const improved = valLoss < bestVal - 1e-5;
    console.log(`\r  epoch ${epoch}/${EPOCHS}  train ${(lossSum / lossN).toFixed(5)}  val ${valLoss.toFixed(5)}  (material-only ${baseline.toFixed(5)})${improved ? '  *best' : ''}          `);

    if (improved) {
      bestVal = valLoss;
      sinceImproved = 0;
      if (bestWeights) bestWeights.forEach(t => t.dispose());
      bestWeights = model.getWeights().map(t => t.clone());
    } else if (++sinceImproved >= PATIENCE) {
      console.log(`  Stopping early — no improvement for ${PATIENCE} epochs.`);
      break;
    } else {
      lr /= 2;
      optimizer.learningRate = lr;
      console.log(`  Learning rate → ${lr.toExponential(2)}`);
    }
  }

  if (bestWeights) {
    model.setWeights(bestWeights);
    bestWeights.forEach(t => t.dispose());
    console.log(`\nRestored the best epoch (val ${bestVal.toFixed(5)}).`);
  }
  console.log(`Training time: ${((Date.now() - tTrain) / 1000 / 60).toFixed(1)} min`);

  const gain = baseline - bestVal;
  console.log(`\nImprovement over material alone: ${gain > 0 ? '-' : '+'}${Math.abs(gain).toFixed(5)} loss` +
              (gain > 0 ? '  (the network is contributing)' : '  (the network is NOT contributing)'));

  // How large are the corrections it actually produces?
  let sumAbs = 0, maxAbs = 0, clamped = 0;
  {
    const predT = model.predict(valXs, { batchSize: 4096 });
    const L = predT.dataSync();
    predT.dispose();
    for (let i = 0; i < val.rows; i++) {
      const cp = Math.abs(L[i] * CP_PER_LOGIT);
      sumAbs += cp;
      if (cp > maxAbs) maxAbs = cp;
      if (cp > CLAMP_CP) clamped++;
    }
  }
  console.log(`Correction size: mean |L| ${(sumAbs / val.rows).toFixed(0)}cp, max ${maxAbs.toFixed(0)}cp, ` +
              `${(clamped / val.rows * 100).toFixed(1)}% would clamp at ±${CLAMP_CP}cp`);

  const passed = runProbes(model);

  valXs.dispose(); valYs.dispose(); valOs.dispose();

  console.log(`\nSaving model to ${MODEL_DIR}...`);
  await model.save(`file://${MODEL_DIR}`);
  fs.writeFileSync(
    path.join(MODEL_DIR, 'normalization.json'),
    JSON.stringify({
      output: 'residual',
      inputs: IN_DIM,
      cpScale: CP_PER_LOGIT,
      clampCp: CLAMP_CP,
      trainedOn: PGN_FILES,
      positions: store.n,
      valLoss: +bestVal.toFixed(5),
      materialOnlyLoss: +baseline.toFixed(5),
    }, null, 2)
  );

  console.log(passed
    ? '\nDone — probes passed, model saved.'
    : '\nDone — model saved, but some probes look off (see above).');
}

main().catch(err => { console.error(err); process.exit(1); });
