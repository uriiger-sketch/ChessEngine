'use strict';
// UI controller — ES module, runs on main thread

import {
  initState, makeMove, getLegalMoves, getGameStatus,
  isInCheck, PIECE_GLYPHS, opposite, positionKey
} from './chess.js';
import { loadModel, isReady as nnIsReady, lastError as nnError } from './neural.js';
import { searchBestMove, searchTopMoves, resetEngine, zobristOf, staticEvalOf } from './engine.js';

// ── State ──────────────────────────────────────────────────────────────────
let gameState     = null;
let humanSide     = 'white';
let aiSide        = 'black';
let legalMoves    = [];
let selectedSq    = null;
let selectedLegal = [];
let thinkTimeMs   = 5000;
let gameOver      = false;
let sideToMove    = 'white';
let aiThinking    = false;
let boardFlipped  = false; // true when playing as Black (board rotated 180°)
let posCounts     = new Map(); // position key → occurrences, for threefold repetition
let gameId        = 0;         // bumped per game so stale search results are ignored

// Zobrist keys of every position played, as [lo, hi, lo, hi, …]. The engine
// needs these to see that a line would repeat a position from earlier in the
// real game, not just within its own search.
let zobristKeys   = [];

// Score the engine reported for its own move, in centipawns from its point of
// view. A searched score is a far better reading than a static one, so the
// evaluation bar prefers it while it is current.
let searchedEval  = null;

// ── Modes ──────────────────────────────────────────────────────────────────
//   play   just the game
//   learn  adds Back: return to the start of your last turn and play again
//   help   Back as well, plus the engine's three best moves on your turn
//
// Modes can be switched at any point in a game; nothing restarts. Back is
// limited to once per move — after using it you must play before using it
// again — so it is a way to try an alternative, not to rewind the game.
const MODES = ['play', 'learn', 'help'];
const MODE_CAPTIONS = {
  play:  'Just you and the engine.',
  learn: 'Back returns you to the start of your last turn.',
  help:  'Your three best moves are shown each turn. Back is available too.',
};
let mode          = loadMode();
let backUsed      = false;
let history       = [];   // snapshot taken before each move

// ── Help mode state ────────────────────────────────────────────────────────
// The best-move search gets exactly the engine's own thinking time, so the
// suggestion is as deep as the move it has to answer. It used to get a fixed
// 1.5s split three ways — a quarter of what the engine had at the 2s setting —
// which is why following the hints lost: that was the engine playing itself
// with a quarter of the time. The two alternatives get a quarter each.
const HINT_COUNT   = 3;
let hints          = [];      // [{move, score, mateIn}] best first
let hintsPending   = false;
let hintReqId      = 0;
let hintsDrawn     = 0;       // how many of `hints` are already on screen

// ── DOM References ─────────────────────────────────────────────────────────
const boardEl      = document.getElementById('board');
const hintLayer    = document.getElementById('hint-layer');
const hintPanel    = document.getElementById('hint-panel');
const statusEl     = document.getElementById('status');
const capturedEl   = document.getElementById('captured-wrap');
const promoModal   = document.getElementById('promo-modal');
const promoChoices = document.getElementById('promo-choices');
const thinkingEl   = document.getElementById('thinking-overlay');
const newGameBtn   = document.getElementById('new-game-btn');
const backBtn      = document.getElementById('back-btn');
const modeSwitch   = document.getElementById('mode-switch');
const modeCaption  = document.getElementById('mode-caption');
const evalFillEl   = document.getElementById('eval-white-fill');
const evalTextEl   = document.getElementById('eval-text');
const confirmModal = document.getElementById('confirm-modal');
const confirmTitle = document.getElementById('confirm-title');
const confirmSub   = document.getElementById('confirm-sub');
const confirmYes   = document.getElementById('confirm-yes');
const confirmNo    = document.getElementById('confirm-no');

// ── Search Workers ─────────────────────────────────────────────────────────
// The engine runs in a Web Worker so a multi-second search never freezes the
// board or the spinner. If module workers are unavailable the engine still
// runs on the main thread — correct, just less smooth.
//
// Help mode gets a second worker of its own. Sharing one would queue the
// engine's reply behind a hint search still in progress; with two, a hint
// search that has gone stale is simply terminated.
let worker      = null;
let workerReqId = 0;
const pendingSearches = new Map();

function makeWorker() {
  return new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
}

try {
  worker = makeWorker();
  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'result') {
      const resolve = pendingSearches.get(msg.id);
      if (resolve) { pendingSearches.delete(msg.id); resolve(msg); }
    }
  };
  worker.onerror = () => { worker = null; };   // fall back to main thread
} catch (_) {
  worker = null;
}

let hintWorker = null;
function getHintWorker() {
  if (hintWorker || !worker) return hintWorker;
  try {
    hintWorker = makeWorker();
    hintWorker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === 'hints') receiveHints(msg.id, msg.hints || [], msg.done !== false);
    };
    hintWorker.onerror = () => { hintWorker = null; };
  } catch (_) {
    hintWorker = null;
  }
  return hintWorker;
}

// Resolves to {move, info}, searching off-thread when possible.
function findBestMove(state, timeLimit, historyKeys) {
  const withNN = nnIsReady();
  if (worker) {
    return new Promise(resolve => {
      const id = ++workerReqId;
      pendingSearches.set(id, resolve);
      worker.postMessage({
        type: 'search', id, state, timeLimit, useNN: withNN, history: historyKeys,
      });
    });
  }
  return new Promise(resolve => {
    setTimeout(() => {
      const move = searchBestMove(state, timeLimit, withNN, { history: historyKeys });
      resolve({ move, info: null });
    }, 20);
  });
}

function resetSearchState() {
  if (worker) worker.postMessage({ type: 'reset' });
  else resetEngine();
}

// ── Neural Network Initialization ──────────────────────────────────────────
// Pure-JS inference (js/neural.js) — no TensorFlow.js, no CDN, works offline.
// The network is always used once loaded; the main thread loads its own copy
// for the evaluation bar and each worker loads one for its searches. The game
// is still fully playable if the model cannot be loaded.
(async function () {
  try {
    await loadModel();
    if (nnIsReady()) {
      updateEvalBar();
      if (!aiThinking && !gameOver) updateStatus();
    } else {
      console.warn('Neural net unavailable:', nnError());
    }
  } catch (e) {
    console.warn('Neural net failed to load:', e);
  }
})();

// ── Board Rendering ────────────────────────────────────────────────────────
// buildBoard places 64 divs in visual order.
// When boardFlipped, visual (vr,vc) → logical (7-vr, 7-vc) so Black's pieces appear at bottom.
function buildBoard() {
  boardEl.innerHTML = '';
  for (let vr = 0; vr < 8; vr++) {
    for (let vc = 0; vc < 8; vc++) {
      const lr = boardFlipped ? 7 - vr : vr;
      const lc = boardFlipped ? 7 - vc : vc;

      const div = document.createElement('div');
      div.className = `sq ${(lr + lc) % 2 === 0 ? 'light' : 'dark'}`;
      div.dataset.r = lr; // logical row stored in DOM
      div.dataset.c = lc;

      // Rank label on visual-left column (vc === 0)
      if (vc === 0) {
        const s = document.createElement('span');
        s.className = 'coord coord-rank';
        s.textContent = 8 - lr;
        div.appendChild(s);
      }
      // File label on visual-bottom row (vr === 7)
      if (vr === 7) {
        const s = document.createElement('span');
        s.className = 'coord coord-file';
        s.textContent = String.fromCharCode(97 + lc);
        div.appendChild(s);
      }

      boardEl.appendChild(div);
    }
  }
}

function getSquareEl(r, c) {
  return boardEl.querySelector(`[data-r="${r}"][data-c="${c}"]`);
}

function renderBoard() {
  boardEl.querySelectorAll('.sq').forEach(sq => {
    const r = +sq.dataset.r;
    const c = +sq.dataset.c;
    const piece = gameState.board[r][c];
    const glyph = PIECE_GLYPHS[piece] || '';

    const old = sq.querySelector('.piece-span');
    if (old) old.remove();

    if (glyph) {
      const span = document.createElement('span');
      span.className = `piece-span ${piece > 0 ? 'piece-w' : 'piece-b'}`;
      span.textContent = glyph;
      sq.appendChild(span);
    }

    sq.classList.remove('selected', 'legal', 'legal-cap', 'last-move-from', 'last-move-to');
  });

  if (gameState.lastMove) {
    const { from: [fr, fc], to: [tr, tc] } = gameState.lastMove;
    getSquareEl(fr, fc)?.classList.add('last-move-from');
    getSquareEl(tr, tc)?.classList.add('last-move-to');
  }

  if (selectedSq) {
    const [sr, sc] = selectedSq;
    getSquareEl(sr, sc)?.classList.add('selected');
    for (const mv of selectedLegal) {
      const el = getSquareEl(mv.to[0], mv.to[1]);
      el?.classList.add(mv.captured !== 0 || mv.enPassant ? 'legal-cap' : 'legal');
    }
  }
}

function renderCaptured() {
  const board = gameState.board;
  const start = { 1: 8, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1 };
  const wCnt = {}, bCnt = {};
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p > 0) wCnt[p] = (wCnt[p] || 0) + 1;
      if (p < 0) bCnt[-p] = (bCnt[-p] || 0) + 1;
    }

  let html = '';
  for (let pt = 5; pt >= 1; pt--) {
    const n = (start[pt] || 0) - (wCnt[pt] || 0);
    for (let i = 0; i < n; i++) html += `<span class="piece-w">${PIECE_GLYPHS[pt]}</span>`;
  }
  html += '<span style="display:inline-block;width:12px"></span>';
  for (let pt = 5; pt >= 1; pt--) {
    const n = (start[pt] || 0) - (bCnt[pt] || 0);
    for (let i = 0; i < n; i++) html += `<span class="piece-b">${PIECE_GLYPHS[-pt]}</span>`;
  }
  capturedEl.innerHTML = html;
}

// ── Evaluation Bar ─────────────────────────────────────────────────────────
// whiteAdv: positive = White is winning (pawn units).
function updateEvalBar() {
  if (!gameState || !evalFillEl || !evalTextEl) return;
  let whiteAdv = 0;

  if (searchedEval !== null) {
    // The engine's own score for the move it just chose, seen from White. A
    // searched score is a much better reading than a static one.
    whiteAdv = (aiSide === 'white' ? searchedEval : -searchedEval) / 100;
  } else {
    try {
      // Otherwise the engine's own leaf evaluation, so the bar and the engine
      // never disagree about what the position is worth.
      whiteAdv = staticEvalOf(gameState, sideToMove, nnIsReady()) / 100;
    } catch (_) {}
  }

  // Lichess-style sigmoid: 50% at 0, ~88% at ±4 pawns
  const prob = 1 / (1 + Math.exp(-whiteAdv / 4));
  evalFillEl.style.width = `${(prob * 100).toFixed(1)}%`;

  const abs = Math.abs(whiteAdv);
  if (abs > 49) {
    evalTextEl.textContent = whiteAdv > 0 ? 'W+' : 'B+';
  } else {
    evalTextEl.textContent = (whiteAdv >= 0 ? '+' : '') + whiteAdv.toFixed(1);
  }
}

// ── Tap Handling ───────────────────────────────────────────────────────────
boardEl.addEventListener('click', onBoardTap);
boardEl.addEventListener('touchstart', e => {
  e.preventDefault();
  const t = e.changedTouches[0];
  const el = document.elementFromPoint(t.clientX, t.clientY);
  if (el) {
    const sq = el.closest('.sq');
    if (sq) sq.dispatchEvent(new MouseEvent('click', { bubbles: true }));
  }
}, { passive: false });

function onBoardTap(e) {
  if (gameOver || sideToMove !== humanSide || aiThinking) return;
  const sqEl = e.target.closest('.sq');
  if (!sqEl) return;
  handleSquareTap(+sqEl.dataset.r, +sqEl.dataset.c);
}

function selectSquare(r, c) {
  selectedSq = [r, c];
  selectedLegal = legalMoves.filter(mv => mv.from[0] === r && mv.from[1] === c);
  renderBoard();
}

function handleSquareTap(r, c) {
  const piece = gameState.board[r][c];
  const friendSign = humanSide === 'white' ? 1 : -1;

  if (!selectedSq) {
    if (Math.sign(piece) !== friendSign) return;
    selectSquare(r, c);
    return;
  }

  const [sr, sc] = selectedSq;
  if (sr === r && sc === c) {
    selectedSq = null; selectedLegal = [];
    renderBoard();
    return;
  }

  const candidates = selectedLegal.filter(mv => mv.to[0] === r && mv.to[1] === c);
  if (candidates.length > 0) {
    if (candidates.some(mv => mv.promotion)) {
      showPromoDialog(candidates, mv => applyMove(mv, humanSide));
    } else {
      applyMove(candidates[0], humanSide);
    }
    return;
  }

  if (Math.sign(piece) === friendSign) {
    selectSquare(r, c);
  } else {
    selectedSq = null; selectedLegal = [];
    renderBoard();
  }
}

// ── Move Application ───────────────────────────────────────────────────────
function applyMove(mv, bySide) {
  if (bySide === humanSide) clearHints();
  history.push(snapshot());

  gameState = makeMove(gameState, mv);
  gameState._sideToMove = opposite(bySide);
  sideToMove = opposite(bySide);
  selectedSq = null; selectedLegal = [];
  recordPosition();

  // The player has committed to a move, so Back is available again.
  if (bySide === humanSide) {
    backUsed = false;
    searchedEval = null;
  }

  renderBoard();
  renderCaptured();
  updateStatus();
  updateEvalBar();
  updateBackBtn();

  if (!gameOver) {
    legalMoves = getLegalMoves(gameState, sideToMove);
    if (sideToMove === aiSide) requestAIMove();
    else requestHints();
  }
}

// ── Repetition Tracking ────────────────────────────────────────────────────
function recordPosition() {
  const k = positionKey(gameState, sideToMove);
  posCounts.set(k, (posCounts.get(k) || 0) + 1);
  const [lo, hi] = zobristOf(gameState, sideToMove);
  zobristKeys.push(lo, hi);
}

function currentRepCount() {
  return posCounts.get(positionKey(gameState, sideToMove)) || 1;
}

// ── History / Back ─────────────────────────────────────────────────────────
function snapshot() {
  return {
    state: JSON.parse(JSON.stringify(gameState)),
    sideToMove,
    gameOver,
    posCounts: new Map(posCounts),
    zobristKeys: zobristKeys.slice(),
    searchedEval,
  };
}

function restore(snap) {
  gameState = JSON.parse(JSON.stringify(snap.state));
  gameState._sideToMove = snap.sideToMove;
  sideToMove   = snap.sideToMove;
  gameOver     = snap.gameOver;
  posCounts    = new Map(snap.posCounts);
  zobristKeys  = snap.zobristKeys.slice();
  searchedEval = snap.searchedEval;
  selectedSq   = null;
  selectedLegal = [];
  legalMoves   = getLegalMoves(gameState, sideToMove);
}

// Index of the most recent position where it was the player's turn.
function lastHumanTurnIndex() {
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].sideToMove === humanSide) return i;
  }
  return -1;
}

function canGoBack() {
  return mode !== 'play' && !backUsed && lastHumanTurnIndex() >= 0;
}

// Return to the start of the player's last turn — undoing their move and the
// engine's reply if it has come — so they can play something else.
function goBack() {
  if (!canGoBack()) return;
  const idx = lastHumanTurnIndex();

  // Invalidate any search still running: its answer is for a position that is
  // about to stop existing.
  gameId++;
  aiThinking = false;
  hideThinking();
  clearHints();

  const snap = history[idx];
  history.length = idx;
  restore(snap);
  backUsed = true;

  renderBoard();
  renderCaptured();
  updateStatus();
  if (!gameOver && statusEl.className === '') {
    setStatus(`${cap(sideToMove)} to move — try another move`);
  }
  updateEvalBar();
  updateBackBtn();
  requestHints();
}

function updateBackBtn() {
  backBtn.disabled = !canGoBack();
}

// ── Help mode: the player's best moves ─────────────────────────────────────
function hintsWanted() {
  return mode === 'help' && gameState !== null && !gameOver && !aiThinking &&
         sideToMove === humanSide;
}

function requestHints() {
  clearHints();
  if (!hintsWanted()) return;

  const id = ++hintReqId;
  hintsPending = true;
  renderHintPanel();

  const payload = {
    type: 'hints', id,
    state: JSON.parse(JSON.stringify(gameState)),
    bestMs: thinkTimeMs,
    restMs: Math.max(300, thinkTimeMs * 0.25),
    count: HINT_COUNT,
    useNN: nnIsReady(),
    history: zobristKeys.slice(),
  };

  const hw = getHintWorker();
  if (hw) {
    hw.postMessage(payload);
  } else {
    // No workers: search here. The board is briefly unresponsive, but the
    // hints still arrive.
    setTimeout(() => {
      if (id !== hintReqId) return;
      const found = searchTopMoves(payload.state, payload.bestMs, payload.count, payload.useNN,
                                   { history: payload.history, restMs: payload.restMs });
      receiveHints(id, found, true);
    }, 30);
  }
}

// Suggestions arrive one at a time, best first. Each arrival only adds to what
// is on screen, so the ones already shown do not flicker or re-animate.
function receiveHints(id, found, done) {
  if (id !== hintReqId || !hintsWanted()) return;   // the position has moved on
  hintsPending = !done;
  hints = found;
  renderHintPanel();
  renderHintLayer();
}

// Drop the current suggestions. A hint search still running is for a position
// that no longer matters, so its worker is terminated rather than left to
// compete with the engine for the CPU; the next request starts a fresh one.
function clearHints() {
  hintReqId++;
  if (hintsPending && hintWorker) { hintWorker.terminate(); hintWorker = null; }
  hintsPending = false;
  hints = [];
  hintsDrawn = 0;
  hintPanel.innerHTML = '';
  while (hintLayer.firstChild) hintLayer.removeChild(hintLayer.firstChild);
}

function formatScore(h) {
  if (h.mateIn !== null && h.mateIn !== undefined) {
    return h.mateIn > 0 ? `mate in ${h.mateIn}` : `mated in ${-h.mateIn}`;
  }
  const p = h.score / 100;
  return (p >= 0 ? '+' : '−') + Math.abs(p).toFixed(1);
}

function renderHintPanel() {
  // Chips for suggestions not yet shown are appended; existing ones stay put.
  const have = hintPanel.querySelectorAll('.hint-chip').length;
  hintPanel.querySelector('.hint-note')?.remove();

  for (let i = have; i < hints.length; i++) {
    const h = hints[i];
    const chip = document.createElement('button');
    chip.className = `hint-chip r${i + 1}`;
    chip.title = 'Tap to pick up this piece';
    chip.innerHTML =
      `<span class="num">${i + 1}</span>` +
      `<span class="san">${toSAN(gameState, sideToMove, h.move)}</span>` +
      `<span class="score">${formatScore(h)}</span>`;
    // Tapping a suggestion picks the piece up, so one more tap plays it.
    bindTap(chip, () => {
      if (!hintsWanted()) return;
      selectSquare(h.move.from[0], h.move.from[1]);
    });
    hintPanel.appendChild(chip);
  }

  if (hintsPending) {
    const note = document.createElement('span');
    note.className = 'hint-note';
    note.innerHTML = hints.length
      ? '<span class="dots"></span>'
      : 'Finding your best move<span class="dots"></span>';
    hintPanel.appendChild(note);
  }
}

// Arrows over the board, numbered and coloured to match the chips.
const SVG_NS = 'http://www.w3.org/2000/svg';
const HINT_COLORS = ['var(--hint-1)', 'var(--hint-2)', 'var(--hint-3)'];
const HINT_WIDTH  = [17, 14, 12];

function renderHintLayer() {

  const centre = (r, c) => {
    const vr = boardFlipped ? 7 - r : r;
    const vc = boardFlipped ? 7 - c : c;
    return [vc * 100 + 50, vr * 100 + 50];
  };

  // Only suggestions not yet drawn. Each new one is inserted beneath those
  // already there, so the best suggestion stays on top where arrows cross.
  for (let i = hintsDrawn; i < hints.length; i++) {
    const mv = hints[i].move;
    const [x1, y1] = centre(mv.from[0], mv.from[1]);
    const [x2, y2] = centre(mv.to[0], mv.to[1]);
    const color = HINT_COLORS[i];
    const w = HINT_WIDTH[i];

    const g = document.createElementNS(SVG_NS, 'g');
    g.setAttribute('class', 'hint-arrow');

    // Mark the destination with a ring in the hint's colour. A flat tint was
    // tried first and mixed into muddy grey on the light squares.
    const vr = boardFlipped ? 7 - mv.to[0] : mv.to[0];
    const vc = boardFlipped ? 7 - mv.to[1] : mv.to[1];
    const ring = document.createElementNS(SVG_NS, 'rect');
    ring.setAttribute('x', vc * 100 + 4); ring.setAttribute('y', vr * 100 + 4);
    ring.setAttribute('width', 92); ring.setAttribute('height', 92);
    ring.setAttribute('rx', 10);
    ring.setAttribute('fill', color);
    ring.setAttribute('fill-opacity', '0.14');
    ring.setAttribute('stroke', color);
    ring.setAttribute('stroke-width', 7);
    g.appendChild(ring);

    // Shaft, stopping short of the square's centre to leave room for the head.
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    const ux = dx / len, uy = dy / len;
    const head = 34, half = w * 1.35;
    const sx = x1 + ux * 30, sy = y1 + uy * 30;
    const bx = x2 - ux * head, by = y2 - uy * head;

    const shaft = document.createElementNS(SVG_NS, 'line');
    shaft.setAttribute('x1', sx); shaft.setAttribute('y1', sy);
    shaft.setAttribute('x2', bx); shaft.setAttribute('y2', by);
    shaft.setAttribute('stroke', color);
    shaft.setAttribute('stroke-width', w);
    shaft.setAttribute('stroke-linecap', 'round');
    shaft.setAttribute('opacity', '0.88');
    g.appendChild(shaft);

    const px = -uy, py = ux;   // perpendicular
    const tip = document.createElementNS(SVG_NS, 'polygon');
    tip.setAttribute('points',
      `${x2},${y2} ${bx + px * half},${by + py * half} ${bx - px * half},${by - py * half}`);
    tip.setAttribute('fill', color);
    tip.setAttribute('opacity', '0.92');
    g.appendChild(tip);

    // Numbered badge in the corner of the destination square.
    const bxc = vc * 100 + 20, byc = vr * 100 + 20;
    const badge = document.createElementNS(SVG_NS, 'circle');
    badge.setAttribute('cx', bxc); badge.setAttribute('cy', byc);
    badge.setAttribute('r', 15);
    badge.setAttribute('fill', color);
    badge.setAttribute('stroke', '#1a1a1a');
    badge.setAttribute('stroke-width', 2.5);
    g.appendChild(badge);

    const num = document.createElementNS(SVG_NS, 'text');
    num.setAttribute('x', bxc); num.setAttribute('y', byc + 6);
    num.setAttribute('text-anchor', 'middle');
    num.setAttribute('font-size', '18');
    num.setAttribute('font-weight', '800');
    num.setAttribute('fill', '#111');
    num.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, sans-serif');
    num.textContent = String(i + 1);
    g.appendChild(num);

    hintLayer.insertBefore(g, hintLayer.firstChild);
  }
  hintsDrawn = hints.length;
}

// Standard algebraic notation with piece glyphs, e.g. "♘f3", "exd5", "O-O".
function toSAN(state, side, mv) {
  if (mv.castle) return mv.castle === 'K' ? 'O-O' : 'O-O-O';

  const file = c => String.fromCharCode(97 + c);
  const sq = (r, c) => file(c) + (8 - r);
  const [fr, fc] = mv.from, [tr, tc] = mv.to;
  const moving = state.board[fr][fc];
  const pt = Math.abs(moving);
  const capture = mv.captured !== 0 || mv.enPassant;

  let san = '';
  if (pt === 1) {
    if (capture) san += file(fc) + 'x';
    san += sq(tr, tc);
    if (mv.promotion) san += '=' + PIECE_GLYPHS[Math.abs(mv.piece)];
  } else {
    san += PIECE_GLYPHS[pt];
    // Disambiguate when another piece of the same kind can reach the square.
    const rivals = getLegalMoves(state, side).filter(o =>
      Math.abs(state.board[o.from[0]][o.from[1]]) === pt &&
      o.to[0] === tr && o.to[1] === tc &&
      (o.from[0] !== fr || o.from[1] !== fc));
    if (rivals.length) {
      const sameFile = rivals.some(o => o.from[1] === fc);
      const sameRank = rivals.some(o => o.from[0] === fr);
      if (!sameFile) san += file(fc);
      else if (!sameRank) san += String(8 - fr);
      else san += sq(fr, fc);
    }
    if (capture) san += 'x';
    san += sq(tr, tc);
  }

  const after = makeMove(state, mv);
  const them = opposite(side);
  if (isInCheck(after, them)) {
    san += getLegalMoves(after, them).length === 0 ? '#' : '+';
  }
  return san;
}

// ── AI ─────────────────────────────────────────────────────────────────────
function requestAIMove() {
  aiThinking = true;
  showThinking();

  const snap    = JSON.parse(JSON.stringify(gameState));
  const keys    = zobristKeys.slice();
  const forSide = aiSide;
  const forGame = gameId;

  findBestMove(snap, thinkTimeMs, keys).then(res => {
    // Discard results from a game or position that has since moved on.
    if (forGame !== gameId) return;

    aiThinking = false;
    hideThinking();

    if (gameOver || sideToMove !== forSide) return;

    const best = res && res.move;
    if (!best) { setStatus('AI has no legal move!'); return; }

    searchedEval = res.info && typeof res.info.score === 'number' ? res.info.score : null;
    applyMove(best, forSide);
  });
}

// ── Status ─────────────────────────────────────────────────────────────────
function updateStatus() {
  const st = getGameStatus(gameState, sideToMove, currentRepCount());
  if (st.over) {
    gameOver = true;
    if (st.result === 'white_wins')      setStatus('White wins by checkmate!', 'mate');
    else if (st.result === 'black_wins') setStatus('Black wins by checkmate!', 'mate');
    else if (st.reason === 'stalemate')  setStatus('Stalemate — draw.', 'draw');
    else                                 setStatus(`Draw — ${st.reason}.`, 'draw');
    return;
  }
  gameOver = false;
  if (isInCheck(gameState, sideToMove)) {
    setStatus(`${cap(sideToMove)} to move — Check!`, 'check');
  } else {
    setStatus(`${cap(sideToMove)} to move`);
  }
}

function setStatus(msg, cls = '') {
  statusEl.textContent = msg;
  statusEl.className = cls;
}

function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

function showThinking() {
  thinkingEl.removeAttribute('hidden');
  setStatus('AI thinking…', 'thinking');
}
function hideThinking() {
  thinkingEl.setAttribute('hidden', '');
}

// ── Promotion Dialog ───────────────────────────────────────────────────────
function showPromoDialog(candidates, callback) {
  const isWhite = humanSide === 'white';
  const opts = [
    { code: 5, glyph: isWhite ? '♕' : '♛' },
    { code: 4, glyph: isWhite ? '♖' : '♜' },
    { code: 3, glyph: isWhite ? '♗' : '♝' },
    { code: 2, glyph: isWhite ? '♘' : '♞' },
  ];
  promoChoices.innerHTML = '';
  for (const { code, glyph } of opts) {
    const div = document.createElement('div');
    div.className = 'promo-piece';
    div.textContent = glyph;
    const pick = () => {
      promoModal.setAttribute('hidden', '');
      const sign = isWhite ? 1 : -1;
      const mv = candidates.find(m => m.piece === sign * code) || candidates[0];
      callback(mv);
    };
    div.addEventListener('click', pick);
    div.addEventListener('touchstart', e => { e.preventDefault(); pick(); }, { passive: false });
    promoChoices.appendChild(div);
  }
  promoModal.removeAttribute('hidden');
}

// ── Confirmation dialog ────────────────────────────────────────────────────
// Resolves true for Yes, false for No, a tap outside the card, or Escape.
let confirmResolve = null;

function confirmDialog(title, sub) {
  if (confirmResolve) confirmResolve(false);   // never stack two dialogs
  confirmTitle.textContent = title;
  confirmSub.textContent = sub;
  confirmModal.removeAttribute('hidden');
  return new Promise(resolve => { confirmResolve = resolve; });
}

function closeConfirm(answer) {
  if (!confirmResolve) return;
  confirmModal.setAttribute('hidden', '');
  const r = confirmResolve;
  confirmResolve = null;
  r(answer);
}

// ── Controls ───────────────────────────────────────────────────────────────
const playWhiteBtn = document.getElementById('play-white-btn');
const playBlackBtn = document.getElementById('play-black-btn');

// Taps are bound for both mouse and touch; touchstart is intercepted so iOS
// does not wait for its click delay.
function bindTap(el, fn) {
  el.addEventListener('click', fn);
  el.addEventListener('touchstart', e => { e.preventDefault(); fn(); }, { passive: false });
}

// A game with moves on the board and no result yet is worth asking about.
function gameInProgress() {
  return history.length > 0 && !gameOver;
}

async function requestNewGame(side) {
  if (gameInProgress()) {
    const title = side === humanSide ? 'Start a new game?' : `Start a new game as ${cap(side)}?`;
    if (!(await confirmDialog(title, 'The current game will be lost.'))) return;
  }
  playWhiteBtn.classList.toggle('active', side === 'white');
  playBlackBtn.classList.toggle('active', side === 'black');
  startNewGame(side);
}

bindTap(playWhiteBtn, () => requestNewGame('white'));
bindTap(playBlackBtn, () => requestNewGame('black'));
bindTap(newGameBtn,   () => requestNewGame(humanSide));
bindTap(backBtn,      () => goBack());

bindTap(confirmYes, () => closeConfirm(true));
bindTap(confirmNo,  () => closeConfirm(false));
confirmModal.addEventListener('click', e => { if (e.target === confirmModal) closeConfirm(false); });
document.addEventListener('keydown', e => {
  if (!confirmResolve) return;
  if (e.key === 'Escape') closeConfirm(false);
  if (e.key === 'Enter')  closeConfirm(true);
});

document.querySelectorAll('[data-time]').forEach(btn => {
  bindTap(btn, () => {
    document.querySelectorAll('[data-time]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    thinkTimeMs = parseInt(btn.dataset.time, 10);
  });
});

// ── Mode switching ─────────────────────────────────────────────────────────
function loadMode() {
  try {
    const m = localStorage.getItem('chessnn.mode');
    if (MODES.includes(m)) return m;
  } catch (_) { /* storage unavailable — fall back to the default */ }
  return 'play';
}

function setMode(next) {
  if (!MODES.includes(next)) return;
  const changed = next !== mode;
  mode = next;
  try { localStorage.setItem('chessnn.mode', mode); } catch (_) {}

  modeSwitch.dataset.mode = mode;
  modeSwitch.querySelectorAll('.mode-btn').forEach(b =>
    b.setAttribute('aria-checked', String(b.dataset.mode === mode)));
  document.body.classList.toggle('mode-help', mode === 'help');

  // Cross-fade the caption rather than snapping it.
  if (changed) {
    modeCaption.classList.add('swap');
    setTimeout(() => {
      modeCaption.textContent = MODE_CAPTIONS[mode];
      modeCaption.classList.remove('swap');
    }, 150);
  } else {
    modeCaption.textContent = MODE_CAPTIONS[mode];
  }

  updateBackBtn();
  if (mode === 'help') { if (!hints.length && !hintsPending) requestHints(); }
  else clearHints();
}

modeSwitch.querySelectorAll('.mode-btn').forEach(btn => {
  bindTap(btn, () => setMode(btn.dataset.mode));
});

// ── Initialization ─────────────────────────────────────────────────────────
function startNewGame(chosenHumanSide = humanSide) {
  gameId++;            // invalidates any search still running for the old game
  aiThinking   = false;
  hideThinking();
  clearHints();
  resetSearchState();  // no knowledge carries over between games
  humanSide    = chosenHumanSide;
  aiSide       = opposite(chosenHumanSide);
  boardFlipped = humanSide === 'black'; // rotate board so player's pieces are always at bottom
  gameState    = initState();
  gameState._sideToMove = 'white';
  sideToMove   = 'white';
  gameOver     = false;
  selectedSq   = null;
  selectedLegal = [];
  posCounts    = new Map();
  zobristKeys  = [];
  history      = [];
  backUsed     = false;
  searchedEval = null;
  recordPosition();
  legalMoves   = getLegalMoves(gameState, 'white');
  buildBoard();     // rebuild grid with correct orientation
  renderBoard();
  renderCaptured();
  updateEvalBar();
  updateBackBtn();
  setStatus('White to move');
  if (aiSide === 'white') requestAIMove(); // AI goes first when playing as Black
  else requestHints();
}

buildBoard();
setMode(mode);
startNewGame();
