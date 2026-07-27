'use strict';
// UI controller — ES module, runs on main thread

import {
  initState, makeMove, getLegalMoves, getGameStatus,
  isInCheck, PIECE_GLYPHS, opposite, evaluatePosition, positionKey
} from './chess.js';
import { loadModel, isReady as nnIsReady, evaluate, lastError as nnError } from './neural.js';
import { searchBestMove } from './engine.js';

// ── State ──────────────────────────────────────────────────────────────────
let gameState     = null;
let humanSide     = 'white';
let aiSide        = 'black';
let legalMoves    = [];
let selectedSq    = null;
let selectedLegal = [];
let thinkTimeMs   = 5000;
let useNN         = true;
let gameOver      = false;
let sideToMove    = 'white';
let aiThinking    = false;
let boardFlipped  = false; // true when playing as Black (board rotated 180°)
let posCounts     = new Map(); // position key → occurrences, for threefold repetition
let gameId        = 0;         // bumped per game so stale search results are ignored

// ── DOM References ─────────────────────────────────────────────────────────
const boardEl     = document.getElementById('board');
const statusEl    = document.getElementById('status');
const capturedEl  = document.getElementById('captured-wrap');
const promoModal  = document.getElementById('promo-modal');
const promoChoices = document.getElementById('promo-choices');
const thinkingEl  = document.getElementById('thinking-overlay');
const newGameBtn  = document.getElementById('new-game-btn');
const nnCb        = document.getElementById('nn-cb');
const evalFillEl  = document.getElementById('eval-white-fill');
const evalTextEl  = document.getElementById('eval-text');

// ── Search Worker ──────────────────────────────────────────────────────────
// The engine runs in a Web Worker so a multi-second search never freezes the
// board or the spinner. If module workers are unavailable the engine still
// runs on the main thread — correct, just less smooth.
let worker      = null;
let workerReqId = 0;
const pendingSearches = new Map();

try {
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'result') {
      const resolve = pendingSearches.get(msg.id);
      if (resolve) { pendingSearches.delete(msg.id); resolve(msg.move); }
    }
  };
  worker.onerror = () => { worker = null; };   // fall back to main thread
} catch (_) {
  worker = null;
}

// Resolves to the chosen move, searching off-thread when possible.
function findBestMove(state, timeLimit, withNN) {
  if (worker) {
    return new Promise(resolve => {
      const id = ++workerReqId;
      pendingSearches.set(id, resolve);
      worker.postMessage({ type: 'search', id, state, timeLimit, useNN: withNN });
    });
  }
  return new Promise(resolve => {
    setTimeout(() => resolve(searchBestMove(state, timeLimit, withNN)), 20);
  });
}

// ── Neural Network Initialization ──────────────────────────────────────────
// Pure-JS inference (js/neural.js) — no TensorFlow.js, no CDN, works offline.
// The main thread loads its own copy for the evaluation bar; the worker loads
// one for the search. The game is fully playable without the model.
(async function () {
  try {
    await loadModel();
    if (nnIsReady()) {
      nnCb.disabled = false;
      updateEvalBar();
      if (!aiThinking && !gameOver) setStatus(`${cap(sideToMove)} to move`);
    } else {
      console.warn('Neural net unavailable:', nnError());
      nnCb.checked  = false;
      nnCb.disabled = true;
      useNN = false;
    }
  } catch (e) {
    console.warn('Neural net failed to load:', e);
    nnCb.checked  = false;
    nnCb.disabled = true;
    useNN = false;
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
// NN (neural.js) returns white-perspective; evaluatePosition is black-perspective in centipawns.
function updateEvalBar() {
  if (!gameState || !evalFillEl || !evalTextEl) return;
  let whiteAdv = 0;
  try {
    const nn = useNN && nnIsReady() ? evaluate(gameState.board) : null;
    whiteAdv = nn !== null ? nn : -evaluatePosition(gameState) / 100;
  } catch (_) {}

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

function handleSquareTap(r, c) {
  const piece = gameState.board[r][c];
  const friendSign = humanSide === 'white' ? 1 : -1;

  if (!selectedSq) {
    if (Math.sign(piece) !== friendSign) return;
    selectedSq = [r, c];
    selectedLegal = legalMoves.filter(mv => mv.from[0] === r && mv.from[1] === c);
    renderBoard();
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
    selectedSq = [r, c];
    selectedLegal = legalMoves.filter(mv => mv.from[0] === r && mv.from[1] === c);
  } else {
    selectedSq = null; selectedLegal = [];
  }
  renderBoard();
}

// ── Move Application ───────────────────────────────────────────────────────
function applyMove(mv, bySide) {
  gameState = makeMove(gameState, mv);
  gameState._sideToMove = opposite(bySide);
  sideToMove = opposite(bySide);
  selectedSq = null; selectedLegal = [];
  recordPosition();

  renderBoard();
  renderCaptured();
  updateStatus();
  updateEvalBar();

  if (!gameOver) {
    legalMoves = getLegalMoves(gameState, sideToMove);
    if (sideToMove === aiSide) requestAIMove();
  }
}

// ── Repetition Tracking ────────────────────────────────────────────────────
function recordPosition() {
  const k = positionKey(gameState, sideToMove);
  posCounts.set(k, (posCounts.get(k) || 0) + 1);
}

function currentRepCount() {
  return posCounts.get(positionKey(gameState, sideToMove)) || 1;
}

// ── AI ─────────────────────────────────────────────────────────────────────
function requestAIMove() {
  aiThinking = true;
  showThinking();

  const snap    = JSON.parse(JSON.stringify(gameState));
  const forSide = aiSide;
  const forGame = gameId;

  findBestMove(snap, thinkTimeMs, useNN && nnIsReady()).then(best => {
    // Discard results from a game that has since been restarted.
    if (forGame !== gameId) return;

    aiThinking = false;
    hideThinking();

    if (gameOver || sideToMove !== forSide) return;

    if (!best) { setStatus('AI has no legal move!'); return; }
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

// ── Controls ───────────────────────────────────────────────────────────────
const playWhiteBtn = document.getElementById('play-white-btn');
const playBlackBtn = document.getElementById('play-black-btn');

function selectColor(side) {
  playWhiteBtn.classList.toggle('active', side === 'white');
  playBlackBtn.classList.toggle('active', side === 'black');
  startNewGame(side);
}

playWhiteBtn.addEventListener('click', () => selectColor('white'));
playBlackBtn.addEventListener('click', () => selectColor('black'));
playWhiteBtn.addEventListener('touchstart', e => { e.preventDefault(); selectColor('white'); }, { passive: false });
playBlackBtn.addEventListener('touchstart', e => { e.preventDefault(); selectColor('black'); }, { passive: false });

newGameBtn.addEventListener('click', () => startNewGame(humanSide));
newGameBtn.addEventListener('touchstart', e => { e.preventDefault(); newGameBtn.click(); }, { passive: false });

nnCb.addEventListener('change', () => { useNN = nnCb.checked; });

document.querySelectorAll('[data-time]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-time]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    thinkTimeMs = parseInt(btn.dataset.time, 10);
  });
  btn.addEventListener('touchstart', e => { e.preventDefault(); btn.click(); }, { passive: false });
});

// ── Initialization ─────────────────────────────────────────────────────────
function startNewGame(chosenHumanSide = humanSide) {
  gameId++;            // invalidates any search still running for the old game
  aiThinking   = false;
  hideThinking();
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
  recordPosition();
  legalMoves   = getLegalMoves(gameState, 'white');
  buildBoard();     // rebuild grid with correct orientation
  renderBoard();
  renderCaptured();
  updateEvalBar();
  setStatus('White to move');
  if (aiSide === 'white') requestAIMove(); // AI goes first when playing as Black
}

buildBoard();
startNewGame();
