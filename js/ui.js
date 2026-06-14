'use strict';
// UI controller — ES module, runs on main thread

import { initState, makeMove, getLegalMoves, getGameStatus, isInCheck, PIECE_GLYPHS, opposite } from './chess.js';
import { loadModel, isReady as nnIsReady } from './neural.js';
import { searchBestMove } from './engine.js';

// ── State ──────────────────────────────────────────────────────────────────
let gameState    = null;
let humanSide    = 'white';
let aiSide       = 'black';
let legalMoves   = [];
let selectedSq   = null; // [r, c] or null
let selectedLegal = [];
let thinkTimeMs  = 5000;
let useNN        = true;
let gameOver     = false;
let sideToMove   = 'white';
let aiThinking   = false;

// ── DOM References ─────────────────────────────────────────────────────────
const boardEl    = document.getElementById('board');
const statusEl   = document.getElementById('status');
const capturedEl = document.getElementById('captured-wrap');
const promoModal = document.getElementById('promo-modal');
const promoChoices= document.getElementById('promo-choices');
const thinkingEl = document.getElementById('thinking-overlay');
const newGameBtn = document.getElementById('new-game-btn');
const nnCb       = document.getElementById('nn-cb');

// ── Neural Network Initialization ──────────────────────────────────────────
// Load TF.js model in background; the game works without it (pure evaluation fallback)
(async function() {
  try {
    if (typeof tf !== 'undefined') {
      await loadModel(tf);
      if (nnIsReady()) {
        setStatus('Ready (NN loaded)');
      } else {
        setStatus('Ready (no NN model found)');
        nnCb.checked = false;
        useNN = false;
      }
    } else {
      setStatus('Ready (TF.js not loaded)');
      nnCb.checked = false;
      useNN = false;
    }
  } catch (_) {
    setStatus('Ready');
    nnCb.checked = false;
    useNN = false;
  }
})();

// ── Board Rendering ────────────────────────────────────────────────────────
function buildBoard() {
  boardEl.innerHTML = '';
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const div = document.createElement('div');
      div.className = `sq ${(r + c) % 2 === 0 ? 'light' : 'dark'}`;
      div.dataset.r = r;
      div.dataset.c = c;

      // Rank label on col 0
      if (c === 0) {
        const rank = document.createElement('span');
        rank.className = 'coord coord-rank';
        rank.textContent = 8 - r;
        div.appendChild(rank);
      }
      // File label on row 7
      if (r === 7) {
        const file = document.createElement('span');
        file.className = 'coord coord-file';
        file.textContent = String.fromCharCode('a'.charCodeAt(0) + c);
        div.appendChild(file);
      }

      boardEl.appendChild(div);
    }
  }
}

function renderBoard() {
  const squares = boardEl.querySelectorAll('.sq');
  squares.forEach(sq => {
    const r = parseInt(sq.dataset.r);
    const c = parseInt(sq.dataset.c);
    const piece = gameState.board[r][c];
    const glyph = PIECE_GLYPHS[piece] || '';

    const existing = sq.querySelector('.piece-span');
    if (existing) existing.remove();

    if (glyph) {
      const span = document.createElement('span');
      span.className = `piece-span ${piece > 0 ? 'piece-w' : 'piece-b'}`;
      span.textContent = glyph;
      sq.appendChild(span);
    }

    sq.classList.remove('selected', 'legal', 'legal-cap', 'last-move-from', 'last-move-to');
  });

  // Last move highlights
  if (gameState.lastMove) {
    const { from: [fr, fc], to: [tr, tc] } = gameState.lastMove;
    getSquareEl(fr, fc).classList.add('last-move-from');
    getSquareEl(tr, tc).classList.add('last-move-to');
  }

  // Selection highlights
  if (selectedSq) {
    const [sr, sc] = selectedSq;
    getSquareEl(sr, sc).classList.add('selected');
    for (const mv of selectedLegal) {
      const el = getSquareEl(mv.to[0], mv.to[1]);
      el.classList.add(mv.captured !== 0 || mv.enPassant ? 'legal-cap' : 'legal');
    }
  }
}

function getSquareEl(r, c) {
  return boardEl.querySelector(`[data-r="${r}"][data-c="${c}"]`);
}

function renderCaptured() {
  const board = gameState.board;
  const startCounts = { 1:8, 2:2, 3:2, 4:2, 5:1, 6:1 };
  const wCount = {}, bCount = {};
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p > 0) wCount[p] = (wCount[p] || 0) + 1;
      if (p < 0) bCount[-p] = (bCount[-p] || 0) + 1;
    }

  let html = '';
  // Black captured white pieces (shown with white glyphs)
  for (let pt = 5; pt >= 1; pt--) {
    const n = (startCounts[pt] || 0) - (wCount[pt] || 0);
    for (let i = 0; i < n; i++) html += `<span class="piece-w">${PIECE_GLYPHS[pt]}</span>`;
  }
  html += '<span style="display:inline-block;width:12px"></span>';
  // White captured black pieces (shown with black glyphs)
  for (let pt = 5; pt >= 1; pt--) {
    const n = (startCounts[pt] || 0) - (bCount[pt] || 0);
    for (let i = 0; i < n; i++) html += `<span class="piece-b">${PIECE_GLYPHS[-pt]}</span>`;
  }
  capturedEl.innerHTML = html;
}

// ── Tap Handling ───────────────────────────────────────────────────────────
boardEl.addEventListener('click', onBoardTap);
boardEl.addEventListener('touchstart', (e) => {
  e.preventDefault();
  const t = e.changedTouches[0];
  const el = document.elementFromPoint(t.clientX, t.clientY);
  if (el) {
    const sq = el.closest('.sq');
    if (sq) {
      sq.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    }
  }
}, { passive: false });

function onBoardTap(e) {
  if (gameOver || sideToMove !== humanSide || aiThinking) return;
  const sqEl = e.target.closest('.sq');
  if (!sqEl) return;
  const r = parseInt(sqEl.dataset.r);
  const c = parseInt(sqEl.dataset.c);
  handleSquareTap(r, c);
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
    const isPromo = candidates.some(mv => mv.promotion);
    if (isPromo) {
      showPromoDialog(candidates, (mv) => applyMove(mv, humanSide));
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

  renderBoard();
  renderCaptured();
  updateStatus();

  if (!gameOver) {
    legalMoves = getLegalMoves(gameState, sideToMove);
    if (sideToMove === aiSide) requestAIMove();
  }
}

// ── AI ─────────────────────────────────────────────────────────────────────
function requestAIMove() {
  aiThinking = true;
  showThinking();

  // Use setTimeout(0) to let the browser render the spinner before blocking
  setTimeout(() => {
    const stateSnapshot = JSON.parse(JSON.stringify(gameState));
    const bestMove = searchBestMove(stateSnapshot, thinkTimeMs, useNN && nnIsReady());
    aiThinking = false;
    hideThinking();

    if (!bestMove) {
      setStatus('AI has no legal move!');
      return;
    }
    applyMove(bestMove, aiSide);
  }, 20);
}

// ── Status ─────────────────────────────────────────────────────────────────
function updateStatus() {
  const status = getGameStatus(gameState, sideToMove);
  if (status.over) {
    gameOver = true;
    if (status.result === 'white_wins') setStatus('White wins by checkmate!', 'mate');
    else if (status.result === 'black_wins') setStatus('Black wins by checkmate!', 'mate');
    else setStatus('Stalemate — draw.', 'draw');
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
  const options = [
    { code: 5, glyph: isWhite ? '♕' : '♛' },
    { code: 4, glyph: isWhite ? '♖' : '♜' },
    { code: 3, glyph: isWhite ? '♗' : '♝' },
    { code: 2, glyph: isWhite ? '♘' : '♞' },
  ];

  promoChoices.innerHTML = '';
  for (const { code, glyph } of options) {
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
    div.addEventListener('touchstart', (e) => { e.preventDefault(); pick(); }, { passive: false });
    promoChoices.appendChild(div);
  }
  promoModal.removeAttribute('hidden');
}

// ── Controls ───────────────────────────────────────────────────────────────
const playWhiteBtn = document.getElementById('play-white-btn');
const playBlackBtn = document.getElementById('play-black-btn');

function selectColor(side) {
  if (side === 'white') {
    playWhiteBtn.classList.add('active');
    playBlackBtn.classList.remove('active');
  } else {
    playBlackBtn.classList.add('active');
    playWhiteBtn.classList.remove('active');
  }
  startNewGame(side);
}

playWhiteBtn.addEventListener('click', () => selectColor('white'));
playBlackBtn.addEventListener('click', () => selectColor('black'));
playWhiteBtn.addEventListener('touchstart', (e) => { e.preventDefault(); selectColor('white'); }, { passive: false });
playBlackBtn.addEventListener('touchstart', (e) => { e.preventDefault(); selectColor('black'); }, { passive: false });

newGameBtn.addEventListener('click', () => startNewGame(humanSide));
newGameBtn.addEventListener('touchstart', (e) => { e.preventDefault(); newGameBtn.click(); }, { passive: false });

nnCb.addEventListener('change', () => { useNN = nnCb.checked; });

document.querySelectorAll('[data-time]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-time]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    thinkTimeMs = parseInt(btn.dataset.time, 10);
  });
  btn.addEventListener('touchstart', (e) => { e.preventDefault(); btn.click(); }, { passive: false });
});

// ── Initialization ─────────────────────────────────────────────────────────
function startNewGame(chosenHumanSide = humanSide) {
  if (aiThinking) return; // don't reset mid-search
  humanSide = chosenHumanSide;
  aiSide = opposite(chosenHumanSide);
  gameState = initState();
  gameState._sideToMove = 'white';
  sideToMove = 'white';
  gameOver = false;
  selectedSq = null;
  selectedLegal = [];
  legalMoves = getLegalMoves(gameState, 'white');
  renderBoard();
  renderCaptured();
  setStatus('White to move');
  // If the AI plays white, fire its opening move immediately
  if (aiSide === 'white') requestAIMove();
}

buildBoard();
startNewGame();
