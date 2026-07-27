'use strict';

// Piece codes: ±1=pawn, ±2=knight, ±3=bishop, ±4=rook, ±5=queen, ±6=king
// Positive = White, Negative = Black
// Board: 0-indexed [row 0..7][col 0..7], row 0 = rank 8 (Black's back rank), row 7 = rank 1 (White's back rank)

const PIECE_VALUES = [0, 100, 320, 330, 500, 900, 0]; // index by abs(piece code)
const SIMPLE_VALUES = [0, 1, 3, 3, 5, 9, 0];

// Both sides use the SAME outline glyphs — CSS classes handle color differentiation
const _PT_GLYPHS = ['', '♙', '♘', '♗', '♖', '♕', '♔']; // index = abs(piece code)
const PIECE_GLYPHS = {};
for (let pt = 0; pt <= 6; pt++) {
  PIECE_GLYPHS[pt]  = _PT_GLYPHS[pt]; // white piece codes  0..6
  PIECE_GLYPHS[-pt] = _PT_GLYPHS[pt]; // black piece codes -6..0 → same glyph
}

// Piece-square tables (from White's perspective, row 0 = rank 8, row 7 = rank 1)
const PST = {
  1: [ // Pawn
    [  0,  0,  0,  0,  0,  0,  0,  0],
    [ 50, 50, 50, 50, 50, 50, 50, 50],
    [ 10, 10, 20, 30, 30, 20, 10, 10],
    [  5,  5, 10, 25, 25, 10,  5,  5],
    [  0,  0,  0, 20, 20,  0,  0,  0],
    [  5, -5,-10,  0,  0,-10, -5,  5],
    [  5, 10, 10,-20,-20, 10, 10,  5],
    [  0,  0,  0,  0,  0,  0,  0,  0]
  ],
  2: [ // Knight
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50]
  ],
  3: [ // Bishop
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20]
  ],
  4: [ // Rook
    [  0,  0,  0,  0,  0,  0,  0,  0],
    [  5, 10, 10, 10, 10, 10, 10,  5],
    [ -5,  0,  0,  0,  0,  0,  0, -5],
    [ -5,  0,  0,  0,  0,  0,  0, -5],
    [ -5,  0,  0,  0,  0,  0,  0, -5],
    [ -5,  0,  0,  0,  0,  0,  0, -5],
    [ -5,  0,  0,  0,  0,  0,  0, -5],
    [  0,  0,  0,  5,  5,  0,  0,  0]
  ],
  5: [ // Queen
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [ -5,  0,  5,  5,  5,  5,  0, -5],
    [  0,  0,  5,  5,  5,  5,  0, -5],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20]
  ],
  6: [ // King (middlegame)
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [ 20, 20,  0,  0,  0,  0, 20, 20],
    [ 20, 30, 10,  0,  0, 10, 30, 20]
  ]
};

function initState() {
  // Row 0 = rank 8 (Black back rank), Row 7 = rank 1 (White back rank)
  const board = [
    [-4,-2,-3,-5,-6,-3,-2,-4],
    [-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0],
    [ 1, 1, 1, 1, 1, 1, 1, 1],
    [ 4, 2, 3, 5, 6, 3, 2, 4]
  ];
  return {
    board,
    wKc: true, wQc: true,   // White kingside/queenside castling rights
    bKc: true, bQc: true,   // Black kingside/queenside castling rights
    lastMove: null,          // {from, to, piece, captured}
    selected: null,          // [r, c] currently selected square
    enPassantTarget: null,   // [r, c] square a pawn can capture en passant
    halfmoveClock: 0         // plies since last capture or pawn move (50-move rule)
  };
}

function cloneState(st) {
  return {
    board: st.board.map(row => [...row]),
    wKc: st.wKc, wQc: st.wQc,
    bKc: st.bKc, bQc: st.bQc,
    lastMove: st.lastMove ? {...st.lastMove} : null,
    selected: st.selected ? [...st.selected] : null,
    enPassantTarget: st.enPassantTarget ? [...st.enPassantTarget] : null,
    halfmoveClock: st.halfmoveClock || 0
  };
}

function inBounds(r, c) {
  return r >= 0 && r < 8 && c >= 0 && c < 8;
}

function sign(x) {
  return x > 0 ? 1 : x < 0 ? -1 : 0;
}

function generateMoves(st, side) {
  const moves = [];
  const board = st.board;
  const friendSign = side === 'white' ? 1 : -1;
  const enemySign = -friendSign;

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p === 0 || sign(p) !== friendSign) continue;
      const pt = Math.abs(p);

      if (pt === 1) { // Pawn
        const forward = friendSign === 1 ? -1 : 1; // White moves up (decreasing row), Black down
        const startRow = friendSign === 1 ? 6 : 1;
        const epRow = friendSign === 1 ? 3 : 4;

        const nr = r + forward;
        if (inBounds(nr, c)) {
          if (board[nr][c] === 0) {
            addPawnMove(moves, r, c, nr, c, p, 0, friendSign);
            if (r === startRow && board[r + 2*forward][c] === 0) {
              moves.push({from:[r,c], to:[r+2*forward,c], piece:p, captured:0});
            }
          }
          for (const dc of [-1, 1]) {
            const cc = c + dc;
            if (!inBounds(nr, cc)) continue;
            if (sign(board[nr][cc]) === enemySign) {
              addPawnMove(moves, r, c, nr, cc, p, board[nr][cc], friendSign);
            } else if (st.enPassantTarget && r === epRow && nr === st.enPassantTarget[0] && cc === st.enPassantTarget[1]) {
              moves.push({from:[r,c], to:[nr,cc], piece:p, captured: friendSign === 1 ? -1 : 1, enPassant:true});
            }
          }
        }

      } else if (pt === 2) { // Knight
        for (const [dr, dc] of [[1,2],[1,-2],[-1,2],[-1,-2],[2,1],[2,-1],[-2,1],[-2,-1]]) {
          const nr = r+dr, nc = c+dc;
          if (inBounds(nr, nc) && sign(board[nr][nc]) !== friendSign)
            moves.push({from:[r,c], to:[nr,nc], piece:p, captured:board[nr][nc]});
        }

      } else if (pt === 3 || pt === 4 || pt === 5) { // Bishop/Rook/Queen
        const dirs = pt === 3 ? [[1,1],[1,-1],[-1,1],[-1,-1]]
                   : pt === 4 ? [[1,0],[-1,0],[0,1],[0,-1]]
                   : [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]];
        for (const [dr, dc] of dirs) {
          let nr = r+dr, nc = c+dc;
          while (inBounds(nr, nc)) {
            if (board[nr][nc] === 0) {
              moves.push({from:[r,c], to:[nr,nc], piece:p, captured:0});
            } else {
              if (sign(board[nr][nc]) === enemySign)
                moves.push({from:[r,c], to:[nr,nc], piece:p, captured:board[nr][nc]});
              break;
            }
            nr += dr; nc += dc;
          }
        }

      } else if (pt === 6) { // King
        for (const [dr, dc] of [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]) {
          const nr = r+dr, nc = c+dc;
          if (inBounds(nr, nc) && sign(board[nr][nc]) !== friendSign)
            moves.push({from:[r,c], to:[nr,nc], piece:p, captured:board[nr][nc]});
        }
        // Castling
        if (side === 'white' && r === 7 && c === 4) {
          if (st.wKc && board[7][5]===0 && board[7][6]===0 && board[7][7]===4
              && !squareAttacked(board,7,4,'black') && !squareAttacked(board,7,5,'black') && !squareAttacked(board,7,6,'black'))
            moves.push({from:[7,4], to:[7,6], piece:6, captured:0, castle:'K'});
          if (st.wQc && board[7][3]===0 && board[7][2]===0 && board[7][1]===0 && board[7][0]===4
              && !squareAttacked(board,7,4,'black') && !squareAttacked(board,7,3,'black') && !squareAttacked(board,7,2,'black'))
            moves.push({from:[7,4], to:[7,2], piece:6, captured:0, castle:'Q'});
        }
        if (side === 'black' && r === 0 && c === 4) {
          if (st.bKc && board[0][5]===0 && board[0][6]===0 && board[0][7]===-4
              && !squareAttacked(board,0,4,'white') && !squareAttacked(board,0,5,'white') && !squareAttacked(board,0,6,'white'))
            moves.push({from:[0,4], to:[0,6], piece:-6, captured:0, castle:'K'});
          if (st.bQc && board[0][3]===0 && board[0][2]===0 && board[0][1]===0 && board[0][0]===-4
              && !squareAttacked(board,0,4,'white') && !squareAttacked(board,0,3,'white') && !squareAttacked(board,0,2,'white'))
            moves.push({from:[0,4], to:[0,2], piece:-6, captured:0, castle:'Q'});
        }
      }
    }
  }
  return moves;
}

function addPawnMove(moves, fr, fc, tr, tc, piece, captured, friendSign) {
  const promotionRow = friendSign === 1 ? 0 : 7;
  if (tr === promotionRow) {
    for (const promo of [5, 4, 3, 2]) {
      moves.push({from:[fr,fc], to:[tr,tc], piece: friendSign * promo, captured, promotion:true});
    }
  } else {
    moves.push({from:[fr,fc], to:[tr,tc], piece, captured});
  }
}

function squareAttacked(board, r, c, attackerSide) {
  const aSign = attackerSide === 'white' ? 1 : -1;
  const pawnDir = aSign === 1 ? 1 : -1; // White pawns attack upward (increasing row from their perspective)

  // Pawn attacks
  const pr = r + pawnDir; // row that an attacker pawn would be on
  if (inBounds(pr, c)) {
    if (inBounds(pr, c-1) && board[pr][c-1] === aSign * 1) return true;
    if (inBounds(pr, c+1) && board[pr][c+1] === aSign * 1) return true;
  }

  // Knight attacks
  for (const [dr, dc] of [[1,2],[1,-2],[-1,2],[-1,-2],[2,1],[2,-1],[-2,1],[-2,-1]]) {
    const nr = r+dr, nc = c+dc;
    if (inBounds(nr,nc) && board[nr][nc] === aSign * 2) return true;
  }

  // Rook/Queen (straight lines)
  for (const [dr, dc] of [[1,0],[-1,0],[0,1],[0,-1]]) {
    let nr = r+dr, nc = c+dc;
    while (inBounds(nr, nc)) {
      if (board[nr][nc] !== 0) {
        if (board[nr][nc] === aSign*4 || board[nr][nc] === aSign*5) return true;
        break;
      }
      nr += dr; nc += dc;
    }
  }

  // Bishop/Queen (diagonals)
  for (const [dr, dc] of [[1,1],[1,-1],[-1,1],[-1,-1]]) {
    let nr = r+dr, nc = c+dc;
    while (inBounds(nr, nc)) {
      if (board[nr][nc] !== 0) {
        if (board[nr][nc] === aSign*3 || board[nr][nc] === aSign*5) return true;
        break;
      }
      nr += dr; nc += dc;
    }
  }

  // King attacks
  for (const [dr, dc] of [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]) {
    const nr = r+dr, nc = c+dc;
    if (inBounds(nr,nc) && board[nr][nc] === aSign*6) return true;
  }

  return false;
}

function findKing(board, side) {
  const code = side === 'white' ? 6 : -6;
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++)
      if (board[r][c] === code) return [r, c];
  return null;
}

function isInCheck(st, side) {
  const kingPos = findKing(st.board, side);
  if (!kingPos) return false;
  return squareAttacked(st.board, kingPos[0], kingPos[1], side === 'white' ? 'black' : 'white');
}

function filterLegal(moves, st, side) {
  return moves.filter(mv => {
    const next = makeMove(st, mv);
    return !isInCheck(next, side);
  });
}

function makeMove(st, mv) {
  const next = cloneState(st);
  const board = next.board;
  const [fr, fc] = mv.from;
  const [tr, tc] = mv.to;
  const piece = mv.piece;

  board[tr][tc] = piece;
  board[fr][fc] = 0;

  // En passant capture: remove the captured pawn
  if (mv.enPassant) {
    const capturedRow = fr; // the pawn captured is on the same row as the moving pawn
    board[capturedRow][tc] = 0;
  }

  // Castling: move the rook
  if (mv.castle) {
    if (tr === 7 && tc === 6) { board[7][5] = 4;  board[7][7] = 0; } // White kingside
    if (tr === 7 && tc === 2) { board[7][3] = 4;  board[7][0] = 0; } // White queenside
    if (tr === 0 && tc === 6) { board[0][5] = -4; board[0][7] = 0; } // Black kingside
    if (tr === 0 && tc === 2) { board[0][3] = -4; board[0][0] = 0; } // Black queenside
  }

  // Update castling rights
  if (Math.abs(piece) === 6) {
    if (piece > 0) { next.wKc = false; next.wQc = false; }
    else           { next.bKc = false; next.bQc = false; }
  }
  if (fr === 7 && fc === 7) next.wKc = false;
  if (fr === 7 && fc === 0) next.wQc = false;
  if (fr === 0 && fc === 7) next.bKc = false;
  if (fr === 0 && fc === 0) next.bQc = false;
  if (tr === 7 && tc === 7) next.wKc = false;
  if (tr === 7 && tc === 0) next.wQc = false;
  if (tr === 0 && tc === 7) next.bKc = false;
  if (tr === 0 && tc === 0) next.bQc = false;

  // Set en passant target if pawn double-pushed
  next.enPassantTarget = null;
  if (Math.abs(piece) === 1 && Math.abs(tr - fr) === 2) {
    next.enPassantTarget = [(fr + tr) / 2, tc];
  }

  // Halfmove clock: resets on a capture or any pawn move, else increments.
  if (mv.captured || mv.enPassant || Math.abs(piece) === 1) next.halfmoveClock = 0;
  else next.halfmoveClock = (st.halfmoveClock || 0) + 1;

  next.lastMove = mv;
  next.selected = null;
  return next;
}

// Compact key identifying a position for repetition detection. Per FIDE, two
// positions are the same only if the side to move, castling rights and en
// passant possibilities also match.
function positionKey(st, side) {
  let s = '';
  for (let r = 0; r < 8; r++) s += st.board[r].join(',') + '/';
  s += side[0];
  s += (st.wKc ? 'K' : '') + (st.wQc ? 'Q' : '') + (st.bKc ? 'k' : '') + (st.bQc ? 'q' : '');
  if (st.enPassantTarget) s += ':' + st.enPassantTarget[0] + st.enPassantTarget[1];
  return s;
}

// Draw by insufficient mating material: K v K, K+minor v K, and K+B v K+B with
// both bishops on the same colour squares.
function insufficientMaterial(board) {
  const minors = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p === 0) continue;
      const pt = Math.abs(p);
      if (pt === 6) continue;
      if (pt === 1 || pt === 4 || pt === 5) return false;  // pawn/rook/queen can mate
      minors.push({ pt, side: Math.sign(p), light: (r + c) % 2 === 0 });
    }
  }
  if (minors.length <= 1) return true;                      // K v K, K+minor v K
  if (minors.length === 2) {
    const [a, b] = minors;
    if (a.pt === 3 && b.pt === 3 && a.side !== b.side && a.light === b.light) return true;
  }
  return false;
}

function getLegalMoves(st, side) {
  return filterLegal(generateMoves(st, side), st, side);
}

// repCount = how many times the current position has already occurred in the
// game (1 = first occurrence). Pass it to enable threefold-repetition detection.
function getGameStatus(st, side, repCount = 1) {
  const legal = getLegalMoves(st, side);

  if (legal.length === 0) {
    if (isInCheck(st, side)) {
      return {over: true, result: side === 'white' ? 'black_wins' : 'white_wins', reason: 'checkmate'};
    }
    return {over: true, result: 'draw', reason: 'stalemate'};
  }

  if (insufficientMaterial(st.board)) {
    return {over: true, result: 'draw', reason: 'insufficient material'};
  }
  if ((st.halfmoveClock || 0) >= 100) {          // 100 plies = 50 full moves
    return {over: true, result: 'draw', reason: 'fifty-move rule'};
  }
  if (repCount >= 3) {
    return {over: true, result: 'draw', reason: 'threefold repetition'};
  }

  return {over: false, result: null, reason: null};
}

function pieceSquareValue(piece, r, c) {
  const pt = Math.abs(piece);
  const isWhite = piece > 0;
  const row = isWhite ? r : 7 - r; // flip for black
  return PST[pt] ? PST[pt][row][c] : 0;
}

function simpleEval(board) {
  let score = 0;
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p > 0) score += SIMPLE_VALUES[p];
      else if (p < 0) score -= SIMPLE_VALUES[-p];
    }
  return score;
}

function evaluatePosition(st) {
  const board = st.board;
  let val = 0;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p === 0) continue;
      const pt = Math.abs(p);
      const side = p > 0 ? 1 : -1;
      val -= side * PIECE_VALUES[pt];
      val -= side * 0.1 * pieceSquareValue(p, r, c);
    }
  }
  // Bishop pair bonus
  let wBishops = 0, bBishops = 0;
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
    if (board[r][c] === 3) wBishops++;
    if (board[r][c] === -3) bBishops++;
  }
  if (wBishops >= 2) val += 50;
  if (bBishops >= 2) val -= 50;

  val += rookFileScore(board);
  val += kingSafety(board);
  val += pawnStructure(board);
  val += passedPawns(board);
  return val;
}

function rookFileScore(board) {
  let score = 0;
  for (let c = 0; c < 8; c++) {
    let wPawn = false, bPawn = false;
    for (let r = 0; r < 8; r++) {
      if (board[r][c] === 1) wPawn = true;
      if (board[r][c] === -1) bPawn = true;
    }
    for (let r = 0; r < 8; r++) {
      if (board[r][c] === 4) {
        if (!wPawn && !bPawn) score += 30;
        else if (!wPawn) score += 15;
      }
      if (board[r][c] === -4) {
        if (!wPawn && !bPawn) score -= 30;
        else if (!bPawn) score -= 15;
      }
    }
  }
  return score;
}

function kingSafety(board) {
  let score = 0;
  const wK = findKing(board, 'white');
  const bK = findKing(board, 'black');
  if (wK) {
    let attackers = 0;
    for (const [dr, dc] of [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]) {
      const nr = wK[0]+dr, nc = wK[1]+dc;
      if (inBounds(nr,nc) && squareAttacked(board, nr, nc, 'black')) attackers++;
    }
    score += attackers * 10;
  }
  if (bK) {
    let attackers = 0;
    for (const [dr, dc] of [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]) {
      const nr = bK[0]+dr, nc = bK[1]+dc;
      if (inBounds(nr,nc) && squareAttacked(board, nr, nc, 'white')) attackers++;
    }
    score -= attackers * 10;
  }
  return score;
}

function pawnStructure(board) {
  // score positive = good for Black
  const wFile = new Int8Array(8), bFile = new Int8Array(8);
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) {
      if (board[r][c] === 1)  wFile[c]++;
      if (board[r][c] === -1) bFile[c]++;
    }

  let score = 0;
  for (let c = 0; c < 8; c++) {
    // Doubled pawn penalty
    if (wFile[c] > 1) score += (wFile[c] - 1) * 20;
    if (bFile[c] > 1) score -= (bFile[c] - 1) * 20;
    // Isolated pawn penalty (no friendly pawns on adjacent files)
    const wAdj = (c > 0 && wFile[c-1]) || (c < 7 && wFile[c+1]);
    const bAdj = (c > 0 && bFile[c-1]) || (c < 7 && bFile[c+1]);
    if (wFile[c] && !wAdj) score += 25; // isolated white pawn = bad for white = good for black
    if (bFile[c] && !bAdj) score -= 25;
  }
  return score;
}

function passedPawns(board) {
  // Passed pawn: no enemy pawns on same or adjacent files ahead of it
  // score positive = good for Black
  let score = 0;
  for (let c = 0; c < 8; c++) {
    for (let r = 1; r < 7; r++) {
      if (board[r][c] === 1) { // White pawn; it advances toward row 0
        let passed = true;
        outer: for (let rr = 0; rr < r; rr++)
          for (let cc = Math.max(0,c-1); cc <= Math.min(7,c+1); cc++)
            if (board[rr][cc] === -1) { passed = false; break outer; }
        if (passed) {
          const advance = 7 - r; // rows from starting rank (1=rank3 to 5=rank7)
          score -= advance * 18; // white passed pawn = good for white = bad for black
        }
      } else if (board[r][c] === -1) { // Black pawn; advances toward row 7
        let passed = true;
        outer: for (let rr = r + 1; rr < 8; rr++)
          for (let cc = Math.max(0,c-1); cc <= Math.min(7,c+1); cc++)
            if (board[rr][cc] === 1) { passed = false; break outer; }
        if (passed) {
          const advance = r;
          score += advance * 18;
        }
      }
    }
  }
  return score;
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

function opposite(side) {
  return side === 'white' ? 'black' : 'white';
}

export {
  initState, cloneState, generateMoves, makeMove, getLegalMoves,
  getGameStatus, isInCheck, squareAttacked, findKing, filterLegal,
  evaluatePosition, simpleEval, boardToVector, opposite, inBounds,
  pieceSquareValue, positionKey, insufficientMaterial,
  PIECE_GLYPHS, PIECE_VALUES, SIMPLE_VALUES
};
