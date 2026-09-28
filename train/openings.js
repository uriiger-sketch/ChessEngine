'use strict';
// Main lines from opening literature, first 9 moves at most.
//
// These are added to the opening book alongside what the masters in the PGN
// files actually played, so the book covers the standard systems even where
// the seven players' games are thin. Names are "Family, Variation"; positions
// shared by several lines of one family are labelled with the family alone.
//
// Every line is replayed move by move when the book is built (train/book.js),
// so a move that is illegal in its position stops the build instead of
// silently corrupting the book.

export const OPENINGS = [
  // ── Open games: 1.e4 e5 ─────────────────────────────────────────────────
  ['Ruy Lopez, Closed',              'e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3'],
  ['Ruy Lopez, Berlin Defence',      'e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 d4 Nd6 Bxc6 dxc6 dxe5 Nf5 Qxd8+ Kxd8'],
  ['Ruy Lopez, Exchange Variation',  'e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O f6 d4 exd4 Nxd4 c5 Nb3 Qxd1 Rxd1'],
  ['Ruy Lopez, Marshall Attack',     'e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O c3 d5 exd5 Nxd5'],
  ['Italian Game, Giuoco Piano',     'e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O Re1 a6 Bb3 Ba7 h3'],
  ['Italian Game, Two Knights',      'e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Na5 Bb5+ c6 dxc6 bxc6 Be2 h6 Nf3 e4'],
  ['Italian Game, Evans Gambit',     'e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Ba5 d4 exd4 O-O Nge7'],
  ['Scotch Game',                    'e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Ba6'],
  ['Four Knights Game',              'e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Bb4 O-O O-O d3 d6 Bg5 Bxc3 bxc3'],
  ['Petrov Defence',                 'e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Nc6 O-O Be7 c4 Nb4'],
  ['Philidor Defence',               'e4 e5 Nf3 d6 d4 Nf6 Nc3 Nbd7 Bc4 Be7 O-O O-O Re1 c6 a4'],
  ['Vienna Game',                    'e4 e5 Nc3 Nf6 f4 d5 fxe5 Nxe4 Nf3 Be7 d4 O-O Bd3 f5'],
  ["King's Gambit Accepted",         'e4 e5 f4 exf4 Nf3 g5 h4 g4 Ne5 Nf6 Bc4 d5 exd5 Bd6'],

  // ── Sicilian Defence: 1.e4 c5 ───────────────────────────────────────────
  ['Sicilian Defence, Najdorf',      'e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7 Qd2'],
  ['Sicilian Defence, Dragon',       'e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 O-O Qd2 Nc6 Bc4'],
  ['Sicilian Defence, Classical',    'e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 e6 Qd2 Be7 O-O-O O-O'],
  ['Sicilian Defence, Scheveningen', 'e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e6 Be2 Be7 O-O O-O f4 Nc6 Be3'],
  ['Sicilian Defence, Sveshnikov',   'e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6 Na3 b5 Bxf6'],
  ['Sicilian Defence, Taimanov',     'e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 Be2 a6 O-O Nf6 Be3 Bb4'],
  ['Sicilian Defence, Kan',          'e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Bd3 Nf6 O-O Qc7 Qe2 d6 c4 g6'],
  ['Sicilian Defence, Rossolimo',    'e4 c5 Nf3 Nc6 Bb5 g6 O-O Bg7 Re1 e5 Bxc6 dxc6 d3 Qe7 a4'],
  ['Sicilian Defence, Alapin',       'e4 c5 c3 Nf6 e5 Nd5 d4 cxd4 Nf3 Nc6 cxd4 d6 Bc4 Nb6 Bb5 dxe5'],
  ['Sicilian Defence, Closed',       'e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6 Be3 e5 Qd2 Nge7'],

  // ── Other replies to 1.e4 ───────────────────────────────────────────────
  ['French Defence, Winawer',        'e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3 Ne7 Qg4 O-O Bd3 Nbc6'],
  ['French Defence, Classical',      'e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e5 Nfd7 Bxe7 Qxe7 f4 O-O Nf3 c5'],
  ['French Defence, Tarrasch',       'e4 e6 d4 d5 Nd2 c5 exd5 Qxd5 Ngf3 cxd4 Bc4 Qd6 O-O Nf6 Nb3 Nc6'],
  ['French Defence, Advance',        'e4 e6 d4 d5 e5 c5 c3 Nc6 Nf3 Qb6 a3 c4 Nbd2 Na5 Rb1 Bd7'],
  ['Caro-Kann Defence, Classical',   'e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3'],
  ['Caro-Kann Defence, Advance',     'e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 c5 Be3 Nd7 O-O Ne7'],
  ['Caro-Kann Defence, Panov',       'e4 c6 d4 d5 exd5 cxd5 c4 Nf6 Nc3 e6 Nf3 Bb4 cxd5 Nxd5 Bd2 Nc6'],
  ['Scandinavian Defence',           'e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5 Bd2 e6'],
  ['Pirc Defence',                   'e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O O-O c6 a4'],
  ['Modern Defence',                 'e4 g6 d4 Bg7 Nc3 d6 Be3 a6 Qd2 Nd7 f4 b5'],
  ['Alekhine Defence',               'e4 Nf6 e5 Nd5 d4 d6 Nf3 Bg4 Be2 e6 O-O Be7 c4 Nb6'],

  // ── Queen's pawn: 1.d4 d5 ───────────────────────────────────────────────
  ["Queen's Gambit Declined",        'd4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 b6 cxd5 Nxd5'],
  ["Queen's Gambit Declined, Exchange", 'd4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 c6 e3 Be7 Bd3 Nbd7 Qc2 O-O'],
  ["Queen's Gambit Accepted",        'd4 d5 c4 dxc4 Nf3 Nf6 e3 e6 Bxc4 c5 O-O a6 dxc5 Bxc5'],
  ['Slav Defence',                   'd4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4 O-O O-O'],
  ['Semi-Slav Defence, Meran',       'd4 d5 c4 c6 Nf3 Nf6 Nc3 e6 e3 Nbd7 Bd3 dxc4 Bxc4 b5 Bd3 Bb7'],
  ['London System',                  'd4 d5 Nf3 Nf6 Bf4 e6 e3 c5 c3 Nc6 Nbd2 Bd6 Bg3 O-O Bd3'],

  // ── Indian defences: 1.d4 Nf6 ───────────────────────────────────────────
  ["King's Indian Defence, Classical", 'd4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Ne1'],
  ['Grünfeld Defence, Exchange',     'd4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 c5 Ne2 Nc6 Be3'],
  ['Nimzo-Indian Defence, Rubinstein', 'd4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O Nc6 a3 Bxc3 bxc3'],
  ["Queen's Indian Defence",         'd4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4+ Bd2 Be7 Bg2 c6 Bc3 d5'],
  ['Bogo-Indian Defence',            'd4 Nf6 c4 e6 Nf3 Bb4+ Bd2 Qe7 g3 Nc6 Nc3 Bxc3 Bxc3 Ne4'],
  ['Catalan Opening',                'd4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O dxc4 Qc2 a6 Qxc4 b5 Qc2 Bb7'],
  ['Benoni Defence, Modern',         'd4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 e4 g6 Nf3 Bg7 Be2 O-O O-O'],
  ['Trompowsky Attack',              'd4 Nf6 Bg5 Ne4 Bf4 c5 f3 Qa5+ c3 Nf6 Nd2 cxd4 Nb3 Qb6'],
  ['Dutch Defence, Leningrad',       'd4 f5 g3 Nf6 Bg2 g6 Nf3 Bg7 O-O O-O c4 d6 Nc3 Qe8'],

  // ── Flank openings ──────────────────────────────────────────────────────
  ['English Opening, Symmetrical',   'c4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 Nf3 e6 O-O Nge7'],
  ['English Opening, Reversed Sicilian', 'c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Be7'],
  ['Réti Opening',                   'Nf3 d5 g3 Nf6 Bg2 e6 O-O Be7 d3 O-O Nbd2 c5 e4 Nc6'],
  ["Bird's Opening",                 'f4 d5 Nf3 Nf6 e3 g6 Be2 Bg7 O-O O-O d3 c5'],
];
