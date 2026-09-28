'use strict';
// Combine rate-network.js runs into one rating: the engine thinking time the
// network is worth.
//
// For each engine time, the network's score becomes an Elo difference. Engine
// strength grows roughly linearly in the logarithm of its thinking time, so a
// straight line is fitted through Elo against log2(time), weighted by games,
// and the time where it crosses zero is where the two play evenly.
//
// Usage: node test/rate-summary.js file1.txt file2.txt …

import fs from 'fs';

const rows = process.argv.slice(2).map(f => {
  const line = fs.readFileSync(f, 'utf8').split('\n').find(l => l.startsWith('SUMMARY '));
  return line ? JSON.parse(line.slice(8)) : null;
}).filter(Boolean).sort((a, b) => a.engineMs - b.engineMs);

// A 0% or 100% score has no finite Elo; nudge it half a game in from the edge.
const eloOf = (score, n) => {
  const p = Math.min(1 - 0.5 / n, Math.max(0.5 / n, score));
  return -400 * Math.log10(1 / p - 1);
};

console.log('engine time | games | network W  D  L | network score | network Elo | engine depth');
console.log('-'.repeat(86));
let total = 0;
const pts = rows.map(r => {
  total += r.games;
  const elo = eloOf(r.networkScore, r.games);
  console.log(`${String(r.engineMs).padStart(8)} ms | ${String(r.games).padStart(5)} | ` +
              `${String(r.networkWins).padStart(9)} ${String(r.draws).padStart(2)} ${String(r.engineWins).padStart(2)} | ` +
              `${(100 * r.networkScore).toFixed(1).padStart(12)}% | ${(elo >= 0 ? '+' : '') + elo.toFixed(0)}`.padEnd(70) +
              ` | ${r.engineDepth}`);
  return { x: Math.log2(r.engineMs), y: elo, w: r.games };
});

// Weighted least squares: y = a + b·x.
const W = pts.reduce((t, p) => t + p.w, 0);
const mx = pts.reduce((t, p) => t + p.w * p.x, 0) / W;
const my = pts.reduce((t, p) => t + p.w * p.y, 0) / W;
const b = pts.reduce((t, p) => t + p.w * (p.x - mx) * (p.y - my), 0) /
          pts.reduce((t, p) => t + p.w * (p.x - mx) ** 2, 0);
const a = my - b * mx;
const evenMs = Math.pow(2, -a / b);

const netMs = rows.reduce((t, r) => t + r.networkMsPerMove * r.games, 0) / total;
const netDepth = rows.reduce((t, r) => t + r.networkDepth * r.games, 0) / total;

console.log('-'.repeat(86));
console.log(`${total} games.  Each doubling of engine time costs the network ${(-b).toFixed(0)} Elo.`);
console.log(`The network (book + ${rows[0].look} moves ahead) plays evenly with the engine at about ` +
            `${evenMs.toFixed(0)} ms per move.`);
console.log(`It spends ${netMs.toFixed(0)} ms per move itself (searching ${netDepth.toFixed(1)} half-moves deep), ` +
            `so it is worth about ${(100 * evenMs / netMs).toFixed(0)}% of its own thinking time in engine terms.`);
console.log('RATING ' + JSON.stringify({ evenMs: +evenMs.toFixed(1), eloPerDoubling: +(-b).toFixed(1),
                                          networkMsPerMove: +netMs.toFixed(0), games: total }));
