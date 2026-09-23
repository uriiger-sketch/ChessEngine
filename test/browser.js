'use strict';
// End-to-end check in a real browser: the engine running in a Web Worker, the
// board reacting to taps, the three modes, hints, Back and confirmation.
//
// Usage: node test/browser.js   (serves the project root on a spare port)

import { createRequire } from 'module';
import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Playwright is installed globally in this environment, not in the project.
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const PORT = 8231;
const sleep = ms => new Promise(r => setTimeout(r, ms));

const TYPES = {
  '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.png': 'image/png', '.bin': 'application/octet-stream',
};

function serve() {
  return http.createServer((req, res) => {
    let p = decodeURIComponent(req.url.split('?')[0]);
    if (p === '/') p = '/index.html';
    const file = path.join(ROOT, p);
    if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404); res.end('not found'); return;
    }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
  }).listen(PORT);
}

let failures = 0;
function check(name, ok, detail) {
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
}

(async () => {
  const server = serve();
  const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
  const page = await browser.newPage();

  const errors = [];
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });

  await page.goto(`http://localhost:${PORT}/index.html`, { waitUntil: 'networkidle' });
  await sleep(2500);

  // ── Version and credit ──────────────────────────────────────────────────
  const version = (await page.textContent('#version')).trim();
  const credit  = (await page.textContent('#credit')).trim();
  check('version is shown', /^v\d/.test(version), version);
  check('credit is shown', /igeru/i.test(credit), credit);

  // The network is always on — there is no switch for it any more.
  check('no neural-net toggle', (await page.$('#nn-cb')) === null);

  await page.click('[data-time="2000"]');

  const status = () => page.textContent('#status');
  const waitForAI = async () => {
    for (let i = 0; i < 150; i++) {
      await sleep(200);
      const s = await status();
      if (!s.includes('thinking')) return s;
    }
    return 'TIMEOUT';
  };

  // Play the human side by taking the first square that offers a legal move.
  const playHuman = () => page.evaluate(() => {
    for (const sq of document.querySelectorAll('#board .sq')) {
      sq.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      const hints = document.querySelectorAll('#board .legal, #board .legal-cap');
      if (hints.length > 0) {
        hints[Math.floor(Math.random() * hints.length)]
          .dispatchEvent(new MouseEvent('click', { bubbles: true }));
        return true;
      }
    }
    return false;
  });

  const boardSig = () => page.evaluate(() =>
    [...document.querySelectorAll('#board .sq')]
      .map(s => (s.querySelector('.piece-span')?.textContent || '.') +
                (s.querySelector('.piece-span')?.className.includes('piece-w') ? 'w' : 'b')).join(''));
  const pieceCount = () => page.evaluate(() =>
    document.querySelectorAll('#board .piece-span').length);
  const mode = () => page.getAttribute('#mode-switch', 'data-mode');
  const arrows = () => page.evaluate(() => document.querySelectorAll('#hint-layer .hint-arrow').length);
  const chips = () => page.evaluate(() =>
    [...document.querySelectorAll('#hint-panel .hint-chip')].map(c => c.textContent.trim()));
  const waitForHints = async () => {
    for (let i = 0; i < 60; i++) {
      await sleep(150);
      if ((await chips()).length > 0) return true;
    }
    return false;
  };

  // ── Play mode by default ────────────────────────────────────────────────
  check('starts in Play mode', (await mode()) === 'play', await mode());
  check('Back disabled in Play mode', await page.isDisabled('#back-btn'));

  console.log('\nPlaying 3 moves in Play mode...');
  for (let i = 1; i <= 3; i++) {
    await playHuman(); await sleep(120);
    console.log(`  move ${i}: ${(await waitForAI()).padEnd(28)} eval ${await page.textContent('#eval-text')}`);
  }
  check('engine answered every move', !(await status()).includes('TIMEOUT'));
  check('no hints outside Help mode', (await arrows()) === 0 && (await chips()).length === 0);

  // ── New Game asks first ─────────────────────────────────────────────────
  const midGame = await boardSig();
  await page.click('#new-game-btn');
  await sleep(250);
  check('New Game asks for confirmation', !(await page.isHidden('#confirm-modal')));
  await page.click('#confirm-no');
  await sleep(250);
  check('"No" closes the dialog', await page.isHidden('#confirm-modal'));
  check('"No" keeps the game', (await boardSig()) === midGame);

  await page.click('#new-game-btn');
  await sleep(250);
  await page.click('#confirm-yes');
  await sleep(400);
  check('"Yes" starts a new game',
        (await pieceCount()) === 32 && (await status()).includes('White to move'), await status());

  // At the starting position there is nothing to lose, so no question.
  await page.click('#new-game-btn');
  await sleep(250);
  check('no confirmation before any move', await page.isHidden('#confirm-modal'));

  // ── Help mode ───────────────────────────────────────────────────────────
  await page.click('.mode-btn[data-mode="help"]');
  check('switches to Help mode', (await mode()) === 'help');
  check('hints appear on the player\'s turn', await waitForHints());
  await sleep(600);    // let the arrows finish fading in
  const c1 = await chips();
  console.log('  hints: ' + c1.join('   '));
  check('three suggestions listed', c1.length === 3, String(c1.length));
  check('three arrows drawn', (await arrows()) === 3, String(await arrows()));

  // Tapping suggestion 1 picks up its piece, and its arrow's square is a legal target.
  const target = await page.evaluate(() => {
    const groups = document.querySelectorAll('#hint-layer .hint-arrow');
    const best = groups[groups.length - 1];           // drawn last = on top = #1
    const rect = best.querySelector('rect');
    return { vr: Math.floor(+rect.getAttribute('y') / 100), vc: Math.floor(+rect.getAttribute('x') / 100) };
  });
  await page.click('#hint-panel .hint-chip.r1');
  await sleep(200);
  const legalAtTarget = await page.evaluate(({ vr, vc }) => {
    const sq = document.querySelectorAll('#board .sq')[vr * 8 + vc];
    return !!document.querySelector('#board .sq.selected') &&
           (sq.classList.contains('legal') || sq.classList.contains('legal-cap'));
  }, target);
  check('suggestion 1 is a legal move for the picked-up piece', legalAtTarget);

  // Play the best suggestion; hints must clear while the engine thinks.
  await page.evaluate(({ vr, vc }) =>
    document.querySelectorAll('#board .sq')[vr * 8 + vc]
      .dispatchEvent(new MouseEvent('click', { bubbles: true })), target);
  await sleep(150);
  check('hints cleared after moving', (await arrows()) === 0);
  await waitForAI();
  check('fresh hints after the engine replies', await waitForHints());
  check('Back available in Help mode', !(await page.isDisabled('#back-btn')));

  // ── Back returns to the start of the player's last turn ─────────────────
  const beforeBack = await boardSig();
  await page.click('#back-btn');
  await sleep(300);
  check('Back changes the position', (await boardSig()) !== beforeBack);
  check('Back hands the turn to the player', /White to move/.test(await status()), await status());
  check('Back is once per move', await page.isDisabled('#back-btn'));
  check('hints return after Back', await waitForHints());

  // ── Learn mode ──────────────────────────────────────────────────────────
  await page.click('.mode-btn[data-mode="learn"]');
  await sleep(350);
  check('switches to Learn mode', (await mode()) === 'learn');
  check('hints removed in Learn mode', (await arrows()) === 0 && (await chips()).length === 0);
  await playHuman(); await sleep(120);
  await waitForAI();
  check('Back re-armed after playing again', !(await page.isDisabled('#back-btn')));
  const beforeBack2 = await boardSig();
  await page.click('#back-btn');
  await sleep(300);
  check('Back works in Learn mode', (await boardSig()) !== beforeBack2);

  // ── Play mode turns Back off; the mode survives a reload ────────────────
  await page.click('.mode-btn[data-mode="play"]');
  await sleep(200);
  check('Back disabled after returning to Play', await page.isDisabled('#back-btn'));
  await page.click('.mode-btn[data-mode="help"]');
  await page.reload({ waitUntil: 'networkidle' });
  await sleep(1500);
  check('mode remembered across reload', (await mode()) === 'help', await mode());
  check('hints shown after reload', await waitForHints());

  // ── Board integrity ─────────────────────────────────────────────────────
  const integrity = await page.evaluate(() => {
    const sqs = document.querySelectorAll('#board .sq');
    let wk = 0, bk = 0;
    sqs.forEach(s => {
      const p = s.querySelector('.piece-span');
      if (p && p.textContent === '♔') p.classList.contains('piece-w') ? wk++ : bk++;
    });
    return { squares: sqs.length, whiteKings: wk, blackKings: bk };
  });
  check('64 squares rendered', integrity.squares === 64, String(integrity.squares));
  check('exactly one king per side',
        integrity.whiteKings === 1 && integrity.blackKings === 1,
        JSON.stringify(integrity));

  // ── Switching colour mid-game confirms, flips, and the engine opens ─────
  await playHuman(); await sleep(120); await waitForAI();
  await page.click('#play-black-btn');
  await sleep(250);
  const title = await page.textContent('#confirm-title');
  check('changing colour mid-game asks first', /Black/.test(title), title);
  await page.click('#confirm-yes');
  await sleep(300);
  const flipped = await page.evaluate(() => {
    const first = document.querySelector('#board .sq');
    return { r: first.dataset.r, c: first.dataset.c };
  });
  check('board flips when playing Black', flipped.r === '7' && flipped.c === '7',
        JSON.stringify(flipped));
  const blackStatus = await waitForAI();
  check('engine opens when player is Black', blackStatus.includes('Black to move'), blackStatus);
  check('hints for Black after the engine opens', await waitForHints());

  // ── Console must be clean ───────────────────────────────────────────────
  check('no console or page errors', errors.length === 0,
        errors.length ? '\n    ' + errors.join('\n    ') : '');

  await browser.close();
  server.close();

  console.log('\n' + '='.repeat(60));
  console.log(failures === 0 ? 'All browser checks passed.' : `${failures} browser check(s) FAILED.`);
  process.exit(failures === 0 ? 0 : 1);
})().catch(e => { console.error(e); process.exit(1); });
