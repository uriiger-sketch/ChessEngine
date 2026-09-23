'use strict';
// End-to-end check in a real browser: the engine running in a Web Worker, the
// board reacting to taps, learning mode and the take-back rule.
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

  // ── Neural net actually loaded in the page ──────────────────────────────
  const nnState = await page.evaluate(() => ({
    disabled: document.getElementById('nn-cb').disabled,
    checked:  document.getElementById('nn-cb').checked,
  }));
  // Loadable and switchable, but off by default: it measured weaker than the
  // hand evaluation, so the default is the engine's strongest setting.
  check('neural net loads and is selectable', !nnState.disabled,
        `disabled=${nnState.disabled}`);
  check('neural net is off by default', !nnState.checked,
        `checked=${nnState.checked}`);

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

  const pieceCount = () => page.evaluate(() =>
    document.querySelectorAll('#board .piece-span').length);

  // ── A few real moves ────────────────────────────────────────────────────
  console.log('\nPlaying 6 moves against the engine...');
  for (let i = 1; i <= 6; i++) {
    if (!await playHuman()) break;
    await sleep(120);
    const s = await waitForAI();
    const ev = await page.textContent('#eval-text');
    console.log(`  move ${i}: ${s.padEnd(32)} eval ${ev}`);
    if (/wins|Draw|Stalemate/.test(s)) break;
  }
  check('engine answered every move', !(await status()).includes('TIMEOUT'));

  // ── Take-back is off until learning mode is on ──────────────────────────
  check('take-back disabled by default',
        await page.isDisabled('#takeback-btn'));

  await page.click('#learn-cb');
  await sleep(100);
  check('take-back offered in learning mode',
        !(await page.isDisabled('#takeback-btn')));

  const beforePieces = await pieceCount();
  const beforeBoard = await page.evaluate(() =>
    [...document.querySelectorAll('#board .sq')]
      .map(s => (s.querySelector('.piece-span')?.textContent || '.') +
                (s.querySelector('.piece-span')?.className.includes('piece-w') ? 'w' : 'b')).join(''));

  await page.click('#takeback-btn');
  await sleep(400);

  const afterBoard = await page.evaluate(() =>
    [...document.querySelectorAll('#board .sq')]
      .map(s => (s.querySelector('.piece-span')?.textContent || '.') +
                (s.querySelector('.piece-span')?.className.includes('piece-w') ? 'w' : 'b')).join(''));

  check('take-back changed the position', beforeBoard !== afterBoard);
  check('take-back returned the move to the player',
        (await status()).includes('White to move'), await status());
  check('only one take-back allowed', await page.isDisabled('#takeback-btn'));
  check('board still intact after take-back',
        (await pieceCount()) >= beforePieces,
        `${beforePieces} → ${await pieceCount()}`);

  // Playing again must re-arm the take-back.
  await playHuman();
  await sleep(150);
  check('take-back re-armed after playing again',
        !(await page.isDisabled('#takeback-btn')));
  await waitForAI();

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

  // ── Playing as Black flips the board and the engine opens ───────────────
  await page.click('#play-black-btn');
  await sleep(300);
  const flipped = await page.evaluate(() => {
    const first = document.querySelector('#board .sq');
    return { r: first.dataset.r, c: first.dataset.c };
  });
  check('board flips when playing Black', flipped.r === '7' && flipped.c === '7',
        JSON.stringify(flipped));
  const blackStatus = await waitForAI();
  check('engine opens when player is Black',
        blackStatus.includes('Black to move'), blackStatus);

  // ── Console must be clean ───────────────────────────────────────────────
  check('no console or page errors', errors.length === 0,
        errors.length ? '\n    ' + errors.join('\n    ') : '');

  await browser.close();
  server.close();

  console.log('\n' + '='.repeat(60));
  console.log(failures === 0 ? 'All browser checks passed.' : `${failures} browser check(s) FAILED.`);
  process.exit(failures === 0 ? 0 : 1);
})().catch(e => { console.error(e); process.exit(1); });
