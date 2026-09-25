'use strict';
// How the app behaves across a release, on a browser that keeps its offline
// cache between launches — which is what a phone does.
//
// A broken update path is what took Help mode down in v2.3: the offline cache
// served a page from one release and a worker from another, and they could not
// understand each other. This test builds a "next release" (the working tree
// with the version bumped) in a temporary folder and checks that:
//   • the next launch after a release runs it, when no game is under way;
//   • a release landing mid-game waits, and the game is untouched;
//   • it is then applied at the next New Game, keeping the chosen colour;
//   • Help mode works at every step.
// The server sends max-age=600 like GitHub Pages, which is part of what made
// stale files stick.
//
// Usage: node test/update.js

import { createRequire } from 'module';
import http from 'http';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');

const HERE = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const PORT = 8243;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.png': 'image/png', '.bin': 'application/octet-stream' };

const CUR = fs.readFileSync(path.join(HERE, 'js/version.js'), 'utf8').match(/APP_VERSION = '([^']+)'/)[1];
const NEXT = CUR.replace(/(\d+)$/, n => String(+n + 1));
const short = v => 'v' + v.replace(/\.0$/, '');

// Build the next release: identical app, bumped version in page and worker.
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'chessnn-next-'));
for (const f of ['index.html', 'manifest.json', 'sw.js', 'css', 'js', 'icons', 'model']) {
  fs.cpSync(path.join(HERE, f), path.join(tmp, f), { recursive: true });
}
for (const [f, re] of [['js/version.js', /APP_VERSION = '[^']+'/], ['sw.js', /const VERSION = '[^']+'/]]) {
  const p = path.join(tmp, f);
  fs.writeFileSync(p, fs.readFileSync(p, 'utf8').replace(re, m => m.replace(/'[^']+'/, `'${NEXT}'`)));
}

let root = HERE;
const server = http.createServer((req, res) => {
  let p = decodeURIComponent(req.url.split('?')[0]);
  if (p === '/') p = '/index.html';
  const f = path.join(root, p);
  if (!f.startsWith(root) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'Content-Type': TYPES[path.extname(f)] || 'application/octet-stream', 'Cache-Control': 'max-age=600' });
  fs.createReadStream(f).pipe(res);
}).listen(PORT);

let failures = 0;
const check = (name, ok, detail) => {
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};

async function helpWorks(p) {
  await p.click('.mode-btn[data-mode="help"]');
  for (let i = 0; i < 70; i++) {
    await sleep(150);
    if ((await p.$$('#hint-panel .hint-chip')).length === 3) return true;
  }
  return false;
}

(async () => {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'chessnn-profile-'));
  const ctx = await chromium.launchPersistentContext(profile, {
    executablePath: process.env.CHROMIUM || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    viewport: { width: 390, height: 844 },
  });
  const url = `http://localhost:${PORT}/index.html`;
  const version = p => p.textContent('#version').then(t => t.trim());

  // Install the current release and let its service worker take over.
  let p = await ctx.newPage();
  await p.goto(url, { waitUntil: 'networkidle' }); await sleep(2500);
  await p.close();
  p = await ctx.newPage();
  await p.goto(url, { waitUntil: 'networkidle' }); await sleep(1500);
  check(`${short(CUR)} installed and running`, (await version(p)) === short(CUR), await version(p));
  check('Help works', await helpWorks(p));
  await p.close();

  // A release lands; the next launch should run it without being asked.
  root = tmp;
  p = await ctx.newPage();
  await p.goto(url, { waitUntil: 'networkidle' }); await sleep(4000);
  check(`next launch runs ${short(NEXT)}`, (await version(p)) === short(NEXT), await version(p));
  check('Help works after the update', await helpWorks(p));
  await p.close();

  // Back to the current release, then a release arriving mid-game.
  root = HERE;
  p = await ctx.newPage();
  await p.goto(url, { waitUntil: 'networkidle' }); await sleep(4000);
  await p.click('.mode-btn[data-mode="play"]');
  await p.click('[data-time="2000"]');
  const play = async () => {
    await p.evaluate(() => {
      for (const sq of document.querySelectorAll('#board .sq')) {
        sq.dispatchEvent(new MouseEvent('click', { bubbles: true }));
        const h = document.querySelectorAll('#board .legal');
        if (h.length) { h[0].dispatchEvent(new MouseEvent('click', { bubbles: true })); return; }
      }
    });
    for (let i = 0; i < 60; i++) { await sleep(200); if (!(await p.textContent('#status')).includes('thinking')) break; }
  };
  await play(); await play();
  const board = () => p.evaluate(() => [...document.querySelectorAll('#board .piece-span')].map(s => s.textContent).join(''));
  const before = await board();
  const vBefore = await version(p);

  root = tmp;
  await p.evaluate(() => navigator.serviceWorker.getRegistration().then(r => r && r.update()));
  await sleep(4000);
  check('a release mid-game does not interrupt it',
        (await version(p)) === vBefore && (await board()) === before, `${await version(p)}, board unchanged`);

  await p.click('#play-black-btn'); await sleep(300);
  await p.click('#confirm-yes');
  await p.waitForLoadState('networkidle'); await sleep(3000);
  check('it is applied at the next New Game', (await version(p)) === short(NEXT), await version(p));
  check('the chosen colour survives the update',
        await p.evaluate(() => document.querySelector('#board .sq').dataset.r === '7'));
  check('Help works for Black after the update', await helpWorks(p));

  await ctx.close();
  server.close();
  fs.rmSync(tmp, { recursive: true, force: true });
  fs.rmSync(profile, { recursive: true, force: true });
  console.log('\n' + '='.repeat(60));
  console.log(failures === 0 ? 'All update checks passed.' : `${failures} update check(s) FAILED.`);
  process.exit(failures === 0 ? 0 : 1);
})().catch(e => { console.error(e); process.exit(1); });
