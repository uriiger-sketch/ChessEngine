// Offline support.
//
// Code and pages are fetched NETWORK-FIRST: whenever the phone is online it
// gets the current files, and the cache is only the fallback for offline play.
// The previous version served everything cache-first, which kept stale code in
// use after an update — and because a page and its workers load at different
// moments, it could hand out a page from one release and a worker from
// another. When the messages between them changed in v2.3, that left Help mode
// silently waiting forever.
//
// The model is the exception: it is large and changes only with a release, so
// it stays cache-first under a cache name tied to the version.
//
// All paths are relative to this worker's scope, so the app works the same at
// a site root or in a subfolder (e.g. a GitHub Pages project site).

const VERSION = '2.4.0';          // keep in step with js/version.js
const CACHE = 'chessnn-' + VERSION;

const CODE = [
  './',
  'index.html',
  'manifest.json',
  'css/style.css',
  'js/version.js',
  'js/chess.js',
  'js/position.js',
  'js/evaluate.js',
  'js/engine.js',
  'js/neural.js',
  'js/ui.js',
  'js/worker.js',
  'icons/icon-192.png',
  'icons/icon-512.png',
];
const MODEL = [
  'model/model.json',
  'model/weights.bin',
  'model/normalization.json',
];

const abs = (p) => new URL(p, self.registration.scope).href;
const isModel = (url) => MODEL.some(p => url.split('?')[0] === abs(p));

self.addEventListener('install', (e) => {
  e.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    // cache: 'reload' goes to the server, not the browser's HTTP cache.
    // Without it a "fresh" install could store copies the browser had kept
    // from the previous release, mixing versions inside one cache.
    const results = await Promise.allSettled([...CODE, ...MODEL].map(async (p) => {
      const res = await fetch(new Request(abs(p), { cache: 'reload' }));
      if (!res.ok) throw new Error(`${p} → HTTP ${res.status}`);
      await cache.put(abs(p), res);
    }));
    const failed = results.filter(r => r.status === 'rejected').map(r => r.reason.message);
    if (failed.length) console.warn('[sw] not cached:', failed.join(', '));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET' || !req.url.startsWith(self.location.origin)) return;

  if (isModel(req.url)) {
    e.respondWith((async () => {
      const cached = await caches.match(req, { ignoreSearch: true });
      if (cached) return cached;
      const res = await fetch(req);
      if (res.ok) (await caches.open(CACHE)).put(req, res.clone());
      return res;
    })());
    return;
  }

  e.respondWith((async () => {
    try {
      // 'no-cache' revalidates with the server rather than trusting the HTTP
      // cache, so an update is picked up immediately. A navigation request
      // cannot be copied with new options, so it is rebuilt from its URL.
      const fresh = req.mode === 'navigate'
        ? new Request(req.url, { cache: 'no-cache', credentials: 'same-origin' })
        : new Request(req, { cache: 'no-cache' });
      const res = await fetch(fresh);
      if (res.ok) {
        const copy = res.clone();
        caches.open(CACHE).then(c => c.put(req.url.split('?')[0], copy));
      }
      return res;
    } catch (_) {
      // Offline: serve the cached copy. Workers are loaded with a version
      // query, so match on the path alone.
      const cached = await caches.match(req, { ignoreSearch: true }) ||
                     (req.mode === 'navigate' ? await caches.match(abs('index.html')) : null);
      if (cached) return cached;
      throw _;
    }
  })());
});
