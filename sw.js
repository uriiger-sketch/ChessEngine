// Bump on every release — a stale cache would keep serving the old JS/model.
const CACHE = 'chess-v6';

// Everything needed to play a full game with no network at all.
const ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/css/style.css',
  '/js/chess.js',
  '/js/position.js',
  '/js/evaluate.js',
  '/js/engine.js',
  '/js/neural.js',
  '/js/ui.js',
  '/js/worker.js',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
];

// The model is required for NN play offline. Inference is pure JS (js/neural.js),
// so there is no cross-origin script to cache — everything needed is same-origin.
const MODEL_ASSETS = [
  '/model/model.json',
  '/model/weights.bin',
  '/model/normalization.json',
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then(async cache => {
      // Cached one at a time rather than with addAll: addAll rejects the whole
      // install if any single entry fails, which would leave the app with no
      // service worker at all rather than a nearly complete cache.
      const results = await Promise.allSettled(
        [...ASSETS, ...MODEL_ASSETS].map(url => cache.add(url))
      );
      const failed = results
        .map((r, i) => (r.status === 'rejected' ? [...ASSETS, ...MODEL_ASSETS][i] : null))
        .filter(Boolean);
      if (failed.length) console.warn('[sw] not cached:', failed.join(', '));
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  // Cache-first for same-origin. Everything the app needs is same-origin.
  if (!e.request.url.startsWith(self.location.origin)) return;

  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(response => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE).then(cache => cache.put(e.request, clone));
        }
        return response;
      });
    })
  );
});
