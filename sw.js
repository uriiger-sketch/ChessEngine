const CACHE = 'chess-v1';
const ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/css/style.css',
  '/js/chess.js',
  '/js/engine.js',
  '/js/neural.js',
  '/js/pgn.js',
  '/js/ui.js',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
];

// Cache model files if they exist
const MODEL_ASSETS = [
  '/model/model.json',
  '/model/weights.bin',
  '/model/normalization.json',
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then(async cache => {
      await cache.addAll(ASSETS);
      // Model files are optional — don't fail install if missing
      await Promise.allSettled(MODEL_ASSETS.map(url =>
        fetch(url).then(r => r.ok ? cache.put(url, r) : null).catch(() => null)
      ));
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
  // Cache-first for same-origin; network for cross-origin (CDN TF.js)
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
