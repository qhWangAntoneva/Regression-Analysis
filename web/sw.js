const CACHE_NAME = "pyodide-cache-v1";

const CDN_ORIGINS = [
  "cdn.jsdelivr.net",
  "unpkg.com",
  "registry.npmmirror.com",
  "pyodide-cdn2.organicstartup.com",
];

function shouldCache(url) {
  try {
    const u = new URL(url);
    return CDN_ORIGINS.includes(u.hostname) && u.pathname.includes("pyodide");
  } catch {
    return false;
  }
}

self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys.map((k) => k !== CACHE_NAME && caches.delete(k))
      );
      await self.clients.claim();
    })()
  );
});

self.addEventListener("fetch", (event) => {
  if (!shouldCache(event.request.url)) return;
  event.respondWith(
    (async () => {
      const cached = await caches.match(event.request);
      if (cached) return cached;
      const response = await fetch(event.request);
      if (response.ok) {
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
      }
      return response;
    })()
  );
});
