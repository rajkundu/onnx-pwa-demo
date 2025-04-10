const CACHE_NAME = 'demo-pwa-cache-v1';
const precacheResources = [
    '/',
    '/index.html',

    // all resources in index.html (in order of appearance)
    'https://cdn.jsdelivr.net/npm/dropzone@5.9.3/dist/min/dropzone.min.js',
    'https://cdn.jsdelivr.net/npm/dropzone@5.9.3/dist/min/dropzone.min.css',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css',
    '/stylesheet.css',
    'https://cdn.jsdelivr.net/npm/tiff.js@1.0.0/tiff.min.js',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.min.js',
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js',
    '/caching.js',
    '/models.js',
    '/index.js',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.5/dist/js/bootstrap.bundle.min.js'
];

self.addEventListener('install', event => {
    console.log(`Service Worker installing '${CACHE_NAME}'`);
    self.skipWaiting();
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            return cache.addAll(precacheResources);
        })
    );
});

self.addEventListener('activate', event => {
    console.log(`Service Worker activating '${CACHE_NAME}'`);
    const whitelist = [CACHE_NAME];
    event.waitUntil(caches.keys().then(cacheNames => Promise.all(
                cacheNames.map(cache => {
                    if (!whitelist.includes(cache)) {
                        return caches.delete(cache);
                    }
                })
            )
        )
    );
    self.clients.claim();
});

self.addEventListener('fetch', (e) => {
    // Cache http and https only, skip unsupported e.g. chrome-extension:// and file://...
    if (!(e.request.url.startsWith('http:') || e.request.url.startsWith('https:'))) {
        return;
    }

    e.respondWith((async () => {
        const cache = await caches.open(CACHE_NAME);

        // force ONNX files to be cache-first since they're large
        if (e.request.url.endsWith(".onnx")) {
            console.log("Returning ONNX from cache");
            const r = await cache.match(e.request);
            if (r) return r;
        }

        try {
            // Fetch
            const response = await fetch(e.request);
            e.waitUntil((async () => {
                try {
                    await cache.put(e.request, response.clone());
                    console.log(`[Service Worker] Cached resource: ${e.request.url}`);
                } catch (err) {
                    console.error(`[Service Worker] Failed to cache ${e.request.url}: ${err}`);
                }
            })());
            return response;
        } catch {
            // Fall back to cache if fetch failed
            const r = await cache.match(e.request);
            if (r) return r;
        }
        return new Response("Service Unavailable", { status: 503, statusText: "Service Unavailable" });
    })());
});
