const CACHE_NAME = 'demo-pwa-cache-v1.0.1';
const precacheResources = [
    './',
    './index.html',

    // all resources in index.html (in order of appearance)
    'https://cdn.jsdelivr.net/npm/dropzone@5.9.3/dist/min/dropzone.min.js',
    'https://cdn.jsdelivr.net/npm/dropzone@5.9.3/dist/min/dropzone.min.css',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.5/dist/css/bootstrap.min.css',
    './stylesheet.css',
    'https://cdn.jsdelivr.net/npm/tiff.js@1.0.0/tiff.min.js',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.min.js',
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js',
    './utils.js',
    './ONNXModel.js',
    './models.js',
    './index.js',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.5/dist/js/bootstrap.bundle.min.js'
];

/**
 * This function is called when the service worker is installed and precaches all
 * resources in precacheResources. All resources integral to the functioning of your
 * ML model offline, including UI elements, should be precached. This ensures that
 * the website works offline from the very next time the user loads it without relying
 * on browser caching.
 */
self.addEventListener('install', event => {
    console.log(`Service Worker installing '${CACHE_NAME}'`);
    self.skipWaiting();
    event.waitUntil(
        caches.open(CACHE_NAME).then(async (cache) => {
            for (const url of precacheResources) {
                const request = new Request(url);
                const response = await fetch(request);
                try {
                    await cache.put(request, response.clone());
                } catch (err) {
                    console.warn(`[Service Worker] Failed to pre-cache resource at '${url}'!`);
                }
            }
        })
    );
});

/**
 * This function is called when a new service worker is activated, and is used here
 * to clean up (i.e., delete) old caches from existing service workers.
 */
self.addEventListener('activate', event => {
    console.log(`Service Worker activating '${CACHE_NAME}'`);
    const whitelist = [CACHE_NAME];
    event.waitUntil(async () => {
        await caches.keys().then(cacheNames => Promise.all(
                cacheNames.map(cache => {
                    if (!whitelist.includes(cache)) {
                        return caches.delete(cache);
                    }
                })
            )
        );
    });
    self.clients.claim();
});

/**
 * This function is called whenever a network request is made, e.g. UI libraries
 * and other dependencies are referenced in index.html. It acts as a proxy between
 * the application and the internet and thus can implement arbitrarily complex
 * logic for determining offline functionality.
 *
 * In the current implementation, most resources are continously fetched whenever
 * the user is online and saved/updated in cache. This way, the next time that the
 * user goes offline, the cache will have the latest versions all resources. This
 * is as opposed to strictly cache-first logic, which would cache resources just
 * once and continue using them indefinitely without updating them.
 */
self.addEventListener('fetch', (e) => {
    // Cache http and https only, skip unsupported e.g. chrome-extension:// and file://...
    if (!(e.request.url.startsWith('http:') || e.request.url.startsWith('https:'))) {
        return;
    }

    e.respondWith((async () => {
        const cache = await caches.open(CACHE_NAME);

        // force ONNX files to be cache-first since they're large
        if (e.request.url.endsWith(".onnx")) {
            const r = await cache.match(e.request);
            if (r) {
                console.log("Returning ONNX from cache");
                return r;
            }
        }

        try {
            // Fetch
            const response = await fetch(e.request);
            await e.waitUntil((async () => {
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
