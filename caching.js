/**
 * A callback function to be called during progression of an iterative process, e.g., downloading a file
 * @callback progressCallback
 * @param {number} progressFraction - decimal value representing completion progress, e.g., 0.6 for 60%
 */

/**
 *
 * @param {string} fetchURL - URL of file to be downloaded using fetch()
 * @param {progressCallback} [progressCallback] - A callback function to be called during the download loop
 * @returns {Promise<Uint8Array>} data from response body, stored as array buffer
 */
async function downloadFileWithChunking(fetchURL, progressCallback=undefined) {
    const response = await fetch(fetchURL);
    const contentLength = response.headers.get("Content-Length");
    const totalSize = contentLength ? parseInt(contentLength, 10) : 0;

    if (totalSize) {
        const reader = response.body.getReader();
        const buffer = new Uint8Array(totalSize);
        let offset = 0;
        let loadedSize = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer.set(value, offset);
            offset += value.length;
            loadedSize += value.length;

            if (progressCallback) {
                progressCallback(loadedSize / totalSize);
            }
        }
        return buffer;
    } else {
        console.warn("Content-Length header is missing. Downloading without chunking.");
        const arrayBuffer = await response.arrayBuffer();
        return new Uint8Array(arrayBuffer);
    }
}

/**
 * Checks whether an ONNXModel object's underlying onnx file exists in the browser’s cache.
 * @param {ONNXModel} model - an ONNXModel object
 * @returns {Promise<boolean>} whether model exists in browser's cache
 */
async function modelIsCached(model) {
    return await caches.match(new Request(model.onnxPath)) ? true : false;
}

/**
 * Deletes all data from the browser cache
 */
async function clearCaches() {
    await caches.keys().then(cacheNames => {
        return Promise.all(
            cacheNames.map(cacheName => {
                return caches.delete(cacheName);
            })
        );
    }).then(function() {
        console.log('All caches cleared');
    }).catch(function(error) {
        console.error('Error clearing caches:', error);
    });
}
