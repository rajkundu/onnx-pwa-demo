async function downloadFileWithChunking(onnx_url, progress_callback=undefined) {
    const response = await fetch(onnx_url);
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

            if (progress_callback) {
                progress_callback(loadedSize / totalSize);
            }
        }
        return buffer;
    } else {
        console.warn("Content-Length header is missing. Downloading without chunking.");
        const arrayBuffer = await response.arrayBuffer();
        return new Uint8Array(arrayBuffer);
    }
}

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
