const DB_NAME = "onnx_model_cache";
const DB_VERSION = 1;
const STORE_NAME = "models";

async function retrieveModel(model_key) {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onupgradeneeded = (event) => {
            let db = event.target.result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME);
            }
        };

        request.onsuccess = (event) => {
            let db = event.target.result;
            let transaction = db.transaction(STORE_NAME, "readonly");
            let store = transaction.objectStore(STORE_NAME);
            let getRequest = store.get(model_key);

            getRequest.onsuccess = () => resolve(getRequest.result);
            getRequest.onerror = () => reject(getRequest.error);
        };

        request.onerror = () => reject(request.error);
    });
}

async function storeModel(buffer, model_key) {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onsuccess = (event) => {
            let db = event.target.result;
            let transaction = db.transaction(STORE_NAME, "readwrite");
            let store = transaction.objectStore(STORE_NAME);
            let putRequest = store.put(buffer, model_key);

            putRequest.onsuccess = () => resolve();
            putRequest.onerror = () => reject(putRequest.error);
        };

        request.onerror = () => reject(request.error);
    });
}

async function loadONNXBufferWithCaching(onnx_url) {
    let storedModel = await retrieveModel(onnx_url);

    var arr = null;
    if (storedModel) {
        console.log(`Loading model from IndexedDB (${onnx_url})`);
        arr = new Uint8Array(storedModel);
    } else {
        console.log(`Fetching model from web (${onnx_url})`);

        const response = await fetch(onnx_url);
        var arrayBuffer = null;

        const progressBar = document.querySelector("#modelDownloadProgress");
        if (progressBar) {
            const contentLength = response.headers.get("Content-Length");
            if (!contentLength) { console.warn("Content-Length header is missing. Cannot track progress accurately."); }
            const totalSize = contentLength ? parseInt(contentLength, 10) : 0;

            let loadedSize = 0;
            const reader = response.body.getReader();
            // Allocate a buffer for the whole file upfront (more efficient)
            const chunks = [];
            let totalBytes = 0;

            // Read chunks and update progress bar
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                chunks.push(value);
                loadedSize += value.length;
                totalBytes += value.length;

                if (totalSize) {
                    const percent = Math.round((loadedSize / totalSize) * 100);
                    if (progressBar) {
                        progressBar.style.width = `${percent}%`;
                        progressBar.innerText = `${percent}%`;
                    }
                }
            }

            console.log("Download complete. Combining chunks...");

            // Create a single Uint8Array to hold all data
            arrayBuffer = new Uint8Array(totalBytes);
            let offset = 0;
            for (const chunk of chunks) {
                arrayBuffer.set(chunk, offset);
                offset += chunk.length;
            }
        } else {
            arrayBuffer = await response.arrayBuffer();
        }

        // Store model for future use
        await storeModel(arrayBuffer, onnx_url);
        console.log("Stored model in IndexedDB");
        arr = new Uint8Array(arrayBuffer);
    }

    return arr;
}

function clearStoredModels() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.deleteDatabase(DB_NAME);

        request.onsuccess = () => {
            console.log("IndexedDB cleared successfully");
            resolve();
        };

        request.onerror = () => {
            console.error("Error clearing IndexedDB", request.error);
            reject(request.error);
        };
    });
}
