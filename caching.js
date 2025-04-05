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

async function downloadFileWithChunking(response, progress_callback=undefined) {
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

async function loadONNXWithCaching(onnx_url, progress_callback=undefined) {
    var model = await retrieveModel(onnx_url);

    if (model) {
        console.log(`Loaded model from IndexedDB (${onnx_url})`);
    } else {
        console.log(`Fetching model from web (${onnx_url})`);

        const response = await fetch(onnx_url);
        model = await downloadFileWithChunking(response, progress_callback);

        // Store model for future use
        await storeModel(model, onnx_url);
        console.log("Stored model in IndexedDB");
    }

    return model;
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
