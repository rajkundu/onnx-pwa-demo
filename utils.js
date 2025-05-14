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
    // console.log(`[Download] content-length/totalSize = ${totalSize}`); // for debugging

    if (totalSize) {
        const reader = response.body.getReader();
        let buffer = new Uint8Array(totalSize);
        let offset = 0;
        let loadedSize = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            // Content-Length header may be erroneously small, so manually expand the download buffer
            if (offset + value.length > buffer.length) {
                const newBuffer = new Uint8Array(Math.max(buffer.length * 1.25, offset + value.length));
                newBuffer.set(buffer, 0);
                buffer = newBuffer;
            }

            buffer.set(value, offset);
            offset += value.length;
            loadedSize += value.length;
            // console.log(`[Download] loadedSize = ${loadedSize}`); // for debugging

            if (progressCallback) {
                progressCallback(loadedSize / buffer.length);
            }
        }
        if (buffer.length > loadedSize) {
            buffer = buffer.slice(0, loadedSize); // this is a copy, not a view, so that the full original buffer can be freed
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

/**
 * Converts a JS file object containing TIFF data to a data URL
 * @param {File} file - JS file object containing TIFF data
 * @returns {Promise<string>} data URL, e.g., for use as the src of an img element
 */
function tiffFileToDataURL(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function (e) {
            const tiffData = e.target.result;
            const tiff = new Tiff({ buffer: tiffData });
            resolve(tiff.toCanvas().toDataURL());
        };
        reader.onerror = reject;
        reader.readAsArrayBuffer(file);
    });
}

/**
 * Converts a TensorFlow tensor into a JSON object for download;
 * e.g., useful for saving JS-preprocessed images to disk for comparison with
 * original preprocessing implementation. JSON is used to avoid introducing
 * changes to pixel values due to compression/decompression/etc., but there are
 * probably more elegant solutions than this.
 * @param {tf.tensor} tensor - TensorFlow tensor to be converted
 */
async function saveTensorToJSON(tensor) {
    const tensorArray = await tensor.array();
    const tensorJSON = JSON.stringify(tensorArray);
    const blob = new Blob([tensorJSON], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'tensorData.json';
    a.click();
    URL.revokeObjectURL(url);
}

/**
 * Helper function for visualizing a TensorFlow tensor by converting it into a
 * dataURL and then setting the src of the passed img element. Scales pixel
 * values to range [0, 255] based on maximum (but NOT minimum) of tensor.
 * @param {tf.tensor} imgTensor - TensorFlow tensor to be visualized
 * @param {HTMLImageElement} imageEl - img element for visualizing the tensor
 */
async function visualizeImgTensor(imgTensor, imageEl) {
    var dispTensor = imgTensor.squeeze();  // Remove batch dimension, shape: [224, 224, 3]

    // Scale pixel values to [0, 255]
    dispTensor = imgTensor.mul(tf.scalar(dispTensor.max().dataSync()[0] <= 1.0 ? 255.0 : 1.0)).clipByValue(0, 255).cast('int32');

    const canvas = document.createElement('canvas');
    await tf.browser.toPixels(dispTensor, canvas);
    imageEl.src = canvas.toDataURL();
}

/**
 * Converts a JS File object containing image data into a TensorFlow tensor
 * @param {File} file - JS File object to be converted
 * @returns {tf.tensor} TensorFlow tensor containing image data
 */
function imageFileToTensor(file) {
    const reader = new FileReader();
    const imgElement = new Image();
    return new Promise(async (resolve, reject) => {
        if (file.name.endsWith(".json")) {
            const reader = new FileReader();

            // Read JSON file as text
            reader.onload = function (e) {
                try {
                    // Parse JSON data
                    const jsonData = JSON.parse(e.target.result);

                    // Check if JSON data is a valid 3D array (HxWx3)
                    if (!Array.isArray(jsonData) || !Array.isArray(jsonData[0]) || !Array.isArray(jsonData[0][0])) {
                        console.error("Invalid JSON structure");
                        return;
                    }

                    // Convert to flattened 1D tf tensor
                    const flatData = jsonData.flat(2);
                    const tensor = tf.tensor(flatData); // Convert to tf tensor

                    // Reshape tensor to tf shape (height, width, 3)
                    const height = jsonData.length;
                    const width = jsonData[0].length;
                    const reshapedTensor = tensor.reshape([height, width, 3]);

                    // Print the tensor or use it for further processing
                    reshapedTensor.print();  // This will print the tensor to the console

                    resolve(reshapedTensor);
                } catch (error) {
                    console.error("Error reading or parsing JSON:", error);
                }
            };

            // Read JSON file as text
            reader.readAsText(file);
        } else if (file.type === "image/tiff") {
            imgElement.src = await tiffFileToDataURL(file);
            imgElement.onload = () => resolve(tf.browser.fromPixels(imgElement).toFloat());
            imgElement.onerror = reject;
        } else {
            reader.onload = e => {
                imgElement.src = e.target.result;
                imgElement.onload = () => resolve(tf.browser.fromPixels(imgElement).toFloat());
                imgElement.onerror = reject;
            };
            reader.readAsDataURL(file);
        }
    });
}

/**
 * Converts an HTML Table Element to comma-separated string and initiates download of resulting .csv file
 * @param {HTMLTableElement} tableEl - HTML Table Element to be converted
 */
function htmlTableToCSV(tableEl) {
    if (!tableEl) {
        console.warn(`Cannot convert invalid table element ${tableEl} to CSV!`);
        return;
    }
    const rows = Array.from(tableEl.querySelectorAll('tr'));
    const csv = rows.map(row =>
        Array.from(row.cells).map(cell => `"${cell.innerText.replace('"', '""')}"`).join(',')
    ).join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'output.csv';
    a.click();
    URL.revokeObjectURL(url);
}

/**
 * Helper function for forcefully ensuring that a Bootstrap Modal hides/closes
 * @param {bootstrap.Modal} bsModal - Bootstrap modal object to be hidden/closed
 */
function forceHideBSModal(bsModal) {
    // Remove from DOM upon being hidden
    bsModal._element.addEventListener('hidden.bs.modal', function () {
        bsModal._element.remove();
    });
    bsModal.hide();
    // in case we're too fast and hide is called during the show animation
    bsModal._element.addEventListener('shown.bs.modal', function () {
        bsModal.hide();
    });
}

/**
 * Creates a modal div (for use with Bootstrap) and appends it to the document body
 * @param {string} modalContentHTML - string of HTML content to be placed within ".modal-content" div
 * @returns {HTMLDivElement} modal div element for use with Bootstrap
 */
function makeModalElement(modalContentHTML) {
    var modalElement = document.createElement('div');
    modalElement.id = "myModal";
    modalElement.classList.add('modal', 'fade');
    modalElement.innerHTML = `
    <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content">
            ${modalContentHTML}
        </div>
    </div>
    `;
    document.body.appendChild(modalElement);
    return modalElement;
}

/**
 * Helper function that resizes an image to fit into a square of a specified size by
 * first padding it with black to make it a square, then resizing it
 * @param {tf.tensor} imgTensor - TensorFlow tensor to be resized
 * @param {int} targetSize - desired side-length of square into which image will be resized
 */
function resizeWithSquarePadding(imgTensor, targetSize) {
    return tf.tidy(() => {
        // Pad image to make shape into a square
        const [h, w] = imgTensor.shape;
        const sizeDiff = Math.abs(h - w);
        const padTop = h < w ? Math.floor(sizeDiff / 2) : 0;
        const padBottom = h < w ? sizeDiff - padTop : 0;
        const padLeft = w < h ? Math.floor(sizeDiff / 2) : 0;
        const padRight = w < h ? sizeDiff - padLeft : 0;
        const imgTensorPadded = tf.pad(imgTensor, [
            [padTop, padBottom],
            [padLeft, padRight],
            [0, 0]
        ]);
        return tf.image.resizeBilinear(imgTensorPadded, [targetSize, targetSize]);
    });
}
