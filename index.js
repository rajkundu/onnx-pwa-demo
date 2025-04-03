async function tiffFileToDataURL(file) {
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

// Function to save tensor to JSON file
async function saveTensorToJSON(tensor) {
    const tensorArray = await tensor.array();
    const tensorJSON = JSON.stringify(tensorArray);
    const blob = new Blob([tensorJSON], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'tensor_data.json';
    a.click();
    URL.revokeObjectURL(url);
}

async function visualizeImgTensor(imgTensor, imageEl) {
    var dispTensor = imgTensor.squeeze();  // Remove batch dimension, shape: [224, 224, 3]

    // Scale pixel values to [0, 255]
    dispTensor = imgTensor.mul(tf.scalar(dispTensor.max().dataSync()[0] <= 1.0 ? 255.0 : 1.0)).clipByValue(0, 255).cast('int32');

    const canvas = document.createElement('canvas');
    await tf.browser.toPixels(dispTensor, canvas);
    imageEl.src = canvas.toDataURL();
}

async function imageFileToTensor(file) {
    const reader = new FileReader();
    const imgElement = new Image();
    return new Promise(async (resolve, reject) => {
        if (file.name.endsWith(".json")) {
            const reader = new FileReader();

            // Step 1: Read the JSON file as text
            reader.onload = function (e) {
                try {
                    // Step 2: Parse the JSON data
                    const jsonData = JSON.parse(e.target.result);

                    // Check if jsonData is a valid 3D array (HxWx3)
                    if (!Array.isArray(jsonData) || !Array.isArray(jsonData[0]) || !Array.isArray(jsonData[0][0])) {
                        console.error("Invalid JSON structure");
                        return;
                    }

                    // Step 3: Flatten the data
                    const flatData = jsonData.flat(2);  // Flatten the HxWx3 array to a 1D array

                    // Step 4: Convert the flat array to a TensorFlow tensor
                    const tensor = tf.tensor(flatData);

                    // Step 5: Reshape the tensor to the original shape (height, width, 3)
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

            // Step 6: Read the file as text
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

async function runModel(model, ortSession, ortInputs) {
    if (model == null || ortSession == null) {
        alert("Please ensure that the model has loaded!");
        return;
    }
    if (ortInputs == null) {
        alert("No input provided!");
    }
    return await ortSession.run(ortInputs);
}

var model = null;
var ortSession = null;
const modelSelect = document.querySelector('#modelSelect');
modelSelect.addEventListener('change', onModelSelectChange);
async function onModelSelectChange(e) {
    model = MODELS[e.target.value];

    var modalElement = document.createElement('div');
    modalElement.classList.add('modal', 'fade');
    modalElement.id = 'loadingModal';
    modalElement.tabIndex = -1;
    modalElement.setAttribute('aria-hidden', 'true');
    modalElement.setAttribute('data-bs-backdrop', 'static');
    modalElement.setAttribute('data-bs-keyboard', 'false');
    modalElement.innerHTML = `
    <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content">
            <div class="modal-body">
                <h4>Loading ${model.name} Model...</h4>
                <div class="progress" style="height: 30px;">
                    <div id="modelDownloadProgress" class="progress-bar bg-success" role="progressbar" style="width: 0%;">0%</div>
                </div>
            </div>
        </div>
    </div>
    `;
    document.body.appendChild(modalElement);
    var modal = new bootstrap.Modal(modalElement);
    modal.show();

    try {
        ortSession = await model.onnx_loader_fn(model.onnx_path);
    } catch {
        model = null;
        return;
    } finally {
        // Remove from DOM upon being hidden
        modalElement.addEventListener('hidden.bs.modal', function () {
            modalElement.remove();
        });
        modal.hide();
        // in case we're too fast and hide is called during the show animation
        modalElement.addEventListener('shown.bs.modal', function () {
            modal.hide();
        });
    }

    // Enable input
    dropzone.element.classList.remove("disabled");
    dropzone.element.querySelector('.dz-message').textContent =  "Drag & drop, or click to browse...";
    dropzone.enable();

    // Clear outputs
    clearOutputs();
}

async function onRunButtonPress() {
    clearOutputs();

    var modalElement = document.createElement('div');
    modalElement.classList.add('modal', 'fade');
    modalElement.id = 'loadingModal';
    modalElement.tabIndex = -1;
    modalElement.setAttribute('aria-hidden', 'true');
    modalElement.setAttribute('data-bs-backdrop', 'static');
    modalElement.setAttribute('data-bs-keyboard', 'false');
    modalElement.innerHTML = `
    <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content">
            <div class="modal-body">
                <h4>Running ${model.name} Model...<span></span></h4>
                <div class="progress" style="height: 30px;">
                    <div id="runProgress" class="progress-bar bg-success" role="progressbar" style="width: 0%;">0%</div>
                </div>
            </div>
        </div>
    </div>
    `;
    document.body.appendChild(modalElement);
    var modal = new bootstrap.Modal(modalElement);
    modal.show();
    const progressBar = modalElement.querySelector("#runProgress");
    const modalHeader = modalElement.querySelector("h4 span");

    allOrtOutputs = [];
    allInferenceTimes = [];
    for (let idx = 0; idx < dropzone.files.length; idx++) {
        let file = dropzone.files[idx];
        const rawInputTensor = await imageFileToTensor(file); // tensor will be in Tensorflow shape = (batch_size * H * W * C)
        const ortInputs = await model.preprocess_fn(rawInputTensor);
        const start = performance.now();  // Start timing
        const ortOutputs = await ortSession.run(ortInputs);
        const end = performance.now();  // End timing
        allOrtOutputs.push(ortOutputs);
        allInferenceTimes.push(end - start);

        const percent = Math.round(((idx+1) / dropzone.files.length) * 100);
        if (progressBar) {
            progressBar.style.width = `${percent}%`;
            progressBar.innerText = `${percent}%`;
        }
        modalHeader.innerText = ` (${idx+1}/${dropzone.files.length})`;
    }

    // Remove from DOM upon being hidden
    modalElement.addEventListener('hidden.bs.modal', function () {
        modalElement.remove();
    });
    modal.hide();
    // in case we're too fast and hide is called during the show animation
    modalElement.addEventListener('shown.bs.modal', function () {
        modal.hide();
    });

    const table = document.querySelector("#multiOutputContainer table");
    const columns = ["filename", "quality", "outputRaw", "inferenceTimeMs"];
    let thead = document.createElement("thead");
    thead.classList.add("table-light");
    thead.innerHTML = `<tr>${columns.map(col => `<th>${col}</th>`).join("")}</tr>`;
    table.appendChild(thead);
    let tbody = document.createElement("tbody");

    var ortOutputTensor, outputRaw, outputSigmoid, outputSigmoidRounded, outputLabel, inferenceTime;
    for (let idx = 0; idx < allOrtOutputs.length; idx++) {
        let ortOutputs = allOrtOutputs[idx];
        let file = dropzone.files[idx];
        ortOutputTensor = ortOutputs[ortSession.outputNames[0]];
        outputRaw = ortOutputTensor.data;
        outputSigmoid = tf.tensor(outputRaw).sigmoid();
        outputSigmoidRounded = outputSigmoid.round().toInt().dataSync()[0];
        outputLabel = model.output_labels[outputSigmoidRounded];

        inferenceTime = allInferenceTimes[idx];

        let tr = document.createElement("tr");
        if (outputLabel.toUpperCase() === "POOR") {
            tr.classList.add("table-danger");
        }
        console.log(file.webkitRelativePath);
        tr.innerHTML = `
            <td>${file.webkitRelativePath ? file.webkitRelativePath : file.name}</td>
            <td>${outputLabel}</td>
            <td>${outputRaw[0].toFixed(8)}</td>
            <td>${allInferenceTimes[idx].toFixed(0)}</td>
        `;
        tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    document.querySelector("#multiOutputContainer").classList.remove("d-none");
}

function clearOutputs() {
    document.querySelector('#outputContainer').innerHTML = "";
    document.querySelector('#outputContainer').classList.add("d-none");
    document.querySelector('#multiOutputContainer').classList.add("d-none");
    document.querySelector('#multiOutputContainer table').innerHTML = "";
}

function exportMultiOutputTableToCSV() {
    const rows = Array.from(document.querySelectorAll('#multiOutputContainer table tr'));
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

var dropzone = null;
document.addEventListener("DOMContentLoaded", async function() {
    MODELS.forEach((model, idx) => {
        let option = document.createElement('option');
        option.value = idx;
        option.textContent = model.name;
        modelSelect.appendChild(option);
    });

    dropzone = new Dropzone("#inputDropzone", {
        url: "javascript:void(0);", // No actual upload, just processing files
        autoProcessQueue: false, // Prevent Dropzone from uploading files
        addRemoveLinks: true,
        dictRemoveFile: "Remove",
        paramName: "file",
        dictDefaultMessage: "Please select an image type above",
        init: function() {
            this.on("addedfile", async function(event) {
                document.querySelectorAll("#modelButtonContainer button").forEach((element) => {
                    element.classList.remove("disabled");
                });
            });
            this.on("reset", async function(event) {
                document.querySelectorAll("#modelButtonContainer button").forEach((element) => {
                    element.classList.add("disabled");
                });
                clearOutputs();
            });
            this.on("thumbnail", async function (file, dataUrl) {
                if (file.type === "image/tiff") {
                    const previewImg = file.previewElement.querySelector(".dz-image img");
                    previewImg.src = await tiffFileToDataURL(file);
                }
            });

            // Disable by default, until a model is selected
            this.disable();
            this.element.classList.add("disabled");
        }
    });
});
