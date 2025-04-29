var activeModel = null;
const modelSelect = document.querySelector('#modelSelect');
async function updateModelSelect() {
    for (var idx = 0; idx < MODELS.length; idx++) {
        // don't access .options directly with index since we may have placeholders, headings, etc.
        const option = modelSelect.querySelector(`option[value="${idx}"]`);
        let model = MODELS[idx];
        if (await modelIsCached(model)) {
            option.textContent = model.name + " (Cached)"; // show this even if online so user knows which need caching
        } else {
            if (navigator.onLine) {
                option.textContent = model.name;
                option.disabled = false;
            } else {
                option.textContent = model.name + " (Unavailable)";
                option.disabled = true;
            }
        }
    };
}
modelSelect.addEventListener('change', async function(e) {
    // if (activeModel && activeModel.ortSession) {
    //     activeModel.ortSession.release();
    // }
    activeModel = MODELS[e.target.value];

    const modalElement = makeModalElement(`
        <div class="modal-body">
            <h4>Loading ${activeModel.name} Model...</h4>
            <div class="progress" style="height: 30px;">
                <div id="modelDownloadProgress" class="progress-bar bg-success" role="progressbar" style="width: 0%;"></div>
            </div>
        </div>
    `);
    modalElement.tabIndex = -1;
    modalElement.setAttribute('aria-hidden', 'true');
    modalElement.setAttribute('data-bs-backdrop', 'static');
    modalElement.setAttribute('data-bs-keyboard', 'false');
    const bsModal = new bootstrap.Modal(modalElement);
    bsModal.show();

    let progressBar = document.querySelector("#modelDownloadProgress");
    try {
        await activeModel.load((progressFraction) => {
            const percent = Math.round(progressFraction * 100);
            progressBar.style.width = `${percent}%`;
            progressBar.innerText = `${percent}%`;
        });
    } catch {
        activeModel = null;
        return;
    } finally {
        forceHideBSModal(bsModal);
    }

    // Enable input
    dropzone.element.classList.remove("disabled");
    dropzone.element.querySelector('.dz-message').textContent =  "Drag & drop, or click to browse...";
    dropzone.enable();

    // Clear outputs
    clearOutputs();

    // Update to show the user that the model is now cached
    updateModelSelect();
});

document.querySelector('#clearCacheButton').addEventListener('click', async function() {
    await clearCaches();

    // Unregister service worker
    if ('serviceWorker' in navigator) {
        await navigator.serviceWorker.getRegistration().then(async function(registration) {
            if (registration) {
                await registration.unregister();
                console.log("Unregistered service worker");
            } else {
                console.log("No service worker to unregister.");
            }
        });
    }

    window.location.reload();
});

document.querySelector('#runButton').addEventListener('click', async function() {
    clearOutputs();

    const modalElement = makeModalElement(`
        <div class="modal-body">
            <h4>Running ${activeModel.name} Model...<span></span></h4>
            <div class="progress" style="height: 30px;">
                <div id="runProgress" class="progress-bar bg-success" role="progressbar" style="width: 0%;">0%</div>
            </div>
        </div>
    `);
    modalElement.tabIndex = -1;
    modalElement.setAttribute('aria-hidden', 'true');
    modalElement.setAttribute('data-bs-backdrop', 'static');
    modalElement.setAttribute('data-bs-keyboard', 'false');
    const bsModal = new bootstrap.Modal(modalElement);
    bsModal.show();
    const progressBar = bsModal._element.querySelector("#runProgress");
    const modalHeader = bsModal._element.querySelector("h4 span");

    // Table to display outputs
    const table = document.querySelector("#multiOutputContainer table");
    // Set up table header
    const columns = ["filename", "quality", "raw", "inferenceTimeMs"];
    let thead = document.createElement("thead");
    thead.classList.add("table-light");
    thead.innerHTML = `<tr>${columns.map(col => `<th>${col}</th>`).join("")}</tr>`;
    table.appendChild(thead);
    // Set up table body (empty)
    let tbody = document.createElement("tbody");
    table.appendChild(tbody);

    for (let idx = 0; idx < dropzone.files.length; idx++) {
        // Run model on input
        let file = dropzone.files[idx];
        const rawInputTensor = await imageFileToTensor(file); // tensor will be in Tensorflow shape = (batch_size * H * W * C)
        const runData = await activeModel.run(rawInputTensor);

        // Update progress bar modal
        const percent = Math.round(((idx+1) / dropzone.files.length) * 100);
        if (progressBar) {
            progressBar.style.width = `${percent}%`;
            progressBar.innerText = `${percent}%`;
        }
        modalHeader.innerText = ` (${idx+1}/${dropzone.files.length})`;

        // Add row to table
        let tr = tbody.insertRow();
        tr.innerHTML = `
            <td>${file.webkitRelativePath ? file.webkitRelativePath : file.name}</td>
            <td>${runData.output.label}</td>
            <td>${runData.output.raw.toFixed(8)}</td>
            <td>${runData.inferenceTimeMs.toFixed(0)}</td>
        `;
        // Conditional styling, poor quality is highlighted as red
        if (runData.output.label.toUpperCase() === "POOR") {
            tr.classList.add("table-danger");
        }
    }

    // Hide modal
    forceHideBSModal(bsModal);

    // Display table
    document.querySelector("#multiOutputContainer").classList.remove("d-none");
});

function clearOutputs() {
    document.querySelector('#multiOutputContainer').classList.add("d-none");
    document.querySelector('#multiOutputContainer table').innerHTML = "";
}

var dropzone = null;
document.addEventListener("DOMContentLoaded", async function() {
    for (var idx = 0; idx < MODELS.length; idx++) {
        const model = MODELS[idx];
        let option = document.createElement('option');
        option.value = idx;
        option.textContent = model.name;
        modelSelect.appendChild(option);
    };

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
                const previewImg = file.previewElement.querySelector(".dz-image img");
                if (file.type === "image/tiff") {
                    previewImg.src = await tiffFileToDataURL(file); // convert TIFF to PNG
                }

                // Free dataURL string; we will just keep the raw buffer
                delete file.dataURL;
            });

            // Disable by default, until a model is selected
            this.disable();
            this.element.classList.add("disabled");
        }
    });

    // Determine whether we're online
    navigator.onLine ? goOnline() : goOffline();
});

// --------- PWA Config ---------- //

// Register PWA Service Worker
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./service-worker.js');
}
async function goOnline() {
    console.log('App is online.');
    document.querySelector('#offlineHeading').innerText = "";
    updateModelSelect();
}
async function goOffline() {
    console.log('App is offline.');
    document.querySelector('#offlineHeading').innerText = " (Offline)";
    updateModelSelect();
}
window.addEventListener('online', goOnline);
window.addEventListener('offline', goOffline);
