// Session options for constructing ONNX Runtime Web (ORT) InferenceSession object
// See https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html
const INFERENCE_SESSION_OPTIONS = {
    // NOTE! WebGL doesn't support dynamic input shapes (e.g., variable batch size).
    // Adding "freeDimensionOverrides: { batch_size: 1 }" to the options list doesn't work.
    // See https://onnxruntime.ai/docs/tutorials/mobile/helpers/make-dynamic-shape-fixed.html
    // and https://github.com/microsoft/onnxruntime/issues/13909
    executionProviders: [
        'webgpu',
        // 'webgl', // disabled; see above
        'cpu'
    ]
}

const FILE_BUCKET_URL = "https://pub-9133bb6240c146bda04d936a663ab7bc.r2.dev/image_quality";
const MODELS = [
    new ONNXModel({
        name: 'GCIPL',
        onnxPath: `${FILE_BUCKET_URL}/GCIPL.onnx`,
        load: async function(progressCallback) {
            this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnxPath, progressCallback), INFERENCE_SESSION_OPTIONS);
        },
        preprocess: async function(inputTensor) {
            var processedTensor = inputTensor;

            const [cropHeight, cropWidth] = [224, 224];

            // Add outer batch size dimension if necessary
            if (processedTensor.shape.length === 3) {
                processedTensor = processedTensor.expandDims(0);
            }

            // Calculate the center crop box dynamically
            const [batch, height, width, channels] = processedTensor.shape;
            const xmin = Math.round((width - cropWidth) / 2.0);
            const ymin = Math.round((height - cropHeight) / 2.0);
            const xmax = xmin + cropWidth;
            const ymax = ymin + cropWidth;

            // Apply center crop and resize (Add batch dimension before crop)
            processedTensor = tf.image.cropAndResize(
                processedTensor,
                tf.tensor([[ymin/height, xmin/width, ymax/height, xmax/width]]), // Normalize x by width, y by height
                [0],
                [cropHeight, cropWidth]
            );

            // Rescale from [0, 255] to [0.0, 1.0]
            processedTensor = processedTensor.div(tf.scalar(255));

            // Normalize: (x - mean) / std
            const mean = tf.tensor([0.485, 0.456, 0.406]);
            const std = tf.tensor([0.229, 0.224, 0.225]);
            processedTensor = processedTensor.sub(mean).div(std);

            // Default Tensorflow shape is (batch_size * H * W * C)
            // but model was trained in PyTorch, so convert tf tensor to shape (batch_size * C * H * W)
            processedTensor = processedTensor.transpose([0, 3, 1, 2]);

            const ortInputs = {
                'input': new ort.Tensor('float32', new Float32Array(processedTensor.dataSync()), processedTensor.shape)
            };

            return ortInputs;
        },
        postprocess: function(ortOutputs) {
            const ortOutputTensor = ortOutputs[this.ortSession.outputNames[0]];
            const outputRaw = ortOutputTensor.data[0];
            const outputSigmoid = tf.tensor(outputRaw).sigmoid();
            const outputSigmoidRounded = outputSigmoid.round().toInt().dataSync()[0];

            const output_labels = ["Good", "Poor"]; // different than the other models!!
            const outputLabel = output_labels[outputSigmoidRounded];

            return { raw: outputRaw, sigmoided: outputSigmoid, label: outputLabel };
        }
    }),

    new ONNXModel({
        name: 'Ang3x3',
        onnxPath: `${FILE_BUCKET_URL}/Ang3x3.onnx`,
        load: async function(progressCallback) {
            this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnxPath, progressCallback), INFERENCE_SESSION_OPTIONS);
        },
        preprocess: async (inputTensor) => {
            var processedTensor = inputTensor;

            // Rescale from [0, 255] to [0.0, 1.0]
            processedTensor = processedTensor.div(tf.scalar(255));

            // Normalize: (x - mean) / std
            const mean = tf.tensor([0.2977059, 0.2977059, 0.2977059]);
            const std = tf.tensor([0.26995874, 0.26995874, 0.26995874]);
            processedTensor = processedTensor.sub(mean).div(std);

            const [resizeHeight, resizeWidth] = [224, 224];
            processedTensor = tf.image.resizeBilinear(processedTensor, [resizeHeight, resizeWidth], false);

            // Add outer batch size dimension if necessary
            if (processedTensor.shape.length === 3) {
                processedTensor = processedTensor.expandDims(0);
            }

            // Default Tensorflow shape is (batch_size * H * W * C)
            // but model was trained in PyTorch, so convert tf tensor to shape (batch_size * C * H * W)
            processedTensor = processedTensor.transpose([0, 3, 1, 2]);

            const ortInputs = {
                'input': new ort.Tensor('float32', new Float32Array(processedTensor.dataSync()), processedTensor.shape)
            };

            return ortInputs;
        },
        postprocess: function(ortOutputs) {
            const ortOutputTensor = ortOutputs[this.ortSession.outputNames[0]];
            const outputRaw = ortOutputTensor.data[0];
            const outputSigmoid = tf.tensor(outputRaw).sigmoid();
            const outputSigmoidRounded = outputSigmoid.round().toInt().dataSync()[0];

            const output_labels = ["Poor", "Good"];
            const outputLabel = output_labels[outputSigmoidRounded];

            return { raw: outputRaw, sigmoided: outputSigmoid, label: outputLabel };
        }
    }),

    new ONNXModel({
        name: 'EDI-OCT',
        onnxPath: `${FILE_BUCKET_URL}/HD21.onnx`,
        load: async function(progressCallback) {
            this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnxPath, progressCallback), INFERENCE_SESSION_OPTIONS);
        },
        preprocess: async (inputTensor) => {
            var processedTensor = inputTensor;

            // Rescale from [0, 255] to [0.0, 1.0]
            processedTensor = processedTensor.div(tf.scalar(255));

            // Normalize: (x - mean) / std
            const mean = tf.tensor([0.1365239,  0.13651993, 0.13652335]);
            const std = tf.tensor([0.10527499, 0.10530869, 0.10528071]);
            processedTensor = processedTensor.sub(mean).div(std);

            const [resizeHeight, resizeWidth] = [224, 224];
            processedTensor = tf.image.resizeBilinear(processedTensor, [resizeHeight, resizeWidth], false);

            // Add outer batch size dimension if necessary
            if (processedTensor.shape.length === 3) {
                processedTensor = processedTensor.expandDims(0);
            }

            // Default Tensorflow shape is (batch_size * H * W * C)
            // but model was trained in PyTorch, so convert tf tensor to shape (batch_size * C * H * W)
            processedTensor = processedTensor.transpose([0, 3, 1, 2]);

            const ortInputs = {
                'input': new ort.Tensor('float32', new Float32Array(processedTensor.dataSync()), processedTensor.shape)
            };

            return ortInputs;
        },
        postprocess: function(ortOutputs) {
            const ortOutputTensor = ortOutputs[this.ortSession.outputNames[0]];
            const outputRaw = ortOutputTensor.data[0];
            const outputSigmoid = tf.tensor(outputRaw).sigmoid();
            const outputSigmoidRounded = outputSigmoid.round().toInt().dataSync()[0];

            const output_labels = ["Poor", "Good"];
            const outputLabel = output_labels[outputSigmoidRounded];

            return { raw: outputRaw, sigmoided: outputSigmoid, label: outputLabel };
        }
    }),

    new ONNXModel ({
        name: 'ONH4.5',
        onnxPath: `${FILE_BUCKET_URL}/ONH4.5.onnx`,
        load: async function(progressCallback) {
            this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnxPath, progressCallback), INFERENCE_SESSION_OPTIONS);
        },
        preprocess: async (inputTensor) => {
            var processedTensor = inputTensor;

            // Rescale from [0, 255] to [0.0, 1.0]
            processedTensor = processedTensor.div(tf.scalar(255));

            // Normalize: (x - mean) / std
            const mean = tf.tensor([0.4014854, 0.4014854, 0.4014854]);
            const std = tf.tensor([0.30258739, 0.30258739, 0.30258739]);
            processedTensor = processedTensor.sub(mean).div(std);

            const [resizeHeight, resizeWidth] = [224, 224];
            processedTensor = tf.image.resizeBilinear(processedTensor, [resizeHeight, resizeWidth], false);

            // Add outer batch size dimension if necessary
            if (processedTensor.shape.length === 3) {
                processedTensor = processedTensor.expandDims(0);
            }

            // Default Tensorflow shape is (batch_size * H * W * C)
            // but model was trained in PyTorch, so convert tf tensor to shape (batch_size * C * H * W)
            processedTensor = processedTensor.transpose([0, 3, 1, 2]);

            const ortInputs = {
                'input': new ort.Tensor('float32', new Float32Array(processedTensor.dataSync()), processedTensor.shape)
            };

            return ortInputs;
        },
        postprocess: function(ortOutputs) {
            const ortOutputTensor = ortOutputs[this.ortSession.outputNames[0]];
            const outputRaw = ortOutputTensor.data[0];
            const outputSigmoid = tf.tensor(outputRaw).sigmoid();
            const outputSigmoidRounded = outputSigmoid.round().toInt().dataSync()[0];

            const output_labels = ["Poor", "Good"];
            const outputLabel = output_labels[outputSigmoidRounded];

            return { raw: outputRaw, sigmoided: outputSigmoid, label: outputLabel };
        }
    }),
]
