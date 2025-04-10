/**
 * @typedef {Object} ONNXModelData
 * @property {string} name - Display name of the model
 * @property {string} onnxPath - Path to the .onnx model file, which can either be remote or local.
 * @property {function} [load] - Optional implementation of load function; see ONNXModel class for details.
 * @property {function} [preprocess] - Custom implementation of input preprocessing function; see ONNXModel class for details.
 * @property {function} [postprocess] - Custom implementation of output postprocessing function; see ONNXModel class for details.
 */

class ONNXModel {
    /**
     * Constructor for ONNXModel object class with optional function overrides.
     * Other functions (e.g., run) can be overwritten manually or patched at runtime.
     * @param {ONNXModelData} data - Initialization data for ONNXModel object
     */
    constructor(data = {}) {
        this.name = data.name;
        this.onnxPath = data.onnxPath;
        this.load = data.load || this.load;
        this.preprocess = data.preprocess || this.preprocess;
        this.postprocess = data.postprocess || this.postprocess;

        // model data is loaded into RAM by load(), not here during construction
        this.ortSession = null;
    }

    /**
     * Default implementation of loader function which loads ONNX model
     * into RAM and creates an ORT inference session.
     */
    async load() {
        this.ortSession = await ort.InferenceSession.create(this.onnxPath);
    }

    /**
     * Default, pass-through implementation of preprocess function.
     * @param {*} rawInput - the raw input to be preprocessed
     * @returns {Promise<Object<string, ort.Tensor>>} input tensor map, i.e., input of ort.InferenceSession.run
     */
    async preprocess(rawInput) {
        return rawInput;
    }

    /**
     * Default implementation of run, which internally performs preprocessing & postprocessing
     * @param {*} rawInput - the raw input to be preprocessed, run, and postprocessed
     * @returns {Promise<Object<string, *>>} object containing postprocessed data and timing metadata from inference
     */
    async run(rawInputs) {
        const preprocessStart = performance.now();
        const ortInputs = await this.preprocess(rawInputs);
        const preprocessEnd = performance.now();
        const ortOutputs = await this.ortSession.run(ortInputs);
        const inferenceEnd = performance.now();
        const processedOutputs = await this.postprocess(ortOutputs);
        const postprocessEnd = performance.now();
        return {
            output: processedOutputs,
            preprocessTimeMs: preprocessEnd-preprocessStart,
            inferenceTimeMs: inferenceEnd-preprocessEnd,
            postprocessTimeMs: postprocessEnd-inferenceEnd
        };
    }

    /**
     * Default, pass-through implementation of postprocess function.
     * @param {Promise<Object<string, ort.Tensor>>} rawOutput - output tensor map, e.g., output of ort.InferenceSession.run
     * @returns {*} any custom structure for holding postprocessed inference data
     */
    async postprocess(rawOutput) {
        return rawOutput;
    }
}

const inferenceSessionOptions = {
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

const COMMON_BUCKET_URL = "https://pub-9133bb6240c146bda04d936a663ab7bc.r2.dev/image_quality";
const MODELS = [
    new ONNXModel({
        name: 'GCIPL',
        onnxPath: `${COMMON_BUCKET_URL}/GCIPL.onnx`,
        load: async function(progressCallback) {
            this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnxPath, progressCallback), inferenceSessionOptions);
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
        onnxPath: `${COMMON_BUCKET_URL}/Ang3x3.onnx`,
        load: async function(progressCallback) {
            this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnxPath, progressCallback), inferenceSessionOptions);
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
        name: 'HD21',
        onnxPath: `${COMMON_BUCKET_URL}/HD21.onnx`,
        load: async function(progressCallback) {
            this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnxPath, progressCallback), inferenceSessionOptions);
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
        onnxPath: `${COMMON_BUCKET_URL}/ONH4.5.onnx`,
        load: async function(progressCallback) {
            this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnxPath, progressCallback), inferenceSessionOptions);
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
