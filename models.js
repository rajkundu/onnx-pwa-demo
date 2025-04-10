class ONNXModel {
    constructor({
            name,
            onnx_path,
            load = async function() {
                this.ortSession = await ort.InferenceSession.create(this.onnx_path);
            },
            preprocess = raw_input => raw_input,
            run = async function(raw_input) {
                const preprocess_start = performance.now();
                const ortInputs = await this.preprocess(raw_input);
                const preprocess_end = performance.now();
                const ortOutputs = await this.ortSession.run(ortInputs);
                const inference_end = performance.now();
                const processed_output = await this.postprocess(ortOutputs);
                const postprocess_end = performance.now();
                return {
                    output: processed_output,
                    preprocessTimeMs: preprocess_end-preprocess_start,
                    inferenceTimeMs: inference_end-preprocess_end,
                    postprocessTimeMs: postprocess_end-inference_end
                };
            },
            postprocess = raw_output => raw_output
        } = {}) {
        this.name = name;
        this.onnx_path = onnx_path;
        this.load = load;
        this.preprocess = preprocess;
        this.postprocess = postprocess;
        this.run = run;

        this.ortSession = null;
    }
}

const inferenceSessionOptions = {
    // NOTE! WebGL doesn't support dynamic input shapes (e.g., variable batch size).
    // adding "freeDimensionOverrides: { batch_size: 1 }" to the options list doesn't help.
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
        onnx_path: `${COMMON_BUCKET_URL}/GCIPL.onnx`,
        load: async function(progress_callback) {
            this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnx_path, progress_callback), inferenceSessionOptions);
        },
        preprocess: async function(input_tensor) {
            var processed_tensor = input_tensor;

            const [cropHeight, cropWidth] = [224, 224];

            // Add outer batch size dimension if necessary
            if (processed_tensor.shape.length === 3) {
                processed_tensor = processed_tensor.expandDims(0);
            }

            // Calculate the center crop box dynamically
            const [batch, height, width, channels] = processed_tensor.shape;
            const xmin = Math.round((width - cropWidth) / 2.0);
            const ymin = Math.round((height - cropHeight) / 2.0);
            const xmax = xmin + cropWidth;
            const ymax = ymin + cropWidth;

            // Apply center crop and resize (Add batch dimension before crop)
            processed_tensor = tf.image.cropAndResize(
                processed_tensor,
                tf.tensor([[ymin/height, xmin/width, ymax/height, xmax/width]]), // Normalize x by width, y by height
                [0],
                [cropHeight, cropWidth]
            );

            // Rescale from [0, 255] to [0.0, 1.0]
            processed_tensor = processed_tensor.div(tf.scalar(255));

            // Normalize: (x - mean) / std
            const mean = tf.tensor([0.485, 0.456, 0.406]);
            const std = tf.tensor([0.229, 0.224, 0.225]);
            processed_tensor = processed_tensor.sub(mean).div(std);

            // Default Tensorflow shape is (batch_size * H * W * C)
            // but model was trained in PyTorch, so convert tf tensor to shape (batch_size * C * H * W)
            processed_tensor = processed_tensor.transpose([0, 3, 1, 2]);

            const ortInputs = {
                'input': new ort.Tensor('float32', new Float32Array(processed_tensor.dataSync()), processed_tensor.shape)
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
        onnx_path: `${COMMON_BUCKET_URL}/Ang3x3.onnx`,
        load: async function(progress_callback) {
            this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnx_path, progress_callback), inferenceSessionOptions);
        },
        preprocess: async (input_tensor) => {
            var processed_tensor = input_tensor;

            // Rescale from [0, 255] to [0.0, 1.0]
            processed_tensor = processed_tensor.div(tf.scalar(255));

            // Normalize: (x - mean) / std
            const mean = tf.tensor([0.2977059, 0.2977059, 0.2977059]);
            const std = tf.tensor([0.26995874, 0.26995874, 0.26995874]);
            processed_tensor = processed_tensor.sub(mean).div(std);

            const [resizeHeight, resizeWidth] = [224, 224];
            processed_tensor = tf.image.resizeBilinear(processed_tensor, [resizeHeight, resizeWidth], false);

            // Add outer batch size dimension if necessary
            if (processed_tensor.shape.length === 3) {
                processed_tensor = processed_tensor.expandDims(0);
            }

            // Default Tensorflow shape is (batch_size * H * W * C)
            // but model was trained in PyTorch, so convert tf tensor to shape (batch_size * C * H * W)
            processed_tensor = processed_tensor.transpose([0, 3, 1, 2]);

            const ortInputs = {
                'input': new ort.Tensor('float32', new Float32Array(processed_tensor.dataSync()), processed_tensor.shape)
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
        onnx_path: `${COMMON_BUCKET_URL}/HD21.onnx`,
        load: async function(progress_callback) {
            this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnx_path, progress_callback), inferenceSessionOptions);
        },
        preprocess: async (input_tensor) => {
            var processed_tensor = input_tensor;

            // Rescale from [0, 255] to [0.0, 1.0]
            processed_tensor = processed_tensor.div(tf.scalar(255));

            // Normalize: (x - mean) / std
            const mean = tf.tensor([0.1365239,  0.13651993, 0.13652335]);
            const std = tf.tensor([0.10527499, 0.10530869, 0.10528071]);
            processed_tensor = processed_tensor.sub(mean).div(std);

            const [resizeHeight, resizeWidth] = [224, 224];
            processed_tensor = tf.image.resizeBilinear(processed_tensor, [resizeHeight, resizeWidth], false);

            // Add outer batch size dimension if necessary
            if (processed_tensor.shape.length === 3) {
                processed_tensor = processed_tensor.expandDims(0);
            }

            // Default Tensorflow shape is (batch_size * H * W * C)
            // but model was trained in PyTorch, so convert tf tensor to shape (batch_size * C * H * W)
            processed_tensor = processed_tensor.transpose([0, 3, 1, 2]);

            const ortInputs = {
                'input': new ort.Tensor('float32', new Float32Array(processed_tensor.dataSync()), processed_tensor.shape)
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
        onnx_path: `${COMMON_BUCKET_URL}/ONH4.5.onnx`,
        load: async function(progress_callback) {
            this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnx_path, progress_callback), inferenceSessionOptions);
        },
        preprocess: async (input_tensor) => {
            var processed_tensor = input_tensor;

            // Rescale from [0, 255] to [0.0, 1.0]
            processed_tensor = processed_tensor.div(tf.scalar(255));

            // Normalize: (x - mean) / std
            const mean = tf.tensor([0.4014854, 0.4014854, 0.4014854]);
            const std = tf.tensor([0.30258739, 0.30258739, 0.30258739]);
            processed_tensor = processed_tensor.sub(mean).div(std);

            const [resizeHeight, resizeWidth] = [224, 224];
            processed_tensor = tf.image.resizeBilinear(processed_tensor, [resizeHeight, resizeWidth], false);

            // Add outer batch size dimension if necessary
            if (processed_tensor.shape.length === 3) {
                processed_tensor = processed_tensor.expandDims(0);
            }

            // Default Tensorflow shape is (batch_size * H * W * C)
            // but model was trained in PyTorch, so convert tf tensor to shape (batch_size * C * H * W)
            processed_tensor = processed_tensor.transpose([0, 3, 1, 2]);

            const ortInputs = {
                'input': new ort.Tensor('float32', new Float32Array(processed_tensor.dataSync()), processed_tensor.shape)
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
