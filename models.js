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

const MODELS = [
    new ONNXModel ({
        name: 'VIT Face Expression Classification',
        onnxPath: "https://huggingface.co/trpakov/vit-face-expression/resolve/main/onnx/model.onnx",
        load: async function(progressCallback) {
            this.ortSession = await ort.InferenceSession.create(await downloadFileWithChunking(this.onnxPath, progressCallback), INFERENCE_SESSION_OPTIONS);
        },
        preprocess: async (inputTensor) => {
            // derived from https://huggingface.co/trpakov/vit-face-expression/blob/main/onnx/preprocessor_config.json
            var processedTensor = inputTensor;

            // Resize to 224x224
            processedTensor = tf.image.resizeBilinear(processedTensor, [224, 224]);

            // Rescale from [0, 255] to [0.0, 1.0]
            processedTensor = processedTensor.div(tf.scalar(255));

            // Normalize: (x - mean) / std
            const mean = tf.tensor([0.5, 0.5, 0.5]);
            const std = tf.tensor([0.5, 0.5, 0.5]);
            processedTensor = processedTensor.sub(mean).div(std);

            // Add outer batch size dimension if necessary
            if (processedTensor.shape.length === 3) {
                processedTensor = processedTensor.expandDims(0);
            }

            // Default Tensorflow shape is (batch_size * H * W * C)
            // but model uses shape (batch_size * C * H * W)
            processedTensor = processedTensor.transpose([0, 3, 1, 2]);

            // Convert tf tensor to ort inputs
            const ortInputs = {
                'pixel_values': new ort.Tensor('float32', new Float32Array(processedTensor.dataSync()), processedTensor.shape)
            };
            return ortInputs;
        },
        postprocess: function(ortOutputs) {
            const ortOutputTensor = ortOutputs[this.ortSession.outputNames[0]];
            const outputRaw = ortOutputTensor.data;

            // derived from https://huggingface.co/trpakov/vit-face-expression/blob/main/onnx/config.json
            const id2label = {
                0: "angry",
                1: "disgust",
                2: "fear",
                3: "happy",
                4: "neutral",
                5: "sad",
                6: "surprise"
              };

              const logits = tf.tensor(outputRaw);
              const probs = tf.softmax(logits);
              const predictedIndex = probs.argMax().dataSync()[0];
              const predictedLabel = id2label[predictedIndex];
              return { raw: logits.dataSync()[predictedIndex], softmaxed: probs, label: predictedLabel };
        }
    })
]
