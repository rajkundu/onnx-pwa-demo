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
