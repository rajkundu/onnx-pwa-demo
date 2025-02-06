const COMMON_BUCKET_URL = "https://pub-9133bb6240c146bda04d936a663ab7bc.r2.dev/image_quality";
const MODELS = [
    {
        name: 'GCIPL',
        onnx_path: `${COMMON_BUCKET_URL}/GCIPL.onnx`,
        onnx_loader_fn: async (model_path) => {
            return await ort.InferenceSession.create(await loadONNXBufferWithCaching(model_path), { executionProviders: ['cpu'] });
        },
        output_labels: ["Good", "Poor"], // different than the other models!!
        preprocess_fn: async (input_tensor) => {
            var processed_tensor = input_tensor;
            if (processed_tensor.shape.length === 3) {
                processed_tensor = processed_tensor.expandDims(0);
            }

            const [cropHeight, cropWidth] = [224, 224];

            // Calculate the center crop box dynamically
            const [batch, height, width, channels] = processed_tensor.shape;
            const xmin = Math.round((width - cropWidth) / 2.0);
            const ymin = Math.round((height - cropHeight) / 2.0);
            const xmax = xmin + cropWidth;
            const ymax = ymin + cropWidth;

            console.log(xmin, ymin, xmax, ymax);

            // Apply center crop and resize (Add batch dimension before crop)
            processed_tensor = tf.image.cropAndResize(
                processed_tensor,
                tf.tensor([[ymin/height, xmin/width, ymax/height, xmax/width]]), // Normalize x by width, y by height
                [0],
                [cropHeight, cropWidth]
            );

            processed_tensor = processed_tensor.div(tf.scalar(255))  // Scale pixel values from [0, 255] to [0.0, 1.0]
            .sub(tf.tensor([0.485, 0.456, 0.406]).reshape([1, 1, 1, 3])) // Normalize w/ mean
            .div(tf.tensor([0.229, 0.224, 0.225]).reshape([1, 1, 1, 3])) // Normalize w/ std

            // Default Tensorflow shape is (batch_size * H * W * C)
            // but model was trained in PyTorch, so use shape (batch_size * C * H * W)
            processed_tensor = processed_tensor.transpose([0, 3, 1, 2]);

            const ortInputs = {
                'input': new ort.Tensor('float32', new Float32Array(processed_tensor.dataSync()), processed_tensor.shape)
            };

            return ortInputs;
        }
    },
    {
        name: 'Ang3x3',
        onnx_path: `${COMMON_BUCKET_URL}/Ang3x3.onnx`,
        onnx_loader_fn: async (model_path) => {
            return await ort.InferenceSession.create(await loadONNXBufferWithCaching(model_path), { executionProviders: ['cpu'] });
        },
        output_labels: ["Poor", "Good"],
        preprocess_fn: async (input_tensor) => {
            var processed_tensor = input_tensor;
            if (processed_tensor.shape.length === 3) {
                processed_tensor = processed_tensor.expandDims(0);
            }

            processed_tensor = processed_tensor.div(tf.scalar(255))  // Scale pixel values from [0, 255] to [0.0, 1.0]
            .sub(tf.tensor([0.2977059, 0.2977059, 0.2977059]).reshape([1, 1, 1, 3])) // Normalize w/ mean
            .div(tf.tensor([0.26995874, 0.26995874, 0.26995874]).reshape([1, 1, 1, 3])) // Normalize w/ std

            const [resizeHeight, resizeWidth] = [224, 224];
            processed_tensor = tf.image.resizeBilinear(processed_tensor, [resizeHeight, resizeWidth], false);

            // Default Tensorflow shape is (batch_size * H * W * C)
            // but model was trained in PyTorch, so use shape (batch_size * C * H * W)
            processed_tensor = processed_tensor.transpose([0, 3, 1, 2]);

            const ortInputs = {
                'input': new ort.Tensor('float32', new Float32Array(processed_tensor.dataSync()), processed_tensor.shape)
            };

            return ortInputs;
        }
    },
    {
        name: 'HD21',
        onnx_path: `${COMMON_BUCKET_URL}/HD21.onnx`,
        onnx_loader_fn: async (model_path) => {
            return await ort.InferenceSession.create(await loadONNXBufferWithCaching(model_path), { executionProviders: ['cpu'] });
        },
        output_labels: ["Poor", "Good"],
        preprocess_fn: async (input_tensor) => {
            var processed_tensor = input_tensor;
            if (processed_tensor.shape.length === 3) {
                processed_tensor = processed_tensor.expandDims(0);
            }

            processed_tensor = processed_tensor.div(tf.scalar(255))  // Scale pixel values from [0, 255] to [0.0, 1.0]
            .sub(tf.tensor([0.1365239,  0.13651993, 0.13652335]).reshape([1, 1, 1, 3])) // Normalize w/ mean
            .div(tf.tensor([0.10527499, 0.10530869, 0.10528071]).reshape([1, 1, 1, 3])) // Normalize w/ std

            const [resizeHeight, resizeWidth] = [224, 224];
            processed_tensor = tf.image.resizeBilinear(processed_tensor, [resizeHeight, resizeWidth], false);

            // Default Tensorflow shape is (batch_size * H * W * C)
            // but model was trained in PyTorch, so use shape (batch_size * C * H * W)
            processed_tensor = processed_tensor.transpose([0, 3, 1, 2]);

            const ortInputs = {
                'input': new ort.Tensor('float32', new Float32Array(processed_tensor.dataSync()), processed_tensor.shape)
            };

            return ortInputs;
        }
    },
    {
        name: 'ONH4.5',
        onnx_path: `${COMMON_BUCKET_URL}/ONH4.5.onnx`,
        onnx_loader_fn: async (model_path) => {
            return await ort.InferenceSession.create(await loadONNXBufferWithCaching(model_path), { executionProviders: ['cpu'] });
        },
        output_labels: ["Poor", "Good"],
        preprocess_fn: async (input_tensor) => {
            var processed_tensor = input_tensor;
            if (processed_tensor.shape.length === 3) {
                processed_tensor = processed_tensor.expandDims(0);
            }

            processed_tensor = processed_tensor.div(tf.scalar(255))  // Scale pixel values from [0, 255] to [0.0, 1.0]
            .sub(tf.tensor([0.4014854, 0.4014854, 0.4014854]).reshape([1, 1, 1, 3])) // Normalize w/ mean
            .div(tf.tensor([0.30258739, 0.30258739, 0.30258739]).reshape([1, 1, 1, 3])) // Normalize w/ std

            const [resizeHeight, resizeWidth] = [224, 224];
            processed_tensor = tf.image.resizeBilinear(processed_tensor, [resizeHeight, resizeWidth], false);

            // Default Tensorflow shape is (batch_size * H * W * C)
            // but model was trained in PyTorch, so use shape (batch_size * C * H * W)
            processed_tensor = processed_tensor.transpose([0, 3, 1, 2]);

            const ortInputs = {
                'input': new ort.Tensor('float32', new Float32Array(processed_tensor.dataSync()), processed_tensor.shape)
            };

            return ortInputs;
        }
    },
]
