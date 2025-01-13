class Layer {
    public int inputNodeCount, outputNodeCount;
    public double[,] weights, costGradientW; 
    public double[,] weightGradients;
    public double[] biases, costGradientB;
    public Func<double, double> activationFunction, derivativeActivationFunction;
    public Layer(int inputNodeCount, int outputNodeCount, Func<double, double> activationFunction, Func<double, double> derivativeActivationFunction, Random rng) {
        this.inputNodeCount = inputNodeCount; this.outputNodeCount = outputNodeCount;
        this.activationFunction = activationFunction; this.derivativeActivationFunction = derivativeActivationFunction;
        weights = new double[outputNodeCount, inputNodeCount]; costGradientW = new double[outputNodeCount, inputNodeCount];
        biases = new double[outputNodeCount]; costGradientB = new double[outputNodeCount];

        for (int x = 0; x < outputNodeCount; x++) {
            for (int y = 0; y < inputNodeCount; y++) {
                weights[x, y] = rng.NextDouble();
                biases[x] = rng.NextDouble();
            }
        }
        inputs = new double[inputNodeCount]; weightedInputs = new double[outputNodeCount]; outputs = new double[outputNodeCount]; 
        weightGradients = new double[outputNodeCount, inputNodeCount];
    }
    public double[] inputs, weightedInputs, outputs;
    public double[] ForwardPass(double[] inputs) {
        this.inputs = inputs;
        for (int x = 0; x < outputNodeCount; ++x) {
            weightedInputs[x] = biases[x];
            for (int y = 0; y < inputNodeCount; ++y) {
                weightedInputs[x] += inputs[y] * weights[x, y];
            }
            outputs[x] = activationFunction(weightedInputs[x]);
        }
        return outputs;
    }
    public double BackwardPass(double[] targets) {
        double errors = 0, costGradient = 0;
        for (int x = 0; x < outputNodeCount; ++x) {
            costGradient = CostFunctions.MeanSquaredErrorDerivative(outputs[x], targets[x]) * derivativeActivationFunction(weightedInputs[x]);
            costGradientB[x] += costGradient;
            for(int y = 0; y < inputNodeCount; ++y) {
                costGradientW[x, y] += costGradient * inputs[y];
                weightGradients[x, y] = costGradient * weights[x, y];
            }
            errors += CostFunctions.MeanSquaredError(outputs[x], targets[x]);
        }
        return errors;
    }
    public void BackwardPass(Layer frontLayer) {
        double costGradient;
        for (int x = 0; x < outputNodeCount; ++x) {
            costGradient = 0;
            for (int z = 0; z < frontLayer.outputNodeCount; ++z) {
                costGradient += frontLayer.weightGradients[z, x];
            }
            costGradient *= derivativeActivationFunction(weightedInputs[x]);
            costGradientB[x] += costGradient;

            for (int y = 0; y < inputNodeCount; ++y) {
                costGradientW[x, y] += inputs[y] * costGradient;
                weightGradients[x, y] = costGradient * weights[x, y];
            }
        }
    }
    public void ApplyGradient(int batchSize, double learnRate) {
        for (int x = 0; x < outputNodeCount; x++) {
            biases[x] -= costGradientB[x] / batchSize * learnRate;
            costGradientB[x] = 0;
            for (int y = 0; y < inputNodeCount; y++) {
                weights[x, y] -= costGradientW[x, y] / batchSize * learnRate;
                costGradientW[x, y] = 0;
            }
        }
    }
}