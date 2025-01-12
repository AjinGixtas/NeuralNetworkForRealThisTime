class Layer {
    public int inputNodeCount, outputNodeCount;
    public double[,] weights; public double[,] costGradientW;
    public double[] biases; public double[] costGradientB;
    public double[] accumulatedGradient;
    public Func<double, double> activationFunction, derivativeActivationFunction;
    public Layer(int inputNodeCount, int outputNodeCount, Func<double, double> activationFunction, Func<double, double> derivativeActivationFunction, Random rng) {
        this.inputNodeCount = inputNodeCount; this.outputNodeCount = outputNodeCount;
        this.activationFunction = activationFunction; this.derivativeActivationFunction = derivativeActivationFunction;
        weights = new double[outputNodeCount, inputNodeCount]; costGradientW = new double[outputNodeCount, inputNodeCount];
        biases = new double[outputNodeCount]; costGradientB = new double[outputNodeCount];

        for (int x = 0; x < inputNodeCount; x++) {
            for (int y = 0; y < outputNodeCount; y++) {
                weights[x, y] = rng.NextDouble();
                biases[y] = rng.NextDouble();
            }
        }
        inputs = new double[inputNodeCount]; weightedInputs = new double[outputNodeCount]; outputs = new double[outputNodeCount];
    }
    public double[] inputs, weightedInputs, outputs;
    public double[] CalculateOutput(double[] inputs) {
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
    // Overload for the output layer
    public double Backwardpass(double[] targets) {
        double errors = 0;
        for (int x = 0; x < outputNodeCount; ++x) {
            errors += CostFunctions.MeanSquaredError(outputs[x], targets[x]);
            accumulatedGradient[x] = CostFunctions.MeanSquaredErrorDerivative(outputs[x], targets[x]) * derivativeActivationFunction(weightedInputs[x]);
        }
        return errors; // We only use this for display purpose, nothing more :D
    }
    // Overload for the hidden layer
    public void Backwardpass(Layer frontLayer) {
        for (int x = 0; x < frontLayer.inputNodeCount; ++x) {
            accumulatedGradient[x] = 0;
            for (int y = 0; y < frontLayer.outputNodeCount; ++y) {
                accumulatedGradient[x] += frontLayer.weights[x, y] * frontLayer.accumulatedGradient[x];
            }
            accumulatedGradient[x] *= frontLayer.derivativeActivationFunction(frontLayer.weightedInputs[x]);
        }
    }
    public void ApplyGradient(double learnRate) {
        for (int x = 0; x < outputNodeCount; x++) {
            biases[x] -= costGradientB[x] * learnRate;
            costGradientB[x] = 0;
            for (int y = 0; y < inputNodeCount; y++) {
                weights[x, y] -= costGradientW[x, y] * learnRate;
                costGradientW[x, y] = 0;
            }
        }
    }
}