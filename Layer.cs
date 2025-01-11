class Layer {
    public int inputNodeCount, outputNodeCount;
    public double[,] weights; public double[,] costGradientW;
    public double[] biases; public double[] costGradientB;
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
            costGradientB[x] = CostFunctions.MeanSquaredErrorDerivative(outputs[x], targets[x]) * derivativeActivationFunction(weightedInputs[x]);
            for (int y = 0; y < inputNodeCount; ++y) {
                costGradientW[x, y] = costGradientB[x] * outputs[x];
            }
        }
        return errors;
    }
    // Overload for the hidden layer
    public void Backwardpass(Layer frontLayer) {

    }
    public void ApplyGradient(double learnRate) {
        for (int x = 0; x < outputNodeCount; x++) {
            biases[x] -= costGradientB[x] * learnRate;
            for (int y = 0; y < inputNodeCount; y++) {
                weights[x, y] -= costGradientW[x, y] * learnRate;
                costGradientW[x, y] = 0;
            }
        }
    }
}
struct LayerData {
    double[,] weights; double[] bias; double[] outputs;
}