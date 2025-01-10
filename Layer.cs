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
        input = new double[inputNodeCount]; weightedInput = new double[outputNodeCount]; output = new double[outputNodeCount];
    }
    public double[] input, weightedInput, output;
    public double[] CalculateOutput(double[] input) {
        this.input = input;
        for (int x = 0; x < outputNodeCount; ++x) {
            weightedInput[x] = biases[x];
            for (int y = 0; y < inputNodeCount; ++y) {
                weightedInput[x] += input[y] * weights[x, y];
            }
            output[x] = activationFunction(weightedInput[x]);
        }
        return output;
    }
}