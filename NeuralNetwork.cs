class NeuralNetwork {
    public Layer[] layers; public int[] layerSizes;
    Random rng = new();

    public NeuralNetwork(int[] layerSizes) {
        this.layerSizes = layerSizes;
        layers = new Layer[layerSizes.Length - 1];
        for (int i = 0; i < layers.Length; i++) {
            layers[i] = new(layerSizes[i], layerSizes[i + 1], ActivationFunctions.Sigmoid, ActivationFunctions.SigmoidDerivative, rng);
        }
    }
    public void ForwardPass(double[] input) {
        double[] layerOutput = layers[0].ForwardPass(input);
        for (int i = 1; i < layers.Length; ++i) {
            layerOutput = layers[i].ForwardPass(layerOutput);
        }
    }
    public double BackwardPass(double[] input, double[] targets) {
        double error = layers[^1].BackwardPass(targets);
        // BACKWARD PROPAGATION!!!!!!!!!!!!!!!!!!!!!!!
        for (int i = layers.Length - 2; i >= 0; --i) {
            layers[i].BackwardPass(layers[i + 1]);
        }
        return error;
    }
    public void ApplyGradient(int batchSize, double learnRate) {
        for(int i = 0; i < layers.Length; ++i) {
            layers[i].ApplyGradient(batchSize, learnRate);
        }
    }
}