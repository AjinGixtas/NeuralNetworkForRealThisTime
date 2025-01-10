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
        double layerOut;
    }
}