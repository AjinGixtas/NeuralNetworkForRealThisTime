public class Layer {
    public readonly int inputNodeCount, outputNodeCount;
    public double[,] weights, costGradientW;
    public double[] biases, costGradientB;
    public readonly Func<double[], double[]> activationFunction, derivativeActivationFunction;
    public readonly Func<double, double, double>? derivativeErrorFunction;
    public Layer(int inputNodeCount, int outputNodeCount, Func<double[], double[]> activationFunction, Func<double[], double[]> derivativeActivationFunction, Func<double, double, double>? derivativeErrorFunction, Random rng) {
        this.inputNodeCount = inputNodeCount; this.outputNodeCount = outputNodeCount;
        this.activationFunction = activationFunction; this.derivativeActivationFunction = derivativeActivationFunction;
        weights = new double[outputNodeCount, inputNodeCount]; costGradientW = new double[outputNodeCount, inputNodeCount];
        biases = new double[outputNodeCount]; costGradientB = new double[outputNodeCount];
        WeightIntialization(rng);

        this.derivativeErrorFunction = derivativeErrorFunction;
    }
    public void ForwardPass(ref LayerDataWB layerDataWB, double[] inputs) {
        layerDataWB.inputs = inputs;
        for (int x = 0; x < outputNodeCount; ++x) {
            layerDataWB.weightedInputs[x] = biases[x];
            for (int y = 0; y < inputNodeCount; ++y) {
                layerDataWB.weightedInputs[x] += inputs[y] * weights[x, y];
            }
        }
        layerDataWB.outputs = activationFunction(layerDataWB.weightedInputs);
    }
    public void InitialBackwardPass(ref LayerDataWB layerDataWB, double[] targets) {
        double[] weightedInputDerivative = derivativeActivationFunction(layerDataWB.weightedInputs);
        for (int x = 0; x < outputNodeCount; ++x) {
            layerDataWB.costGradients[x] = derivativeErrorFunction(layerDataWB.outputs[x], targets[x]) * weightedInputDerivative[x];
            for (int y = 0; y < inputNodeCount; ++y) {
                layerDataWB.weightGradients[x, y] = layerDataWB.costGradients[x] * layerDataWB.inputs[y];
            }
        }
    }
    public void HiddenLayerBackwardPass(ref LayerDataWB layerDataWB, ref LayerDataWB frontLayerDataWB, int frontLayerOutputNodeCount) {
        Array.Clear(layerDataWB.costGradients, 0, layerDataWB.costGradients.Length);
        double[] weightedInputDerivative = derivativeActivationFunction(layerDataWB.weightedInputs); 
        for (int x = 0; x < outputNodeCount; ++x) {
            for (int z = 0; z < frontLayerOutputNodeCount; ++z) {
                layerDataWB.costGradients[x] += frontLayerDataWB.weightGradients[z, x];
            }
            layerDataWB.costGradients[x] *= weightedInputDerivative[x];

            for (int y = 0; y < inputNodeCount; ++y) {
                layerDataWB.weightGradients[x, y] = layerDataWB.costGradients[x] * layerDataWB.inputs[y];
            }
        }
    }
    public void AccumulateGradient(ref LayerDataWB layerDataWB) {
        for (int x = 0; x < outputNodeCount; ++x) {
            costGradientB[x] += layerDataWB.costGradients[x];
            for (int y = 0; y < inputNodeCount; ++y) {
                costGradientW[x, y] += layerDataWB.weightGradients[x, y];
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
        Array.Clear(costGradientB, 0, costGradientB.Length);
        Array.Clear(costGradientW, 0, costGradientW.Length);
    }
    public void WeightIntialization(Random rng) {
        for (int x = 0; x < outputNodeCount; x++) {
            for (int y = 0; y < inputNodeCount; y++) {
                weights[x, y] = RandomInNormalDistribution(rng, 0, 1) / Math.Sqrt(inputNodeCount);
            }
        }
        double RandomInNormalDistribution(Random rng, double mean, double standardDeviation) {
            return Math.Sqrt(-2.0 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2.0 * Math.PI * (1 - rng.NextDouble())) * standardDeviation + mean;
        }
        return;
    }
}
