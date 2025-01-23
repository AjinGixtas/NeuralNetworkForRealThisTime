public class Layer {
    public readonly int inputNodeCount, outputNodeCount;
    public double[,] weights, costGradientW;
    public double[] biases, costGradientB;
    public readonly Func<double, double> activationFunction, derivativeActivationFunction;
    public Layer(int inputNodeCount, int outputNodeCount, Func<double, double> activationFunction, Func<double, double> derivativeActivationFunction, Random rng) {
        this.inputNodeCount = inputNodeCount; this.outputNodeCount = outputNodeCount;
        this.activationFunction = activationFunction; this.derivativeActivationFunction = derivativeActivationFunction;
        weights = new double[outputNodeCount, inputNodeCount]; costGradientW = new double[outputNodeCount, inputNodeCount];
        biases = new double[outputNodeCount]; costGradientB = new double[outputNodeCount];
        WeightIntialization(rng);
    }
    public void ForwardPass(ref LayerDataWB layerDataWB, double[] inputs) {
        layerDataWB.inputs = inputs;
        for (int x = 0; x < outputNodeCount; ++x) {
            layerDataWB.weightedInputs[x] = biases[x];
            for (int y = 0; y < inputNodeCount; ++y) {
                layerDataWB.weightedInputs[x] += inputs[y] * weights[x, y];
            }
            layerDataWB.outputs[x] = activationFunction(layerDataWB.weightedInputs[x]);
        }
    }
    public void BackwardPass(ref LayerDataWB layerDataWB, double[] targets) {
        for (int x = 0; x < outputNodeCount; ++x) {
            layerDataWB.costGradients[x] = CostFunctions.MeanSquaredErrorDerivative(layerDataWB.outputs[x], targets[x]) * derivativeActivationFunction(layerDataWB.weightedInputs[x]);
            for (int y = 0; y < inputNodeCount; ++y) {
                layerDataWB.weightGradients[x, y] = layerDataWB.costGradients[x] * layerDataWB.inputs[y];
            }
        }
    }
    public void BackwardPass(ref LayerDataWB layerDataWB, ref LayerDataWB frontLayerDataWB, int frontLayerOutputNodeCount) {
        Array.Clear(layerDataWB.costGradients, 0, layerDataWB.costGradients.Length);
        for (int x = 0; x < outputNodeCount; ++x) {
            for (int z = 0; z < frontLayerOutputNodeCount; ++z) {
                layerDataWB.costGradients[x] += frontLayerDataWB.weightGradients[z, x];
            }
            layerDataWB.costGradients[x] *= derivativeActivationFunction(layerDataWB.weightedInputs[x]);

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
            double x1 = 1 - rng.NextDouble();
            double x2 = 1 - rng.NextDouble();

            double y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
            return y1 * standardDeviation + mean;
        }
        return;
        weights = new double[,] {
            { .1, .2, .3, .4},
            { .15, .25, .35, .45 },
            { .45, .35, .25, .15 }
        };
        return;
    }
}