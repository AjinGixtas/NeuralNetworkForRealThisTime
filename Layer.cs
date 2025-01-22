<<<<<<< Updated upstream
class Layer {
    public int inputNodeCount, outputNodeCount;
    public double[,] weights, costGradientW; 
    public double[,] weightGradients;
=======
public class Layer {
    public readonly int inputNodeCount, outputNodeCount;
    public double[,] weights, costGradientW;
>>>>>>> Stashed changes
    public double[] biases, costGradientB;
    public readonly Func<double, double> activationFunction, derivativeActivationFunction;
    public Layer(int inputNodeCount, int outputNodeCount, Func<double, double> activationFunction, Func<double, double> derivativeActivationFunction, Random rng) {
        this.inputNodeCount = inputNodeCount; this.outputNodeCount = outputNodeCount;
        this.activationFunction = activationFunction; this.derivativeActivationFunction = derivativeActivationFunction;
        weights = new double[outputNodeCount, inputNodeCount]; costGradientW = new double[outputNodeCount, inputNodeCount];
        biases = new double[outputNodeCount]; costGradientB = new double[outputNodeCount];

        for (int x = 0; x < outputNodeCount; x++) {
            biases[x] = rng.NextDouble();
            for (int y = 0; y < inputNodeCount; y++) {
                weights[x, y] = rng.NextDouble();
            }
        }
<<<<<<< Updated upstream
        inputs = new double[inputNodeCount]; weightedInputs = new double[outputNodeCount]; outputs = new double[outputNodeCount]; 
        weightGradients = new double[outputNodeCount, inputNodeCount];
=======
>>>>>>> Stashed changes
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
        double sumSquaredLossDerivative = 0.0;
        for(int x = 0; x < outputNodeCount; ++x) {
            sumSquaredLossDerivative += CostFunctions.MeanSquaredErrorDerivative(layerDataWB.outputs[x], targets[x]);
        }
        sumSquaredLossDerivative /= outputNodeCount;
        for (int x = 0; x < outputNodeCount; ++x) {
<<<<<<< Updated upstream
            costGradient = CostFunctions.MeanSquaredErrorDerivative(outputs[x], targets[x]) * derivativeActivationFunction(weightedInputs[x]);
            costGradientB[x] += costGradient;
            for(int y = 0; y < inputNodeCount; ++y) {
                costGradientW[x, y] += costGradient * inputs[y];
                weightGradients[x, y] = costGradient * weights[x, y];
=======
            layerDataWB.costGradients[x] = sumSquaredLossDerivative * derivativeActivationFunction(layerDataWB.weightedInputs[x]);
            for (int y = 0; y < inputNodeCount; ++y) {
                layerDataWB.weightGradients[x, y] = layerDataWB.costGradients[x] * weights[x, y];
>>>>>>> Stashed changes
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
                layerDataWB.weightGradients[x, y] = layerDataWB.costGradients[x] * weights[x, y];
            }
        }
    }
    public void AccumulateGradient(ref LayerDataWB layerDataWB) {
        for (int x = 0; x < outputNodeCount; ++x) {
            costGradientB[x] += layerDataWB.costGradients[x];
            for (int y = 0; y < inputNodeCount; ++y) {
                costGradientW[x, y] += layerDataWB.costGradients[x] * layerDataWB.inputs[y];
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
}