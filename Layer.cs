using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
class Layer {
	int batchSize, inputSize, neuronCount, x, y, k;
	double[,] weights, preActivationOutputs, postActivationoutputs; public double[] biases;
	Func<double, double> activationFunction, activationFunctionDerivative;

	public Layer(int batchSize, int neuronCount, int inputSize, Func<double, double> activationFunction, Func<double, double> activationFunctionDerivative) {
		Random random = new Random();
		this.batchSize = batchSize; this.neuronCount = neuronCount; this.inputSize = inputSize;
		this.activationFunction = activationFunction; this.activationFunctionDerivative = activationFunctionDerivative
		this.weights = new double[inputSize][neuronCount];
		this.preActivationOutputs = new double[batchSize][neuronCount];
		this.postActivationoutputs = new double[batchSize][neuronCount];
		this.biases  = new double[neuronCount];
		for(y = 0; y < neuronCount; ++y) {
			for(x = 0; x < inputSize; ++x) {
				weights[x, y] = random.Next(-1.0, 1.0);
			}
		}
		for(y = 0; y < neuronCount; ++y) {
			biases[y] = random.Next(-1.0, 1.0);
		}
	}

	public double[,] ForwardPass(double[,] inputs) {
		MatrixMath.
		for(y = 0; y < neuronCount; ++y) {
			for(x = 0; x < batchSize; ++x) {
				preActivationOutputs[x, y] = biases[y]
				for(k = 0; k < inputSize; ++k) { preActivationOutputs += inputs[x, k] * weights[k, y]; }
				postActivationoutputs[x, y] = activationFunction(preActivationOutputs[x, y]);
			}
		}
		return postActivationoutputs;
	}

	double[,] preActivationLossDerivatives, weightsLossDerivatives, biasesLossDerivatives;
	// lossDerivative is a neuronCount * batchSize matrix.
	public void Backpropogation(double[,] postActivationLossDerivatives) {
		for(y = 0; y < batchSize; ++y) {
			for(x = 0; x < batchSize; ++x) {
				preActivationLossDerivatives[x, y] = postActivationLossDerivatives[x, y] * activationFunctionDerivative(preActivationOutputs[x, y]);
			}
		}
		for(y = 0; y < batchSize; ++y) {
			for(x = 0; x < batchSize; ++x) {
				
			}
		}
	}
}