using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
class Layer {
	public Matrix<double> weights, outputs;
	public Vector<double> biases;
	Func<double, double> activationFunction;
	public Layer(int inputSize, int neuronCount, Func<double, double> activationFunction) {
		Random random = new Random();
		weights = Matrix<double>.Build.Dense(inputSize, neuronCount, (inputSize, neuronCount) => ContinuousUniform.Sample(-.5, .5));
		biases = Vector<double>.Build.Dense(neuronCount, 0);
		this.activationFunction = activationFunction;
	}
	public Matrix<double> ForwardPass(Matrix<double> inputs) {
		outputs = inputs.Multiply(weights); 
		for(int i = 0; i < outputs.RowCount; ++i) outputs.SetRow(i, outputs.Row(i).Add(biases));
		outputs = ActivationFunctions.ApplyFunction(outputs, activationFunction);
		return outputs;
	}
}