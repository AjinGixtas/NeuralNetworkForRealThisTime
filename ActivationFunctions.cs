using MathNet.Numerics.LinearAlgebra;
public static class ActivationFunctions {
	public static double Sigmoid(double x) { return 1.0 / (1.0 + Math.Exp(-x)); }
	public static double SigmoidDerivative(double x) {
		double sigmoid = Sigmoid(x);
		return sigmoid * (1 - sigmoid);
	}
	public static double Tanh(double x) { return Math.Tanh(x); }
	public static double TanhDerivative(double x) {
		double tanh = Tanh(x);
		return 1 - tanh * tanh;
	}
	public static double ReLU(double x) { return Math.Max(0, x); }
	public static double ReLUDerivative(double x) { return x > 0 ? 1 : 0; }
    public static double[] Softmax(double[] x) {
        double[] softmax = new double[x.Length];
        double max = x[0];
        double sumExp = 0.0;
        for (int i = 1; i < x.Length; i++) if (x[i] > max) max = x[i];
        for (int i = 0; i < x.Length; i++) { softmax[i] = Math.Exp(x[i] - max); sumExp += softmax[i]; }
        for (int i = 0; i < x.Length; i++) softmax[i] /= sumExp;
        return softmax;
    }
    public static Matrix<double> ApplyFunction(Matrix<double> matrix, Func<double, double> func) { return matrix.Map(func); }
}