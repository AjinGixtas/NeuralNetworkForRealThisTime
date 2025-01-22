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
}
public static class CostFunctions {
    public static double MeanSquaredError(double output, double target) { return .5 * Math.Pow(output - target, 2.0); } // Not used anywhere (so far), just help with code comprehension)
    public static double MeanSquaredErrorDerivative(double output, double target) { return output - target; }
}
public static class LearnrateEquation {
    public static double ExponentialDecay(double initialLearnRate, double decayRate, int iteration) { return initialLearnRate * Math.Pow(double.E, -decayRate * iteration); }
    public static double StepDecay(double initialLearnRate, double decayFactor, double decayCycle, int iteration) { return initialLearnRate * Math.Pow(decayFactor, Math.Floor(iteration / decayCycle)); }
    public static double LinearDecay(double initialLearnRate, double decayRate, int iteration) { return initialLearnRate - decayRate * iteration; }
    public static double CosineAnnealing(double minLearnRate, double maxLearnRate, int maxIteration, int iteration) { return minLearnRate + .5 * (maxLearnRate - minLearnRate) * (1 + Math.Cos(iteration * Math.PI / maxIteration)); }
}
public static class MatrixMath {
    public static void Add(ref double[,] a, double[,] b) {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
            throw new Exception("a and b dimensions must be the same");

        int I = a.GetLength(0), J = a.GetLength(1);
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                a[i, j] += b[i, j];
            }
        }
    }

    public static void Subtract(ref double[,] a, double[,] b) {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
            throw new Exception("a and b dimensions must be the same");

        int I = a.GetLength(0), J = a.GetLength(1);
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                a[i, j] -= b[i, j];
            }
        }
    }

    public static void ScalarMultiply(ref double[,] a, double b) {
        int I = a.GetLength(0), J = a.GetLength(1);
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                a[i, j] *= b;
            }
        }
    }

    public static void DotProduct(double[,] a, double[,] b, ref double[,] c) {
        if (a.GetLength(1) != b.GetLength(0))
            throw new Exception("a width must be equal to b height");

        int I = a.GetLength(0), J = b.GetLength(1), K = a.GetLength(1);
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                c[i, j] = 0;
                for (int k = 0; k < K; ++k) {
                    c[i, j] += a[i, k] * b[k, j];
                }
            }
        }
    }

    public static void Transpose(double[,] a, ref double[,] b) {
        int I = a.GetLength(0), J = a.GetLength(1);
        if (I != b.GetLength(1) || J != b.GetLength(0)) throw new Exception("a width must be equal to b height");
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                b[j, i] = a[i, j];
            }
        }
    }
}
