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
}
public static class MatrixMath {
    public static double[,] Add(double[,] a, double[,] b) {
        if(a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
            throw new Exception("a and b dimension must be the same");
        
        int I = a.GetLength(0), J = a.GetLength(1);
        for(int i = 0; i < I; ++i) {
            for(int j = 0; j < J; ++j) {
                a[i, j] += b[i, j];
            }
        }
        return a;
    }

    public static double[,] Subtract(double[,] a, double[,] b) {
        if(a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
            throw new Exception("a and b dimension must be the same");
        
        int I = a.GetLength(0), J = a.GetLength(1);
        for(int i = 0; i < I; ++i) {
            for(int j = 0; j < J; ++j) {
                a[i, j] -= b[i, j];
            }
        }
        return a;
    }

    public static double[,] ScalarMultiply(double[,] a, double b) {
        int I = a.GetLength(0), J = a.GetLength(1);
        for(int i = 0; i < I; ++i) {
            for(int j = 0; j < J; ++j) {
                a[i, j] *= b;
            }
        }
        return a;
    }

    public static double[,] DotProduct(double[,] a, double[,] b) {
        if(a.GetLength(1) != b.GetLength(0))
            throw new Exception("a width must be equal to b height");

        int I = a.GetLength(0), J = b.GetLength(1), K = a.GetLength(1);
        double[,] c = new double[I, J];
        for(int i = 0; i < I; ++i) {
            for(int j = 0; j < J; ++j) {
                for(int k = 0; k < K; ++k) {
                    c[i, j] += a[i, k] * b[k, j];
                }
            }
        }
        return c;
    }

    public static double[,] Transpose(double[,] a) {
        int I = a.GetLength(0), J = a.GetLength(1);
        double[,] b = new double[a.GetLength(1), a.GetLength(0)];
        for(int i = 0; i < I; ++i) {
            for(int j = 0; j < J; ++j) {
                b[i, j] = a[j, i];
            }
        }
        return b;
    }
}
