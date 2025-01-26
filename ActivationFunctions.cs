using System;

public static class ActivationFunctions {
    public static double[] Sigmoid(double[] input) {
        double[] output = new double[input.Length];
        for (int i = 0; i < input.Length; ++i)
            output[i] = 1.0 / (1.0 + Math.Exp(-input[i]));
        return output;
    }
    public static double[] SigmoidDerivative(double[] input) {
        double[] sigmoid = Sigmoid(input);
        double[] output = new double[input.Length];
        for (int i = 0; i < input.Length; ++i)
            output[i] = sigmoid[i] * (1 - sigmoid[i]);
        return output;
    }

    // Hyperbolic Tangent (Tanh) function
    public static double[] Tanh(double[] input) {
        double[] output = new double[input.Length];
        for (int i = 0; i < input.Length; ++i)
            output[i] = Math.Tanh(input[i]);
        return output;
    }

    // Derivative of Tanh function
    public static double[] TanhDerivative(double[] input) {
        double[] tanh = Tanh(input);
        double[] output = new double[input.Length];
        for (int i = 0; i < input.Length; ++i)
            output[i] = 1 - (tanh[i] * tanh[i]);
        return output;
    }

    // Rectified Linear Unit (ReLU) function
    public static double[] ReLU(double[] input) {
        double[] output = new double[input.Length];
        for (int i = 0; i < input.Length; ++i)
            output[i] = Math.Max(0, input[i]);
        return output;
    }

    // Derivative of ReLU function
    public static double[] ReLUDerivative(double[] input) {
        double[] output = new double[input.Length];
        for (int i = 0; i < input.Length; ++i)
            output[i] = input[i] > 0 ? 1 : 0;
        return output;
    }

    // Leaky ReLU function
    public static double[] LeakyReLU(double[] input, double alpha = 0.01) {
        double[] output = new double[input.Length];
        for (int i = 0; i < input.Length; ++i)
            output[i] = input[i] > 0 ? input[i] : alpha * input[i];
        return output;
    }

    // Derivative of Leaky ReLU function
    public static double[] LeakyReLUDerivative(double[] input, double alpha = 0.01) {
        double[] output = new double[input.Length];
        for (int i = 0; i < input.Length; ++i)
            output[i] = input[i] > 0 ? 1 : alpha;
        return output;
    }

    // Softmax function
    public static double[] Softmax(double[] input) {
        double[] output = new double[input.Length];
        double max = double.MinValue;
        double sum = 0.0;

        for (int i = 0; i < input.Length; ++i)
            if (input[i] > max) max = input[i];

        for (int i = 0; i < input.Length; ++i) {
            output[i] = Math.Exp(input[i] - max);
            sum += output[i];
        }

        for (int i = 0; i < input.Length; ++i)
            output[i] /= sum;

        return output;
    }
    // Softmax derivative (NEVER use this damn thing, I can't understand the way its output map to, so don't even try -_-)
    public static double[][] SoftmaxDerivative(double[] input) {
        int length = input.Length;
        double[][] derivative = new double[length][];
        double[] softmaxOutput = Softmax(input);
        for (int i = 0; i < length; ++i) {
            derivative[i] = new double[length];

            for (int j = 0; j < length; ++j) {
                if (i == j) {
                    derivative[i][j] = softmaxOutput[i] * (1 - softmaxOutput[i]);
                } else {
                    derivative[i][j] = -softmaxOutput[i] * softmaxOutput[j];
                }
            }
        }
        return derivative;
    }
    // Use this thing in Softmax + Cross-entropy for the derivative activation function, thank me later :3
    public static double[] ConstantDerivative(double[] input) {
        double[] output = Enumerable.Repeat(1.0, input.Length).ToArray();
        return output;
    }
}
public static class CostFunctions {
    public static double MeanSquaredError(double output, double target) {
        return 0.5 * Math.Pow(output - target, 2.0);
    }
    public static double MeanSquaredErrorDerivative(double output, double target) {
        return output - target;
    }

    public static double MeanAbsoluteError(double output, double target) {
        return Math.Abs(output - target);
    }
    public static double MeanAbsoluteErrorDerivative(double output, double target) {
        return output > target ? 1 : -1;
    }

    public static double HuberLoss(double output, double target, double delta = 1.0) {
        double error = output - target;
        return Math.Abs(error) <= delta ? 0.5 * error * error : delta * (Math.Abs(error) - 0.5 * delta);
    }
    public static double HuberLossDerivative(double output, double target, double delta = 1.0) {
        double error = output - target;
        return Math.Abs(error) <= delta ? error : delta * Math.Sign(error);
    }
    public static double CrossEntropy(double predicted, double target) {
        return -target * Math.Log(predicted) - (1 - target) * Math.Log(1 - predicted);
    }
    public static double CrossEntropyDerivative(double predicted, double target) {
        return (predicted - target) / (predicted * (1 - predicted));
    }
    // If you want to know how it work, google it :)
    public static double SoftmaxAndCrossEntropyDerivative(double predicted, double target) {
        return predicted - target;
    }
}

public static class LearnrateEquation {
    public static double ExponentialDecay(double initialLearnRate, double decayRate, int iteration) {
        return initialLearnRate * Math.Pow(Math.E, -decayRate * iteration);
    }

    public static double StepDecay(double initialLearnRate, double decayFactor, double decayCycle, int iteration) {
        return initialLearnRate * Math.Pow(decayFactor, Math.Floor(iteration / decayCycle));
    }

    public static double LinearDecay(double initialLearnRate, double decayRate, int iteration) {
        return Math.Max(0, initialLearnRate - decayRate * iteration);
    }

    public static double CosineAnnealing(double minLearnRate, double maxLearnRate, int maxIteration, int iteration) {
        return minLearnRate + 0.5 * (maxLearnRate - minLearnRate) * (1 + Math.Cos(iteration * Math.PI / maxIteration));
    }

    public static double PolynomialDecay(double initialLearnRate, double decayRate, int iteration, int power) {
        return initialLearnRate / Math.Pow(1 + decayRate * iteration, power);
    }

    public static double CyclicalLearningRate(double baseLearnRate, double maxLearnRate, int stepSize, int iteration) {
        double cycle = Math.Floor(1 + iteration / (2.0 * stepSize));
        double x = Math.Abs(iteration / stepSize - 2 * cycle + 1);
        return baseLearnRate + (maxLearnRate - baseLearnRate) * Math.Max(0, (1 - x));
    }
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
