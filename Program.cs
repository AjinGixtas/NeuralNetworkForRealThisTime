
public class Program {
    public static void Main() {
        NeuralNetwork neuralNetwork = new([2, 3, 3, 2]);
        KeyValuePair<double[], double[]> inputAndTarget = GenerateTrainingData();
        neuralNetwork.ForwardPass(inputAndTarget.Key);
        Console.WriteLine(neuralNetwork.BackwardPass(inputAndTarget.Key, inputAndTarget.Value));
        int batchSize = 5, epoch = 10000; double learnRate = 1.0 / 3.0;
        for (int i = 0; i < epoch; ++i) {
            for (int j = 0; j < batchSize; ++j) {
                inputAndTarget = GenerateTrainingData();
                neuralNetwork.ForwardPass(inputAndTarget.Key);
                neuralNetwork.BackwardPass(inputAndTarget.Key, inputAndTarget.Value);
            }
            neuralNetwork.ApplyGradient(batchSize, learnRate);
        }
        inputAndTarget = GenerateTrainingData();
        neuralNetwork.ForwardPass(inputAndTarget.Key);
        Console.WriteLine(neuralNetwork.BackwardPass(inputAndTarget.Key, inputAndTarget.Value));
    }
    public static KeyValuePair<double[], double[]> GenerateTrainingData() {
        Random rng = new Random();
        KeyValuePair<double[], double[]> output = new KeyValuePair<double[], double[]>();
        double a = rng.NextDouble(), b = rng.NextDouble();
        double _a = (5 * Math.Pow(a, 2) + 2 * Math.Pow(b, 2)) < 1 ? 1.0 : 0.0;
        double _b = 1.0 - _a;
        output = new KeyValuePair<double[], double[]>([a,b], [_a, _b]);
        return output;
    }
}