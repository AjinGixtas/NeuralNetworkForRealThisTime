using System;

public static class PiMirror
{
    public static void Run() {
        NeuralNetwork neuralNetwork = new([256, 64, 10]);

    }
    public static (double[][], double[][]) GenerateTrainingData(string filePath, int batchSize, int inputSize) {
        double[][] input = new double[batchSize][], target = new double[batchSize][];
        Random random = new();
        using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read))
        using (StreamReader reader = new StreamReader(fs)) {
            int startingIndex, endingIndex; char[] buffer = new char[inputSize];
            for (int i = 0; i < batchSize; ++i) {
                input[i] = new double[inputSize];
                startingIndex = random.Next(1_000_000_000-inputSize); endingIndex = startingIndex + inputSize;
                fs.Seek(startingIndex, SeekOrigin.Begin);
                for(int j = startingIndex; j < endingIndex; ++j) {

                }
            }
            fs.Seek(1000000, SeekOrigin.Begin);

            char[] buffer = new char[100];
            reader.Read(buffer, 0, buffer.Length);
            Console.WriteLine(new string(buffer));
        }
        return (input, target);
    }
}
