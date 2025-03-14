using System;

public static class PiMirror
{
    public static void GenerateModel() {
        NeuralNetwork feedNetwork = new([1, 3*16*16, 1]);
        int dataSize = 8000;
        double[][] inputs = new double[dataSize][], targets = new double[dataSize][];
        GenerateCustomNoiseData(dataSize, ref inputs, ref targets);
        int batchSize = 10;
        double[][] inputBatch = new double[batchSize][], targetBatch = new double[batchSize][];
        for(int i = 0; i < 5000; ++i) {
            GenerateNoiseTrainingBatch(batchSize, ref inputBatch, ref targetBatch, ref inputs, ref targets);
            feedNetwork.Train(batchSize, inputBatch, targetBatch, .75, 0, 0);
        }
        feedNetwork.ExportConfiguration("PiMirror_Model_Config.txt");
    }
    static (double[][], double[][]) GenerateCustomNoiseData(int length, ref double[][] inputs, ref double[][] targets) {
        Random random = new();
        for(int i = 0; i < length; ++i) { 
            inputs[i] =  [ Math.PI ]; 
            targets[i] = (random.Next() < .75 ? [Math.PI] : [ Math.PI + (random.NextDouble()-.5)*LearnrateEquation.CosineAnnealing(0.0, Math.PI, 50, i)*2.0 ]); 
        }
        return (inputs, targets);
    }
    static void GenerateNoiseTrainingBatch(int length, ref double[][] inputBatch, ref double[][] targetBatch, ref double[][] inputs, ref double[][] targets) {
        Random random = new Random();
        for (int i = 0; i < length; ++i) {
            int index = random.Next(0, inputs.Length);
            inputBatch[i] = inputs[index];
            targetBatch[i] = targets[index];
        }
    }
}
