﻿
using SkiaSharp;
public class Program {
    public static void Main() {
        Console.WriteLine("Hello! Let's begin >:J");
        //MNISTRun();
        IrisRun();
    }
    public static void MNISTRun() {
        var (inputs, targets) = DataLoader.LoadMNIST("TrainingData/MNIST/train-images-idx3-ubyte.gz", "TrainingData/MNIST/train-labels-idx1-ubyte.gz");
        Random random = new(); NeuralNetwork neuralNetwork = new([784, 300, 10]);
        double[][] inputBatch = [], targetBatch = []; 

        int batchSize = 5, iteration = 4000; double learnRate = .975, regularizationRate = .3;
        for (int i = 0; i < iteration; ++i) {
            DataLoader.GenerateDataBatch(batchSize, ref inputBatch, ref targetBatch, ref inputs, ref targets);
            learnRate = LearnrateEquation.CosineAnnealing(.005, .975, iteration, i);
            neuralNetwork.Train(batchSize, inputBatch, targetBatch, learnRate, regularizationRate);
            neuralNetwork.ApplyGradient(batchSize, learnRate);
        }
    }
    public static void IrisRun() {
        var (inputs, targets) = DataLoader.LoadIrisDataset("TrainingData/Iris/iris.data");
        Random random = new(); NeuralNetwork neuralNetwork = new([4, 3]);
        int batchSize = 5, iteration = 4000; double learnRate = .8, regularizationRate = 0;
        
        double[][] inputBatch = [], targetBatch = []; 
        for (int i = 0; i < iteration; ++i) {
            DataLoader.GenerateDataBatch(batchSize, ref inputBatch, ref targetBatch, ref inputs, ref targets);
            neuralNetwork.Train(batchSize, inputBatch, targetBatch, learnRate, regularizationRate);
            neuralNetwork.ApplyGradient(batchSize, learnRate);
        }
    }
}
public class HeatmapGenerator {
    public static void CreateHeatmap(double[,] data, string filePath) {
        int width = data.GetLength(1);
        int height = data.GetLength(0);

        using (var bitmap = new SKBitmap(width, height)) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double value = data[y, x];
                    SKColor color = ValueToColor(value);
                    bitmap.SetPixel(x, y, color);
                }
            }

            using (var image = SKImage.FromBitmap(bitmap))
            using (var dataStream = image.Encode(SKEncodedImageFormat.Png, 100))
            using (var fileStream = File.OpenWrite(filePath)) {
                dataStream.SaveTo(fileStream);
            }
        }
    }

    private static SKColor ValueToColor(double value) {
        // Map value to a gradient (e.g., blue to red)
        byte r = (byte)(value * 255);
        byte b = (byte)(255 - r);
       return new SKColor(r, 0, b);
    }
}