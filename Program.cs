
using SkiaSharp;
public class Program {
    public static void Main() {
        MNISTRun();
    }
    public static void MNISTRun() {
        Console.WriteLine("Hello!");
        char[] displayChars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
        Random random = new();
        string imagesPath = "TrainingData/MNIST/train-images-idx3-ubyte.gz", labelsPath = "TrainingData/MNIST/train-labels-idx1-ubyte.gz";
        var (images, labels) = MNISTReader.LoadMNIST(imagesPath, labelsPath);
        NeuralNetwork neuralNetwork = new([784, 300, 10]);
        int batchSize = 5, iteration = 3000; double learnRate = .975, regularizationRate = .3;
        var (imageVectors, labelVectors) = GenerateMNISTTrainingData();
        var (input, target) = GenerateMNISTBatch(batchSize);

        for (int i = 0; i < iteration; ++i) {
            (input, target) = GenerateMNISTBatch(batchSize);
            learnRate = LearnrateEquation.CosineAnnealing(.001, .975, iteration, i);
            neuralNetwork.Train(batchSize, input, target, learnRate, regularizationRate);
            neuralNetwork.ApplyGradient(batchSize, learnRate);
        }

        (double[][], double[][]) GenerateMNISTTrainingData() {
            double[][] imageVectors = new double[images.Length][], labelVectors = new double[labels.Length][];
            for(int i = 0; i < images.Length; ++i) {
                imageVectors[i] = new double[28 * 28];
                for(int j = 0; j < images[i].Length; ++j) imageVectors[i][j] = images[i][j] / 255.0;
                labelVectors[i] = new double[10];
                labelVectors[i][labels[i]] = 1.0;
            }
            return (imageVectors, labelVectors);
        }
        (double[][], double[][]) GenerateMNISTBatch(int batchSize) {
            Random random = new();
            double[][] imageBatches = new double[batchSize][], labelBatches = new double[batchSize][];
            for(int i = 0; i < batchSize; ++i) {
                imageBatches[i] = imageVectors[random.Next(0, imageVectors.Length)];
                labelBatches[i] = labelVectors[random.Next(0, labelVectors.Length)];
            }
            return (imageBatches, labelBatches);
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