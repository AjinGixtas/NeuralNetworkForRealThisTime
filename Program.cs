
using SkiaSharp;
public class Program {
    public static void Main() {
        Console.WriteLine("Hello! Let's begin >:J");
        BreastCancerWisconsinRun();
        //MNISTRun();
        //IrisRun();
    }
    public static void MNISTRun() {
        Console.WriteLine("MNIST number data set");
        var (trainInputs, trainTargets) = DataLoader.LoadMNIST("TrainingData/MNIST/train-images-idx3-ubyte.gz", "TrainingData/MNIST/train-labels-idx1-ubyte.gz");
        var (testInputs, testTargets) = DataLoader.LoadMNIST("TestingData/MNIST/t10k-images-idx3-ubyte.gz", "TestingData/MNIST/t10k-labels-idx1-ubyte.gz");
        NeuralNetwork neuralNetwork = new([784, 300, 10]);
        double[][] inputBatch = [], targetBatch = []; 

        int batchSize = 8, iteration = 100; double learnRate = .975, regularizationRate = .3;
        for (int i = 0; i < iteration; ++i) {
            DataLoader.GenerateRandomDataBatch(batchSize, ref inputBatch, ref targetBatch, ref trainInputs, ref trainTargets);
            learnRate = LearnrateEquation.CosineAnnealing(.005, .975, iteration, i);
            neuralNetwork.Train(batchSize, inputBatch, targetBatch, learnRate, regularizationRate);
        }
        iteration = 10_000 / batchSize;
        int totalMark = 0; double totalError = 0.0;
        for(int i = 0; i < iteration; ++i) {
            DataLoader.GenerateDeterminedDataBatch(batchSize, ref inputBatch, ref targetBatch, ref testInputs, ref testTargets, i * batchSize);
            
            var (error, mark) = neuralNetwork.Test(batchSize, inputBatch, targetBatch);
            totalMark += mark; totalError += error;
        }
        Console.WriteLine($"Mark: {totalMark}/{10_000}, error: {totalError}/{10_000}");
    }
    public static void IrisRun() {
        Console.WriteLine("Iris data set");
        var (inputs, targets) = DataLoader.LoadIrisDataset("TrainingData/Iris/iris.data");
        NeuralNetwork neuralNetwork = new([4, 3]);
        int batchSize = 8, iteration = 5000; double learnRate = 0.0, regularizationRate = 0;
        
        double[][] inputBatch = [], targetBatch = []; 
        for (int i = 0; i < iteration; ++i) {
            learnRate = LearnrateEquation.CosineAnnealing(.005, .975, iteration, i);
            DataLoader.GenerateRandomDataBatch(batchSize, ref inputBatch, ref targetBatch, ref inputs, ref targets);
            neuralNetwork.Train(batchSize, inputBatch, targetBatch, learnRate, regularizationRate);
        }
    }
    public static void BreastCancerWisconsinRun() {
        Console.WriteLine("Breast cancer data set");
        var (inputs, targets) = DataLoader.LoadBreastCancerWisconsinDataset("TrainingData/BreastCancerWisconsin/data.csv");
        var (trainInputs, trainTargets, testInputs, testTargets) = DataLoader.SplitData(inputs, targets, .15);
        NeuralNetwork neuralNetwork = new([30, 40, 50, 35, 2]);
        double[][] inputBatch = [], targetBatch = [];
        int batchSize = 8, iteration = 100; double learnRate = .975, regularizationRate = .3;
        for (int i = 0; i < iteration; ++i) {
            DataLoader.GenerateRandomDataBatch(batchSize, ref inputBatch, ref targetBatch, ref trainInputs, ref trainTargets);
            learnRate = LearnrateEquation.CosineAnnealing(.005, .975, iteration, i);
            neuralNetwork.Train(batchSize, inputBatch, targetBatch, learnRate, regularizationRate);
        }

        iteration = trainInputs.Length / batchSize;
        int totalMark = 0; double totalError = 0.0;
        for(int i = 0; i < iteration; ++i) {
            DataLoader.GenerateDeterminedDataBatch(batchSize, ref inputBatch, ref targetBatch, ref testInputs, ref testTargets, i * batchSize);
            var (error, mark) = neuralNetwork.Test(batchSize, inputBatch, targetBatch);
            totalMark += mark; totalError += error;
        }
        Console.WriteLine($"Mark: {totalMark}/{trainTargets.Length}, error: {totalError}/{trainTargets.Length * 2}");
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