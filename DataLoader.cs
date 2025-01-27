using System;
using System.IO.Compression;

public class DataLoader {
    public static void GenerateRandomDataBatch(int batchSize, ref double[][] inputsBatchContainer, ref double[][] targetsBatchContainer, ref double[][] inputs, ref double[][] targets) {
        Random random = new();
        inputsBatchContainer = new double[batchSize][]; targetsBatchContainer = new double[batchSize][];
        for (int i = 0; i < batchSize; ++i) {
            int index = random.Next(0, inputs.Length);
            inputsBatchContainer[i] = inputs[index];
            targetsBatchContainer[i] = targets[index];
        }
    }
    public static void GenerateDeterminedDataBatch(int batchSize, ref double[][] inputsBatchContainer, ref double[][] targetsBatchContainer, ref double[][] inputs, ref double[][] targets, int lowerBoundIndex) { // [lowerBound, lowerBound + batchSize)
        batchSize = Math.Min(batchSize, inputs.Length - lowerBoundIndex);
        inputsBatchContainer = new double[batchSize][]; targetsBatchContainer = new double[batchSize][];
        for (int i = 0; i < batchSize; ++i) {
            inputsBatchContainer[i] = inputs[lowerBoundIndex + i]; // Fix out-of-bound array error. We need a better way to divide and generate test batch. Late B)
            targetsBatchContainer[i] = targets[lowerBoundIndex + i];
        }
    }
    
    public static (double[][] trainInputs, double[][] trainTargets, double[][] testInputs, double[][] testTargets) SplitData(double[][] inputs, double[][] targets, double testSize = 0.2) {
        int totalSize = inputs.Length, testCount = (int)(totalSize * testSize), trainCount = totalSize - testCount;
        Random random = new();
        int[] indices = Enumerable.Range(0, totalSize).OrderBy(_ => random.Next()).ToArray();
        double[][] trainInputs = new double[trainCount][], trainTargets = new double[trainCount][], testInputs = new double[testCount][], testTargets = new double[testCount][];

        for (int i = 0; i < trainCount; ++i) {
            trainInputs[i] = inputs[indices[i]];
            trainTargets[i] = targets[indices[i]];
        }

        for (int i = 0; i < testCount; ++i) {
            testInputs[i] = inputs[indices[trainCount + i]];
            testTargets[i] = targets[indices[trainCount + i]];
        }

        return (trainInputs, trainTargets, testInputs, testTargets);
    }
    
    public static (double[][] images, double[][] labels) LoadMNIST(string imagesPath, string labelsPath) {
        byte[][] images = ReadImages(imagesPath);
        byte[] labels = ReadLabels(labelsPath);
        return GenerateMNISTTrainingData();
        static byte[][] ReadImages(string filePath) {
            FileStream fileStream = new(filePath, FileMode.Open, FileAccess.Read);
            GZipStream gZipStream = new(fileStream, CompressionMode.Decompress);
            BinaryReader binaryReader = new BinaryReader(gZipStream);
            int magicNumber = ReadInt32BigEndian(binaryReader);
            if (magicNumber != 2051) throw new InvalidDataException("Invalid MNIST image file.");
            int numImages = ReadInt32BigEndian(binaryReader), numRows = ReadInt32BigEndian(binaryReader), numCols = ReadInt32BigEndian(binaryReader);
            byte[][] images = new byte[numImages][];
            for (int i = 0; i < numImages; i++) images[i] = binaryReader.ReadBytes(numRows * numCols);
            return images;
        }
        static byte[] ReadLabels(string filePath) {
            FileStream fileStream = new(filePath, FileMode.Open, FileAccess.Read);
            GZipStream gZipStream = new(fileStream, CompressionMode.Decompress);
            BinaryReader binaryReader = new BinaryReader(gZipStream);
            int magicNumber = ReadInt32BigEndian(binaryReader);
            if (magicNumber != 2049) throw new InvalidDataException("Invalid MNIST image file.");
            int numLabels = ReadInt32BigEndian(binaryReader);
            return binaryReader.ReadBytes(numLabels);

        }
        static int ReadInt32BigEndian(BinaryReader binaryReader) {
            byte[] bytes = binaryReader.ReadBytes(4);
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
        (double[][], double[][]) GenerateMNISTTrainingData() {
            double[][] imageVectors = new double[images.Length][], labelVectors = new double[labels.Length][];
            for (int i = 0; i < images.Length; ++i) {
                imageVectors[i] = new double[28 * 28];
                for (int j = 0; j < images[i].Length; ++j) imageVectors[i][j] = images[i][j] / 255.0;
                labelVectors[i] = new double[10];
                labelVectors[i][labels[i]] = 1.0;
            }
            return (imageVectors, labelVectors);
        }
    }
    public static (double[][], double[][]) LoadIrisDataset(string filePath) {
        double[][] inputs = new double[150][], targets = new double[150][];
        int index = 0;
        using (var reader = new StreamReader(filePath)) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 50; ++j) {
                    inputs[index] = new double[4]; targets[index] = new double[3];
                    string[] line = reader.ReadLine().Split(',');
                    for (int k = 0; k < 4; ++k) inputs[index][k] = double.Parse(line[k]);
                    targets[index][i] = 1.0;
                    ++index;
                }
            }
        }
        return (inputs, targets);
    }
    public static (double[][] inputs, double[][] targets) LoadBreastCancerWisconsinDataset(string filePath) {
        var inputs = new List<double[]>();
        var targets = new List<double[]>();

        using (var reader = new StreamReader(filePath)) {
            string line;
            line = reader.ReadLine();
            while ((line = reader.ReadLine()) != null) {
                var data = line.Split(',');
                double[] features = data.Skip(2).Select(double.Parse).ToArray(); // Skip 'id' and 'diagnosis'
                double[] label = new double[2];
                label[data[1] == "M" ? 1 : 0] = 1.0;
                inputs.Add(features);
                targets.Add(label);
            }
        }

        return (inputs.ToArray(), targets.ToArray());
    }
}
