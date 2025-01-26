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
        inputsBatchContainer = new double[batchSize][]; targetsBatchContainer = new double[batchSize][];
        for (int i = 0; i < batchSize; ++i) {
            inputsBatchContainer[i] = inputs[lowerBoundIndex + i];
            targetsBatchContainer[i] = targets[lowerBoundIndex + i];
        }
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
}
