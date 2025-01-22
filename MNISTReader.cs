using System;
using System.IO.Compression;

public class MNISTReader {
    public static (byte[][] images, byte[] labels) LoadMNIST(string imagesPath, string labelsPath) {
        byte[][] images = ReadImages(imagesPath);
        byte[] labels = ReadLabels(labelsPath);
        return (images, labels);
    }
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
}
