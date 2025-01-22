<<<<<<< Updated upstream
﻿class NeuralNetwork {
=======
﻿using Newtonsoft.Json;
using System.Text.Json;

public class NeuralNetwork {
>>>>>>> Stashed changes
    public Layer[] layers; public int[] layerSizes;
    Random rng = new();
    NeuralNetworkDataWB[] neuralNetworkDataWBs = new NeuralNetworkDataWB[0];
    int batchSize = 0;
    public NeuralNetwork(int[] layerSizes) {
        this.layerSizes = layerSizes;
        layers = new Layer[layerSizes.Length - 1];
        for (int i = 0; i < layers.Length; i++) {
            layers[i] = new(layerSizes[i], layerSizes[i + 1], ActivationFunctions.Sigmoid, ActivationFunctions.SigmoidDerivative, rng);
        }
    }
<<<<<<< Updated upstream
    public void ForwardPass(double[] input) {
        double[] layerOutput = layers[0].ForwardPass(input);
=======
    public void ForwardPass(ref NeuralNetworkDataWB neuralNetworkDataWB, double[] input) {
        layers[0].ForwardPass(ref neuralNetworkDataWB.layerDataWBs[0], input);
>>>>>>> Stashed changes
        for (int i = 1; i < layers.Length; ++i) {
            layers[i].ForwardPass(ref neuralNetworkDataWB.layerDataWBs[i], neuralNetworkDataWB.layerDataWBs[i-1].outputs);
        }
    }
<<<<<<< Updated upstream
    public double BackwardPass(double[] input, double[] targets) {
        double error = layers[^1].BackwardPass(targets);
        // BACKWARD PROPAGATION!!!!!!!!!!!!!!!!!!!!!!!
=======
    public void BackwardPass(ref NeuralNetworkDataWB neuralNetworkDataWB, double[] input, double[] targets) {
        layers[^1].BackwardPass(ref neuralNetworkDataWB.layerDataWBs[^1], targets);
>>>>>>> Stashed changes
        for (int i = layers.Length - 2; i >= 0; --i) {
            layers[i].BackwardPass(ref neuralNetworkDataWB.layerDataWBs[i], ref neuralNetworkDataWB.layerDataWBs[i+1], layers[i+1].outputNodeCount);
        }
    }
    public void ApplyGradient(int batchSize, double learnRate) {
        for(int i = 0; i < layers.Length; ++i) {
            layers[i].ApplyGradient(batchSize, learnRate);
        }
    }
<<<<<<< Updated upstream
=======
    public void Train(int batchSize, double[][] inputs, double[][] targets, double learnRate, double regularizationRate) {
        double[] errors = new double[batchSize];
        if (batchSize != this.batchSize || neuralNetworkDataWBs == null) {
            neuralNetworkDataWBs = new NeuralNetworkDataWB[batchSize];
            for(int i = 0; i < neuralNetworkDataWBs.Length; ++i) neuralNetworkDataWBs[i] = new NeuralNetworkDataWB(this);
        }
        Parallel.For(0, batchSize, (i) => {
            ForwardPass(ref neuralNetworkDataWBs[i], inputs[i]);
            BackwardPass(ref neuralNetworkDataWBs[i], inputs[i], targets[i]);
        });
        //for (int i = 0; i < batchSize; ++i) {
        //    File.WriteAllText($"neuralNetworkData{index}-{i}.json", JsonConvert.SerializeObject(neuralNetworkDataWBs[i].layerDataWBs));
        //}
        //SerializableLayerDataWB[] serializableLayerDataWBs = new SerializableLayerDataWB[layers.Length];
        //for (int i = 0; i < layers.Length; ++i) {
        //    serializableLayerDataWBs[i] = new SerializableLayerDataWB(layers[i]);
        //}
        //File.WriteAllText($"neuralNetworkData{index}.json", JsonConvert.SerializeObject(serializableLayerDataWBs, Formatting.Indented));
        for (int j = 0; j < layers.Length; ++j) {
            for (int i = 0; i < batchSize; ++i) {
                layers[j].AccumulateGradient(ref neuralNetworkDataWBs[i].layerDataWBs[j]);
            }
            layers[j].ApplyGradient(batchSize, learnRate);
        }

        double totalError = 0.0, error = 0.0, biggestOutput = 0.0; int answerNeuralNetworkGotCorrect = 0, correctAnswer = 0, answer = 0;
        for(int i = 0; i < batchSize; ++i) {
            error = 0.0; biggestOutput = 0.0;
            for(int j = 0; j < targets[i].Length; ++j) { if (targets[i][j] == 1.0) { correctAnswer = j; } }
            for(int j = 0; j < neuralNetworkDataWBs[i].layerDataWBs[^1].outputs.Length; ++j) {
                error += CostFunctions.MeanSquaredError(neuralNetworkDataWBs[i].layerDataWBs[^1].outputs[j], targets[i][j]);
                if(biggestOutput < neuralNetworkDataWBs[i].layerDataWBs[^1].outputs[j]) { answer = j; biggestOutput = neuralNetworkDataWBs[i].layerDataWBs[^1].outputs[j]; }
            }
            if (correctAnswer == answer) answerNeuralNetworkGotCorrect += 1;
            totalError += error;
        }
        Console.WriteLine($"Mark: {answerNeuralNetworkGotCorrect}/{batchSize}, Error: {totalError}/{batchSize*neuralNetworkDataWBs[0].layerDataWBs[^1].outputs.Length/2.0}");
    }
}
public struct NeuralNetworkDataWB {
    public LayerDataWB[] layerDataWBs;
    public NeuralNetworkDataWB(NeuralNetwork neuralNetwork) {
        layerDataWBs = new LayerDataWB[neuralNetwork.layers.Length];
        for (int i = 0; i < layerDataWBs.Length; ++i) {
            layerDataWBs[i] = new LayerDataWB(neuralNetwork.layers[i]);
        }
    }
}
public struct LayerDataWB(Layer layer) {
    public double[] inputs = new double[layer.inputNodeCount], weightedInputs = new double[layer.outputNodeCount], outputs = new double[layer.outputNodeCount];
    public double[] costGradients = new double[layer.outputNodeCount];
    public double[,] weightGradients = new double[layer.outputNodeCount, layer.inputNodeCount];
}
public struct SerializableLayerDataWB(Layer layer) {
    public double[,] weights = layer.weights, costGradientW = layer.costGradientW;
    public double[] biases = layer.biases, costGradientB = layer.costGradientB;
>>>>>>> Stashed changes
}