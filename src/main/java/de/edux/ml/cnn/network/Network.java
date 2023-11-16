package de.edux.ml.cnn.network;

import de.edux.ml.cnn.functions.Optimizer;
import de.edux.ml.cnn.layer.Layer;
import de.edux.ml.cnn.tensor.Tensor4D;

import java.util.List;

/**
 * CNN use mini batches for training.
 * CNN use Categorical Cross Entropy as loss function.
 * CNN use Stochastic Gradient Descent as optimizer.
 * CNN use ReLu as activation function for hidden layers.
 * CNN use Softmax as activation function for the last layer.
 */
public class Network extends Layer implements Trainable {

    private final List<Layer> layers;
    private final Tensor4D[] trainImages;
    private final Tensor4D[] trainLabels;
    private final int epochs;
    private final Optimizer optimizer;
    private final double learningRate;
    private final int batchSize;

    public Network(
            List<Layer> layers,
            Tensor4D[] trainImages,
            Tensor4D[] trainLabels,
            int batchSize,
            int epochs,
            Optimizer optimizer,

            double learningRate) {
        this.trainImages = trainImages;
        this.trainLabels = trainLabels;
        this.batchSize = batchSize;
        this.epochs = epochs;
        this.optimizer = optimizer;
        this.learningRate = learningRate;
        this.layers = layers;
    }

    public Tensor4D forward(Tensor4D inputs) {
        Tensor4D output = inputs;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    @Override
    public Tensor4D backward(Tensor4D gradient) {
        Tensor4D backpropagatedGradient = gradient;
        for (int i = layers.size() - 1; i >= 0; i--) {
            backpropagatedGradient = layers.get(i).backward(backpropagatedGradient);
        }
        return backpropagatedGradient;
    }

/*
    public class Tensor4D {
        private int batches;
        private int channels;
        private int rows;
        private int cols;
        private double[][][][] data;

        public Tensor4D(int batches, int channels, int rows, int cols) {
            this.batches = batches;
            this.channels = channels;
            this.rows = rows;
            this.cols = cols;
            this.data = new double[batches][channels][rows][cols];
        }
*/

    @Override
    public void train() {
        for (int i = 0; i < epochs; i++) {
            for (int j = 0; j < trainImages.length; j += batchSize) {
                Tensor4D batchImages = new Tensor4D(batchSize, 1, 28, 28);
                Tensor4D batchLabels = new Tensor4D(batchSize, 1, 1, 10);
                for (int k = 0; k < batchSize; k++) {
                    batchImages.getData()[k] = trainImages[j + k].getData()[0];
                    batchLabels.getData()[k] = trainLabels[j + k].getData()[0];
                }
                Tensor4D output = forward(batchImages);
                Tensor4D input = backward(output);
            }
        }
    }

    @Override
    public double evaluate(Tensor4D[] input, Tensor4D[] target) {
        // Evaluation des Netzwerks
        return 0; // Implementieren Sie die Evaluierungsmethode
    }

}
