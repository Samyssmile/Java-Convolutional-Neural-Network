package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.Tensor4D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ThreadLocalRandom;

public class FlattenLayer extends Layer {
    private static final Logger LOG = LoggerFactory.getLogger(FlattenLayer.class);
    private int originalBatches;
    private int originalChannels;
    private int originalRows;
    private int originalCols;


    @Override
    public Tensor4D forward(Tensor4D input) {
        LOG.debug("FlattenLayer forward");

        originalBatches = input.getBatches();
        originalChannels = input.getChannels();
        originalRows = input.getRows();
        originalCols = input.getCols();

        return input.flatten();
    }

    @Override
    public Tensor4D backward(Tensor4D input) {
        LOG.debug("FlattenLayer backward");

        Tensor4D reshapedTensor = new Tensor4D(originalBatches, originalChannels, originalRows, originalCols);

        // Rückumwandlung des geflachten Tensors in seine ursprüngliche Form
        for (int batch = 0; batch < originalBatches; batch++) {
            for (int channel = 0; channel < originalChannels; channel++) {
                for (int row = 0; row < originalRows; row++) {
                    for (int col = 0; col < originalCols; col++) {
                        int flatIndex = channel * originalRows * originalCols + row * originalCols + col;
                        reshapedTensor.getData()[batch][channel][row][col] = input.getData()[batch][0][0][flatIndex];
                    }
                }
            }
        }

        return reshapedTensor;
    }

}


/*
 *
 * /**
 *
 * CNN use mini batches for training.
 * CNN use Categorical Cross Entropy as loss function.
 * CNN use Stochastic Gradient Descent as optimizer.
 * CNN use ReLu as activation function for hidden layers.
 * CNN use Softmax as activation function for the last layer.
 *//*

public class Tensor4D {
    private int batches;
    private int channels;
    private int rows;
    private int cols;
    private double[][][][] data;

    private final ThreadLocalRandom random = ThreadLocalRandom.current();

    public Tensor4D(int batches, int channels, int rows, int cols) {
        this.batches = batches;
        this.channels = channels;
        this.rows = rows;
        this.cols = cols;
        this.data = new double[batches][channels][rows][cols];
    }

    public de.edux.ml.cnn.tensor.Tensor4D flatten() {
        int flattenedRows = 1;
        int flattenedCols = channels * rows * cols;
        de.edux.ml.cnn.tensor.Tensor4D flattenedTensor = new de.edux.ml.cnn.tensor.Tensor4D(batches, 1, flattenedRows, flattenedCols);

        for (int batch = 0; batch < batches; batch++) {
            for (int channel = 0; channel < channels; channel++) {
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        int flatIndex = channel * rows * cols + row * cols + col;
                        flattenedTensor.data[batch][0][0][flatIndex] = this.data[batch][channel][row][col];
                    }
                }
            }
        }
        return flattenedTensor;
    }*/
