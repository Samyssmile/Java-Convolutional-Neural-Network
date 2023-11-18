package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.Tensor4D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * CNN use mini batches for training.
 * CNN use Categorical Cross Entropy as loss function.
 * CNN use Stochastic Gradient Descent as optimizer.
 * CNN use ReLu as activation function for hidden layers.
 * CNN use Softmax as activation function for the last layer.
 */
public class ConvolutionalLayer extends Layer {
    private final Tensor4D filters;
    private int numberOfFilters;
    private int filterSize;
    private int stride;
    private int padding;
    private static final Logger LOG = LoggerFactory.getLogger(ConvolutionalLayer.class);
    private Tensor4D originalInput;

    public ConvolutionalLayer(
            int numberOfFilters, int filterSize, int stride, int padding, int channels) {
        this.numberOfFilters = numberOfFilters;
        this.filterSize = filterSize;
        this.stride = stride;
        this.padding = padding;

        this.filters = new Tensor4D(numberOfFilters, channels, filterSize, filterSize);
        initializeFilters(this.filters);
    }

    private void initializeFilters(Tensor4D filters) {
        filters.randomHE();
    }

    @Override
    public Tensor4D forward(Tensor4D input) {
        LOG.debug("ConvolutionalLayer forward");
        this.originalInput = input;
        return input.convolve(filters, stride, padding);
    }

    @Override
    public Tensor4D backward(Tensor4D inputGradient) {
        LOG.debug("ConvolutionalLayer backward");

        // Gradienten in Bezug auf die Filter berechnen
        // Diese Berechnung hängt von Ihrer spezifischen Implementierung der Faltung ab
        Tensor4D gradientWrtFilters = calculateGradientWrtFilters(inputGradient, originalInput);

        // Gradienten in Bezug auf den Eingabetensor berechnen
        // Dies erfordert eine volle Faltung (Full Convolution) zwischen inputGradient und den Filtern
        Tensor4D gradientWrtInput = calculateGradientWrtInput(inputGradient, filters);

        // Optional: Aktualisieren Sie die Filter basierend auf gradientWrtFilters

        return gradientWrtInput;
    }

    private Tensor4D calculateGradientWrtFilters(Tensor4D inputGradient, Tensor4D originalInput) {
        // Die Dimensionen für den Gradiententensor der Filter
        int gradFilterBatches = this.numberOfFilters; // Anzahl der Filter
        int gradFilterChannels = originalInput.getChannels(); // Anzahl der Kanäle im ursprünglichen Eingabetensor
        int gradFilterRows = this.filterSize;
        int gradFilterCols = this.filterSize;

        // Initialisieren des Gradiententensors für die Filter
        Tensor4D gradFilters = new Tensor4D(gradFilterBatches, gradFilterChannels, gradFilterRows, gradFilterCols);

        // Durchführen der Gradientenberechnung für jeden Filter
        for (int filterNum = 0; filterNum < gradFilterBatches; filterNum++) {
            for (int channel = 0; channel < gradFilterChannels; channel++) {
                for (int row = 0; row < gradFilterRows; row++) {
                    for (int col = 0; col < gradFilterCols; col++) {
                        double gradientSum = 0.0;

                        // Iterieren über den Eingabegradienten und den ursprünglichen Eingabetensor
                        for (int batch = 0; batch < originalInput.getBatches(); batch++) {
                            for (int i = 0; i < inputGradient.getRows(); i++) {
                                for (int j = 0; j < inputGradient.getCols(); j++) {
                                    // Berechnen der Position im Eingabetensor
                                    int inputRow = i * this.stride + row;
                                    int inputCol = j * this.stride + col;

                                    // Überprüfen der Grenzen
                                    if (inputRow < originalInput.getRows() && inputCol < originalInput.getCols()) {
                                        gradientSum += originalInput.getData()[batch][channel][inputRow][inputCol] *
                                                inputGradient.getData()[batch][filterNum][i][j];
                                    }
                                }
                            }
                        }

                        gradFilters.getData()[filterNum][channel][row][col] = gradientSum;
                    }
                }
            }
        }

        return gradFilters;
    }


    private Tensor4D calculateGradientWrtInput(Tensor4D inputGradient, Tensor4D filters) {
        // Berechnen der Dimensionen für den Gradiententensor des Eingabetensors
        int gradInputBatches = inputGradient.getBatches();
        int gradInputChannels = filters.getChannels(); // Anzahl der Kanäle in den Filtern
        int gradInputRows = (inputGradient.getRows() - 1) * this.stride -2 * padding + filters.getRows();
        int gradInputCols = (inputGradient.getCols() - 1) * this.stride -2*padding+ filters.getCols();

        // Initialisieren des Gradiententensors für den Eingabetensor
        Tensor4D gradInput = new Tensor4D(gradInputBatches, gradInputChannels, gradInputRows, gradInputCols);

        // Durchführen der vollen Faltung zwischen inputGradient und den Filtern
        for (int batch = 0; batch < gradInputBatches; batch++) {
            for (int channel = 0; channel < gradInputChannels; channel++) {
                for (int row = 0; row < gradInputRows; row++) {
                    for (int col = 0; col < gradInputCols; col++) {
                        double gradientSum = 0.0;

                        // Iterieren über die Filter und den Eingabegradienten
                        for (int filterNum = 0; filterNum < filters.getBatches(); filterNum++) {
                            for (int i = 0; i < filters.getRows(); i++) {
                                for (int j = 0; j < filters.getCols(); j++) {
                                    // Berechnen der Position im Eingabegradienten
                                    int inputRow = row - i;
                                    int inputCol = col - j;

                                    // Überprüfen der Grenzen
                                    if (inputRow >= 0 && inputRow < inputGradient.getRows() && inputCol >= 0 && inputCol < inputGradient.getCols()) {
                                        gradientSum += filters.getData()[filterNum][channel][i][j] *
                                                inputGradient.getData()[batch][filterNum][inputRow][inputCol];
                                    }
                                }
                            }
                        }

                        gradInput.getData()[batch][channel][row][col] = gradientSum;
                    }
                }
            }
        }

        return gradInput;
    }
}


/**
 *
 * CNN use mini batches for training.
 * CNN use Categorical Cross Entropy as loss function.
 * CNN use Stochastic Gradient Descent as optimizer.
 * CNN use ReLu as activation function for hidden layers.
 * CNN use Softmax as activation function for the last layer.
 */
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

    public Tensor4D convolve(Tensor4D filter, int stride, int padding) {

        return output;
    }*/
