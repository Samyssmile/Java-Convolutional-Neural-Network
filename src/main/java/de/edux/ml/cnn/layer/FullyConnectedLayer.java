package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.Tensor4D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

public class FullyConnectedLayer extends Layer {

    private static final Logger LOG = LoggerFactory.getLogger(FullyConnectedLayer.class);
    private final double learningRate;

    private Tensor4D weights;
    private Tensor4D biases;
    private int inputSize;
    private int outputSize;
    private Tensor4D lastInput;
    private Random random = new Random();

    public FullyConnectedLayer(int inputSize, int outputSize, double learningRate) {
        this.learningRate = learningRate;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.initializeWeights();

    }

    private void initializeWeights() {
        double stdDev = Math.sqrt(2.0 / inputSize);

        // Initialisieren der Gewichte und Biases mit He-Initialisierung
        this.weights = new Tensor4D(1, outputSize, inputSize, 1);
        this.biases = new Tensor4D(1, outputSize, 1, 1);

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                this.weights.getData()[0][i][j][0] = stdDev * random.nextGaussian();
            }
            this.biases.getData()[0][i][0][0] = 0;  // Biases werden oft mit 0 initialisiert
        }
    }


    @Override
    public Tensor4D forward(Tensor4D input) {
        LOG.debug("FullyConnectedLayer forward");
        this.lastInput = input;
        return input.multiply(weights).addBiases(biases);
    }


    @Override
    public Tensor4D backward(Tensor4D errorGradient) {
        return null;
    }
}