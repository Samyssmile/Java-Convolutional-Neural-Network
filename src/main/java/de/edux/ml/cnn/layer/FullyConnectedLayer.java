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
        this.weights.randomHE();
        this.biases = new Tensor4D(1, outputSize, 1, 1);

        for (int i = 0; i < outputSize; i++) {
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
        // Gradienten für Gewichte berechnen
        Tensor4D transposedLastInput = this.lastInput.transpose();
        Tensor4D weightGradient = transposedLastInput.multiply(errorGradient);

        Tensor4D averagedWeightGradient = weightGradient.averageOverBatches(); // Methode 'averageOverBatches()' muss implementiert sein

        Tensor4D biasGradient = errorGradient.sumOverBatchesForBiases();

        // Fehlergradienten für den vorherigen Layer berechnen
        Tensor4D prevLayerErrorGradient =  errorGradient.multiply(this.weights.transpose()); // Pseudocode

        // Optional: Aktualisierung der Gewichte und Biases hier (oder in einem separaten Schritt)
        this.weights = updateWeights(this.weights, weightGradient); // Pseudocode
        this.biases = updateBiases(this.biases, biasGradient); // Pseudocode*/

        return prevLayerErrorGradient;
    }

    private Tensor4D updateWeights(Tensor4D weights, Tensor4D weightGradient) {
        for (int i = 0; i < weights.getChannels(); i++) {
            for (int j = 0; j < weights.getRows(); j++) {
                for (int k = 0; k < weights.getCols(); k++) {
                    weights.getData()[0][i][j][k] -= learningRate * weightGradient.getData()[0][i][j][k];
                }
            }
        }
        return weights;
    }


    private Tensor4D updateBiases(Tensor4D biases, Tensor4D biasGradient) {
        for (int i = 0; i < biases.getChannels(); i++) {
            for (int j = 0; j < biases.getRows(); j++) {
                for (int k = 0; k < biases.getCols(); k++) {
                    biases.getData()[0][i][j][k] -= learningRate * biasGradient.getData()[0][i][j][k];
                }
            }
        }
        return biases;
    }




}