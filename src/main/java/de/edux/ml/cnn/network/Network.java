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

    @Override
    public void train(Tensor4D[] inputs, Tensor4D[] targets) {
        for (int i = 0; i < epochs; i++) {
            double totalLoss = 0.0;

            for (int j = 0; j < trainImages.length; j += batchSize) {
                Tensor4D batchImages = new Tensor4D(batchSize, 1, 28, 28);
                Tensor4D batchLabels = new Tensor4D(batchSize, 1, 1, 10);
                // Vorwärtsdurchlauf
                Tensor4D output = forward(batchImages);

                // Berechnung des Verlustes und des Gradienten
                Tensor4D lossGradient = calculateLossGradient(output, batchLabels); // Diese Methode müssen Sie implementieren

                // Rückwärtsdurchlauf
                backward(lossGradient);
                double batchLoss = calculateLoss(output, batchLabels);
                totalLoss += batchLoss;

            }

            double accurca = evaluate(inputs, targets);

            // Mittleren Verlust für die Epoche ausgeben
            double averageLoss = totalLoss / (trainImages.length / batchSize);
            System.out.println("Epoch " + i + ", Loss: " + totalLoss + ", Accuracy: " + accurca+"%");
        }
    }

    private double calculateLoss(Tensor4D output, Tensor4D batchLabels) {
        double totalLoss = 0.0;

        for (int batch = 0; batch < output.getBatches(); batch++) {
            for (int channel = 0; channel < output.getChannels(); channel++) {
                // Die Kreuzentropie für jeden Output berechnen
                double predicted = output.getData()[batch][channel][0][0];
                double actual = batchLabels.getData()[batch][0][0][channel];
                totalLoss += -actual * Math.log(predicted); // Kreuzentropie-Formel
            }
        }

        return totalLoss / output.getBatches(); // Mittelwert des Verlustes über alle Batches
    }


    private Tensor4D calculateLossGradient(Tensor4D output, Tensor4D batchLabels) {
        Tensor4D lossGradient = new Tensor4D(output.getBatches(), output.getChannels(), output.getRows(), output.getCols());

        for (int batch = 0; batch < output.getBatches(); batch++) {
            for (int channel = 0; channel < output.getChannels(); channel++) {
                // Beachten Sie, dass die Labels in der 'cols'-Dimension von 'batchLabels' liegen
                lossGradient.getData()[batch][channel][0][0] = output.getData()[batch][channel][0][0] - batchLabels.getData()[batch][0][0][channel];
            }
        }

        return lossGradient;
    }


    private int calculateCorrectPredictions(Tensor4D output, Tensor4D target) {
        int correct = 0;

        for (int batch = 0; batch < output.getBatches(); batch++) {
            int predictedLabel = argMax(output.getData()[batch]); // Implementieren Sie die argMax-Methode
            int actualLabel = argMax(target.getData()[batch]); // Implementieren Sie die argMax-Methode

            if (predictedLabel == actualLabel) {
                correct++;
            }
        }

        return correct;
    }

    private int argMax(double[][][] dataArray) {
        int maxIndex = 0;
        double max = dataArray[0][0][0];

        for (int i = 0; i < dataArray.length; i++) {
            if (dataArray[i][0][0] > max) {
                max = dataArray[i][0][0];
                maxIndex = i;
            }
        }

        return maxIndex;
    }



    @Override
    public double evaluate(Tensor4D[] inputs, Tensor4D[] targets) {
        int correctPredictions = 0;

        for (int i = 0; i < inputs.length; i++) {
            Tensor4D output = forward(inputs[i]);
            correctPredictions += calculateCorrectPredictions(output, targets[i]); // Diese Methode müssen Sie implementieren
        }

        return ((double) correctPredictions / inputs.length)*100;
    }

    /**
     * Führt eine Vorhersage für ein einzelnes Bild durch.
     * @param image Das Bild, für das eine Vorhersage gemacht werden soll.
     * @return Die Vorhersage des Netzwerks.
     */
    public Tensor4D predict(Tensor4D image) {
        Tensor4D output = image;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }



}
