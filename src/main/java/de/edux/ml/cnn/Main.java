package de.edux.ml.cnn;

import de.edux.ml.cnn.functions.Channels;
import de.edux.ml.cnn.functions.Optimizer;
import de.edux.ml.cnn.layer.*;
import de.edux.ml.cnn.network.Network;
import de.edux.ml.cnn.network.NetworkBuilder;
import de.edux.ml.cnn.tensor.Tensor4D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;

public class Main {
    private static final Logger LOG = LoggerFactory.getLogger(Main.class);

    private static final int EPOCHS = 5;
    private static final double LEARNING_RATE = 0.01;
    private static final int BATCH_SIZE = 100;
    String trainImagePath = "mnist" + File.separator + "train-images-idx3-ubyte";
    String trainLabelPath = "mnist" + File.separator + "train-labels-idx1-ubyte";
    String testImagePath = "mnist" + File.separator + "t10k-images-idx3-ubyte";
    String testLabelPath = "mnist" + File.separator + "t10k-labels-idx1-ubyte";
    Tensor4D[] trainImages = loadImages(trainImagePath, 60000);
    Tensor4D[] trainLabels = loadLabels(trainLabelPath, 60000);

    Tensor4D[] testImages = loadImages(testImagePath, 10000);
    Tensor4D[] testLabels = loadLabels(testLabelPath, 10000);

    public static void main(String[] args) {

        new Main().run();
    }

    private void run() {
        LOG.debug("Start training");
        Network network =
                new NetworkBuilder()
                        .addLayer(new InputLayer(28, 28, 1))
                        .addLayer(new ConvolutionalLayer(8, 3, 1, 1, 3))
                        .addLayer(new FlattenLayer())
                        .build(trainImages, trainLabels, BATCH_SIZE, EPOCHS, Optimizer.SGD, LEARNING_RATE);

        // start training (batch size = 100, epochs = 5, optimizer = SGD, learning rate = 0.01)
        network.train();
        network.evaluate(testImages, testLabels);
    }

    private static Tensor4D[] loadImages(String imagePath, int limit) {
        try {
            FileInputStream fileStream = new FileInputStream(imagePath);
            BufferedInputStream bufferedStream = new BufferedInputStream(fileStream);

            // Skip the header
            bufferedStream.skip(16);

            int imageSize = 28 * 28;
            byte[] buffer = new byte[imageSize];

            Tensor4D[] images = new Tensor4D[limit];
            for (int i = 0; i < limit; i++) {
                if (bufferedStream.read(buffer) == -1) break;
                images[i] = byteArrayToTensor4D(buffer);
            }

            bufferedStream.close();
            return images;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private static Tensor4D byteArrayToTensor4D(byte[] array) {
        // Korrekt initialisieren: 1 Batch, 1 Kanal, 28 Reihen, 28 Spalten
        Tensor4D tensor = new Tensor4D(1, 1, 28, 28);
        for (int i = 0; i < array.length; i++) {
            tensor.getData()[0][0][i / 28][i % 28] = (array[i] & 0xFF) / 255.0f; // Normalisieren
        }
        return tensor;
    }

    private static Tensor4D[] loadLabels(String labelPath, int limit) {
        try {
            FileInputStream fileStream = new FileInputStream(labelPath);
            BufferedInputStream bufferedStream = new BufferedInputStream(fileStream);

            // Skip the header
            bufferedStream.skip(8);

            Tensor4D[] labels = new Tensor4D[limit];
            for (int i = 0; i < limit; i++) {
                int labelValue = bufferedStream.read();
                if (labelValue == -1) break;
                labels[i] = oneHotEncode(labelValue, 10); // 10 Klassen für MNIST
            }

            bufferedStream.close();
            return labels;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private static Tensor4D oneHotEncode(int value, int numClasses) {
        Tensor4D Tensor4D = new Tensor4D(1, 1, 1, numClasses);
        Tensor4D.getData()[0][0][0][value] = 1.0f;
        return Tensor4D;
    }
}
