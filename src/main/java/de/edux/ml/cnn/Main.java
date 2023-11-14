package de.edux.ml.cnn;

import de.edux.ml.cnn.functions.Channels;
import de.edux.ml.cnn.functions.Optimizer;
import de.edux.ml.cnn.layer.*;
import de.edux.ml.cnn.network.Network;
import de.edux.ml.cnn.network.NetworkBuilder;
import de.edux.ml.cnn.tensor.Tensor;
import java.io.*;

public class Main {
  private static final int EPOCHS = 5;
  private static final double LEARNING_RATE = 0.01;
  private static final int BATCH_SIZE = 100;

  public static void main(String[] args) {
    String trainImagePath = "mnist" + File.separator + "train-images-idx3-ubyte";
    String trainLabelPath = "mnist" + File.separator + "train-labels-idx1-ubyte";
    String testImagePath = "mnist" + File.separator + "t10k-images-idx3-ubyte";
    String testLabelPath = "mnist" + File.separator + "t10k-labels-idx1-ubyte";

    Tensor[] trainImages = loadImages(trainImagePath, 60000);
    Tensor[] trainLabels = loadLabels(trainLabelPath, 60000);

    Tensor[] testImages = loadImages(testImagePath, 10000);
    Tensor[] testLabels = loadLabels(testLabelPath, 10000);

    System.out.println("MNIST data loaded.");

    Network network =
        new NetworkBuilder()
            .addLayer(new ConvolutionalLayer(8, 3, 1, 1, Channels.GRAY))
            .addLayer(new ReLuLayer())
            .addLayer(new MaxPoolingLayer(2, 2))
            .addLayer(new FlattenLayer())
            .addLayer(new FullyConnectedLayer(1568, 10, Channels.GRAY))
            .addLayer(new SoftmaxLayer())
            .build(trainImages, trainLabels, BATCH_SIZE, EPOCHS, Optimizer.SGD, LEARNING_RATE);

    // start training (batch size = 100, epochs = 5, optimizer = SGD, learning rate = 0.01)
    network.train();
    network.evaluate(testImages, testLabels);
  }

  private static Tensor[] loadImages(String imagePath, int limit) {
    try {
      FileInputStream fileStream = new FileInputStream(imagePath);
      BufferedInputStream bufferedStream = new BufferedInputStream(fileStream);

      // Skip the header
      bufferedStream.skip(16);

      int imageSize = 28 * 28;
      byte[] buffer = new byte[imageSize];

      Tensor[] images = new Tensor[limit];
      for (int i = 0; i < limit; i++) {
        if (bufferedStream.read(buffer) == -1) break;
        images[i] = byteArrayToTensor(buffer);
      }

      bufferedStream.close();
      return images;
    } catch (IOException e) {
      e.printStackTrace();
      return null;
    }
  }

  private static Tensor byteArrayToTensor(byte[] array) {
    Tensor tensor = new Tensor(1, 28, 28, 1); // Für MNIST: 1 Batch, 28x28 Größe, 1 Kanal
    for (int i = 0; i < array.length; i++) {
      tensor.getData()[0][i / 28][i % 28][0] = (array[i] & 0xFF) / 255.0f; // Normalisieren
    }
    return tensor;
  }

  private static Tensor[] loadLabels(String labelPath, int limit) {
    try {
      FileInputStream fileStream = new FileInputStream(labelPath);
      BufferedInputStream bufferedStream = new BufferedInputStream(fileStream);

      // Skip the header
      bufferedStream.skip(8);

      Tensor[] labels = new Tensor[limit];
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

  private static Tensor oneHotEncode(int value, int numClasses) {
    Tensor tensor = new Tensor(1, 1, 1, numClasses);
    tensor.getData()[0][0][0][value] = 1.0f;
    return tensor;
  }
}
