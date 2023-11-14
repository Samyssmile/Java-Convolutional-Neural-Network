package de.edux.ml.cnn.network;

import de.edux.ml.cnn.layer.Layer;
import de.edux.ml.cnn.functions.Optimizer;
import de.edux.ml.cnn.tensor.Tensor;

import java.util.List;

public class Network extends Layer implements Trainable {

  private final List<Layer> layers;
  private final Tensor[] trainImages;
  private final Tensor[] trainLabels;
  private final int epochs;
  private final Optimizer optimizer;
  private final double learningRate;
  private final int batchSize;

  public Network(
      List<Layer> layers,
      Tensor[] trainImages,
      Tensor[] trainLabels,
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

  public Tensor forward(Tensor inputs) {
    Tensor output = inputs;
    for (Layer layer : layers) {
      output = layer.forward(output);
    }
    return output;
  }

  @Override
  public Tensor backward(Tensor gradient) {
    Tensor backpropagatedGradient = gradient;
    for (int i = layers.size() - 1; i >= 0; i--) {
      backpropagatedGradient = layers.get(i).backward(backpropagatedGradient);
    }
    return backpropagatedGradient;
  }

  @Override
  public void train() {
    // Implementieren Sie die Trainingsmethode
    // Mini Batch Gradient Descent

    // TODO implement start training with mini batch gradient descent
    for(int i = 0; i < epochs; i++) {
      for(int j = 0; j < trainImages.length; j++) {
          forward(trainImages[j]);
      }
    }
  }

  @Override
  public double evaluate(Tensor[] input, Tensor[] target) {
    // Evaluation des Netzwerks
    return 0; // Implementieren Sie die Evaluierungsmethode
  }

}
