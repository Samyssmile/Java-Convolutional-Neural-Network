package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.functions.Channels;
import de.edux.ml.cnn.tensor.Tensor;

public class FullyConnectedLayer extends Layer {

  /**
   * @param inputSize
   * @param outputSize
   * @param channels 1 for grayscale, 3 for RGB
   */
  public FullyConnectedLayer(int inputSize, int outputSize, Channels channels) {}

  @Override
  public Tensor forward(Tensor input) {
    return null;
  }

  @Override
  public Tensor backward(Tensor input) {
    return null;
  }
}
