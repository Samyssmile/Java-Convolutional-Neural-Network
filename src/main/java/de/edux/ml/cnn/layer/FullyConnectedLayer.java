package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.functions.Channels;
import de.edux.ml.cnn.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FullyConnectedLayer extends Layer {

  private static final Logger LOG = LoggerFactory.getLogger(FullyConnectedLayer.class);


  /**
   * @param inputSize
   * @param outputSize
   * @param channels 1 for grayscale, 3 for RGB
   */
  public FullyConnectedLayer(int inputSize, int outputSize, Channels channels) {}

  @Override
  public Tensor forward(Tensor input) {
    LOG.debug("FullyConnectedLayer forward");
    return null;
  }

  @Override
  public Tensor backward(Tensor input) {
    LOG.debug("FullyConnectedLayer backward");
    return null;
  }
}
