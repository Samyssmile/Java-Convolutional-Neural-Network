package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.functions.Channels;
import de.edux.ml.cnn.tensor.Tensor4D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FullyConnectedLayer extends Layer {

  private static final Logger LOG = LoggerFactory.getLogger(FullyConnectedLayer.class);

  public FullyConnectedLayer(int inputSize, int outputSize, Channels channels) {}
  @Override
  public Tensor4D forward(Tensor4D input) {
    LOG.debug("FullyConnectedLayer forward");
    return null;
  }

  @Override
  public Tensor4D backward(Tensor4D input) {
    LOG.debug("FullyConnectedLayer backward");
    return null;
  }
}
