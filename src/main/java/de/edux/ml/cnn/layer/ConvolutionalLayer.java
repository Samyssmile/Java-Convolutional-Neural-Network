package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.functions.Channels;
import de.edux.ml.cnn.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ConvolutionalLayer extends Layer {

  private static final Logger LOG = LoggerFactory.getLogger(ConvolutionalLayer.class);

  public ConvolutionalLayer(
      int numberOfFilters, int filterSize, int stride, int padding, Channels channels) {}

  @Override
  public Tensor forward(Tensor input) {
    LOG.debug("ConvolutionalLayer forward");
    return null;
  }

  @Override
  public Tensor backward(Tensor input) {
    LOG.debug("ConvolutionalLayer backward");
    return null;
  }
}
