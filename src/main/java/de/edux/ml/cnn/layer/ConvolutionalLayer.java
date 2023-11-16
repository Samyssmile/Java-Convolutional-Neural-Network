package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.functions.Channels;
import de.edux.ml.cnn.tensor.Tensor4D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ConvolutionalLayer extends Layer {

  private static final Logger LOG = LoggerFactory.getLogger(ConvolutionalLayer.class);

  public ConvolutionalLayer(
      int numberOfFilters, int filterSize, int stride, int padding, Channels channels) {}

  @Override
  public Tensor4D forward(Tensor4D input) {
    LOG.debug("ConvolutionalLayer forward");
    return null;
  }

  @Override
  public Tensor4D backward(Tensor4D input) {
    LOG.debug("ConvolutionalLayer backward");
    return null;
  }
}
