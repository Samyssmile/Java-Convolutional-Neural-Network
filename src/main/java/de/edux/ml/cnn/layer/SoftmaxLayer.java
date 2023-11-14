package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SoftmaxLayer extends Layer{
    private static final Logger LOG = LoggerFactory.getLogger(MaxPoolingLayer.class);

    @Override
    public Tensor forward(Tensor input) {
        LOG.debug("SoftmaxLayer forward");
        return null;
    }

    @Override
    public Tensor backward(Tensor input) {
        LOG.debug("SoftmaxLayer backward");
        return null;
    }
}
