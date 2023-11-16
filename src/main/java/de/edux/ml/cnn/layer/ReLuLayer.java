package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.Tensor4D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ReLuLayer extends Layer{
    private static final Logger LOG = LoggerFactory.getLogger(MaxPoolingLayer.class);
    @Override
    public Tensor4D forward(Tensor4D input) {
        LOG.debug("ReLuLayer forward");
        return null;
    }

    @Override
    public Tensor4D backward(Tensor4D input) {
        LOG.debug("ReLuLayer backward");
        return null;
    }
}
