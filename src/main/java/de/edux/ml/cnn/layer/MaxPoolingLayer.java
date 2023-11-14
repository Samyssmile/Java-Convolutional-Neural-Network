package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MaxPoolingLayer extends Layer {
    private static final Logger LOG = LoggerFactory.getLogger(MaxPoolingLayer.class);


    public MaxPoolingLayer(int poolSize, int stride) {

    }

    @Override
    public Tensor forward(Tensor input) {
        LOG.debug("MaxPoolingLayer forward");
        return null;
    }

    @Override
    public Tensor backward(Tensor input) {
        LOG.debug("MaxPoolingLayer backward");
        return null;
    }
}
