package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.Tensor4D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DropoutLayer extends Layer {
    private static final Logger LOG = LoggerFactory.getLogger(DropoutLayer.class);


    @Override
    public Tensor4D forward(Tensor4D input) {
        return null;
    }

    @Override
    public Tensor4D backward(Tensor4D input) {
        return null;
    }
}
