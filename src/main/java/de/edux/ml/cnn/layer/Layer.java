package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.Tensor;

public abstract class Layer {

    public abstract Tensor forward(Tensor input);

    public abstract Tensor backward(Tensor input);

}
