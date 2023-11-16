package de.edux.ml.cnn.layer;


import de.edux.ml.cnn.tensor.Tensor4D;

public abstract class Layer {

    public abstract Tensor4D forward(Tensor4D input);

    public abstract Tensor4D backward(Tensor4D input);

}
