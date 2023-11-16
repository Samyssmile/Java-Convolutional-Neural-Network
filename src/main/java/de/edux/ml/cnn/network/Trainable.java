package de.edux.ml.cnn.network;

import de.edux.ml.cnn.tensor.Tensor4D;

public interface Trainable {
    void train();
    double evaluate(Tensor4D[] input, Tensor4D[] target);
}
