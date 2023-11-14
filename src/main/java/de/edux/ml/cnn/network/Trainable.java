package de.edux.ml.cnn.network;

import de.edux.ml.cnn.tensor.Tensor;

public interface Trainable {
    void train();

    double evaluate(Tensor[] input, Tensor[] target);
}
