package de.edux.ml.cnn.network;

import de.edux.ml.cnn.layer.Layer;
import de.edux.ml.cnn.functions.Optimizer;
import de.edux.ml.cnn.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {

  private List<Layer> layers;

  public NetworkBuilder() {
    this.layers = new ArrayList<>();
  }

  public NetworkBuilder addLayer(Layer layer) {
    this.layers.add(layer);
    return this;
  }

  public Network build(
      Tensor[] trainImages,
      Tensor[] trainLabels,
      int batchSize,
      int epochs,
      Optimizer optimizer,
      double learningRate) {

    return new Network(
        layers, trainImages, trainLabels, batchSize, epochs, optimizer, learningRate);
  }
}
