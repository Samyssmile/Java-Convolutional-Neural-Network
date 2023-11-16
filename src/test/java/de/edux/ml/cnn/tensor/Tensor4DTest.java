package de.edux.ml.cnn.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Tensor4DTest {

  @Test
  void testConvolveSimpleCase() {
    // Erstellen Sie einen einfachen Tensor4D
    Tensor4D inputTensor = new Tensor4D(1, 1, 3, 3); // 1 Batch, 1 Channel, 3x3 Größe
    inputTensor.setData(new double[][][][]{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});

    // Erstellen Sie einen einfachen Filter
    Tensor4D filter = new Tensor4D(1, 1, 2, 2); // 1 Output Channel, 1 Input Channel, 2x2 Größe
    filter.setData(new double[][][][]{{{{1, 0}, {0, -1}}}});

    // Convolve ohne Padding, Stride 1
    Tensor4D result = inputTensor.convolve(filter, 1, 0);

    // Erwartetes Ergebnis
    double[][][][] expected = {{{{-4, -4}, {-4, -4}}}};

    assertArrayEquals(expected, result.getData(), "Convolution result is not as expected");
  }
}
