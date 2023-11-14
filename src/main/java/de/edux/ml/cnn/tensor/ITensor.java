package de.edux.ml.cnn.tensor;

public interface ITensor {
    // Zugriffsmethode für das innere Array
    float[][][][] getData();

    // Elementweise Addition
    Tensor add(Tensor other);

    // Skalar-Multiplikation
    Tensor multiply(float scalar);

    // Faltungsfunktion
    Tensor convolve(Tensor kernel);

    // MaxPooling-Operation
    Tensor maxPooling(int poolHeight, int poolWidth);

    // Flatten-Operation
    Tensor flatten();

    // Dense-Operation (Vollständig verbundene Schicht)
    Tensor dense(float[][] weights, float[] bias);
}
