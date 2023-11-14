package de.edux.ml.cnn.tensor;

public class Tensor implements ITensor {

  private final float[][][][] data; // 4D-Array: [Batch][Höhe][Breite][Kanäle]

  public Tensor(int batch, int height, int width, int channels) {
    this.data = new float[batch][height][width][channels];
  }

  // Zugriffsmethode für das innere Array
  @Override
  public float[][][][] getData() {
    return data;
  }

  // Elementweise Addition
  @Override
  public Tensor add(Tensor other) {
    if (this.data.length != other.data.length
        || this.data[0].length != other.data[0].length
        || this.data[0][0].length != other.data[0][0].length
        || this.data[0][0][0].length != other.data[0][0][0].length) {
      throw new IllegalArgumentException("Dimensionen der Tensoren stimmen nicht überein");
    }

    Tensor result =
        new Tensor(data.length, data[0].length, data[0][0].length, data[0][0][0].length);
    for (int i = 0; i < data.length; i++) {
      for (int j = 0; j < data[i].length; j++) {
        for (int k = 0; k < data[i][j].length; k++) {
          for (int l = 0; l < data[i][j][k].length; l++) {
            result.data[i][j][k][l] = this.data[i][j][k][l] + other.data[i][j][k][l];
          }
        }
      }
    }

    return result;
  }

  // Skalar-Multiplikation
  @Override
  public Tensor multiply(float scalar) {
    Tensor result =
        new Tensor(data.length, data[0].length, data[0][0].length, data[0][0][0].length);

    for (int i = 0; i < data.length; i++) {
      for (int j = 0; j < data[i].length; j++) {
        for (int k = 0; k < data[i][j].length; k++) {
          for (int l = 0; l < data[i][j][k].length; l++) {
            result.data[i][j][k][l] = this.data[i][j][k][l] * scalar;
          }
        }
      }
    }

    return result;
  }

  // Faltungsfunktion
  @Override
  public Tensor convolve(Tensor kernel) {
    int kernelHeight = kernel.data[0].length;
    int kernelWidth = kernel.data[0][0].length;

    // Ausgabetensor hat die gleiche Batch-Größe und Kanalanzahl, aber verkleinerte Höhe und Breite
    Tensor result =
        new Tensor(
            data.length,
            data[0].length - kernelHeight + 1,
            data[0][0].length - kernelWidth + 1,
            data[0][0][0].length);

    for (int b = 0; b < data.length; b++) { // für jeden Batch
      for (int h = 0; h < data[0].length - kernelHeight + 1; h++) { // für jede Höhenposition
        for (int w = 0; w < data[0][0].length - kernelWidth + 1; w++) { // für jede Breitenposition
          for (int c = 0; c < data[0][0][0].length; c++) { // für jeden Kanal
            float sum = 0;
            for (int kh = 0; kh < kernelHeight; kh++) { // über die Höhe des Kernels
              for (int kw = 0; kw < kernelWidth; kw++) { // über die Breite des Kernels
                sum += this.data[b][h + kh][w + kw][c] * kernel.data[0][kh][kw][c];
              }
            }
            result.data[b][h][w][c] = sum;
          }
        }
      }
    }
    return result;
  }

  // MaxPooling-Operation
  @Override
  public Tensor maxPooling(int poolHeight, int poolWidth) {
    int newHeight = data[0].length / poolHeight;
    int newWidth = data[0][0].length / poolWidth;

    Tensor result = new Tensor(data.length, newHeight, newWidth, data[0][0][0].length);

    for (int b = 0; b < data.length; b++) { // für jeden Batch
      for (int h = 0; h < newHeight; h++) { // für jede neue Höhenposition
        for (int w = 0; w < newWidth; w++) { // für jede neue Breitenposition
          for (int c = 0; c < data[0][0][0].length; c++) { // für jeden Kanal
            float max = -Float.MAX_VALUE;
            for (int ph = 0; ph < poolHeight; ph++) { // über die Höhe des Pooling-Fensters
              for (int pw = 0; pw < poolWidth; pw++) { // über die Breite des Pooling-Fensters
                int currentHeight = h * poolHeight + ph;
                int currentWidth = w * poolWidth + pw;
                max = Math.max(max, data[b][currentHeight][currentWidth][c]);
              }
            }
            result.data[b][h][w][c] = max;
          }
        }
      }
    }
    return result;
  }

  public Tensor relu() {
    Tensor result =
        new Tensor(data.length, data[0].length, data[0][0].length, data[0][0][0].length);

    for (int i = 0; i < data.length; i++) {
      for (int j = 0; j < data[i].length; j++) {
        for (int k = 0; k < data[i][j].length; k++) {
          for (int l = 0; l < data[i][j][k].length; l++) {
            result.data[i][j][k][l] = Math.max(0, this.data[i][j][k][l]);
          }
        }
      }
    }

    return result;
  }

  // Flatten-Operation
  @Override
  public Tensor flatten() {
    int batchSize = data.length;
    int flattenedSize = data[0].length * data[0][0].length * data[0][0][0].length;

    Tensor result = new Tensor(batchSize, 1, 1, flattenedSize);

    for (int b = 0; b < batchSize; b++) {
      int flatIndex = 0;
      for (int h = 0; h < data[b].length; h++) {
        for (int w = 0; w < data[b][h].length; w++) {
          for (int c = 0; c < data[b][h][w].length; c++) {
            result.data[b][0][0][flatIndex++] = this.data[b][h][w][c];
          }
        }
      }
    }

    return result;
  }

  // Dense-Operation (Vollständig verbundene Schicht)
  @Override
  public Tensor dense(float[][] weights, float[] bias) {
    if (data[0][0][0].length != weights[0].length) {
      throw new IllegalArgumentException(
          "Die Anzahl der Merkmale im Tensor und die Anzahl der Gewichte müssen übereinstimmen.");
    }

    int batchSize = data.length;
    int outputSize = weights.length;
    Tensor result = new Tensor(batchSize, 1, 1, outputSize);

    for (int b = 0; b < batchSize; b++) {
      for (int i = 0; i < outputSize; i++) {
        float sum = bias[i];
        for (int j = 0; j < data[b][0][0].length; j++) {
          sum += data[b][0][0][j] * weights[i][j];
        }
        result.data[b][0][0][i] = sum;
      }
    }

    return result;
  }

  public static float crossEntropyLoss(Tensor predictions, Tensor labels) {
    if (predictions.data.length != labels.data.length
        || predictions.data[0][0][0].length != labels.data[0][0][0].length) {
      throw new IllegalArgumentException(
          "Die Dimensionen von Vorhersagen und Labels müssen übereinstimmen.");
    }

    float sumLoss = 0.0f;
    int numObservations = predictions.data.length;
    int numClasses = predictions.data[0][0][0].length;

    for (int i = 0; i < numObservations; i++) {
      for (int j = 0; j < numClasses; j++) {
        sumLoss += -labels.data[i][0][0][j] * Math.log(predictions.data[i][0][0][j]);
      }
    }

    return sumLoss / numObservations;
  }
}
