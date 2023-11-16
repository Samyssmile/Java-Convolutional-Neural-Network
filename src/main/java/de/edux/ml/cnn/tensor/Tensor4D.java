package de.edux.ml.cnn.tensor;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 *
 * CNN use mini batches for training.
 * CNN use Categorical Cross Entropy as loss function.
 * CNN use Stochastic Gradient Descent as optimizer.
 * CNN use ReLu as activation function for hidden layers.
 * CNN use Softmax as activation function for the last layer.
 */
public class Tensor4D {
  private int batches;
  private int channels;
  private int rows;
  private int cols;
  private double[][][][] data;

  private final ThreadLocalRandom random = ThreadLocalRandom.current();

  public Tensor4D(int batches, int channels, int rows, int cols) {
    this.batches = batches;
    this.channels = channels;
    this.rows = rows;
    this.cols = cols;
    this.data = new double[batches][channels][rows][cols];
  }

    public Tensor4D flatten() {
        int flattenedRows = 1;
        int flattenedCols = channels * rows * cols;
        Tensor4D flattenedTensor = new Tensor4D(batches, 1, flattenedRows, flattenedCols);

        for (int batch = 0; batch < batches; batch++) {
            for (int channel = 0; channel < channels; channel++) {
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        int flatIndex = channel * rows * cols + row * cols + col;
                        flattenedTensor.data[batch][0][0][flatIndex] = this.data[batch][channel][row][col];
                    }
                }
            }
        }
        return flattenedTensor;
    }

  public Tensor4D convolve(Tensor4D filter, int stride, int padding) {
    // Adjust the input dimensions based on padding
    int paddedRows = this.rows + 2 * padding;
    int paddedCols = this.cols + 2 * padding;

    // Output dimensions
    int outputBatches = this.batches;
    int outputChannels =
        filter.batches; // Assuming output channels are represented by filter's batch dimension
    int outputRows = (paddedRows - filter.rows) / stride + 1;
    int outputCols = (paddedCols - filter.cols) / stride + 1;

    // Initialize the output tensor
    Tensor4D output = new Tensor4D(outputBatches, outputChannels, outputRows, outputCols);

    // Apply padding if necessary
    double[][][][] paddedInput = applyPadding(padding);

    // Perform the convolution
    for (int batch = 0; batch < this.batches; batch++) {
      for (int outChannel = 0; outChannel < outputChannels; outChannel++) {
        for (int row = 0; row < outputRows; row++) {
          for (int col = 0; col < outputCols; col++) {
            double sum = 0.0;
            for (int inChannel = 0; inChannel < this.channels; inChannel++) {
              for (int filterRow = 0; filterRow < filter.rows; filterRow++) {
                for (int filterCol = 0; filterCol < filter.cols; filterCol++) {
                  int inputRow = row * stride + filterRow;
                  int inputCol = col * stride + filterCol;
                  sum +=
                      paddedInput[batch][inChannel][inputRow][inputCol]
                          * filter.data[outChannel][inChannel][filterRow][filterCol];
                }
              }
            }
            output.data[batch][outChannel][row][col] = sum;
          }
        }
      }
    }
    return output;
  }

  private double[][][][] applyPadding(int padding) {
    if (padding == 0) {
      return this.data;
    }

    int paddedRows = this.rows + 2 * padding;
    int paddedCols = this.cols + 2 * padding;
    double[][][][] paddedData = new double[this.batches][this.channels][paddedRows][paddedCols];

    for (int batch = 0; batch < this.batches; batch++) {
      for (int channel = 0; channel < this.channels; channel++) {
        for (int row = 0; row < this.rows; row++) {
          for (int col = 0; col < this.cols; col++) {
            paddedData[batch][channel][row + padding][col + padding] =
                this.data[batch][channel][row][col];
          }
        }
      }
    }

    return paddedData;
  }

  public void setBatches(int batches) {
    this.batches = batches;
  }

  public void setChannels(int channels) {
    this.channels = channels;
  }

  public void setRows(int rows) {
    this.rows = rows;
  }

  public void setCols(int cols) {
    this.cols = cols;
  }

  public void setData(double[][][][] data) {
    this.data = data;
  }

  public int getBatches() {
    return batches;
  }

  public int getChannels() {
    return channels;
  }

  public int getRows() {
    return rows;
  }

  public int getCols() {
    return cols;
  }

  public double[][][][] getData() {
    return data;
  }

  public void randomHE() {
    for (int i = 0; i < batches; i++) {
      for (int j = 0; j < channels; j++) {
        for (int k = 0; k < rows; k++) {
          for (int l = 0; l < cols; l++) {
            data[i][j][k][l] = random.nextGaussian() * Math.sqrt(2.0 / (rows * cols));
          }
        }
      }
    }
  }

}
