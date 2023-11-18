package de.edux.ml.cnn.tensor;

import de.edux.ml.cnn.layer.FullyConnectedLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ThreadLocalRandom;

/**
 * CNN use mini batches for training.
 * CNN use Categorical Cross Entropy as loss function.
 * CNN use Stochastic Gradient Descent as optimizer.
 * CNN use ReLu as activation function for hidden layers.
 * CNN use Softmax as activation function for the last layer.
 * Fully connected layer use He initialization.
 * Fully connected layer updates weights and biases.
 */
public class Tensor4D {

    private static final Logger LOG = LoggerFactory.getLogger(Tensor4D.class);

    private int batches;
    private int channels;
    private int rows;
    private int cols;
    private double[][][][] data;

    private final ThreadLocalRandom random = ThreadLocalRandom.current();

    /**
     *
     * @param batches
     * @param channels
     * @param rows
     * @param cols
     */
    public Tensor4D(int batches, int channels, int rows, int cols) {
        this.batches = batches;
        this.channels = channels;
        this.rows = rows;
        this.cols = cols;
        this.data = new double[batches][channels][rows][cols];
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


    public Tensor4D multiply(double scalar) {
        Tensor4D result = new Tensor4D(batches, channels, rows, cols);
        for (int batch = 0; batch < batches; batch++) {
            for (int channel = 0; channel < channels; channel++) {
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        result.data[batch][channel][row][col] = this.data[batch][channel][row][col] * scalar;
                    }
                }
            }
        }
        return result;
    }

    public Tensor4D subtract(Tensor4D tensor) {
        if (this.batches != tensor.batches || this.channels != tensor.channels || this.rows != tensor.rows || this.cols != tensor.cols) {
            LOG.error("Dimension mismatch: this Tensor [{}][{}][{}][{}], other Tensor [{}][{}][{}][{}]",
                    this.batches, this.channels, this.rows, this.cols,
                    tensor.batches, tensor.channels, tensor.rows, tensor.cols);
            throw new IllegalArgumentException("Tensor dimensions must match for subtraction");
        }
        Tensor4D result = new Tensor4D(batches, channels, rows, cols);
        for (int batch = 0; batch < batches; batch++) {
            for (int channel = 0; channel < channels; channel++) {
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        result.data[batch][channel][row][col] = this.data[batch][channel][row][col] - tensor.data[batch][channel][row][col];
                    }
                }
            }
        }
        return result;
    }


    public Tensor4D transpose() {
        // Create a new tensor with swapped rows and cols dimensions
        Tensor4D transposed = new Tensor4D(this.batches, this.channels, this.cols, this.rows);

        // Transpose each 2D matrix (rows x cols) within each batch and channel
        for (int batch = 0; batch < this.batches; batch++) {
            for (int channel = 0; channel < this.channels; channel++) {
                for (int i = 0; i < this.rows; i++) {
                    for (int j = 0; j < this.cols; j++) {
                        transposed.data[batch][channel][j][i] = this.data[batch][channel][i][j];
                    }
                }
            }
        }

        return transposed;
    }

    public Tensor4D sumOverBatches() {
        Tensor4D sum = new Tensor4D(1, 1, 1, this.cols);

        for (int i = 0; i < this.cols; i++) {
            double total = 0;
            for (int batch = 0; batch < this.batches; batch++) {
                total += this.data[batch][0][0][i];
            }
            sum.data[0][0][0][i] = total;
        }

        return sum;
    }

    public Tensor4D dot(Tensor4D tensor) {
        // Check if the dimensions are compatible for dot product
        if (this.cols != tensor.rows) {
            LOG.error("Dimension mismatch: this Tensor [{}][{}][{}][{}], other Tensor [{}][{}][{}][{}]",
                    this.batches, this.channels, this.rows, this.cols,
                    tensor.batches, tensor.channels, tensor.rows, tensor.cols);
            throw new IllegalArgumentException("Dimensions are not compatible for dot product.");
        }

        // Define dimensions for the result tensor
        int resultBatches = this.batches;
        int resultChannels = tensor.channels;
        int resultRows = this.rows;
        int resultCols = tensor.cols;

        // Initialize the result tensor
        Tensor4D result = new Tensor4D(resultBatches, resultChannels, resultRows, resultCols);

        // Perform the dot product
        for (int batch = 0; batch < resultBatches; batch++) {
            for (int channel = 0; channel < resultChannels; channel++) {
                for (int row = 0; row < resultRows; row++) {
                    for (int col = 0; col < resultCols; col++) {
                        double sum = 0.0;
                        for (int k = 0; k < this.cols; k++) {
                            sum += this.data[batch][channel][row][k] * tensor.data[batch][channel][k][col];
                        }
                        result.data[batch][channel][row][col] = sum;
                    }
                }
            }
        }

        return result;
    }

    /**
     * Multipliziert diesen Tensor mit einem anderen Tensor.
     * @param other Der andere Tensor, mit dem multipliziert wird.
     * @return Das Ergebnis der Multiplikation.
     */
    public Tensor4D multiply2(Tensor4D other) {
        // Überprüfen der Dimensionen für die Multiplikation
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Dimensionen stimmen nicht überein für die Multiplikation");
        }

        int newBatches = this.batches;
        int newChannels = other.channels;
        int newRows = this.rows;
        int newCols = other.cols;

        Tensor4D result = new Tensor4D(newBatches, newChannels, newRows, newCols);

        // Durchführen der Multiplikation
        for (int batch = 0; batch < newBatches; batch++) {
            for (int channel = 0; channel < newChannels; channel++) {
                for (int row = 0; row < newRows; row++) {
                    for (int col = 0; col < newCols; col++) {
                        double sum = 0.0;
                        for (int k = 0; k < this.cols; k++) {
                            sum += this.data[batch][0][row][k] * other.data[0][channel][k][col];
                        }
                        result.data[batch][channel][row][col] = sum;
                    }
                }
            }
        }

        return result;
    }

    public Tensor4D sumOverBatchesForBiases() {
        Tensor4D sum = new Tensor4D(1, this.channels, 1, 1);

        for (int channel = 0; channel < this.channels; channel++) {
            double total = 0.0;
            for (int batch = 0; batch < this.batches; batch++) {
                total += this.data[batch][channel][0][0];
            }
            sum.data[0][channel][0][0] = total;
        }

        return sum;
    }


    /**
     * Führt eine Matrix-Vektor-Multiplikation mit einem anderen Tensor4D durch.
     * @param other Der Tensor, mit dem multipliziert wird.
     * @return Das Ergebnis der Multiplikation als neuer Tensor4D.
     */
    public Tensor4D multiply(Tensor4D other) {
        // Sicherstellen, dass die Multiplikation möglich ist
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Die Spalten des ersten Tensors müssen mit den Zeilen des zweiten Tensors übereinstimmen.");
        }

        // Der resultierende Tensor hat die Dimensionen [this.batches, other.channels, this.rows, other.cols]
        Tensor4D result = new Tensor4D(this.batches, other.channels, this.rows, other.cols);

        // Durchführen der Matrix-Vektor-Multiplikation für jeden Batch
        for (int batch = 0; batch < this.batches; batch++) {
            for (int channel = 0; channel < other.channels; channel++) {
                for (int row = 0; row < this.rows; row++) {
                    double sum = 0.0;
                    for (int k = 0; k < this.cols; k++) {
                        sum += this.data[batch][0][row][k] * other.data[0][channel][k][0];
                    }
                    result.data[batch][channel][row][0] = sum;
                }
            }
        }

        return result;
    }


    /**
     * Berechnet den Durchschnitt über die Batch-Dimension.
     * @return Ein neuer Tensor, der den Durchschnitt über die Batch-Dimension darstellt.
     */
    public Tensor4D averageOverBatches() {
        // Die neue Batch-Dimension wird 1 sein
        int newBatches = 1;

        Tensor4D averaged = new Tensor4D(newBatches, this.channels, this.rows, this.cols);

        // Durchführen des Durchschnitts über die Batches
        for (int channel = 0; channel < this.channels; channel++) {
            for (int row = 0; row < this.rows; row++) {
                for (int col = 0; col < this.cols; col++) {
                    double sum = 0.0;
                    for (int batch = 0; batch < this.batches; batch++) {
                        sum += this.data[batch][channel][row][col];
                    }
                    averaged.data[0][channel][row][col] = sum / this.batches;
                }
            }
        }

        return averaged;
    }


    /**
     * Führt eine Batch-weise Matrix-Vektor-Multiplikation mit einem anderen Tensor4D durch.
     * @param other Der Tensor, mit dem multipliziert wird.
     * @return Das Ergebnis der Multiplikation als neuer Tensor4D.
     */
    public Tensor4D batchedMatrixVectorMultiply(Tensor4D other) {
        // Sicherstellen, dass die Multiplikation möglich ist
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Die Spalten des ersten Tensors müssen mit den Zeilen des zweiten Tensors übereinstimmen.");
        }

        // Der resultierende Tensor hat die Dimensionen [this.batches, this.channels, other.cols, 1]
        Tensor4D result = new Tensor4D(this.batches, this.channels, other.cols, 1);

        // Durchführen der Batch-weise Matrix-Vektor-Multiplikation
        for (int batch = 0; batch < this.batches; batch++) {
            for (int i = 0; i < this.channels; i++) {
                for (int j = 0; j < other.cols; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < this.cols; k++) {
                        sum += this.data[batch][i][0][k] * other.data[0][j][k][0];
                    }
                    result.data[batch][i][0][j] = sum;
                }
            }
        }

        return result;
    }

    /**
     * Applies the softmax function to the last dimension of the 4D tensor.
     *
     * @return Tensor4D The result after applying the softmax function.
     */
    public Tensor4D softmax() {
        Tensor4D result = new Tensor4D(batches, channels, rows, cols);

        for (int b = 0; b < batches; b++) {
            for (int c = 0; c < channels; c++) {
                for (int r = 0; r < rows; r++) {
                    double[] softmaxVector = softmax(this.data[b][c][r]);
                    result.data[b][c][r] = softmaxVector;
                }
            }
        }

        return result;
    }

    /**
     * Applies the softmax function to a vector.
     *
     * @param input The input vector.
     * @return double[] The softmax transformed vector.
     */
    private double[] softmax(double[] input) {
        double[] output = new double[input.length];
        double sum = 0.0;

        // Exponentiate and sum
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.exp(input[i]);
            sum += output[i];
        }

        // Normalize
        for (int i = 0; i < output.length; i++) {
            output[i] /= sum;
        }

        return output;
    }


    public Tensor4D addBiases(Tensor4D biases) {
        // Assuming biases' shape is [1][1][1][outputSize]
        for (int batch = 0; batch < batches; batch++) {
            for (int i = 0; i < cols; i++) {
                this.data[batch][0][0][i] += biases.data[0][0][0][i];
            }
        }

        return this;
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

    @Override
    public String toString() {
        return "Tensor4D{" +
                "batches=" + batches +
                ", channels=" + channels +
                ", rows=" + rows +
                ", cols=" + cols +
                '}';
    }
}
