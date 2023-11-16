package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.Tensor4D;

/**
 * CNN use mini batches for training.
 * CNN use Categorical Cross Entropy as loss function.
 * CNN use Stochastic Gradient Descent as optimizer.
 * CNN use ReLu as activation function for hidden layers.
 * CNN use Softmax as activation function for the last layer.
 */
public class InputLayer extends Layer {
    private final int inputWidth;
    private final int inputHeight;
    private final int channels;

    public InputLayer(int inputWidth, int inputHeight, int channels) {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.channels = channels;
    }

    @Override
    public Tensor4D forward(Tensor4D input) {
        return validateDimnsions(input);
    }

    private Tensor4D validateDimnsions(Tensor4D input) {
        if (input.getRows() != inputHeight || input.getCols() != inputWidth || input.getChannels() != channels) {
            throw new IllegalArgumentException("Input dimensions are not valid");
        }
        return input;
    }

    @Override
    public Tensor4D backward(Tensor4D input) {
        return input;
    }
}

