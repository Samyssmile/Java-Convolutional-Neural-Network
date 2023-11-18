package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.Tensor4D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SoftmaxLayer extends Layer {
    private static final Logger LOG = LoggerFactory.getLogger(SoftmaxLayer.class);

    @Override
    public Tensor4D forward(Tensor4D input) {
        LOG.debug("SoftmaxLayer forward");
        return input.softmax();
    }

    private Tensor4D softmax(Tensor4D input) {
        Tensor4D output = new Tensor4D(input.getBatches(), input.getChannels(), input.getRows(), input.getCols());

        for (int batch = 0; batch < input.getBatches(); batch++) {
            for (int row = 0; row < input.getRows(); row++) {
                for (int col = 0; col < input.getCols(); col++) {
                    // Compute the exponential values and sum them
                    double sumExp = 0.0;
                    for (int channel = 0; channel < input.getChannels(); channel++) {
                        sumExp += Math.exp(input.getData()[batch][channel][row][col]);
                    }

                    // Apply softmax transformation
                    for (int channel = 0; channel < input.getChannels(); channel++) {
                        double expValue = Math.exp(input.getData()[batch][channel][row][col]);
                        output.getData()[batch][channel][row][col] = expValue / sumExp;
                    }
                }
            }
        }

        return output;
    }


    @Override
    public Tensor4D backward(Tensor4D dL_dy) {
        LOG.debug("SoftmaxLayer backward");

        // Assuming dL_dy contains the gradient of the loss with respect to the output of the softmax layer
        // and softmaxOutput is the output of the softmax layer from the forward pass.
        Tensor4D softmaxOutput = this.forward(dL_dy);

        // Compute the gradient for softmax layer
        // This gradient will be passed to previous layers
        Tensor4D gradient = computeSoftmaxGradient(softmaxOutput, dL_dy);

        return gradient;
    }

    private Tensor4D computeSoftmaxGradient(Tensor4D softmaxOutput, Tensor4D dL_dy) {
        Tensor4D gradient = new Tensor4D(softmaxOutput.getBatches(), softmaxOutput.getChannels(), softmaxOutput.getRows(), softmaxOutput.getCols());

        for (int b = 0; b < softmaxOutput.getBatches(); b++) {
            for (int c = 0; c < softmaxOutput.getChannels(); c++) {
                for (int r = 0; r < softmaxOutput.getRows(); r++) {
                    for (int col = 0; col < softmaxOutput.getCols(); col++) {
                        // Gradient for softmax and cross-entropy is the difference between softmax output and true labels
                        gradient.getData()[b][c][r][col] = softmaxOutput.getData()[b][c][r][col] - dL_dy.getData()[b][c][r][col];
                    }
                }
            }
        }

        return gradient;
    }


}
