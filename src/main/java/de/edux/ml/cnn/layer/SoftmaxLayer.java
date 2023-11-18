package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.Tensor4D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SoftmaxLayer extends Layer{
    private static final Logger LOG = LoggerFactory.getLogger(MaxPoolingLayer.class);

    @Override
    public Tensor4D forward(Tensor4D input) {
        LOG.debug("SoftmaxLayer forward");
        return input.softmax();
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
