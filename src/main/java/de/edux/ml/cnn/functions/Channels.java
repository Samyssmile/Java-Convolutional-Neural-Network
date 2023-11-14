package de.edux.ml.cnn.functions;

public enum Channels {
  RGB(3),
  GRAY(1);

  Channels(int channels) {
    this.channels = channels;
  }

  private final int channels;
}
