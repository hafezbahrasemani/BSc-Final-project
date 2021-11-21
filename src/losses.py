import tensorflow as tf
import neural_structured_learning as nsl


class GANLoss:
    def __init__(self):
        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)

        # real_loss = nsl.lib.jensen_shannon_divergence(tf.ones_like(real_output), real_output, axis=None)
        # fake_loss = nsl.lib.jensen_shannon_divergence(tf.zeros_like(real_output), fake_output, axis=None)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        # return nsl.lib.jensen_shannon_divergence(tf.zeros_like(fake_output), fake_output, axis=None)
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
