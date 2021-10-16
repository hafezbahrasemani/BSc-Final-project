import tensorflow as tf
from src.config import GANConfig


class GANOpt:
    """
        Define functions to return Adam optimizer for both networks
    """
    def __init__(self):
        pass

    def get_generator_opt(self):
        return tf.keras.optimizers.Adam(1e-4, beta_1=GANConfig.BETA_1, beta_2=GANConfig.BETA_2)

    def get_discriminator(self):
        return tf.keras.optimizers.Adam(1e-4, beta_1=GANConfig.BETA_1, beta_2=GANConfig.BETA_2)
