import tensorflow as tf


class ResidualBlock(tf.keras.Model):
    """
        create a class Residual block based on Residual Networks definition
    """
    def __init__(self, dim):
        super().__init__()
        self.res_block = tf.keras.Sequential([
            tf.keras.layers.ReLU(True),
            tf.keras.layers.Conv1D(dim, dim, 5, padding='same'),
            tf.keras.layers.ReLU(True),
            tf.keras.layers.Conv1D(dim, dim, 5, padding='same'),
        ])

    def call(self, input_data, **kwargs):
        output = self.res_block(input_data)
        return input_data + (0.3 * output)


class GeneratorNetwork(tf.keras.Model):
    def __init__(self, dim, pass_length):
        super(GeneratorNetwork, self).__init__()

        self.dim = dim
        self.pass_length = pass_length

        # instantiate a Sequential Model
        self.generator_res_block_model = tf.keras.models.Sequential()

        # first linear layer
        self.first_linear_layer = tf.keras.layers.Dense(pass_length, activation='linear', input_shape=[dim*pass_length, ])

        # residual blocks in a sequential order
        self.generator_res_block_model.add(ResidualBlock(dim=dim))
        self.generator_res_block_model.add(ResidualBlock(dim=dim))
        self.generator_res_block_model.add(ResidualBlock(dim=dim))
        self.generator_res_block_model.add(ResidualBlock(dim=dim))
        self.generator_res_block_model.add(ResidualBlock(dim=dim))

        # convolutional 1D layer
        """
        
        """
        self.conv_1d_layer = tf.keras.layers.Conv1D(dim, 1, padding='valid')

        # last soft max layer
        self.softmax_layer = tf.keras.layers.Softmax(axis=1)

    def call(self, input_noise, **kwargs):
        """

        :param input_noise: noise input of some sample generated passwords
        :param kwargs:
        :return: the generated passwords for an iteration
        """

        # feed first layer with noise data
        output = self.first_linear_layer(input_noise)

        # reshape the result of linear layer
        output = tf.reshape(output, [-1, 2, self.dim])

        # feed residual blocks by output from reshape stage
        output = self.generator_res_block_model(output)
        # output = tf.reshape(output, (1, 32, 8))

        # feed resulted data to convolutional layer
        output = self.conv_1d_layer(output)

        # transpose operation on the resulted output
        output = tf.transpose(output)

        # feed softmax layer with transposed output
        output = self.softmax_layer(output)
        # output = tf.reshape(output, [2, 1, 32])

        return output


class DiscriminatorNetwork(tf.keras.Model):
    def __init__(self, dim, pass_length):
        super(DiscriminatorNetwork, self).__init__()
        self.dim = dim
        self.pass_length = pass_length

        self.block = tf.keras.Sequential([
            ResidualBlock(dim=dim),
            ResidualBlock(dim=dim),
            ResidualBlock(dim=dim),
            ResidualBlock(dim=dim),
            ResidualBlock(dim=dim),
        ])
        self.conv1d = tf.keras.layers.Conv1D(dim, 1, padding='valid')
        self.linear = tf.keras.layers.Dense(dim, activation='linear', input_shape=(dim*pass_length, ))

    def call(self, input_data, **kwargs):
        output = tf.transpose(input_data)
        # , [0, 2, 1]
        output = self.conv1d(output)
        output = self.block(output)
        output = tf.reshape(output, (-1, 64, 4))
        output = self.linear(output)
        return output
