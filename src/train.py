import datetime
import string
import time

import tensorflow as tf
from keras.layers import TextVectorization
from keras.utils.np_utils import to_categorical

from src.config import GANConfig
from src.losses import GANLoss
from src.networks import GeneratorNetwork, DiscriminatorNetwork
from src.optimizer import GANOpt
import numpy as np


class TrainGAN:
    def __init__(self):
        self.generator = GeneratorNetwork(dim=GANConfig.LAYER_DIM, pass_length=GANConfig.OUTPUT_SEQ_LENGTH)
        self.discriminator = DiscriminatorNetwork(dim=GANConfig.LAYER_DIM, pass_length=GANConfig.OUTPUT_SEQ_LENGTH)
        self.gan_loss = GANLoss()
        self.gan_opt = GANOpt()
        self.generator_opt: tf.keras.optimizers.Adam = self.gan_opt.get_generator_opt()
        self.discriminator_opt: tf.keras.optimizers.Adam = self.gan_opt.get_generator_opt()
        self.vocab_size = None

    # @tf.function
    def train_step(self, passwords):
        """
        this would be called on each iteration
            > Here we use tensorflow GradiantTape to record operations for differentiation for each epoch.

        :param passwords:
        :return:
        """
        # generates a new set of random values every time:
        tf.random.set_seed(5)
        z = tf.constant(tf.random.normal([GANConfig.NOISE_INPUT_SIZE, 1, self.generator.pass_length], dtype=tf.dtypes.float32))
        # z = tf.random.normal([GANConfig.NOISE_INPUT_SIZE, 1, self.generator.pass_length], 0, 1)  # noise input for generator
        #   seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # passwords = tf.strings.unicode_decode(passwords, input_encoding='UTF-8')
            for _ in range(GANConfig.DISC_ITERATIONS_PER_GEN_ITERATIONS):
                padded_passwords = []
                vocabulary = set(" ")  # start with the initial padding char
                # for p in passwords:
                #     current_p = p.numpy().decode('utf-8')
                    # if len(current_p) <= GANConfig.OUTPUT_SEQ_LENGTH:
                    #     padded_passwords.append(current_p.ljust(GANConfig.OUTPUT_SEQ_LENGTH))
                        # vocabulary |= set(current_p)  # |= is the union set operation.
                vocabulary = [char for char in string.printable]
                vocabulary.append('<unk>')
                self.vocab_size = 128
                char2id = dict((c, i) for i, c in enumerate(vocabulary))

                # set unknown chars to <unk> character with index 100
                encoded_passwords = [[char2id.get(c) if char2id.get(c) else 100 for c in password.decode('utf-8')] for password in passwords.numpy()]
                one_hot_encoded = [tf.constant(to_categorical(p, num_classes=self.vocab_size)) for p in encoded_passwords]
                numpy_one_hot = np.array(one_hot_encoded)

                # real_input = tf.reshape(numpy_one_hot, [2, 1, 32])
                real_output = self.discriminator.call(input_data=numpy_one_hot)

            generated_passwords = self.generator.call(input_noise=z)
            generated = tf.reshape(generated_passwords, [-1, 2, 128])
            fake_output = self.discriminator.call(input_data=generated)

            gen_loss = self.gan_loss.generator_loss(fake_output)
            disc_loss = self.gan_loss.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_opt.apply_gradients(zip(
                gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_opt.apply_gradients(zip(
                gradients_of_discriminator,
                self.discriminator.trainable_variables))
        return gen_loss, disc_loss

    def train(self, dataset, epochs):
        # fixed_seed = tf.random.normal(0, 1, (1, 32), dtype=tf.dtypes.float32)  # noise input for generator
        # tf.compat.v1.disable_eager_execution()

        start = time.time()
        vocabulary = self._get_vocabulary()

        def generate_samples():
            samples = session.run(fake_inputs)
            samples = np.argmax(samples, axis=2)
            decoded_samples = []
            for i in xrange(len(samples)):
                decoded = []
                for j in range(len(samples[i])):
                    decoded.append(vocabulary[samples[i][j]])
                decoded_samples.append(tuple(decoded))
            return decoded_samples

        for epoch in range(epochs):
            epoch_start = time.time()

            gen_loss_list = []
            disc_loss_list = []

            for batch in dataset:
                t = self.train_step(batch)
                gen_loss_list.append(t[0])
                disc_loss_list.append(t[1])

            g_loss = sum(gen_loss_list) / len(gen_loss_list)
            d_loss = sum(disc_loss_list) / len(disc_loss_list)

            epoch_elapsed = time.time() - epoch_start
            print(f'Epoch {epoch + 1}, gen loss={g_loss},disc loss={d_loss}, {self.hms_string(epoch_elapsed)}')
            # self.save_generated_passwords(epoch, fixed_seed)

        elapsed = time.time() - start
        print(f'Training time: {elapsed}')

    def hms_string(self, sec_elapsed):
        h = int(sec_elapsed / (60 * 60))
        m = int((sec_elapsed % (60 * 60)) / 60)
        s = sec_elapsed % 60
        return "{}:{:>02}:{:>05.2f}".format(h, m, s)

    def _get_vocabulary(self):
        vocabulary = [char for char in string.printable]
        vocabulary.append('<unk>')
        char2id = dict((c, i) for i, c in enumerate(vocabulary))
        return char2id

    def save_generated_passwords(self, epoch, seed):
            pass

    def preprocess_dataset(self, max_features):
        vectorized_layer = TextVectorization(
            standardize='custom_standardization',
            max_tokens=max_features,
            # split=char_split,  # word_split or char_split
            output_mode="int",
            output_sequence_length=GANConfig.OUTPUT_SEQ_LENGTH,
        )

        def vectorized_text(text) -> object:
            text = tf.expand_dims(text, -1)
            return tf.squeeze(vectorized_layer(text))