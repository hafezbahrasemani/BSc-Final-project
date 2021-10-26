import datetime
import time

import tensorflow as tf
from keras.layers import TextVectorization
from keras.utils.np_utils import to_categorical

from src.config import GANConfig
from src.losses import GANLoss
from src.networks import GeneratorNetwork, DiscriminatorNetwork
from src.optimizer import GANOpt
import numpy as np
import tensorflow_hub as hub
import tensorflow_datasets as tfds


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

        z = tf.random.normal([GANConfig.NOISE_INPUT_SIZE, 1, self.generator.pass_length], 0, 1)  # noise input for generator
        #   seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # password = tf.strings.unicode_decode(password, input_encoding='UTF-8')
            # passwords = tf.cast(passwords, tf.int64)
            for _ in range(GANConfig.DISC_ITERATIONS_PER_GEN_ITERATIONS):
                # embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
                # hub_layer = hub.KerasLayer(embedding, input_shape=[GANConfig.BACH_SIZE],
                #                            dtype=tf.string, trainable=True)
                # passwords = hub_layer(passwords)

                padded_passwords = []
                charset = set(" ")  # start with the initial padding char
                for p in passwords:
                    padded_passwords.append(p.numpy().decode('utf-8').ljust(GANConfig.OUTPUT_SEQ_LENGTH, " "))
                    charset |= set(p.numpy().decode('utf-8'))  # |= is the union set operation.

                # Convert characters to integers
                self.vocab_size = len(charset)
                char2id = dict((c, i) for i, c in enumerate(charset))

                vectorization_layer = tf.keras.layers.TextVectorization(
                    max_tokens=5000,
                    output_mode='int',
                    output_sequence_length=GANConfig.OUTPUT_SEQ_LENGTH,
                    vocabulary=padded_passwords
                )

                # vectorization_layer.adapt(padded_passwords)
                model = tf.keras.models.Sequential()

                model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
                model.add(vectorization_layer)
                output = model.predict(padded_passwords)
                # One hot encode the passwords
                encoded_passwords = [[char2id[c] for c in password] for password in padded_passwords]
                one_hot_encoded = [tf.constant(to_categorical(p, num_classes=self.vocab_size)) for p in encoded_passwords]

                # resh = tf.reshape(one_hot_encoded[0], [10 * self.vocab_size])
                #Tensor("one_hot:0", shape=(64, 10, 51), dtype=float32)
                real_inputs_discrete = tf.compat.v1.placeholder(tf.int32, shape=[GANConfig.BACH_SIZE, GANConfig.OUTPUT_SEQ_LENGTH])
                tf.one_hot(real_inputs_discrete, self.vocab_size)

                real_input = tf.reshape(one_hot_encoded, [2, 1, 32])
                real_output = self.discriminator.call(input_data=real_input)

            generated_passwords = self.generator.call(input_noise=z)

            fake_output = self.discriminator.call(input_data=generated_passwords)

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

        start = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            gen_loss_list = []
            disc_loss_list = []

            for batch in dataset:
                t = self.train_step(batch['password'])
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
