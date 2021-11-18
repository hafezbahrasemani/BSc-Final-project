import datetime
import string
import time
from math import log

import tensorflow as tf
from keras.layers import TextVectorization
from keras.utils.np_utils import to_categorical

from src.config import GANConfig
from src.losses import GANLoss
from src.networks import GeneratorNetwork, DiscriminatorNetwork
from src.optimizer import GANOpt
import numpy as np
from numpy import save


class TrainGAN:
    def __init__(self):
        self.generator = GeneratorNetwork(dim=GANConfig.LAYER_DIM, pass_length=GANConfig.OUTPUT_SEQ_LENGTH)
        self.discriminator = DiscriminatorNetwork(dim=GANConfig.LAYER_DIM, pass_length=GANConfig.OUTPUT_SEQ_LENGTH)
        self.gan_loss = GANLoss()
        self.gan_opt = GANOpt()
        self.generator_opt: tf.keras.optimizers.Adam = self.gan_opt.get_generator_opt()
        self.discriminator_opt: tf.keras.optimizers.Adam = self.gan_opt.get_generator_opt()
        self.vocab_size = None
        self.char2id = None

    # @tf.function
    def train_step(self, passwords, epoch):
        """
        this would be called on each iteration
            > Here we use tensorflow GradiantTape to record operations for differentiation for each epoch.

        :param epoch:
        :param passwords:
        :return:
        """
        # generates a new set of random values every time:
        tf.random.set_seed(5)
        z = tf.random.uniform(shape=[GANConfig.NOISE_INPUT_SIZE, self.vocab_size, GANConfig.OUTPUT_SEQ_LENGTH],
                              minval=0, maxval=1, dtype=tf.float32)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            for _ in range(GANConfig.DISC_ITERATIONS_PER_GEN_ITERATIONS):
                encoded_passwords = [[self.char2id.get(c) for c in password.decode('utf-8')] for password in
                                     passwords.numpy()]
                one_hot_encoded = [tf.constant(to_categorical(p, num_classes=self.vocab_size)) for p in
                                   encoded_passwords]
                numpy_one_hot = np.array(one_hot_encoded)

                # Pass real passwords to discriminator for producing real output, this will be used for disc_loss calculations
                real_output = self.discriminator.call(input_data=numpy_one_hot)

            # Every time pass noisy passwords to generator, so this will generated new ones
            generated_passwords = self.generator.call(input_noise=z)
            if epoch % 100 == 0:
                save(f"{GANConfig.PROBABILITY_DIR}/epoch_{epoch}_prod.npy", generated_passwords)

            generated = tf.reshape(generated_passwords, [128, 10, self.vocab_size])
            generated_argmax = np.argmax(generated, axis=-1)

            # pass generator output to discriminator
            generated = tf.reshape(generated_passwords, [128, 10, self.vocab_size])
            fake_output = self.discriminator.call(input_data=generated)

            # calculate both generator and discriminator losses
            gen_loss = self.gan_loss.generator_loss(fake_output)
            disc_loss = self.gan_loss.discriminator_loss(real_output, fake_output)

            # compute gradient based on computed losses
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            # apply adam optimizer on both networks
            self.generator_opt.apply_gradients(zip(
                gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_opt.apply_gradients(zip(
                gradients_of_discriminator,
                self.discriminator.trainable_variables))

            # self.generator.summary()
            # self.discriminator.summary()

        return gen_loss, disc_loss, generated_argmax

    def train(self, dataset, char2id, epochs):
        self.generator.build(input_shape=[])
        self.discriminator.build(input_shape=[])

        generated = None
        start = time.time()

        self.char2id = char2id
        self.vocab_size = len(char2id)

        for epoch in range(epochs):
            epoch_start = time.time()

            start_time_str = datetime.datetime.now().strftime(format="%Y-%m-%dT%H-%M-%S")

            print(f"epoch {epoch} started at {start_time_str}")
            gen_loss_list = []
            disc_loss_list = []

            for batch in dataset:
                gen_loss, disc_loss, generated = self.train_step(batch, epoch)
                gen_loss_list.append(gen_loss)
                disc_loss_list.append(disc_loss)

            g_loss = sum(gen_loss_list) / len(gen_loss_list)
            d_loss = sum(disc_loss_list) / len(disc_loss_list)

            epoch_elapsed = time.time() - epoch_start
            print(f"saving losses and generated data for epoch {epoch}")
            losses_string = f'Epoch {epoch}, gen loss={g_loss},disc loss={d_loss}, {self.hms_string(epoch_elapsed)}'
            self.save_losses(file_name=f"epoch_{epoch}_losses", losses_string=losses_string)

            # convert generated passwords vector to password strings, then save them to a text file
            current_time_str = datetime.datetime.now().strftime(format="%Y%m%d-%H%M%S")
            self.save_generated_passwords(generated, f"generated-password_epoch-{str(epoch)}_{current_time_str}")

        elapsed = time.time() - start
        print(f'Training time: {elapsed}')

    @staticmethod
    def hms_string(sec_elapsed):
        h = int(sec_elapsed / (60 * 60))
        m = int((sec_elapsed % (60 * 60)) / 60)
        s = sec_elapsed % 60
        return "{}:{:>02}:{:>05.2f}".format(h, m, s)

    def _get_vocabulary(self):
        # vocabulary = [char for char in string.printable]
        # vocabulary.append('<unk>')
        # char2id = dict((c, i) for i, c in enumerate(vocabulary))
        return self.char2id

    def _convert_password_float_vector_to_string(self, generated_password_vector):
        char2id = self.char2id
        id2char = {}
        for key, val in char2id.items():
            id2char[val] = key
        password = ''
        for char_id in generated_password_vector:
            password += str(id2char.get(char_id) if id2char.get(char_id) else " ")
        return password

    def save_losses(self, file_name, losses_string):
        file = open(f'{GANConfig.LOSSES_DIR}/{file_name}.txt', 'a')
        file.write(losses_string)
        file.write("\n")
        file.close()

    def save_generated_passwords(self, passwords, file_name):
        file = open(f'{GANConfig.GENERATED_DIR}/{file_name}.txt', 'w')
        for password in passwords:
            word = self._convert_password_float_vector_to_string(password)
            file.write(word)
            file.write("\n")
        file.close()

    def _get_probability_of_character(self, character):
        """
            The soft-max output of PassGAN
            acts as posterior distribution over character set.

            here we calculate the probability of each character from the conditional posterior distribution.
        :param character:
        :return:
        """
        return 0.5

    def get_prediction_probability(self, password: str, charset: list):
        """
            Algorithm 1: Get prediction probability
            Result: score
            Input : password, model, charmap,;
            prob = 1;
            for char in password do
                char_prob = model.getProbability(char);
                prob = prob × char_prob
            end
            score = -log(prob)

        :param password:
        :param charset:
        :return:
        """
        prob = 1
        for char in password:
            char_prob = self._get_probability_of_character(character=char)
            prob *= char_prob
        score = -log(prob)
        return score
