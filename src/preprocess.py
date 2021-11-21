import pickle

import tensorflow as tf
import tensorflow_datasets as tfds

from src.config import GANConfig
import numpy as np


class PreprocessingPipeLine:
    def __init__(self):
        pass

    def load_and_cache_dataset(self):
        if GANConfig.INITIAL_TRAINING:
            ds, ds_info = tfds.load(name=GANConfig.DATASET_NAME,
                                    split=f'train[:{GANConfig.INITIAL_TRAIN_SIZE}]', with_info=True)
        else:
            ds, ds_info = tfds.load(name=GANConfig.DATASET_NAME,
                                    split='train[:1000000]', with_info=True)
        ds = ds.shuffle(1000, reshuffle_each_iteration=True)
        ds, char2id = self.choose_passwords_of_length_10_or_less(ds)
        ds = ds.batch(GANConfig.BACH_SIZE, drop_remainder=True)
        ds = ds.cache()

        self.save_charset_to_memory(charset=char2id)
        return ds, char2id

    def choose_passwords_of_length_10_or_less(self, dataset):
        ds = []
        vocabulary = set(" ")
        for data in dataset:
            try:
                word: str = data['password'].numpy().decode("utf-8")
                if len(word) <= 10:
                    ds.append(word.ljust(10))
                    vocabulary |= set(word)

            except Exception:
                pass
        char2id = dict((c, i) for i, c in enumerate(vocabulary))

        return tf.data.Dataset.from_tensor_slices(ds), char2id

    def save_charset_to_memory(self, charset):
        with open('charset.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(charset, f, pickle.HIGHEST_PROTOCOL)
