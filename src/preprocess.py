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
                                    split='train[:2500000]'
                           )
        ds = ds.shuffle(1000, reshuffle_each_iteration=True)
        # ds = ds.apply(tf.data.experimental.unbatch())
        ds = self.choose_passwords_of_length_10_or_less(ds)
        # ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        # ds = ds.shuffle(100000, reshuffle_each_iteration=True)
        # ds = ds.batch(64)
        # Cache dataset for future use
        ds = ds.batch(GANConfig.BACH_SIZE, drop_remainder=True)

        ds = ds.cache()
        return ds, ds_info

    def choose_passwords_of_length_10_or_less(self, dataset):
        ds = []
        for data in dataset:
            try:
                word: str = data['password'].numpy().decode("utf-8")
                if len(word) <= 10:
                    ds.append(word.ljust(10))
            except Exception:
                pass
        # def sequence_len_filter(data):
        #     for p in data.get('password'):
        #         if len(p.numpy().decode("utf-8")) <= 10:
        #             return data
        #
        # return [sequence_len_filter(x) for x in ds]
        return tf.data.Dataset.from_tensor_slices(ds)

    def call(self):
        initial_ds, dataset_info = self.load_and_cache_dataset()

        def sequence_length_filter(password: tf.Tensor):
            if len(password.get("password").decode("utf-8")) <= 10:
                return password

        ds = initial_ds.as_numpy_iterator()

        ds = ds.filter(sequence_length_filter)
        print("[info] dataset elements spec: ", initial_ds.element_spec)

        # ds = initial_ds.filter()
        # for data in initial_ds:
        #     print(data)
        # ds = list(filter(lambda x: len(x) <= 10, initial_ds))
        # for password in ds:
        #     print(password.get("password").decode("utf-8"))


# pre_processing_pipeline = PreprocessingPipeLine()
# dataset, dataset_info = pre_processing_pipeline.load_and_cache_dataset()
#
# # dataset = pre_processing_pipeline.choose_passwords_of_length_10_or_less(ds=dataset)
# ds = []
# print("[info] train dataset length: ", len(dataset))
# # print(dataset_info.splits.total_num_examples)
# for data in dataset:
#     try:
#         word: str = data.get('password').numpy().decode("utf-8")
#         if len(word) <= 10:
#             ds.append(word.ljust(10))
#     except Exception:
#         pass
#
# print("[info] less than 10 characters train dataset length: ", len(ds))
#
#
# def padding_password_to_length_10(password: str):
#     return password.ljust(10)
#
#
# print("[info] start padding passwords to length 10 ...")
#
# map_object = map(padding_password_to_length_10, ds)
# ds = list(map_object)
#
# print("[info] padding ended ...")
#
# ds = np.array(ds)
#
# print("[info] convert list dataset to numpy array done.")
#
#
# def append_random_variations_of_passwords_to_dataset(ds):
#     print("[info] dataset length before appending password variations: ", len(ds))
#
#     print("[info] dataset length after appending password variations: ", len(ds))
#     pass
