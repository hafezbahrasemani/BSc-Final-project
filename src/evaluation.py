import pickle
from math import log

from numpy import load
from zxcvbn import zxcvbn

from src.config import GANConfig
import numpy as np


class Evaluation:
    def __init__(self):
        pass

    def _get_probability_of_character(self, character, index):
        data = load(f"{GANConfig.PROBABILITY_DIR}/epoch_40_prod.npy")
        char2id = self.load_charset_from_memory()
        char_id = char2id.get(character)
        prob = np.reshape(data[-1], (GANConfig.OUTPUT_SEQ_LENGTH, len(char2id)))
        return prob[index - 1][char_id - 1]

    def get_prediction_probability(self, password: str):
        prob = 1
        for index, char in enumerate(password):
            char_prob = self._get_probability_of_character(character=char, index=index)
            prob *= char_prob
        score = -log(prob, 10)
        zxcvbn_guess_log = zxcvbn(password=password).get("guesses_log10")
        print(zxcvbn(password=password))
        print(f"zxcvbn guess log 10: {zxcvbn_guess_log}")
        print(f"our guess log 10: {score}")

        return score

    def load_charset_from_memory(self):
        char2id = {}
        with open(f"{GANConfig.CHARSET_DIR}/charset.pickle", "rb") as fin:
            while True:
                try:
                    small_dict = pickle.load(fin)
                except EOFError:
                    break
                char2id.update(small_dict)
        return char2id


if __name__ == '__main__':
    evaluation = Evaluation()
    evaluation.get_prediction_probability(password="1234567890")