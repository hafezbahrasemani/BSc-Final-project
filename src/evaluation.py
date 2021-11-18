from zxcvbn import zxcvbn


class Evaluation:
    def __init__(self):
        pass

    def calculate_password_strength(self, password):
        return zxcvbn(password, user_inputs=[])

