import datetime


class GANConfig:
    DATASET_NAME = "rock_you"
    NOISE_INPUT_SIZE = 128  # noise input size
    BACH_SIZE = 128
    EPOCHS = 199000
    LAYER_DIM = 128
    GRADIENT_PENALTY = 10
    OUTPUT_SEQ_LENGTH = 10
    DISC_ITERATIONS_PER_GEN_ITERATIONS = 10  # How many discriminator iterations per generator iteration

    INITIAL_TRAIN_SIZE = 10000  # Train size for starting training in local environment
    INITIAL_TRAINING = True  # specify loading initial training or 2.5M passwords for actual training

    # Adam Optimizer"s hyper-parameters
    LEARNING_RATE = 0.0001
    BETA_1 = 0.5
    BETA_2 = 0.9

    # Directories
    LOSSES_DIR = "./losses"
    GENERATED_DIR = "./generated"
    PROBABILITY_DIR = "./probabilities"
