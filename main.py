from src.preprocess import PreprocessingPipeLine
from src.train import TrainGAN

if __name__ == '__main__':
    pre_processing_pipeline = PreprocessingPipeLine()
    dataset, char2id = pre_processing_pipeline.load_and_cache_dataset()

    training = TrainGAN()
    training.train(dataset=dataset, char2id=char2id, epochs=10)