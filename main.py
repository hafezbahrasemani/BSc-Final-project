from src.preprocess import PreprocessingPipeLine
from src.train import TrainGAN

if __name__ == '__main__':
    pre_processing_pipeline = PreprocessingPipeLine()
    dataset, dataset_info = pre_processing_pipeline.load_and_cache_dataset()

    training = TrainGAN()
    training.train(dataset=dataset, epochs=10)