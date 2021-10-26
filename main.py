from src.preprocess import PreprocessingPipeLine
from src.train import TrainGAN

if __name__ == '__main__':
    pre_processing_pipeline = PreprocessingPipeLine()
    dataset, dataset_info = pre_processing_pipeline.load_and_cache_dataset()
    ds = []
    # for batch in dataset:
    #     for data in batch['password'].numpy():
    #         try:
    #             word: str = data.decode("utf-8")
    #             if len(word) <= 10:
    #                 ds.append(word.ljust(10))
    #         except Exception:
    #             pass
    # dataset = pre_processing_pipeline.choose_passwords_of_length_10_or_less(ds=dataset)

    training = TrainGAN()
    training.train(dataset=dataset, epochs=100)