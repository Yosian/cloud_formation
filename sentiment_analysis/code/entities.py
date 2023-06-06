from dataclasses import dataclass


@dataclass
class Parameters:
    training_data_bucket: str = 'sentiment-training-data-bucket'
    training_data_path: str = 'emotion.csv'
    training_data_local_path: str = 'data'

