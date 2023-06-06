import os
import pickle
import re
from typing import List

import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import pad_sequences
from nltk.corpus import stopwords
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer

from sentiment_analysis.code.entities import Parameters
from sentiment_analysis.code.infrastructure import HandleS3Objects


def check_if_file_present(path_to_file: str) -> bool:
    # Check if the file exists
    return os.path.exists(path_to_file)


def make_training_dataframe(file: str) -> pd.DataFrame:
    """
    This function creates a Pandas DataFrame from a file.

    Args:
        file (str): The path to the file.

    Returns:
        pd.DataFrame: The DataFrame.
    """

    # Check if the file exists.
    if not check_if_file_present(path_to_file=file):
        # If the file does not exist, download it from the S3 bucket.
        HandleS3Objects(bucket=Parameters.training_data_bucket,
                        origin=Parameters.training_data_path,
                        destination=Parameters.training_data_local_path).obtain_file_from_bucket()

    # Read the file into a DataFrame.
    temp = pd.read_csv(file, encoding='ISO-8859-1',  # nrows=100000,
                       skiprows=lambda i: i % 50 != 0,
                       names=["target", "ids", "date", "flag", "user", "text"])

    return temp


def clean_text(text: str) -> str:
    """
    This function cleans the text by removing punctuation, stop words, and converting all words to lowercase.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """

    # Remove punctuation.
    text = re.sub('[,.!?;:]', '', text)

    # Convert all words to lowercase.
    text = text.lower()

    # Remove stop words.
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text


def clean_text_in_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the text in the rows of a Pandas dataframe.

    Args:
        df (pd.DataFrame): The dataframe to clean.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """

    # Create a new column in the dataframe for the cleaned text.
    frame['cleaned_text'] = frame['text'].apply(clean_text)

    return frame[['target', 'cleaned_text']]


class TrainModel:
    def __init__(self, frame: str, max_features: int = 10000, embedding_dim: int = 128):
        self.frame = frame
        self.tokeniser = Tokenizer(num_words=10000)
        self.sequences = None
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.model = None

    def tokenise(self):
        self.tokeniser.fit_on_texts(self.frame['cleaned_text'])

    def pad_sequences(self):
        self.sequences = pad_sequences(
            self.tokeniser.texts_to_sequences(self.frame['cleaned_text']), maxlen=60)

    def train_model(self):
        # Create the model
        self.model = Sequential([
            Embedding(self.max_features + 1, self.embedding_dim),
            Dropout(0.2),
            GlobalAveragePooling1D(),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='accuracy', patience=2)

        # Train the model
        self.model.fit(self.sequences, self.frame['target'],
                       epochs=10, callbacks=[early_stopping])

    def save_model(self):
        with open("model.pkl", "wb") as f:
            pickle.dump(self.model, f)
            print("model saved as model.pkl")

        with open("tokeniser.pkl", "wb") as f:
            pickle.dump(self.tokeniser, f)
            print("model saved as tokeniser.pkl")


class PredictSentiment:
    def __init__(self, list_of_sentences: List) -> List:
        self.tokenised_sentences = None
        self.sequences = None
        self.tokeniser = None
        self.model = None
        self.list_of_sentences = list_of_sentences

    def tokenise(self):
        # check if tokeniser is present
        if not check_if_file_present(path_to_file="tokeniser.pkl"):
            HandleS3Objects(origin="model/tokeniser.pkl",
                            bucket="sentiment-training-data-bucket",
                            destination="tokeniser.pkl").obtain_file_from_bucket()
        self.tokeniser = pickle.load(open("tokeniser.pkl", "rb"))
        # self.tokenised_sentences = self.tokeniser.fit_on_texts(self.list_of_sentences)

    # def pad_sequences(self):
    #     self.sequences = pad_sequences(
    #         self.tokeniser.texts_to_sequences(self.list_of_sentences), maxlen=60)

    def predict(self):
        # check if model is present
        if not check_if_file_present(path_to_file="model.pkl"):
            HandleS3Objects(origin="model/model.pkl",
                            bucket="sentiment-training-data-bucket",
                            destination="model.pkl").obtain_file_from_bucket()
        self.model = pickle.load(open("model.pkl", "rb"))
        # predict list
        l_emotion_pred = []
        for sentence in self.list_of_sentences:
            tokenised_sentence = self.tokeniser.texts_to_sequences([sentence])
            l_emotion_pred.append(self.model.predict(tokenised_sentence).item(0))
        return l_emotion_pred


