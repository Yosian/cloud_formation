from sentiment_analysis.code.infrastructure import HandleS3Objects
from sentiment_analysis.code.repository import make_training_dataframe, \
    clean_text_in_dataframe, TrainModel, PredictSentiment

# get training dataframe
df = make_training_dataframe(file='data.csv')
df['target'] = [0 if label == 0 else 1 for label in df.target.values]

# neural network #####################################################################
df = clean_text_in_dataframe(frame=df)

tm = TrainModel(frame=df)

tm.tokenise()
tm.pad_sequences()
tm.train_model()
tm.save_model()

HandleS3Objects(origin="model.pkl",
                bucket="sentiment-training-data-bucket",
                destination="model/model.pkl").upload_file_to_bucket()

HandleS3Objects(origin="tokeniser.pkl",
                bucket="sentiment-training-data-bucket",
                destination="model/tokeniser.pkl").upload_file_to_bucket()

# predict
l_sentences = ["this is shit", "I love this stuff, it is awesome", "I hate this crap",
               "what a beautiful day we have today", "this tv show is ok",
               "I did not like this film at all", "neutral comment"]
predict = PredictSentiment(list_of_sentences=l_sentences)
predict.tokenise()
l_emotions = predict.predict()
results = dict(zip(l_sentences, l_emotions))
