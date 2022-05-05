from sklearn.feature_extraction.text import HashingVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from nltk.stem.porter import PorterStemmer
import pickle
import re
from nltk import probability
import numpy as np
import pandas as pd
from tqdm import tqdm
from apps.tweetml.fetch_tweets import *
import nltk
nltk.download('stopwords')

all_tweets, positives = [], []
positive_tweet_trend, negative_tweet_trend = [], []
overall_trend = []
recent_prob_list = []

def processing_tweets():
    def preprocess_tweet(text):
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        text = re.sub('[\W]+', ' ', text.lower())
        text = text+' '.join(emoticons).replace('-', '')
        return text

    tqdm.pandas()
    df = pd.read_csv('suicidal_data.csv')
    df['tweet'] = df['tweet'].progress_apply(preprocess_tweet)
    porter = PorterStemmer()

    def tokenizer_porter(text):
        return [porter.stem(word) for word in text.split()]

    stop = stopwords.words('english')

    def tokenizer(text):
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\(|D|P)', text.lower())
        text = re.sub('[\W]+', ' ', text.lower())
        text += ' '.join(emoticons).replace('-', '')
        tokenized = [w for w in tokenizer_porter(text) if w not in stop]
        return tokenized

    vect = HashingVectorizer(decode_error='ignore', n_features=2**21,
                             preprocessor=None, tokenizer=tokenizer)

    clf = SGDClassifier(loss='log', random_state=1)

    X = df["tweet"].to_list()
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0)  # cross validation check

    X_train = vect.transform(X_train)
    X_test = vect.transform(X_test)

    classes = np.array([0, 1])
    clf.partial_fit(X_train, y_train, classes=classes)

    print('Accuracy: %.3f' % clf.score(X_test, y_test))
    clf = clf.partial_fit(X_test, y_test)

    # label = {0:'negative', 1:'positive'}
    # example = ["When he optimized the solution I fall in love ❤️ with data structures."]
    # X = vect.transform(example)
    # print('Prediction: %s\nProbability: %.2f%%'
    #       %(label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))
    count = 0
    for tweet in get_tweets():
        tweet_link = tweet[len(tweet)-24:len(tweet)]
        label = {0: 'negative', 1: 'positive'}
        X = vect.transform([tweet])
        prediction, prob = label[clf.predict(X)[0]], np.max(
            clf.predict_proba(X))*100
        if prediction == "positive":
            positives.append([tweet, prediction, "{:.2f}".format(prob), tweet_link])
            positive_tweet_trend.append(prob)
            overall_trend.append(prob)
            if(count<10):
                recent_prob_list.append(prob)
                count+=1
        else:
            all_tweets.append([tweet, prediction, "{:.2f}".format(prob), tweet_link])
            negative_tweet_trend.append(prob)
            overall_trend.append(100-prob)
            if(count<10):
                recent_prob_list.append(100-prob)
                count+=1


# if len(positives) == 0:
#     print("There are no suicidal tweets")
# else:
#     print(positives)
# for i in range(len(positives)):
#     print('Tweet: %%s \nPrediction: %s\nProbability: %.2f%%' %(positives[i][0],positives[i][1]), positives[i][2])
