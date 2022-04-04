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
import fetch_tweets
import nltk
nltk.download('stopwords')


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
[w for w in tokenizer_porter(
    'my life is meaningless i just want to end my life so badly my life is completely empty') if w not in stop]


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

positives = []
for tweet in fetch_tweets.get_tweets():
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([tweet])
    prediction, prob = label[clf.predict(X)[0]], np.max(
        clf.predict_proba(X))*100
    if prediction == "positive":
        positives.append([tweet, label[clf.predict(X)[0]],
                         np.max(clf.predict_proba(X))*100])
    print("tweet :", tweet)
    print('Prediction: %s\nProbability: %.2f%%' %
          (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))
if len(positives) == 0:
    print("There are no suicidal tweets")
else:
    print(positives)
    # for i in range(len(positives)):
    #     print('Tweet: %%s \nPrediction: %s\nProbability: %.2f%%' %(positives[i][0],positives[i][1]), positives[i][2])
