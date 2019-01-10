import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History
from keras import optimizers

from keras.regularizers import l2

from keras.datasets import imdb


def onehot_encoding(data, size):
    onehot = np.zeros((len(data), size))
    for i, d in enumerate(data):
        onehot[i,d] = 1.
    return onehot


def prob_to_sentiment(y):

    if(prob>0.9): return "fantastica"
    elif(prob>0.75): return "ottima"
    elif(prob>0.55): return "buona"
    elif(prob>0.45): return "neutrale"
    elif(prob>0.25): return "negativa"
    elif(prob>0.1): return "brutta"
    else: return "pessima"


(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)

X_train.shape
X_test.shape

word_index = imdb.get_word_index()
word_index['love']

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
reverse_word_index.get(14-3)

decoded_review = [reverse_word_index.get(i-3, '?') for i in X_train[0]]
decoded_review = ' '.join(decoded_review)
decoded_review

X_train_oh = onehot_encoding(X_train, 5000)
X_test_oh = onehot_encoding(X_test, 5000)

X_train_oh.shape

from keras.layers import Dropout

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(5000,), kernel_regularizer=l2(0.1)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01) ))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_oh, y_train, epochs=100, batch_size=512)

model.evaluate(X_test_oh, y_test)

review = "This movie will blow your mind and break your heart - and make you desperate to go back for more. Brave, brilliant and better than it has any right to be."

from re import sub

review = sub(r'[^\w\s]','',review)

review = review.lower()

review = review.split(" ")

review_array = []

for word in review:
    if(word in word_index):
        index = word_index[word]
        if(index <= 5000):
            review_array.append(index+3)


x = onehot_encoding([review_array], 5000)
x.shape

prob = model.predict(x)
prob
print("La recensione è %s" % prob_to_sentiment(prob))
