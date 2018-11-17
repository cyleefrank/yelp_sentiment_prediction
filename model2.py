# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

'''
Read reviews from a JSON-formatted file into an array.
'''
import json
import string
import re
import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.optimizers import SGD, Adam
from sklearn.metrics import accuracy_score

lines = [];
num_pos = 0;
num_neg = 0;
num_total = 75000;

with open('/Users/edgarleung/Downloads/yelp_dataset/yelp_academic_dataset_review.json', 'r') as f:
    for line in f:
        if (len(lines) >= (num_total * 2)):
            break;

        json_info = json.loads(line);

        if json_info['stars'] > 3:
            if num_pos > num_total:
                continue;
            num_pos = num_pos + 1;
        elif json_info['stars'] < 3:
            if num_neg > num_total:
                continue;
            num_neg = num_neg + 1;
        else:
            continue;

        lines.append(json.loads(line));

'''
Separate line data into reviews and labels
'''
reviews = [line['text'] for line in lines];

stars = [line['stars'] for line in lines];
labels = ['1' if star > 3 else '0' for star in stars];

'''
Clean each document by removing unnecesary characters and splitting by space.
'''


def clean_document(doco):
    punctuation = string.punctuation + '\n\n';
    punc_replace = ''.join([' ' for s in punctuation]);
    doco_clean = doco.replace('-', ' ');
    doco_alphas = re.sub(r'\W +', '', doco_clean)
    trans_table = str.maketrans(punctuation, punc_replace);
    doco_clean = ' '.join([word.translate(trans_table) for word in doco_alphas.split(' ')]);
    doco_clean = doco_clean.split(' ');
    doco_clean = [word.lower() for word in doco_clean if len(word) > 0];

    return doco_clean;


# Generate a cleaned reviews array from original review texts
review_cleans = [clean_document(doc) for doc in reviews];
sentences = [' '.join(r) for r in review_cleans]

# Use a Keras Tokenizer and fit on the sentences

tokenizer = Tokenizer();
tokenizer.fit_on_texts(sentences);
text_sequences = np.array(tokenizer.texts_to_sequences(sentences));
sequence_dict = tokenizer.word_index;
word_dict = dict((num, val) for (val, num) in sequence_dict.items());

# We get a map of encoding-to-word in sequence_dict

# Generate encoded reviews
reviews_encoded = [];
for i, review in enumerate(review_cleans):
    reviews_encoded.append([sequence_dict[x] for x in review]);

# Plot a Histogram of length of reviews
lengths = [len(x) for x in reviews_encoded];
with plt.xkcd():
    plt.hist(lengths, bins=range(100))



# Truncate and Pad reviews at a Maximum cap of 60 words.
max_cap = 60;
X = pad_sequences(reviews_encoded, maxlen=max_cap, truncating='post')

# Obtain a One-hot Y array for each review label.
Y = np.array([[0,1] if '0' in label else [1,0] for label in labels])

# Get a randomized sequence of positions to shuffle reviews
np.random.seed(1024);
random_posits = np.arange(len(X))
np.random.shuffle(random_posits);

# Shuffle X and Y
X = X[random_posits];
Y = Y[random_posits];

# Divide the reviews into Training, Dev, and Test data.
train_cap = int(0.85 * len(X));
dev_cap = int(0.93 * len(X));

X_train, Y_train = X[:train_cap], Y[:train_cap];
X_dev, Y_dev = X[train_cap:dev_cap], Y[train_cap:dev_cap];
X_test, Y_test = X[dev_cap:], Y[dev_cap:]



model = Sequential();
model.add(Embedding(len(word_dict)+1, max_cap, input_length=max_cap));
model.add(LSTM(60, return_sequences=True, recurrent_dropout=0.5));
model.add(Dropout(0.5))
model.add(LSTM(60, recurrent_dropout=0.5));
model.add(Dense(60, activation='relu'));
model.add(Dense(2, activation='softmax'));
print(model.summary());

optimizer = Adam(lr=0.01, decay=0.001);
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# fit model
model.fit(X_train, Y_train, batch_size=64, epochs=10, validation_data=(X_dev, Y_dev))


# Obtain predictions
predictions = model.predict_classes(X_test)

# Convert Y_test to the same format as predictions
actuals = [0 if y[0] == 1 else 1 for y in Y_test];

# Use SkLearn's Metrics module
accuracy_score(predictions, actuals)