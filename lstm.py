#Importing libraries
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import json
import pandas as pd

##Coverting JSON to pandas dataframe

def convert(x):
    ob = json.loads(x)
    for k, v in ob.items():
        if isinstance(v, list):
            ob[k] = ','.join(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                ob['%s_%s' % (k, kk)] = vv
            del ob[k]
    return ob
json_filename='review.json'
with open(json_filename,'rb') as f:
    data = f.readlines()

#Converting into pandas dataframe and filtering only text and ratings given by the users
df = pd.DataFrame([convert(line) for line in data])
data = df[['text', 'stars']]

#I have considered a rating above 3 as positive and less than or equal to 3 as negative.

data['sentiment']=['pos' if (x>3) else 'neg' for x in data['stars']]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
data['text']= [x.encode('ascii') for x in data['text']]

tokenizer = Tokenizer(nb_words = 2500, split=' ')
tokenizer.fit_on_texts(data['text'].values)
#print(tokenizer.word_index)  # To see the dicstionary
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 300
batch_size= 32

##Buidling the LSTM network

model = Sequential()
model.add(Embedding(2500, embed_dim,input_length = X.shape[1], dropout=0.1))
model.add(LSTM(lstm_out, dropout_U=0.1, dropout_W=0.1))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

Y = pd.get_dummies(data['sentiment']).values
X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.20, random_state = 36)

#Here we train the Network.

model.fit(X_train, Y_train, batch_size =batch_size, nb_epoch = 1,  verbose = 5)

# Measuring score and accuracy on validation set

score,acc = model.evaluate(X_valid, Y_valid, verbose = 2, batch_size = batch_size)
print("Logloss score: %.2f" % (score))
print("Validation set Accuracy: %.2f" % (acc))
