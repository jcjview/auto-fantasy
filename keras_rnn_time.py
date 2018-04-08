import numpy
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

# load ascii text and covert to lowercase
filename = "shakespeare.txt"
raw_text=[]
with open(filename,encoding='utf-8') as fp:
    for line in fp:
        if(len(line)<40):continue
        line=line.replace(" ","_")
        raw_list=[s for s in line]
        raw_text.append(" ".join(raw_list))

seq_length = 50
tokenizer = Tokenizer(filters='!"#$%&()*+-/:;<=>@[\\]^`{|}~\t\n')
tokenizer.fit_on_texts(raw_text)
sequences = tokenizer.texts_to_sequences(raw_text)
x=pad_sequences(sequences,maxlen=seq_length)
print(x.shape)
y = numpy.copy(x)
y[:, :-1] = x[:, 1:]
y = np_utils.to_categorical(y)
n_vocab=len(tokenizer.word_index)
char_to_int=tokenizer.word_index
int_to_char={}
for key,value in char_to_int.items():
    int_to_char[value]=key

import pickle
with open('vab.pkl3', 'wb') as f:
     pickle.dump(tokenizer.word_index, f)

x = x / float(n_vocab)
print(x.shape)
print(y.shape)
n_patterns = len(x)
x = numpy.reshape(x, (n_patterns, seq_length, 1))
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length,1), return_sequences=True))
model.add(TimeDistributed(Dense(y.shape[2], activation='softmax')))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1024)
model.save_weights('weights.h5')