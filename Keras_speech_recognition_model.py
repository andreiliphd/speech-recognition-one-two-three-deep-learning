import librosa
from os import listdir
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM

def loadSound(path):
    soundList = listdir(path)
    loadedSound = []
    for sound in soundList:
        Y, sr = librosa.load(path + sound)
        loadedSound.append(librosa.feature.mfcc(Y, sr=sr))
    return np.array(loadedSound)

one = loadSound('./voice_123/one/')
one = loadSound('./voice_123/one/')
two = loadSound('./voice_123/two/')
three = loadSound('./voice_123/three/')
X = np.concatenate((one, two, three), axis=0)
one_label = np.concatenate((np.ones(10), np.zeros(10), np.zeros(10)))
two_label = np.concatenate((np.zeros(10), np.ones(10), np.zeros(10)))
three_label = np.concatenate((np.zeros(10), np.zeros(10), np.ones(10)))
y = np.concatenate((one_label[:, None], two_label[:, None], three_label[:, None]), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential()
model.add(LSTM(units=256,return_sequences=True))
model.add(LSTM(units=128,return_sequences=True))
model.add(LSTM(units=64,return_sequences=True))
model.add(LSTM(units=32))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10000,validation_data=(X_test, y_test))
