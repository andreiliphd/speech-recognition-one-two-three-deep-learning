import librosa
from os import listdir
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F


def loadSound(path):
    soundList = listdir(path)
    loadedSound = []
    for sound in soundList:
        Y, sr = librosa.load(path + sound)
        loadedSound.append(librosa.feature.mfcc(Y, sr=sr))
    return np.array(loadedSound)


one = loadSound('./voice_123/one/')
two = loadSound('./voice_123/two/')
three = loadSound('./voice_123/three/')
X = np.concatenate((one, two, three), axis=0)
one_label = np.concatenate((np.ones(10), np.zeros(10), np.zeros(10)))
two_label = np.concatenate((np.zeros(10), np.ones(10), np.zeros(10)))
three_label = np.concatenate((np.zeros(10), np.zeros(10), np.ones(10)))
y = np.concatenate((np.repeat(0, 10), np.repeat(1, 10), np.repeat(2, 10)), axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
X_train = X_train.swapaxes(1,0)
X_test = X_test.swapaxes(1,0)
X_train_torch = torch.from_numpy(X_train).float()
X_test_torch = torch.from_numpy(X_test).float()
y_train_torch = torch.from_numpy(y_train).long()
y_test_torch = torch.from_numpy(y_test).long()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm1 = nn.LSTM(input_size=87, hidden_size=256)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64)
        self.lstm4 = nn.LSTM(input_size=64, hidden_size=32)
        self.fc1 = nn.Linear(in_features=32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=3)

    def forward(self, x):
        x = torch.tanh(self.lstm1(x)[0])
        x = torch.tanh(self.lstm2(x)[0])
        x = torch.tanh(self.lstm3(x)[0])
        x = torch.tanh(self.lstm4(x)[0][0])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model = RNN()
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(10000):
    y_pred = model(X_train_torch)
    loss = loss_fn(y_pred, y_train_torch)
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        def accuracy_calc(X, y, type):
            correct = 0
            total_correct = 0
            outputs = model(X).detach().numpy()
            label = y.detach().numpy()
            for number in range(outputs.shape[0]):
                correct = np.argmax(outputs[number]) == label[number]
                total_correct += correct
            print(type + ' accuracy: ' + str(total_correct/outputs.shape[0] * 100) + '%')
        accuracy_calc(X_train_torch, y_train_torch, "Training")
        accuracy_calc(X_test_torch, y_test_torch, "Testing")
