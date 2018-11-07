import librosa
from os import listdir
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

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
X.shape
one_label = np.concatenate((np.ones(10), np.zeros(10), np.zeros(10)))
two_label = np.concatenate((np.zeros(10), np.ones(10), np.zeros(10)))
three_label = np.concatenate((np.zeros(10), np.zeros(10), np.ones(10)))
y = np.concatenate((one_label[:, None], two_label[:, None], three_label[:, None]), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train = X_train.swapaxes(1,0)
X_test = X_test.swapaxes(1,0)
y_train.shape
X_train_placeholder = tf.placeholder(dtype=tf.float32,shape=[None, None, 87])
y_train_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 3])

def RNN(x, output_size, num_hidden, timesteps):
    output = tf.Variable((0, 0), trainable=False, validate_shape=False, dtype=tf.float32)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, activation="tanh")
    state = lstm_cell.zero_state(batch_size=output_size, dtype=tf.float32)
    for batch in range(timesteps):
        output, state = lstm_cell(x[batch], state)
    return output

nn = RNN(X_train_placeholder, 27, 256, 20)
nn = tf.layers.dense(nn, 128, activation='relu')
nn = tf.layers.dense(nn, 64, activation='relu')
nn = tf.layers.dense(nn, 32, activation='relu')
nn = tf.layers.dense(nn, 3, activation='relu')

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train_placeholder, logits=nn))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        _, loss_val = sess.run([optimizer,cross_entropy], feed_dict={X_train_placeholder: X_train, y_train_placeholder:y_train})
        if i%2 == 0:
            matches = tf.equal(tf.argmax(nn,1),tf.argmax(y_train_placeholder,1))
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            print('Currently on step {}'.format(i))
            print('Loss: ', str(loss_val))
            print('Training accuracy is:')
            print(sess.run(acc,feed_dict={X_train_placeholder: X_train, y_train_placeholder: y_train}))
