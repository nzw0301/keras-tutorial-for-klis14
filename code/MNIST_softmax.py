import numpy as np
np.random.seed(13)

# Kerasの読み込み
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Activation
from keras.utils import np_utils

# 入力層の次元数 (28x28)
nb_unit_size = 784

# 出力層の数(0~9までの10)
nb_class = 10

# MNIST の読み込み
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# one-hot representation
y_train = np_utils.to_categorical(y_train, nb_class)
y_test = np_utils.to_categorical(y_test, nb_class)

# 画像を28x28 => 784のベクトルに変換
X_train = X_train.reshape(X_train.shape[0], nb_unit_size)
X_test  = X_test.reshape(X_test.shape[0], nb_unit_size)

# scaling (0~255から0~1.の間の値を取るように変換)
X_train = X_train.astype('float32')/255
X_test  = X_test.astype('float32')/255

# 0か9までを分類するパーセプトロン
model = Sequential()  
model.add(Dense(10, input_dim=nb_unit_size))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer="adadelta",
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, verbose=2, validation_split=0.15)
