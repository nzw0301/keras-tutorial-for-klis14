import numpy as np
np.random.seed(13)

# Kerasの読み込み
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Activation
from keras.utils import np_utils

# 入力層の次元数 (28x28)
nb_unit_size = 784

# MNIST の読み込み
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 0か1の数字のみ取り出す
indices_train = y_train < 2
indices_test  = y_test < 2

X_train = X_train[indices_train]
y_train = y_train[indices_train]
X_test = X_test[indices_test]
y_test = y_test[indices_test]

# 画像を28x28 => 784のベクトルに変換
X_train = X_train.reshape(X_train.shape[0], nb_unit_size)
X_test  = X_test.reshape(X_test.shape[0], nb_unit_size)

# scaling (0~255から0~1.の間の値を取るように変換)
X_train = X_train.astype('float32')/255
X_test  = X_test.astype('float32')/255

# 0か1かを判定する単純パーセプトロン
model = Sequential()  
model.add(Dense(1, input_dim=nb_unit_size))
model.add(Activation("sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer="adadelta",
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, verbose=2,  validation_split=0.15)
