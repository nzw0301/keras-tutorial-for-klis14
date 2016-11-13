import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
np.random.seed(2)

# 入力データの都合上，(1,4)の行列に変換
X = np.array([20, 10, 4, 0]).reshape((1, 4))
y = np.array([1.]) # 正解ラベル

model = Sequential()
model.add(Dense(1, input_shape=(4,)))
model.add(Activation("sigmoid"))

#2値分類なので，binary_crossentropy
model.compile(optimizer="sgd", loss="binary_crossentropy")

model.fit(X, y, verbose=2)

