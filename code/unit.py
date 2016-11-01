import numpy as np
np.random.seed(1112) # 再現性のため，値はなんでもいい

def sigmoid(u):
  return 1/(1+np.exp(-u))


x = np.arange(-5, 5)
w = np.random.rand(10)

print(sigmoid(w.dot(x)))
