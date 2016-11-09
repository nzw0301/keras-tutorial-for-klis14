import numpy as np

def sigmoid(u):
  return 1/(1+np.exp(-u))

if __name__ == '__main__':
  x = np.array([20, 10, 4, 0])
  w = np.array([0.1, -0.2, -0.3, 1])
  b = 1.

  u = w.dot(x)
  z = sigmoid(u+b)
  print(z)
