import numpy as np
from unit import sigmoid


def binary_cross_entropy(y, z):
  return -y*np.log2(z) - (1-y)*np.log2(1-z)

if __name__ == '__main__':
  x = np.array([20, 10, 4, 0])
  w = np.array([0.1, -0.2, -0.3, 1])
  b = 1.

  u = w.dot(x)
  z = sigmoid(u+b)

  print("z={}".format(z))
  for y in [0, 1]:
    print("y={}:".format(y), binary_cross_entropy(y, z))
