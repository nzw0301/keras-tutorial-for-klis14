import numpy as np

from unit import sigmoid
from loss import binary_cross_entropy

if __name__ == '__main__':
  np.random.seed(1112)

  x, y  = np.array([20, 10, 4, 0]), 1.
  w, b = np.array([0.1, -0.2, -0.3, 1]), 1.

  eta = .01

  print(sigmoid(w.dot(x)+b))

  for i in range(100):
    z = sigmoid(w.dot(x)+b)
    print(i, binary_cross_entropy(y, z))

    # update
    for j, w_j in enumerate(w):
      w[j] -= eta*(z-y)*x[j]
    b -= eta*(z-y)

  print(sigmoid(w.dot(x)+b))


