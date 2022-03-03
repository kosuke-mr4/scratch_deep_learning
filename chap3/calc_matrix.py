import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)  # (2, 3)

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)  # (3, 2)

AB = np.dot(A, B)
print(AB)
# [[22 28]
#  [49 64]]

## Emulated NN

X = np.array([1, 2])
print(X.shape)  # (2,)

W = np.array([[1, 3, 5], [2, 4, 6]])
print(W.shape)  # (2, 3)

Y = np.dot(X, W)
print(Y)  # [ 5 11 17]
