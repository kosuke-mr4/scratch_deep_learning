import numpy as np

x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
y = np.argmax(x, axis=1)

print(y)  # [1 2 1 0]

y1 = np.argmax(x, axis=0)
print(y1)  # [3 0 1] # ↓で見てる？
