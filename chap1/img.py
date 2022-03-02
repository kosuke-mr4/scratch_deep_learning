import matplotlib.pyplot as plt
from matplotlib.image import imread
from numpy import imag

img = imread("dataset/neko.png")
plt.imshow(img)

plt.show()
