from filters import *
import numpy as np

image = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

kernel = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

out = conv_nested(image, kernel)
print(out)