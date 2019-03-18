import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Code example using rect option with the grabCut method

img = cv.imread('images\\messi.jpg')
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (50, 50, 450, 290)
iterations = 5

mask, bgdModel, fgdModel = cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterations, cv.GC_INIT_WITH_RECT)

mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
img = img*mask[:, :, np.newaxis]

plt.imshow(img), plt.colorbar(), plt.show()
