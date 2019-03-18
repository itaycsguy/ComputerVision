import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


"""
    TODO: Part of this assignment unique implementation - 
    more then binary segmentation
"""
class Segments:
    def __init__(self):
        pass

    def get_all_segments(self):
        pass

    def add_segment(self):
        pass



"""
    TODO: Basic model which is expand the 'Segmentation_Example.py' to class view - 
    need to use it with interactive refinements + mask input by the user
"""
class ImModel:
    def __init__(self, img_name, rect , iterations):
        self._img = cv.imread("images\\" + img_name)
        self._mask = np.zeros(self._img.shape[:2], np.uint8)
        self._rect = rect
        self._bgdModel = np.zeros((1, 65), np.float64)
        self._fgdModel = np.zeros((1, 65), np.float64)
        self._iterations = iterations
        self._mode = cv.GC_INIT_WITH_RECT
        if not rect:
            self._mode = cv.GC_INIT_WITH_MASK


    def get_result_img(self):
        mask, bgdModel, fgdModel = cv.grabCut(self._img,
                                              self._mask,
                                              self._rect,
                                              self._bgdModel,
                                              self._fgdModel,
                                              self._iterations,
                                              self._mode)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        img = self._img * mask[:, :, np.newaxis]
        return img


    def plot_result(self, img):
        plt.imshow(img), plt.colorbar(), plt.show()


if __name__ == "__main__":
    rect = (50, 50, 450, 290)
    iterations = 5
    imModel = ImModel("messi.jpg", rect, iterations)
    img = imModel.get_result_img()
    imModel.plot_result(img)
