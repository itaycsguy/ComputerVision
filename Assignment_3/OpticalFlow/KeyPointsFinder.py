import cv2
import numpy as np


class KeyPointsFinder:

    """
        [1] cell_size - size of block to divide the image, default: 8 -> (8, 8)
        [2] bin_n     - number of bins for the histogram at each cell, default: 16
        [3] win_size  - size the image (expecting to 2^x type of size), default: (128, 64)
    """
    def __init__(self, image_path, cell_size=8, bin_n=9, win_size=(128, 64)):
        # These are parameters which required from the program to get initially
        self._cell_size = cell_size     # Descriptor cell computation on the image after resizing
        self._bin_n = bin_n             # Number of bins
        self._win_size = win_size
        self._real_image = cv2.imread(image_path)
        self._resized_image = cv2.resize(self._real_image, self._win_size, interpolation=cv2.INTER_CUBIC)
        self._feature_vectors = None


    """
        Return the original image
    """
    def get_real_image(self):
        return self._real_image


    """
        Return the resized image
    """
    def get_resized_image(self):
        return self._resized_image


    """
        Return a reshaped structure of the HOG descriptor
        [1] raw_hist - the raw data that HOG finds
    """
    def __reshape(self, raw_hist):
        hists = list()
        for i in range(0, len(raw_hist), self._bin_n):
            hists.append(raw_hist[i:(i + self._bin_n)].ravel())

        return hists


    """
        Reducing one dimensionality
        [1] multi_feature_vectors - list[list[list[number]]] type of structure to list[list[number]]
    """
    def __reduce_dimension(self, multi_feature_vectors):
        reduced = []
        for single_hog in multi_feature_vectors:
            for hist in single_hog:
                reduced.append(hist)

        return reduced


    """
        HOG descriptor 
        [1] image - an openCV image
    """
    def get_native_hog(self, image):
        winSize = (image.shape[1], image.shape[0])
        blockSize = (self._cell_size, self._cell_size)
        blockStride = (self._cell_size, self._cell_size)
        cellSize = (self._cell_size, self._cell_size)
        nbins = self._bin_n
        derivAperture = 1
        winSigma = -1.0
        histogramNormType = 1
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        signedGradient = 0
        hog_instance = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                                         derivAperture, winSigma, histogramNormType, L2HysThreshold,
                                         gammaCorrection, nlevels, signedGradient)

        # compute(img[, winStride[, padding[, locations]]]) -> descriptors
        winStride = (self._cell_size, self._cell_size)
        padding = (self._cell_size, self._cell_size)
        # correlated with n-bins number -> not working at all:
        locations = []
        hist = hog_instance.compute(image, winStride, padding, locations)
        # return => |hist| = 1152 * 9 = |descriptor| * 9? => 128 blocks * 9 bins * 9?
        return self.__reshape(hist)


    """
        Generating all possible key-points at the current executed image
    """
    def generate_key_points(self):
        self._feature_vectors = np.asarray(self.__reduce_dimension(self.get_native_hog(self._resized_image)), dtype=np.float32)


    """
        Get the n strongest key points from all image that were found
        [1] img - an image
        [2] n   - number of key points to find
    """
    def get_strongest(self, img, n):
        data_len = len(self._feature_vectors)
        # Descriptor vector per block
        data_len = data_len / (self._bin_n * self._cell_size**2)
        blocks_idx = np.flip(np.argsort(np.asarray(self._feature_vectors)))[:n]
        blocks_loc = self._cell_size**2 * np.asarray(blocks_idx / data_len, dtype=np.uint32)
        rows = np.asarray(blocks_loc / img.shape[0], dtype=np.uint32)
        cols = np.asarray(blocks_loc / img.shape[1], dtype=np.uint32)

        key_points_idx = list()
        for x, y in zip(rows, cols):
            key_points_idx.append((x, y))

        return key_points_idx


    """
        Finding n key points at the image
    """
    def get_key_points(self, n):
        self.generate_key_points()
        return finder.get_strongest(self._resized_image, n)


    """
        Plot found key points on the image
        [1] key_points - an array of key points
    """
    def plot_key_points(self, key_points):
        for point in key_points:
            cv2.circle(self._resized_image, point, 5, (0, 255, 0), -1)
        cv2.imshow('HOG Image Strongest Key Points', self._resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    sys_path = "D:\\PycharmProjects\\ComputerVision\\Assignment_3\\OpticalFlow\\Datasets\\"
    image = sys_path + "image001.jpg"
    finder = KeyPointsFinder(image)
    key_points = finder.get_key_points(50)
    finder.plot_key_points(key_points)

