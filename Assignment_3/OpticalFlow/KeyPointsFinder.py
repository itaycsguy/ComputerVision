import cv2
import numpy as np


class KeyPointsFinder:

    """
    KeyPointsFinder(image_path[, cell_size=8[, bin_n=9[, win_size=(128, 64)]]]) -> HOG descriptor
    .   @brief Computes the hog descriptor per an image.
    .   @param cell_size Size of the divided image.
    .   @param bin_n Size of the histogram per block.
    .   @param win_size Size of the scaled image (cols, rows).
    """
    def __init__(self, image_path, cell_size=8, bin_n=9, win_size=(128, 64)):
        # These are parameters which required from the program to get initially
        self._cell_size = cell_size     # Descriptor cell computation on the image after resizing
        self._bin_n = bin_n             # Number of bins
        self._win_size = win_size
        self._real_image = cv2.imread(image_path)
        self._scaled_image = cv2.resize(self._real_image, self._win_size, interpolation=cv2.INTER_CUBIC)
        self._hog_descriptor = None
        self._grad = None
        self._angleOfs = None


    """
    get_real_image() -> original image
    .   @brief Getting the original input image.
    """
    def get_real_image(self):
        return self._real_image


    """
    get_scaled_image() -> scaled image
    .   @brief Getting the scaled input image.
    """
    def get_scaled_image(self):
        return self._scaled_image


    """
    __reshape(raw_hist) -> reshaped hog descriptor
    .   @brief Reshaping the hog descriptor.
    .   @param raw_hist A raw data normalized histogram.
    """
    def __reshape(self, raw_hist):
        hists = list()
        for i in range(0, len(raw_hist), self._bin_n):
            hists.append(raw_hist[i:(i + self._bin_n)].ravel())

        return hists


    """
    __reduce_dimension(multi_feature_vectors) -> reduced dimension array
    .   @brief Reduces a dimension.
    .   @param multi_feature_vectors A multi-dimensional array of features.
    """
    def __reduce_dimension(self, multi_feature_vectors):
        reduced = []
        for single_hog in multi_feature_vectors:
            for hist in single_hog:
                reduced.append(hist)

        return reduced


    """
    get_native_hog(image) -> HOG descriptor
    .   @brief Computes the hog descriptor per an image.
    .   @param image An image.
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
        winStride = winSize
        padding = (self._cell_size, self._cell_size)
        # correlated with n-bins number -> not working at all:
        locations = []
        hist = hog_instance.compute(image, winStride, padding, locations)
        grad = np.zeros(winStride, dtype=np.float32)
        angleOfs = np.zeros(winStride, dtype=np.float32)
        grad, angleOfs = hog_instance.computeGradient(image, grad, angleOfs)
        return self.__reshape(hist), self.__reshape(grad), self.__reshape(angleOfs)


    """
    generate_key_points() -> None
    .   @brief Computes all possible key-points at the current image.
    """
    def generate_key_points(self):
        self._hog_descriptor, self._grad, self._angleOfs = self.get_native_hog(self._scaled_image)
        self._hog_descriptor = np.asarray(self.__reduce_dimension(self._hog_descriptor), dtype=np.float32)


    """
    get_strongest(img[, n]]) -> key_points indices
    .   @brief Computes the strongest key-points from HOG descriptor.
    .   @param img An image.
    .   @param n Number of key points to find.
    """
    def get_strongest(self, n):
        row_blocks = self._scaled_image.shape[1] / self._cell_size
        col_blocks = self._scaled_image.shape[0] / self._cell_size
        descriptor_size = len(self._hog_descriptor)
        descriptor_size_per_block = int(descriptor_size / (row_blocks * col_blocks))

        descriptors_strength = list()
        for i in range(0, descriptor_size, descriptor_size_per_block):
            descriptors_strength.append(np.sum(self._hog_descriptor[i:(i + descriptor_size_per_block)]))

        # sorted by the maximum
        sorted_blocks_idx = np.flip(np.argsort(descriptors_strength))
        int_blocks = np.asarray(sorted_blocks_idx / col_blocks, dtype=np.uint32)
        rows = int_blocks * self._cell_size
        rows = rows[:n]
        cols = np.asarray(np.asarray((sorted_blocks_idx / col_blocks) - int_blocks, dtype=np.float32) * col_blocks * self._cell_size, dtype=np.uint32)
        cols = cols[:n]

        # TODO:
        """
        need to pick up stronger locations 
        at the maximum block if it exists 
        until the first less then the next maximum block will arrived
        """

        locations = list()
        for x, y in zip(rows, cols):
            point = (x, y)
            locations.append(point)

        return locations


    """
    get_key_points(n) -> key_points
    .   @brief Computes n key-points as a wrapper function.
    .   @param n Number of key points to find.
    """
    def get_key_points(self, n):
        self.generate_key_points()
        return finder.get_strongest(n)


    """
    plot_key_points(key_points) -> None
    .   @brief Plotting the key-points on the image.
    .   @param key_points Key-points that have been found.
    """
    def plot_key_points(self, key_points):
        for point in key_points:
            cv2.circle(self._scaled_image, point, 5, (0, 255, 0), -1)
        cv2.imshow('HOG Image Strongest Key Points', self._scaled_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    sys_path = "D:\\PycharmProjects\\ComputerVision\\Assignment_3\\OpticalFlow\\Datasets\\"
    image = sys_path + "image001.jpg"
    finder = KeyPointsFinder(image)
    key_points = finder.get_key_points(5)
    finder.plot_key_points(key_points)

