## Students And Developers: Itay Guy, 305104184 & Elias Jadon, 207755737

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
    get_copy_real_image() -> copy of the original image
    .   @brief Generating a copy of the original input image.
    """
    def get_copy_real_image(self):
        return self._real_image.copy()


    """
    reshape(raw_hist) -> reshaped hog descriptor
    .   @brief Reshaping the hog descriptor.
    .   @param raw_hist A raw data normalized histogram.
    """
    def reshape(self, raw_hist):
        hists = list()
        for i in range(0, len(raw_hist), self._bin_n):
            hists.append(raw_hist[i:(i + self._bin_n)].ravel())

        return hists


    """
    reduce_dimension(multi_feature_vectors) -> reduced dimension array
    .   @brief Reduces a dimension.
    .   @param multi_feature_vectors A multi-dimensional array of features.
    """
    def reduce_dimension(self, multi_feature_vectors):
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
        return self.reshape(hist), grad, angleOfs


    """
    generate_key_points() -> None
    .   @brief Computes all possible key-points at the current image.
    """
    def generate_key_points(self):
        self._hog_descriptor, self._grad, self._angleOfs = self.get_native_hog(self._scaled_image)
        self._hog_descriptor = np.asarray(self.reduce_dimension(self._hog_descriptor), dtype=np.float32)


    def fast(self):
        pass


    """
    get_strongest(img[, n]]) -> key_points indices
    .   @brief Computes the strongest key-points from HOG descriptor.
    .   @param img An image.
    .   @param n Number of key points to find.
    .   @param fast If to use with a smarter algorithm [default=True]
    """
    def get_strongest(self, n, fast=False):
        rows_size = self._scaled_image.shape[0]
        row_blocks = rows_size / self._cell_size
        cols_size = self._scaled_image.shape[1]
        col_blocks = cols_size / self._cell_size
        descriptor_size = len(self._hog_descriptor)
        descriptor_size_per_block = int(descriptor_size / (row_blocks * col_blocks))

        gx = self._grad[:, :, 0]
        gy = self._grad[:, :, 1]
        g = np.sqrt(np.power(gx, 2) + np.power(gy, 2))

        locations = list()
        values = list()
        if not fast:
            # test case which is running over all gx,gy pairs on the image
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    locations.append((j, i))
                    values.append(g[i, j])
        else:
            descriptors_strength = list()
            for i in range(0, descriptor_size, descriptor_size_per_block):
                descriptors_strength.append(np.mean(np.square(self._hog_descriptor[i:(i + descriptor_size_per_block)])))

            # sorted blocks by the maximum
            block_entries = np.flip(np.argsort(descriptors_strength))
            rows = np.asarray((block_entries / col_blocks), dtype=np.uint32) * self._cell_size
            cols = np.asarray(block_entries % col_blocks, dtype=np.uint32) * self._cell_size

            for x, y in zip(rows, cols):
                if (x, y) not in locations:
                    for i in range(self._cell_size):
                        for j in range(self._cell_size):
                            locations.append((y + j, x + i))
                            values.append(g[x + i, y + j])

        points = list()
        # sorted point values by the maximum
        all_peaks = np.flip(np.argsort(values))
        # considering the number of point the user asked for
        for i in all_peaks[:n]:
            points.append(locations[i])

        return points


    """
    get_key_points(n) -> key_points
    .   @brief Computes n key-points as a wrapper function.
    .   @param n Number of key points to find.
    .   @param fast If to use with a smarter algorithm [default=True]
    .   @param ret_scaled If to return the scaled points [default=False]
    """
    def get_key_points(self, n, fast=True, ret_scaled=False):
        self.generate_key_points()
        key_points = self.get_strongest(n, fast)
        if not ret_scaled:
            for i, _ in enumerate(key_points):
                y_rel = key_points[i][1] / self._scaled_image.shape[0]
                x_rel = key_points[i][0] / self._scaled_image.shape[1]
                key_points[i] = (int(y_rel * self._real_image.shape[1]), int(x_rel * self._real_image.shape[0]))
        return key_points



    """
    plot_key_points(key_points) -> None
    .   @brief Plotting the key-points on the image.
    .   @param key_points Key-points that have been found.
    """
    def plot_key_points(self, key_points):
        copy_image = self._real_image.copy()
        for point in key_points:
            cv2.circle(copy_image, point, 5, (0, 255, 0), -1)
        cv2.imshow('HOG Image Strongest Key Points', copy_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    sys_path = "D:\\PycharmProjects\\ComputerVision\\Assignment_3\\OpticalFlow\\Datasets\\"
    # image = sys_path + "vehicle_test.jpg"
    image = sys_path + "highway.jpg"
    finder = KeyPointsFinder(image)
    key_points = finder.get_key_points(1500)
    finder.plot_key_points(key_points)