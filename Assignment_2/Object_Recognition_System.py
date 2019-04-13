## Students And Developers: Itay Guy, 305104184 & Elias Jadon, 207755737

import os
import cv2
import numpy as np
import dlib
import pickle
import argparse


trainImageDirName = "C://Users//user//Documents//GitHub//ComputerVision//Assignment_2//Datasets"
testImageDirName = "/Datasets/Testset"


"""
    A class who holds all the training data + labels
"""
class Database:
    # Defined at the published assignment
    ALLOWED_DIRS = {"airplane": 0, "elephant": 1, "motorbike": 2}

    """
        Making an hash which holds the data efficiently and conveniently
    """
    def __init__(self, dir_path):
        self._root_dir = dir_path
        self._image_hash = {}
        for outer_dir in os.listdir(dir_path):
            if outer_dir.lower() in Database.ALLOWED_DIRS.keys():
                inner_dir_path = dir_path + "//" + outer_dir
                for image_name in os.listdir(inner_dir_path):
                    self._image_hash[outer_dir + "//" + image_name] = Database.ALLOWED_DIRS[outer_dir.lower()]


    """
        Return root training-set directory
    """
    def get_root_dir(self):
        return self._root_dir


    """
        Return designed hash table contain all db content
    """
    def get_data_hash(self):
        return self._image_hash


    """
        Return number of directories who is participate of the process
    """
    def get_sets_amount(self):
        return len(Database.ALLOWED_DIRS)



class Features:
    __PICKLE_LOC = "var//Features.pkl"

    """
        [1] database  - a Database object
        [2] k         - clusters amount, default: 10
        [3] cell_size - size of block to divide the image, default: 8 -> (8, 8)
        [4] bin_n     - number of bins for the histogram at each cell, default: 9
        [5] win_size  - size the image (expecting to 2^x type of size), default: (64, 128)
    """
    def __init__(self, database, k=10, cell_size=8, bin_n=9, win_size=(64, 128)):
        # Database reference
        self._database = database
        # These are parameters which required from the program to get initially
        self._cell_size = cell_size     # Descriptor cell computation on the image after resizing
        self._bin_n = bin_n             # Number of bins
        self._win_size = win_size
        self._K = k                     # kmeans algorithm parameter
        # meta information
        self._feature_vectors = None
        self._feature_vectors_labels = None
        # computed information
        self._centers = None
        self._labels = None
        self._bows = None



    """
        HOG descriptor for list of images by using the native API from openCV 
        [1] images - list of openCV opened images
    """
    def native_hog_compute(self, images):
        samples = []
        winSize = (self._win_size[1], self._win_size[0])
        blockSize = (self._cell_size, self._cell_size)
        blockStride = (self._cell_size, self._cell_size)
        cellSize = (self._cell_size, self._cell_size)
        nbins = self._bin_n
        derivAperture = 1
        winSigma = 4.0
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                                derivAperture, winSigma, histogramNormType, L2HysThreshold,
                                gammaCorrection, nlevels)
        # compute(img[, winStride[, padding[, locations]]]) -> descriptors
        winStride = (self._cell_size, self._cell_size)
        padding = (self._cell_size, self._cell_size)
        locations = ((10, 20), (30, 30), (50, 50), (70, 70),
                     (90, 90), (110, 110), (130, 130), (150, 150),
                     (170, 170), (190, 190))
        for image in images:
            hist = hog.compute(image, winStride, padding, locations)
            samples.append(hist)
        return np.float32(samples)



    """
        HOG descriptor for an image by using the native API from openCV 
        [1] image - an opened openCV image
    """
    def get_native_hog(self, image):
        winSize = (image.shape[1], image.shape[0])
        blockSize = (self._cell_size, self._cell_size)
        blockStride = (self._cell_size, self._cell_size)
        cellSize = (self._cell_size, self._cell_size)
        nbins = self._bin_n
        derivAperture = 1
        winSigma = 4.0
        histogramNormType = 1
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 1
        nlevels = 64
        hog_instance = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                                         derivAperture, winSigma, histogramNormType, L2HysThreshold,
                                         gammaCorrection, nlevels)

        # compute(img[, winStride[, padding[, locations]]]) -> descriptors
        winStride = (self._cell_size, self._cell_size)
        padding = (self._cell_size, self._cell_size)
        locations = []  # correlated with n-bins number
        hist = hog_instance.compute(image, winStride, padding, locations)
        return hist


    """
        Return by HOG method an HOG descriptor for each cell_size sub-image of the input image
        [1] image_instance - an opened openCV image
    """
    def __get_HOG_desctiptor(self, image_instance):
        # Gradients computations
        gx = cv2.Sobel(image_instance, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(image_instance, cv2.CV_32F, 0, 1)

        # Degrees computations [0 - 180]
        mag, ang = cv2.cartToPolar(gx, gy)

        # bin separation by _bin_n steps
        bin = np.int32(self._bin_n * ang / (2 * np.pi))

        hists = []
        # The cell_size can be separated to cell_x, cell_y which at this case they are the same for always
        for i in range(0, int(image_instance.shape[0] / self._cell_size)):
            # Run along cols
            for j in range(0, int(image_instance.shape[1] / self._cell_size)):
                # Run along rows
                bin_cells = bin[(i * self._cell_size): (i * self._cell_size + self._cell_size), (j * self._cell_size): (j * self._cell_size + self._cell_size)]
                mag_cells = mag[(i * self._cell_size): (i * self._cell_size + self._cell_size), (j * self._cell_size): (j * self._cell_size + self._cell_size)]

                # Generate the an histogram by quantized degrees
                cell_hists = np.asarray([np.bincount(b.ravel(), m.ravel(), self._bin_n) for b, m in zip(bin_cells, mag_cells)])

                # Transform to Hellinger kernel [Normalization issue]
                eps = 1e-7
                cell_hists /= cell_hists.sum() + eps
                cell_hists = np.sqrt(cell_hists)
                cell_hists /= cv2.norm(cell_hists) + eps

                for hist in cell_hists.tolist():
                    hists.append(hist)

        ## There are few differences between this implementation and the native HOG from opencv -> take a pick:
        # print(np.asarray(hists).shape)
        # print(self.get_hog(image_instance).shape)
        # return the descriptor which is fully describe the image features
        return hists


    """
        Return 2D feature vectors due to its image-based arrangement
        [1] multi_feature_vectors - list[list[list[number]]] type of structure to list[list[number]]
    """
    def _reduce_to_2D(self, multi_feature_vectors):
        reduced = []
        for single_hog in multi_feature_vectors:
            for hist in single_hog:
                reduced.append(hist)

        return reduced


    """
        Generating all blocks descriptors in the DB by HOG method and quantize it through kmeans algorithm
    """
    def generate_visual_word_dict(self):
        if self._database is None:
            print("No available database is found.")
            return
        data = self._database.get_data_hash()
        self._feature_vectors = []
        self._feature_vectors_labels = list()
        for image_name, image_label in zip(data.keys(), data.values()):
            # Coloured image
            image_instance = cv2.imread(self._database.get_root_dir() + "//" + image_name)
            image_instance = cv2.resize(image_instance, self._win_size, interpolation=cv2.INTER_CUBIC)
            self._feature_vectors.append(self.__get_HOG_desctiptor(image_instance))
            self._feature_vectors_labels.append(image_label)

        feature_vector_instances = np.asarray(self._reduce_to_2D(self._feature_vectors), dtype=np.float32)

        # Here - DB training-set is transferred to features vectors representation
        # Clustering into K groups at our problem
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # Input all visual word from the DB
        return_value, self._labels, self._centers = cv2.kmeans(feature_vector_instances, self._K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


    """
        Return the closest center index in terms of MSE to a block descriptor
        -   How close is a parameter that is required by the program and make the histogram changeable
        [1] single_descriptor       - an histogram
        [2] centers_descriptors_arr - list of histograms, could be the centers which were discovered by kmeans algorithm one step before 
    """
    def __get_simi_class(self, single_descriptor, centers_descriptors_arr):
        best_mse = np.Inf
        min_idx = -1
        for curr_idx, cd_item in enumerate(centers_descriptors_arr):
            curr_mse = np.square(single_descriptor - cd_item).mean(axis=0)  # pseudo-code: (1/n) * sum((ai-bi)^2)
            if curr_mse < best_mse:
                best_mse = curr_mse
                min_idx = curr_idx
        return min_idx


    """
        Return feature histogram for each image from the Database
    """
    def generate_bows(self):
        if self._centers is None:
            print("'gen_visual_word_dict()' method should be used as a prior step.")
            return None
        self._bows = list()
        for fv_image in self._feature_vectors:
            path_hists = np.zeros(self._K, dtype=np.uint32)
            for block_descriptor in fv_image:
                best_class = self.__get_simi_class(block_descriptor, self._centers)
                if best_class != -1:
                    path_hists[best_class] += 1
            self._bows.append(path_hists.tolist())

        return self._bows


    """
        For each image there is an histogram of features which has quantized by clustering algorithm to K centers -> histogram of size K
    """
    def get_bows(self):
        return self._bows

    """
        Foreach image there is (height/cell_size * width/cell_size) features
    """
    def get_feature_vectors_by_image(self):
        return self._feature_vectors


    """
        By the image DB directory name
    """
    def get_feature_vectors_labels_by_image(self):
        return self._feature_vectors_labels


    """
        Can be used after kmeans execution [as part of the algorithm output]
    """
    def get_quantized_center(self):
        return self._centers


    """
        Can be used after kmeans execution [as part of the algorithm output]
    """
    def get_quantized_labels(self):
        return self._labels


    """
        Save the output data from Feature class methods to pickle file for next using -> [bows, labels]
    """
    def save(self):
        obj_data = [self._bows, self._feature_vectors_labels]
        if not os.path.exists("var"):
            os.mkdir("var")
        pickle.dump(obj_data, open(Features.__PICKLE_LOC, "wb+"))


    """
        Load [bows, labels] from saved pickle file at 'var' directory
    """
    def load(self):
        feature_obj_data = pickle.load(open(Features.__PICKLE_LOC, "rb"))
        self._bows = feature_obj_data[0]
        self._feature_vectors_labels = feature_obj_data[1]
        return self._bows, self._feature_vectors_labels



class Classifier:

    """
        [1] features -  a Feature class object
        [2] c        - margin flexible variable (the actual width)
    """
    def __init__(self, features, c=10):
        self._features = features
        self._c = c
        self._svm_instance = dlib.svm_c_trainer_radial_basis()  # svm_c_trainer_linear() -> the linear case
        self._svm_instance.set_c(self._c)
        # self._svm_instance.be_verbose()                         # linear field
        self._decision_function = None



    """
        Return dlib vectors array of dlib vector list
        [1] data - list[list[number]] to dlib.vectors[dlib.vector[number]]
    """
    def __prepare_data(self, data):
        vecs = dlib.vectors()
        for item in data:
            vecs.append(dlib.vector(item))

        return vecs


    """
        Training the classifier
    """
    def train(self):
        # prepare the data for being as type of dlib objects
        data = self.__prepare_data(self._features.get_bows())                       # -> dlib.vectors
        labels = dlib.array(self._features.get_feature_vectors_labels_by_image())   # -> dlib.array
        self._decision_function = self._svm_instance.train(data, labels)            # -> dlib._decision_function_radial_basis
        return self._decision_function


    def test(self):
        pass




if __name__ == "__main__":

    ## Must object to initial the program
    db_instance = Database(trainImageDirName)
    # Must object to handle data as features
    feature_instance = Features(db_instance)
    ## Feature extraction process which is necessary while no pre-processing have been made yet
    # feature_instance.generate_visual_word_dict()
    # feature_instance.generate_bows()
    # feature_instance.save()
    feature_instance.load()
    classifier_instance = Classifier(feature_instance)
    classifier_instance.train()