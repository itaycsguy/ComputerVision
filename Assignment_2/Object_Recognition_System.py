## Students And Developers: Itay Guy, 305104184 & Elias Jadon, 207755737

import os
import cv2
import numpy as np
import dlib
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import operator
import argparse

trainImageDirName = ".//Datasets"
testImageDirName = ".//Datasets//Testset"


"""
    A class who holds all the training data + labels
"""
class Database:
    PICKLE_DIR = "var"
    MIN_ADD_DB = 0

    # Defined at the published assignment [initial classes by assignment design]
    ALLOWED_DIRS = {"airplane": 0, "elephant": 1, "motorbike": 2}

    """
        Making an hash which holds the data efficiently and conveniently
    """
    def __init__(self, dir_path, FINAL_CLASSES=True, datasets=None):
        if not os.path.exists(dir_path):
            print(dir_path, "does not exist.")
            return
        self._root_dir = dir_path
        self._image_hash = {}
        if datasets is not None:
            for idx, item in enumerate(datasets):
                datasets[idx] = item[0]
        self._added_datasets = datasets
        self.append_datasets(datasets)
        if FINAL_CLASSES:
            self.load_datasets_from_dir()


    """
        Return the specific classes who were just added to the hash
    """
    def get_added_datasets_names(self):
        return self._added_datasets


    """
        Print out the avaliable datasets
    """
    def show_avaliable_datasets(self):
        dir_names = "Avaliable datasets => ("
        key_length = len(Database.ALLOWED_DIRS.keys())
        for i, dir_name in enumerate(Database.ALLOWED_DIRS.keys()):
            dir_names += dir_name
            if i < (key_length-1):
                dir_names += ", "
        dir_names += ")"
        print(dir_names)


    """
        Loading the datasets from the indicated directory
    """
    def load_datasets_from_dir(self, dir_path=None):
        if dir_path is None:
            dir_path = self._root_dir
        for outer_dir in os.listdir(dir_path):
            if outer_dir.lower() in Database.ALLOWED_DIRS.keys():
                inner_dir_path = dir_path + "//" + outer_dir
                for image_name in os.listdir(inner_dir_path):
                    self._image_hash[outer_dir + "//" + image_name] = Database.ALLOWED_DIRS[outer_dir.lower()]


    """
        Appending the new dataset names to the Hash for being use as samples to the algorithm
    """
    def append_datasets(self, datasets):
        if datasets is not None:
            for dataset in datasets:
                if not (dataset.lower() in Database.ALLOWED_DIRS):
                    Database.ALLOWED_DIRS[dataset.lower()] = max(Database.ALLOWED_DIRS.values()) + 1


    """
        Return the initial hash with the trained classes names [sorted by key value]
    """
    def get_target_names(self):
        return sorted(Database.ALLOWED_DIRS.items(), key=operator.itemgetter(1))


    """
        Return the initial hash with the trained classes values [sorted]
    """
    def get_target_values(self):
        return sorted(Database.ALLOWED_DIRS.values())


    def class2name(self, num_class):
        for name, val in Database.ALLOWED_DIRS.items():
            if val == num_class:
                return name

        return ""


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
    @staticmethod
    def get_sets_amount():
        return len(Database.ALLOWED_DIRS)



class Features:
    __PICKLE_FILE = "Features.pkl"
    DEF_SVM_QUANTIZATION = 40
    DEF_NN_QUANTIZATION = 370      # optimum k clusters
    MIN_QUANTIZATION = 1


    """
        [1] database  - a Database object
        [2] k         - clusters amount, default: 10
        [3] cell_size - size of block to divide the image, default: 8 -> (8, 8)
        [4] bin_n     - number of bins for the histogram at each cell, default: 9
        [5] win_size  - size the image (expecting to 2^x type of size), default: (64, 128)
    """
    def __init__(self, database, k=DEF_SVM_QUANTIZATION, cell_size=8, bin_n=9, win_size=(64, 128)):
        # Database reference
        self._database = database
        # These are parameters which required from the program to get initially
        self._cell_size = cell_size     # Descriptor cell computation on the image after resizing
        self._bin_n = bin_n             # Number of bins
        self._win_size = win_size
        self._K = k                     # kmeans algorithm parameter
        # meta information
        self._feature_vectors = None
        # the same as self._feature_vectors except to the issue it is flatted to 2D structure
        self._feature_vector_instances_2D = None
        self._feature_vectors_labels = None
        # computed information
        self._patches_centers = None
        self._patches_labels = None
        self._bows = None
        self._test_bows = None


    """
        Print out the current k parameter value
    """
    def show_current_k(self):
        print("Current K to the clustering algorithm:", self._K)


    """
        Return a reshape structure of the HOG API returning
    """
    def __reshape(self, raw_hist):
        hists = list()
        for i in range(0, len(raw_hist), self._bin_n):
            hists.append(raw_hist[i:(i + self._bin_n)].ravel())

        return hists


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
        histogramNormType = 1
        L2HysThreshold = 0.2
        gammaCorrection = 0
        nlevels = 64
        signedGradient = 0
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                                derivAperture, winSigma, histogramNormType, L2HysThreshold,
                                gammaCorrection, nlevels, signedGradient)
        # compute(img[, winStride[, padding[, locations]]]) -> descriptors
        winStride = (self._cell_size, self._cell_size)
        padding = (self._cell_size, self._cell_size)
        locations = []
        for image in images:
            hist = hog.compute(image, winStride, padding, locations)
            samples.append(self.__reshape(hist))
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
        winSigma = -1.0
        histogramNormType = 1
        L2HysThreshold = 0.2
        gammaCorrection = 0
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
        return self.__reshape(hist)


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
    def generate_visual_word_dict(self, NEED_CLUSTERING=True):
        if self._database is None:
            print("No available database is found.")
            return
        data = self._database.get_data_hash()
        self._feature_vectors = []
        self._feature_vectors_labels = list()
        self._patchs_labels = list()
        for image_name, image_label in zip(data.keys(), data.values()):
            # Coloured image
            image_instance = cv2.imread(self._database.get_root_dir() + "//" + image_name)
            image_instance = cv2.resize(image_instance, self._win_size, interpolation=cv2.INTER_CUBIC)
            self._feature_vectors.append(self.get_native_hog(image_instance))
            self._feature_vectors_labels.append(image_label)

        self._feature_vector_instances_2D = np.asarray(self._reduce_to_2D(self._feature_vectors), dtype=np.float32)
        # Here - DB training-set is transferred to features vectors representation
        if NEED_CLUSTERING:
            # Clustering into K groups at our problem
            self.cluster_data()

    """
        Clustering the data to k groups using kmeans algorithm
    """
    def cluster_data(self, k=None):
        if k is None:
            k = self._K
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        return_value, labels, self._patches_centers = cv2.kmeans(self._feature_vector_instances_2D, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        self._patches_labels = labels.ravel()


    """
        Make the k clusters to be changeable for the user execution
    """
    def set_K(self, k):
        self._K = k


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
    def generate_bows(self, feature_vectors, train_mode=True):
        if self._patches_centers is None:
            print("'gen_visual_word_dict()' method should be used as a prior step.")
            return None

        bows = list()
        if not train_mode:
            bows = np.zeros(self._K, dtype=np.uint32)

        patch_loc = 0
        for fv_image in feature_vectors:
            if train_mode:
                patch_hists = np.zeros(self._K, dtype=np.uint32)
                for _ in range(0, len(fv_image)):
                    best_class = self._patches_labels[patch_loc]
                    patch_hists[best_class] += 1
                    patch_loc += 1
                bows.append(patch_hists.tolist())
            else:
                best_class = self.__get_simi_class(fv_image, self._patches_centers)
                if best_class != -1:
                    bows[best_class] += 1

        if train_mode:
            self._bows = bows
        else:
            self._test_bows = bows

        return bows


    """
        Return the defined win_size to use at HOG descriptor computation
    """
    def get_win_size(self):
        return self._win_size


    """
        For each image there is an histogram of features which has quantized by clustering algorithm to K centers -> histogram of size K
    """
    def get_bows(self):
        return self._bows


    """
        Return the computed test bow
    """
    def get_test_bows(self):
        return self._test_bows


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
        Return the DB reference
    """
    def get_database(self):
        return self._database


    """
        Save the output data from Feature class methods to pickle file for next using -> [bows, labels, centers]
    """
    def save(self):
        obj_data = [self._bows, self._feature_vectors_labels, self._patches_centers]
        if not os.path.exists("var"):
            os.mkdir("var")
        pickle.dump(obj_data, open(Database.PICKLE_DIR + "//" + Features.__PICKLE_FILE, "wb+"))


    """
        Load [bows, labels, centers] from saved pickle file at 'var' directory
    """
    def load(self):
        feature_obj_data = pickle.load(open(Database.PICKLE_DIR + "//" + Features.__PICKLE_FILE, "rb"))
        self._bows = feature_obj_data[0]
        self._feature_vectors_labels = feature_obj_data[1]
        self._patches_centers = feature_obj_data[2]
        return self._bows, self._feature_vectors_labels, self._patches_centers



class Classifier:
    # C = 1
    __PICKLE_FILE = "Classifier.pkl"
    THRESHOLD = 39              # optimum margin with of SVM
    MIN_THRESHOLD = 0.01
    NN_THRESH = 4700            # optimum distance for nearest-neighbor
    LINEAR_SVM = 0
    NN = 1

    """
        [1] features -  a Feature class object
        [2] c        - margin flexible variable (the actual width)
    """
    def __init__(self, features, type=LINEAR_SVM):
        self._features = features
        self._type = type
        if type == Classifier.LINEAR_SVM:
            self._threshold = Classifier.THRESHOLD
            self._svm_instance = dlib.svm_c_trainer_linear()    # -> the linear case
            self._svm_instance.set_c(self._threshold)
        else:
            self._nn_thresh = Classifier.NN_THRESH
        self._decision_functions = None

        # For SVM Structural problem
        # self.num_samples = 0
        # self.num_dimensions = 0
        # self.samples = None
        # self.labels = None
        # self.weights = None

        classes_num = len(Database.ALLOWED_DIRS.keys())
        # self._confusion_matrix_structural_SVM = np.zeros((classes_num, classes_num, ), dtype=np.uint32)
        self._confusion_matrix = np.zeros((classes_num, classes_num, ), dtype=np.uint32)
        self._recognition_results = None


    """
        Compute PSI(x,label)
    """
    # def make_psi(self, x, label):
    #     psi = dlib.vector()
    #     psi.resize(self.num_dimensions)
    #     dims = len(x)
    #     if label == 0:
    #         for i in range(0, dims):
    #             psi[i] = x[i]
    #     elif label == 1:
    #         for i in range(dims, 2 * dims):
    #             psi[i] = x[i - dims]
    #     elif label == 2:  # the label must be 2
    #         for i in range(2 * dims, 3 * dims):
    #             psi[i] = x[i - 2 * dims]
    #     return psi



    # def get_truth_joint_feature_vector(self, idx):
    #     return self.make_psi(self.samples[idx], self.labels[idx])


    """
        Compute the dot product between the two vectors a and b
    """
    # def dot(self, a, b):
    #     return sum(i * j for i, j in zip(a, b))


    # def separation_oracle(self, idx, current_solution):
    #     samp = self.samples[idx]
    #     dims = len(samp)
    #     scores = [0, 0, 0]
    #     # compute scores for each of the three classifiers
    #     scores[0] = self.dot(current_solution[0:dims], samp)
    #     scores[1] = self.dot(current_solution[dims:2*dims], samp)
    #     scores[2] = self.dot(current_solution[2*dims:3*dims], samp)
    #     # correct label and a loss of 0 if we get the right label.
    #     if self.labels[idx] != 0:
    #         scores[0] += 1
    #     if self.labels[idx] != 1:
    #         scores[1] += 1
    #     if self.labels[idx] != 2:
    #         scores[2] += 1
    #     # Now figure out which classifier has the largest loss-augmented score.
    #     max_scoring_label = scores.index(max(scores))
    #     # And finally record the loss that was associated with that predicted
    #     # label. Again, the loss is 1 if the label is incorrect and 0 otherwise.
    #     if max_scoring_label == self.labels[idx]:
    #         loss = 0
    #     else:
    #         loss = 1
    #     # Finally, return the loss and PSI vector corresponding to the label
    #     # we just found.
    #     psi = self.make_psi(samp, max_scoring_label)
    #     return loss, psi


    """
        Given the 9-dimensional weight vector which defines a 3 class classifier,
        predict the class of the given 3-dimensional sample vector.   Therefore, the
        output of this function is either 0, 1, or 2 (i.e. one of the three possible
        labels)
    """
    # def predict_label(self, sample):
    #     # The individual classifier scores are stored in scores and the highest scoring
    #     # index is returned as the label.
    #     w0 = self.weights[0:3]
    #     w1 = self.weights[3:6]
    #     w2 = self.weights[6:9]
    #     scores = [self.dot(w0, sample), self.dot(w1, sample), self.dot(w2, sample)]
    #     max_scoring_label = scores.index(max(scores))
    #     return max_scoring_label


    # def __update_problem(self, samples, labels):
    #     self.num_samples = len(samples)
    #     self.num_dimensions = len(samples)*3
    #     self.samples = samples
    #     self.labels = labels


    """
        Set a new threshold to the NN distance
    """
    def set_nn_thresh(self, thresh):
        self._nn_thresh = thresh


    """
        Return dlib vectors array of dlib vector list
        [1] data - list[list[number]] to dlib.vectors[dlib.vector[number]]
    """
    def __prepare_dlib_data(self, data):
        vecs = dlib.vectors()
        for item in data:
            vecs.append(dlib.vector(item))

        return vecs


    """
        Return labels as expected by dlib API
    """
    def __prepare_dlib_labels(self, labels, target_value):
        vals = dlib.array()
        for label in labels:
            if label == target_value:
                vals.append(+1)
            else:
                vals.append(-1)

        return vals


    """
        Return labels as expected by dlib API
    """
    def __prepare_labels(self, labels, target_value):
        vals = list()
        for label in labels:
            if label == target_value:
                vals.append(+1)
            else:
                vals.append(-1)

        return np.asarray(vals)


    """
        Training the classifier
    """
    def train(self):
        if self._type == Classifier.LINEAR_SVM:
            # -> dlib.vectors
            data = self.__prepare_dlib_data(self._features.get_bows())

            # Could be: airplane, elephant, motorbike
            labels = list()
            actual_labels = self._features.get_feature_vectors_labels_by_image()
            targets = self._features.get_database().get_target_values()
            for target in targets:
                # -> dlib.array => [-1,1]
                labels.append(self.__prepare_dlib_labels(actual_labels, target))

            self._decision_functions = list()
            for label in labels:
                self._decision_functions.append(self._svm_instance.train(data, label))

            """learning by structural SVM method."""
            # self.__update_problem(self._features.get_bows(), actual_labels)
            # self.weights = dlib.solve_structural_svm_problem(self)

        return self._decision_functions


    """
        Implementation to the NN classifier
    """
    def __NN_activation(self, bow):
        data = self._features.get_bows()
        labels = self.__prepare_labels(self._features.get_feature_vectors_labels_by_image(), self._features.get_target_value())
        best_mse = np.Inf
        closest_class = 0
        for curr_idx, bow_item in enumerate(data):
            curr_mse = np.square(bow - bow_item).mean(axis=0)  # pseudo-code: (1/n) * sum((ai-bi)^2)
            if curr_mse < self._nn_thresh and curr_mse < best_mse:
                best_mse = curr_mse
                closest_class = labels[curr_idx]
        return closest_class


    """
        Implementation to the multi-class NN classifier
    """
    def __multi_NN_activation(self, bow):

        data = self._features.get_bows()
        labels = self._features.get_feature_vectors_labels_by_image()
        best_mse = np.Inf
        closest_class = 0
        for curr_idx, bow_item in enumerate(data):
            curr_mse = np.square(bow - bow_item).mean(axis=0)  # pseudo-code: (1/n) * sum((ai-bi)^2)
            if curr_mse < self._nn_thresh and curr_mse < best_mse:
                best_mse = curr_mse
                closest_class = labels[curr_idx]
        return closest_class


    """
        Executing the whole recognition process as same as we did to at the training step
        *** At this process we use the linear classifier which it looks better then radial basis 
    """
    def recognizer(self, testset_filter=None):
        if not os.path.exists(testImageDirName):
            print(testImageDirName, "directory does not exist.")
            return

        y_true = list()
        y_score = list()
        self._recognition_results = {}
        for image_name in os.listdir(testImageDirName):
            if testset_filter is None or testset_filter in image_name.lower():
                image_path = testImageDirName + "//" + image_name
                image = cv2.imread(image_path)
                image = cv2.resize(image, self._features.get_win_size(), interpolation=cv2.INTER_CUBIC)

                # Extracting features from an input new test image & computing the visual words
                feature_vectors = self._features.get_native_hog(image)

                bow = self._features.generate_bows(feature_vectors, False)
                prediction_activation = 0
                if self._type == Classifier.LINEAR_SVM:
                    # Build single BOW of dlib type
                    dlib_bow = dlib.vector(bow)

                    # Classifying a BOW using the classifier we have built in step 4
                    # -> [0, 1, 2] by the activation function
                    prediction_activation = self.__multi_activation(dlib_bow)
                else:
                    prediction_activation = self.__multi_NN_activation(bow)

                # -> [0, 1, 2] by the activation function
                actual_activation = self.__pseudo_activation(image_name)

                y_true.append(actual_activation)
                y_score.append(prediction_activation)

                # Given 2 classes determination -> need to determine for kind of binary problem
                self.__update_multi_confusion_matrix(prediction_activation, actual_activation)

                #  A raw data image -> 1:1 [no another image with the same name on that test session]
                self._recognition_results[image_name] = prediction_activation

                # self._confusion_matrix_structural_SVM[self.predict_label(bow), actual_activation] += 1

                # image = cv2.imread(image_path)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # bottomLeftCornerOfText = (50, 70)
                # fontScale = 1
                # fontColor = (255, 255, 255)
                # lineType = 2
                #
                #
                # prediction = 'Error'
                # print(prediction_activation)
                # if (prediction_activation[0] == 0):
                #     prediction = 'Airplane'
                # if (prediction_activation[0] == 1):
                #     prediction = 'Elephant'
                # if (prediction_activation[0] == 2):
                #     prediction = 'MotorBike'
                # cv2.putText(image, prediction,
                #             bottomLeftCornerOfText,
                #             font,
                #             fontScale,
                #             fontColor,
                #             lineType)
                # cv2.imshow("Test Images", image)
                # cv2.waitKey(0)

        # cv2.destroyAllWindows()
        return y_true, y_score


    """
        Operating an activation function to determine the classification of dlib_bow
        [1] dlib_bow - bow conversion to dlib type
        Return [class, score]
    """
    def __multi_activation(self, dlib_bow):
        float_values = list()
        # By assignment design there 3 options: airplane, elephant and motorbike
        for f in self._decision_functions:
            float_values.append(f(dlib_bow))

        # here we have 3 classes probability
        idx = np.argmax(float_values)
        # return value of form: (0/1/2, 0.0-1.0)
        return idx, float_values[idx]



    """
        Pseudo activation operation - taking an image, split it than determine it's class from Database class hash
        [1] image name - input test image name
        Return [class, score]
    """
    def __pseudo_activation(self, image_name):
        # Actual_activation
        target_class = Database.ALLOWED_DIRS[image_name.split("_")[0]]
        return target_class


    """
        Set the margin width to the SVM classifier
    """
    def set_threshold(self, threshold):
        self._threshold = threshold
        self._svm_instance.set_c(threshold)


    """
        Updating the confusion matrix to multi-class
    """
    def __update_multi_confusion_matrix(self, prediction, actual):      # prediction = [0,1,2] , actual = [0,1,2]
        if self._type == Classifier.LINEAR_SVM:
            prediction = prediction[0]
        self._confusion_matrix[prediction, actual] += 1


    """
        Saving: confusion matrix & recognition results hash to a pickle file to var directory
    """
    def save(self):
        # Saving the results
        data = [self._confusion_matrix, self._recognition_results]
        pickle.dump(data, open(Database.PICKLE_DIR + "//" + Classifier.__PICKLE_FILE, "wb+"))


    """
        Loading: confusion matrix from index 0 & recognition results from index 1 to their class variables
    """
    def load(self):
        data = pickle.load(open(Database.PICKLE_DIR + "//" + Classifier.__PICKLE_FILE, "rb"))
        self._confusion_matrix = data[0]
        self._recognition_results = data[1]
        return self._confusion_matrix, self._recognition_results


    """
        get the accuracy
    """
    def get_test_accuracy(self):
        TP = np.sum(np.diag(self._confusion_matrix))
        result = TP / self._confusion_matrix.sum()
        print("Accuracy: TP /#all_data = {} / {} = {} ".format(TP, self._confusion_matrix.sum(), result))
        return result


    """
        Return the precision to the airplane class
    """
    def get_test_precision(self,  num_class):     # num_class = [0 ,1 ,2]
        sub_TP = self._confusion_matrix[num_class, num_class]

        # sum over rows
        sub_FP = np.sum(self._confusion_matrix, axis=1)[num_class] - self._confusion_matrix[num_class, num_class]

        world_val = sub_TP + sub_FP
        if world_val != 0:
            result = sub_TP / world_val
        else:
            result = np.Inf

        class_name = self._features.get_database().class2name(num_class)
        print("{} Precision: TP / (TP + FP) = {}/({} + {}) = {} ".format(class_name, sub_TP, sub_TP, sub_FP, result))
        return result


    """
        Return the recall to the airplane class
    """
    def get_test_recall(self, num_class):    # num_class = [0 ,1 ,2]
        sub_TP = self._confusion_matrix[num_class, num_class]

        # sum over cols
        sub_FN = np.sum(self._confusion_matrix, axis=0)[num_class] - self._confusion_matrix[num_class, num_class]

        world_val = sub_TP + sub_FN
        if world_val != 0:
            result = sub_TP / world_val
        else:
            result = np.Inf

        class_name = self._features.get_database().class2name(num_class)
        print("{} Recall: TP / (TP + FN) = {}/({} + {}) = {} ".format(class_name, sub_TP, sub_TP, sub_FN, result))
        return result


    """
        display out the threshold
    """
    def show_test_threshold(self):
        if self._type == Classifier.LINEAR_SVM:
            print("Current C for the SVM margin width:", self._threshold)
        else:
            print("Current NN threshold:", self._nn_thresh)


    """
        display out the confusion matrix
    """
    def show_test_confusion_matrix(self):
        print("Current confusion matrix:\n", np.matrix(self._confusion_matrix))
        # print("Current Structural SVM confusion matrix:\n", self._confusion_matrix_structural_SVM)


    """
        Plot the desired Curve for choosing the optimal parameter
        - accuracy, precision, recall
    """
    @staticmethod
    def Accuracy_Sets(accuracy, datasets_list):
        fig, axs = plt.subplots(1, 1, constrained_layout=True)
        axs.set_ylim(datasets_list[0], datasets_list[len(datasets_list) - 1])
        axs.set_ylim(0.0, 1.0)
        axs.plot(datasets_list, accuracy, 'red', '--', linewidth=2)
        axs.set_xlabel('#Dataset')
        axs.set_ylabel('Accuracy')
        axs.legend(['data'], loc='best')
        fig.suptitle('Accuracy Function', fontsize=16)
        plt.show()


    @staticmethod
    def compute_optimal_pair_iter(precision, recall, var):
        class_optimum = list()
        for p_arr, r_arr in zip(precision, recall):
            summation_class = list()
            for p, r, v in zip(p_arr, r_arr, var):
                summation_class.append((p + r, p, r, v))
            class_optimum.append(max(summation_class, key=operator.itemgetter(0))[1:])  # form of (precision, recall, var)
        return class_optimum



    """
        Plot the desired ROC Curve for choosing the optimal parameter
        - accuracy, precision, recall
    """
    @staticmethod
    def ROC_Curve(optimum, accuracy, precision, recall, dependent_var, dependent_var_name, classifier, confusionRows=3):
        linear_x = [0.0, 0.5, 1.0]
        linear_y = [1.0, 0.5, 0.0]

        fig, axs = plt.subplots(2, 1, constrained_layout=True)

        # print("recall =", str(recall))
        # print("precision =", str(precision))
        # print("accuracy =", str(accuracy))
        # print("dependent_var =", str(dependent_var))
        # print("optimum =", str(optimum))

        for i in range(0, confusionRows):
            base_line, = axs[0].plot(recall[i], precision[i], '-', linewidth=1)
            # if optimum:
                # p, r = optimum[i][:2]
                # axs[0].plot(p, r, 'o', color=base_line.get_color())

        axs[0].plot(linear_x, linear_y, '--', linewidth=0.5)

        # info_arr = ['Airplane', 'Opt.', 'Elephant', 'Opt.', 'Motorbike', 'Opt.', 'Linear line']
        info_arr = ['Airplane', 'Elephant', 'Motorbike', 'Linear line']
        if optimum:
            plt.text(0.03, 0.5, 'Airplane Optimum Threshold: ' + str(optimum[0][2]) + '\nElephant Optimum Threshold: ' + str(optimum[1][2]) + '\nMotorbike Optimum Threshold: ' + str(optimum[2][2]), fontsize=14, transform=plt.gcf().transFigure)

        axs[0].set_xlim(0.0, 1.0)
        axs[0].set_ylim(0.0, 1.0)
        axs[0].set_xlabel('Recall')
        axs[0].set_ylabel('Precision')
        axs[0].legend(info_arr, loc='best')

        fig.suptitle('ROC Curve', fontsize=16)

        axs[1].set_ylim(0.0, 1.0)
        axs[1].plot(dependent_var, accuracy, '-', linewidth=2)
        axs[1].set_xlabel('Dependent-Variable: ' + dependent_var_name.lower())
        axs[1].set_ylabel('Accuracy')
        axs[1].legend(['Data', 'Opt.'], loc='best')

        plt.show()



"""
    Get all more optional datasets from the entry directory that is defined
    [1] count - number of additional datasets
"""
def get_all_rest_datasets(count=0):
    if not os.path.exists(trainImageDirName):
        print(trainImageDirName, "does not exit.")
        exit(-1)
    new_dir_names = list()
    for dir in os.listdir(trainImageDirName):
        if len(new_dir_names) == count:
            break
        dir_lower = dir.lower()
        if dir_lower not in Database.ALLOWED_DIRS.keys() and (os.path.realpath(testImageDirName) != os.path.realpath(trainImageDirName + "//" + dir)):
            new_dir_names.append([dir_lower])

    return new_dir_names


"""
    Running the system using determined parameters
"""
def driver(classifier_type, additional_datasets, k, threshold, testset_filter=None, SLEEP_TIME_OUT=3):
    db_instance = Database(trainImageDirName, FINAL_CLASSES=True, datasets=additional_datasets)
    db_instance.show_avaliable_datasets()
    feature_instance = Features(db_instance, k=k)
    feature_instance.generate_visual_word_dict(NEED_CLUSTERING=True)
    feature_instance.generate_bows(feature_instance.get_feature_vectors_by_image())
    feature_instance.save()
    feature_instance.load()
    classifier_instance = None
    if classifier_type == Classifier.LINEAR_SVM:
        classifier_instance = Classifier(feature_instance, type=Classifier.LINEAR_SVM)
        classifier_instance.set_threshold(threshold)
        classifier_instance.train()
    else:
        classifier_instance = Classifier(feature_instance, type=Classifier.NN)
        classifier_instance.set_nn_thresh(threshold)
    y_true, y_score = classifier_instance.recognizer(testset_filter)
    classifier_instance.save()
    classifier_instance.load()

    accuracy = classifier_instance.get_test_accuracy()
    precision = []
    recall = []
    for i in range(0 , classifier_instance._confusion_matrix.shape[0]):
        precision.append(classifier_instance.get_test_precision(i))
        recall.append(classifier_instance.get_test_recall(i))

    feature_instance.show_current_k()
    classifier_instance.show_test_threshold()
    classifier_instance.show_test_confusion_matrix()
    time.sleep(SLEEP_TIME_OUT)
    return y_true, y_score, accuracy, precision, recall


"""
    Determine the parameter to execute over
    [1] dataset_amount=?
    [2] k=?
    [3] c=?
"""
def run(additional_datasets, classifier, testset_filter=None, **kwargs):
    threshold = Classifier.THRESHOLD
    quantization = Features.DEF_SVM_QUANTIZATION
    if classifier == Classifier.NN:
        quantization = Features.DEF_NN_QUANTIZATION
        threshold = Classifier.NN_THRESH
    for key, value in kwargs.items():
        if classifier == Classifier.LINEAR_SVM:
            if key.lower() == "threshold" and np.float32(value) > Classifier.MIN_THRESHOLD:
                threshold = np.float32(value)
        elif key.lower() == "threshold" and np.float32(value) > 0:
            threshold = np.float32(value)
        if key.lower() == "quantization" and np.uint32(value) > Features.MIN_QUANTIZATION:
            quantization = np.uint32(value)

    return driver(classifier, additional_datasets, quantization, threshold, testset_filter)


def run_by_threshold(classifier, init, loop_length, LOAD=False, confusionRows=3):
    y_true_glob = list()
    y_scores_glob = list()
    accuracy = list()
    precision = []
    recall = []
    datasets = list()
    DEP_VAR_NAME = "THRESHOLD"
    DEP_VAR = None
    if classifier == Classifier.NN:
        DEP_VAR = np.arange(init, init + 10 * loop_length, 100.0)
    else:
        DEP_VAR = np.arange(init, init + loop_length, 1.0)

    classifier_name = "LINEAR_SVM"
    if classifier == Classifier.NN:
        classifier_name = "NN"
    run_data_path = Database.PICKLE_DIR + "//Run_data_" + classifier_name + "_" + DEP_VAR_NAME + ".pkl"
    for _ in range(loop_length):
        datasets.append(list())
    if LOAD:
        data = pickle.load(open(run_data_path, "rb"))
        y_true_glob = data[0]
        y_scores_glob = data[1]
        accuracy = data[2]
        precision = data[3]
        recall = data[4]
    else:

        for ds_amt, dep_var in zip(datasets, DEP_VAR):
            y_true, y_score, acc, prec, rec = run(ds_amt, classifier, testset_filter=None, threshold=dep_var)

            accuracy.append(acc)
            for i in range(0, confusionRows):
                if len(precision) == i:
                    precision.append([prec[i]])
                else:
                    precision[i].append(prec[i])

                if len(recall) == i:
                    recall.append([rec[i]])
                else:
                    recall[i].append(rec[i])

            for true, score in zip(y_true, y_score):
                y_true_glob.append(true)
                y_scores_glob.append(score)

            print("")
        pickle.dump([y_true_glob, y_scores_glob, accuracy, precision, recall, DEP_VAR], open(run_data_path, "wb+"))

    precision_sorted = np.sort(precision)
    recall_sorted = np.fliplr(np.sort(recall))

    optimum = Classifier.compute_optimal_pair_iter(precision, recall, DEP_VAR)
    Classifier.ROC_Curve(optimum, accuracy, precision_sorted, recall_sorted, DEP_VAR, DEP_VAR_NAME, classifier, confusionRows=confusionRows)



def run_by_quantization(classifier, init, loop_length, LOAD=False, confusionRows=3):
    y_true_glob = list()
    y_scores_glob = list()
    accuracy = list()
    precision = []
    recall = []


    datasets = list()
    DEP_VAR_NAME = "QUANTIZATION"
    DEP_VAR = None
    if classifier == Classifier.NN:
        DEP_VAR = np.arange(init, init + loop_length, 100.0)
    else:
        DEP_VAR = np.arange(init, init + loop_length, 1.0)

    classifier_name = "LINEAR_SVM"
    if classifier == Classifier.NN:
        classifier_name = "NN"
    run_data_path = Database.PICKLE_DIR + "//Run_data_" + classifier_name + "_" + DEP_VAR_NAME + ".pkl"
    for _ in range(loop_length):
        datasets.append(list())
    if LOAD:
        data = pickle.load(open(run_data_path, "rb"))
        y_true_glob = data[0]
        y_scores_glob = data[1]
        accuracy = data[2]
        precision = data[3]
        recall = data[4]
    else:
        for ds_amt, dep_var in zip(datasets, DEP_VAR):
            y_true, y_score, acc, prec, rec = run(ds_amt, classifier, testset_filter=None, quantization=dep_var)

            accuracy.append(acc)
            for i in range(0, confusionRows):
                if len(precision) == i:
                    precision.append([prec[i]])
                else:
                    precision[i].append(prec[i])

                if len(recall) == i:
                    recall.append([rec[i]])
                else:
                    recall[i].append(rec[i])

            for true, score in zip(y_true, y_score):
                y_true_glob.append(true)
                y_scores_glob.append(score)

            print("")
        pickle.dump([y_true_glob, y_scores_glob, accuracy, precision, recall, DEP_VAR], open(run_data_path, "wb+"))


    precision_sorted = np.sort(precision)
    recall_sorted = np.fliplr(np.sort(recall))

    print(Classifier.compute_optimal_pair_iter(precision, recall, DEP_VAR))
    Classifier.ROC_Curve(None, accuracy, precision_sorted, recall_sorted, DEP_VAR, DEP_VAR_NAME, classifier, confusionRows=confusionRows)



def run_by_multi_datasets(classifier, LOAD=False, confusionRows=3):
    y_true_glob = list()
    y_scores_glob = list()
    accuracy = list()
    precision = []
    recall = []

    run_data_path = Database.PICKLE_DIR
    datasets = [get_all_rest_datasets(0), get_all_rest_datasets(1), get_all_rest_datasets(2), get_all_rest_datasets(3)]
    classifier_name = "LINEAR_SVM"
    if classifier == Classifier.NN:
        classifier_name = "NN"
    run_data_path = Database.PICKLE_DIR + "//Datasets_Run_data_" + classifier_name + ".pkl"
    if LOAD:
        data = pickle.load(open(run_data_path, "rb"))
        y_true_glob = data[0]
        y_scores_glob = data[1]
        accuracy = data[2]
        precision = data[3]
        recall = data[4]
    else:
        for ds_amt in datasets:
            y_true, y_score, acc, prec, rec = run(ds_amt, classifier)

            accuracy.append(acc)
            for i in range(0, confusionRows):
                if len(precision) == i:
                    precision.append([prec[i]])
                else:
                    precision[i].append(prec[i])

                if len(recall) == i:
                    recall.append([rec[i]])
                else:
                    recall[i].append(rec[i])

            for true, score in zip(y_true, y_score):
                y_true_glob.append(true)
                y_scores_glob.append(score)

            print("")
        pickle.dump([y_true_glob, y_scores_glob, accuracy, precision, recall], open(run_data_path, "wb+"))

    Classifier.Accuracy_Sets(accuracy, list(range(3, 7)))



def run_assignment_requirements():
    # find an optimum threshold: => Found: airplane: 4700.0, elephant: 100, motorbike:100.0
    # run_by_threshold(Classifier.NN, init=100.0, loop_length=1000, LOAD=True)
    # run_by_quantization(Classifier.NN, init=10, loop_length=1000, LOAD=True)
    # run_by_multi_datasets(Classifier.NN, LOAD=True)

    # run(list(), Classifier.NN, testset_filter='airplane')
    # run(list(), Classifier.NN, testset_filter='elephant')
    # run(list(), Classifier.NN, testset_filter='motorbike')
    # run(list(), Classifier.NN)

    # run_by_threshold(Classifier.LINEAR_SVM, init=10.0, loop_length=40, LOAD=True)
    # run_by_quantization(Classifier.LINEAR_SVM, init=10, loop_length=40, LOAD=True)
    # run_by_multi_datasets(Classifier.LINEAR_SVM, LOAD=True)

    # run(list(), Classifier.LINEAR_SVM, testset_filter='airplane')
    # run(list(), Classifier.LINEAR_SVM, testset_filter='elephant')
    # run(list(), Classifier.LINEAR_SVM, testset_filter='motorbike')
    pass



def run_optimum():
    run(list(), Classifier.NN)
    # run(list(), Classifier.LINEAR_SVM)


if __name__ == "__main__":

    # run_assignment_requirements()
    run_optimum()


