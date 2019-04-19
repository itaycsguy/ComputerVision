## Students And Developers: Itay Guy, 305104184 & Elias Jadon, 207755737

import os
import cv2
import numpy as np
import dlib
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse

trainImageDirName = ".//Datasets"
testImageDirName = ".//Datasets//Testset"


"""
    A class who holds all the training data + labels
"""
class Database:
    PICKLE_DIR = "var"
    MIN_ADD_DB = 0

    # Defined at the published assignment
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
                Database.ALLOWED_DIRS[dataset.lower()] = max(Database.ALLOWED_DIRS.values()) + 1


    def get_target1_class(self):
        return Database.ALLOWED_DIRS["airplane"]


    def get_target2_class(self):
        return Database.ALLOWED_DIRS["elephant"]


    def get_target3_class(self):
        return Database.ALLOWED_DIRS["motorbike"]


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
    __PICKLE_FILE = "Features.pkl"
    DEF_K = 10  # optimum k clusters
    MIN_K = 1


    """
        [1] database  - a Database object
        [2] k         - clusters amount, default: 10
        [3] cell_size - size of block to divide the image, default: 8 -> (8, 8)
        [4] bin_n     - number of bins for the histogram at each cell, default: 9
        [5] win_size  - size the image (expecting to 2^x type of size), default: (64, 128)
    """
    def __init__(self, database, k=DEF_K, cell_size=8, bin_n=9, win_size=(64, 128)):
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
        Return 'airplane'
    """
    def get_target1_value(self):
        return self._database.get_target1_class()


    """
        Return 'elephant'
    """
    def get_target2_value(self):
        return self._database.get_target2_class()


    """
        Return 'motorbike'
    """
    def get_target3_value(self):
        return self._database.get_target3_class()


    """
        Return the target class
    """
    def get_target_value(self):
        return self._database.get_target_class()


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
        Can be used after kmeans execution [as part of the algorithm output]
    """
    def get_quantized_center(self):
        return self._patches_centers


    """
        Can be used after kmeans execution [as part of the algorithm output]
    """
    def get_quantized_labels(self):
        return self._patches_labels


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
    __PICKLE_FILE = "Classifier.pkl"
    DEF_C = 10.0    # optimum margin with of SVM
    MIN_C = 0.01
    DEF_NN_THRESH = 250     # optimum distance for nearest-neighbor
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
            self._c = Classifier.DEF_C
            self._svm_instance = dlib.svm_c_trainer_linear()    # -> the linear case
            self._svm_instance.set_c(self._c)
        else:
            self._nn_thresh = Classifier.DEF_NN_THRESH
        self._decision_function = None
        self._decision_function1 = None
        self._decision_function2 = None
        self._decision_function3 = None
        classes_num = len(Database.ALLOWED_DIRS.keys())
        self._confusion_matrix = np.zeros((classes_num, classes_num, ), dtype=np.uint32)
        self._recognition_results = None


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
            # prepare the data for being as type of dlib objects
            # -> dlib.vectors
            data = self.__prepare_dlib_data(self._features.get_bows())
            # -> dlib.array => [-1,1]
            # labels = self.__prepare_dlib_labels(self._features.get_feature_vectors_labels_by_image(), self._features.get_target_value())
            actual_labels = self._features.get_feature_vectors_labels_by_image()
            labels1 = self.__prepare_dlib_labels(actual_labels, self._features.get_target1_value())     # airplane
            labels2 = self.__prepare_dlib_labels(actual_labels, self._features.get_target2_value())     # elephant
            labels3 = self.__prepare_dlib_labels(actual_labels, self._features.get_target3_value())     # motorbike
            # -> dlib._decision_function_radial_basis
            # self._decision_function = self._svm_instance.train(data, labels)
            self._decision_function1 = self._svm_instance.train(data, labels1)
            self._decision_function2 = self._svm_instance.train(data, labels2)
            self._decision_function3 = self._svm_instance.train(data, labels3)

        return self._decision_function1, self._decision_function2, self._decision_function3


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
    def recognizer(self):
        if not os.path.exists(testImageDirName):
            print(testImageDirName, "directory does not exist.")
            return

        y_true = list()
        y_score = list()
        self._recognition_results = {}
        for image_name in os.listdir(testImageDirName):
            # if "airplane" not in image_name.lower() and "elephant" not in image_name.lower():
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
                prediction_activation = self.__multi_activation(dlib_bow)    # self.__activation(dlib_bow)         # -> [0, 1, 2] by the activation function
            else:
                prediction_activation = self.__multi_NN_activation(bow) # self.__NN_activation(bow)

            actual_activation = self.__pseudo_activation(image_name)    # -> [0, 1, 2] by the activation function

            y_true.append(actual_activation)
            y_score.append(prediction_activation)

            # Given 2 classes determination -> need to determine for kind of binary problem
            self.__update_multi_confusion_matrix(prediction_activation, actual_activation)

            #  A raw data image -> 1:1 [no another image with the same name on that test session]
            self._recognition_results[image_name] = prediction_activation


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
        float_value1 = self._decision_function1(dlib_bow) #airplane
        float_value2 = self._decision_function2(dlib_bow) #Elephant
        float_value3 = self._decision_function3(dlib_bow) # motorbike
        # here we have 3 classes probability
        classes = [float_value1, float_value2, float_value3]
        idx = np.argmax(classes)
        # return value of form: (0/1/2, 0.0-1.0)
        return idx, classes[idx]



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
    def set_C(self, c):
        self._c = c


    """
        Updating the confusion matrix to multi-class
    """
    def __update_multi_confusion_matrix(self, prediction, actual): # prediction = [0,1,2] , actual = [0,1,2]
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
        TP = self._confusion_matrix[0, 0] + self._confusion_matrix[1, 1] + self._confusion_matrix[2, 2]
        result = TP / self._confusion_matrix.sum()
        print("Accuracy: TP /#all_data = {} / {} = {} ".format(TP, self._confusion_matrix.sum(), result))
        return result


    """
        Return the precision to the airplane class
    """
    def get_test_precision_airplane(self):
        TP_A = self._confusion_matrix[0, 0]
        FP_A = self._confusion_matrix[0, 1] + self._confusion_matrix[0, 2]
        result = TP_A / (TP_A + FP_A)
        print("Precision: TP_A / (TP_A + FN_A) = {}/({} + {}) = {} ".format(TP_A, TP_A, FP_A, result))
        return result


    """
        Return the precision to the elephant class
    """
    def get_test_precision_elephant(self):
        TP_A = self._confusion_matrix[1, 1]
        FP_A = self._confusion_matrix[1, 0] + self._confusion_matrix[1, 2]
        result = TP_A / (TP_A + FP_A)
        print("Precision: TP_B / (TP_B + FN_B) = {}/({} + {}) = {} ".format(TP_A, TP_A, FP_A, result))
        return result


    """
        Return the precision to the motorbike class
    """
    def get_test_precision_motorbike(self):
        TP_A = self._confusion_matrix[2, 2]
        FP_A = self._confusion_matrix[2, 0] + self._confusion_matrix[2, 1]
        result = TP_A / (TP_A + FP_A)
        print("Precision: TP_C / (TP_C + FN_C) = {}/({} + {}) = {} ".format(TP_A, TP_A, FP_A, result))
        return result


    """
        Return the recall to the airplane class
    """
    def get_test_recall_airplane(self):
        TP_A = self._confusion_matrix[0, 0]
        FN_A = self._confusion_matrix[1, 0] + self._confusion_matrix[2, 0]
        result = TP_A / (TP_A + FN_A)
        print("Recall: TP_A / (TP_A + FN_A) = {}/({} + {}) = {} ".format(TP_A, TP_A, FN_A, result))
        return result


    """
        Return the recall to the elephant class
    """
    def get_test_recall_elephant(self):
        TP_A = self._confusion_matrix[1, 1]
        FN_A = self._confusion_matrix[0, 1] + self._confusion_matrix[2, 1]
        result = TP_A / (TP_A + FN_A)
        print("Recall: TP_B / (TP_B + FN_B) = {}/({} + {}) = {} ".format(TP_A, TP_A, FN_A, result))
        return result


    """
        Return the recall to the motorbike class
    """
    def get_test_recall_motorbike(self):
        TP_A = self._confusion_matrix[2, 2]
        FN_A = self._confusion_matrix[0, 2] + self._confusion_matrix[1, 2]
        result = TP_A / (TP_A + FN_A)
        print("Recall: TP_C / (TP_C + FN_C) = {}/({} + {}) = {} ".format(TP_A, TP_A, FN_A, result))
        return result


    """
        display out c
    """
    def show_test_c(self):
        if self._type == Classifier.LINEAR_SVM:
            print("Current C for the SVM margin width:", self._c)
        else:
            print("Current NN threshold:", self._nn_thresh)


    """
        display out the confusion matrix
    """
    def show_test_confusion_matrix(self):
        print("Current confusion matrix:\n", self._confusion_matrix)



    """
        Return the optimum paramter under accuracy condition
    """
    @staticmethod
    def find_optimum_by_accuracy(accuracy, dependent_var, dependent_name):
        max_acc = 0
        var = -1
        for y, x in zip(accuracy, dependent_var):
            if max_acc < y:
                max_acc = y
                var = x

        if max_acc > 0 and var > -1:
            print("An optimal parameter by accuracy was found " + dependent_name + ":", var)

        return var, max_acc


    # @staticmethod
    # def find_optimum_by_ROC_CURVE(recall, precision, dependent_var, dependent_name):
    #     idx0 = np.argmax(precision[0])
    #     idx1 = np.argmax(precision[1])
    #     idx2 = np.argmax(precision[2])
    #
    #     print("An optimal parameter by ROC Curve was found " + dependent_name + ":", dependent_var[idx])
    #     return dependent_var[idx], recall[idx], precision[idx]


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


    """
        Plot the desired ROC Curve for choosing the optimal parameter
        - accuracy, precision, recall
    """
    @staticmethod
    def ROC_Curve(accuracy, precision, recall, dependent_var, dependent_var_name):
        linear_x = [0.0, 0.5, 1.0]
        linear_y = [1.0, 0.5, 0.0]

        fig, axs = plt.subplots(2, 1, constrained_layout=True)

        print("recall = " + str(recall))
        print("recall[0] = " + str(recall[0]))
        print("recall[1] = " + recall[1])
        print("recall[2] = " + recall[2])
        print("precision = " + precision)
        print("precision[0] = " + precision[0])
        print("precision[1] = " + precision[1])
        print("precision[2] = " + precision[2])

        axs[0].plot(recall[0], precision[0], '--', linewidth=2)
        axs[0].plot(recall[1], precision[1], '--', linewidth=2)
        axs[0].plot(recall[2], precision[2], '--', linewidth=2)

        axs[0].plot(linear_x, linear_y, '-', linewidth=0.5)

        info_arr = ['data', 'boundary']

        axs[0].set_xlim(0.0, 1.0)
        axs[0].set_ylim(0.0, 1.0)
        axs[0].set_xlabel('Recall')
        axs[0].set_ylabel('Precision')
        axs[0].legend(info_arr, loc='best')

        fig.suptitle('ROC Curve', fontsize=16)

        axs[1].set_ylim(0.0, 1.0)
        axs[1].plot(dependent_var, accuracy, '-', linewidth=2)
        axs[1].set_xlabel('Dependent-Variable: ' + dependent_var_name)
        axs[1].set_ylabel('Accuracy')
        axs[1].legend(['data', 'optimum'], loc='best')

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
def driver(classifier_type, additional_datasets, k, c, SLEEP_TIME_OUT=3):
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
        classifier_instance.set_C(c)
        classifier_instance.train()
    else:
        classifier_instance = Classifier(feature_instance, type=Classifier.NN)
        classifier_instance.set_nn_thresh(c)
    y_true, y_score = classifier_instance.recognizer()
    classifier_instance.save()
    classifier_instance.load()

    accuracy = classifier_instance.get_test_accuracy()
    precision = [classifier_instance.get_test_precision_airplane(),
                 classifier_instance.get_test_precision_elephant(),
                 classifier_instance.get_test_precision_motorbike()
                 ]
    recall = [classifier_instance.get_test_recall_airplane(),
              classifier_instance.get_test_recall_elephant(),
              classifier_instance.get_test_recall_motorbike()
              ]

    feature_instance.show_current_k()
    classifier_instance.show_test_c()
    classifier_instance.show_test_confusion_matrix()
    time.sleep(SLEEP_TIME_OUT)
    return y_true, y_score, accuracy, precision, recall


"""
    Determine the parameter to execute over
    [1] dataset_amount=?
    [2] k=?
    [3] c=?
"""
def run(additional_datasets, classifier, **kwargs):
    k = Features.DEF_K
    classifier_thresh = Classifier.DEF_C
    if classifier == Classifier.NN:
        classifier_thresh = Classifier.DEF_NN_THRESH
    c = classifier_thresh
    for key, value in kwargs.items():
        if classifier == Classifier.LINEAR_SVM:
            if key.lower() == "c" and np.float32(value) > Classifier.MIN_C:
                c = np.float32(value)
        elif key.lower() == "c" and np.float32(value) > 0:
                c = np.float32(value)
        if key.lower() == "k" and np.uint32(value) > Features.MIN_K:
            k = np.uint32(value)

    return driver(classifier, additional_datasets, k, c)


def run_by_c(classifier, init, loop_length, LOAD=False):
    y_true_glob = list()
    y_scores_glob = list()
    accuracy = list()
    precision = list()
    recall = list()
    precision_airplane = list()
    precision_elephant = list()
    precision_motorbike = list()
    recall_airplane = list()
    recall_elephant = list()
    recall_motorbike = list()
    datasets = list()
    DEP_VAR_NAME = "C"
    DEP_VAR = None
    if classifier == Classifier.NN:
        DEP_VAR = np.arange(init, init + 10 * loop_length, 100.0)
    else:
        DEP_VAR = np.arange(init, init + loop_length, 1.0)

    run_data_path = Database.PICKLE_DIR + "//Run_data_" + DEP_VAR_NAME + ".pkl"
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
            y_true, y_score, acc, prec, rec = run(ds_amt, classifier, c=dep_var)

            accuracy.append(acc)
            precision_airplane.append(prec[0])
            precision_elephant.append(prec[1])
            precision_motorbike.append(prec[2])
            recall_airplane.append(rec[0])
            recall_elephant.append(rec[1])
            recall_motorbike.append(rec[2])

            for true, score in zip(y_true, y_score):
                y_true_glob.append(true)
                y_scores_glob.append(score)

            print("")
        precision = [precision_airplane,precision_elephant,precision_motorbike]
        recall = [recall_airplane,recall_elephant,recall_motorbike]

        pickle.dump([y_true_glob, y_scores_glob, accuracy, precision, recall, DEP_VAR], open(run_data_path, "wb+"))

    precision_sorted = [precision_airplane.sort(),precision_elephant.sort(),precision_motorbike.sort()]
    recall_sorted = [recall_airplane.sort(),recall_elephant.sort(),recall_motorbike.sort()]

    Classifier.ROC_Curve(accuracy, precision_sorted, recall_sorted, DEP_VAR, DEP_VAR_NAME)



def run_by_k(classifier, init, loop_length, LOAD=False):
    y_true_glob = list()
    y_scores_glob = list()
    accuracy = list()
    precision = list()
    recall = list()
    precision_airplane = list()
    precision_elephant = list()
    precision_motorbike = list()
    recall_airplane = list()
    recall_elephant = list()
    recall_motorbike = list()
    datasets = list()
    DEP_VAR_NAME = "K"
    DEP_VAR = None
    if classifier == Classifier.NN:
        DEP_VAR = np.arange(init, init + loop_length, 100.0)
    else:
        DEP_VAR = np.arange(init, init + loop_length, 1.0)

    run_data_path = Database.PICKLE_DIR + "//Run_data_" + DEP_VAR_NAME + ".pkl"
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
            y_true, y_score, acc, prec, rec = run(ds_amt, classifier, k=dep_var)

            accuracy.append(acc)
            precision_airplane.append(prec[0])
            precision_elephant.append(prec[1])
            precision_motorbike.append(prec[2])
            recall_airplane.append(rec[0])
            recall_elephant.append(rec[1])
            recall_motorbike.append(rec[2])


            for true, score in zip(y_true, y_score):
                y_true_glob.append(true)
                y_scores_glob.append(score)

            print("")
        precision = [precision_airplane, precision_elephant, precision_motorbike]
        recall = [recall_airplane, recall_elephant, recall_motorbike]


        pickle.dump([y_true_glob, y_scores_glob, accuracy, precision, recall, DEP_VAR], open(run_data_path, "wb+"))

    precision_sorted = [precision_airplane.sort(), precision_elephant.sort(), precision_motorbike.sort()]
    recall_sorted = [recall_airplane.sort(), recall_elephant.sort(), recall_motorbike.sort()]
    print("dfbdfbdfprecision = " + str(precision_sorted))
    
    Classifier.ROC_Curve(accuracy, precision_sorted, recall_sorted, DEP_VAR, DEP_VAR_NAME)


def run_by_multi_datasets(classifier, LOAD=False):
    y_true_glob = list()
    y_scores_glob = list()
    accuracy = list()
    precision = list()
    recall = list()

    run_data_path = Database.PICKLE_DIR
    datasets = [get_all_rest_datasets(0), get_all_rest_datasets(1), get_all_rest_datasets(2), get_all_rest_datasets(3)]
    run_data_path = Database.PICKLE_DIR + "//Datasets_Run_data.pkl"
    DEP_VAR = np.zeros(4)
    if LOAD:
        data = pickle.load(open(run_data_path, "rb"))
        y_true_glob = data[0]
        y_scores_glob = data[1]
        accuracy = data[2]
        precision = data[3]
        recall = data[4]
    else:
        for ds_amt, dep_var in zip(datasets, DEP_VAR):
            y_true, y_score, acc, prec, rec = run(ds_amt, classifier, c=dep_var)

            accuracy.append(acc)
            precision.append(prec)
            recall.append(rec)

            for true, score in zip(y_true, y_score):
                y_true_glob.append(true)
                y_scores_glob.append(score)

            print("")

        pickle.dump([y_true_glob, y_scores_glob, accuracy, precision, recall, DEP_VAR], open(run_data_path, "wb+"))

    precision_sorted = precision.copy()
    precision_sorted.sort()
    recall_sorted = recall.copy()
    recall_sorted.sort(reverse=True)

    Classifier.Accuracy_Sets(accuracy, list(range(3, 7)))



if __name__ == "__main__":

    # run_by_multi_datasets(Classifier.NN, LOAD=False)
    run_by_multi_datasets(Classifier.NN, LOAD=False) # BUG in appending more directories
    # run_by_k(Classifier.NN, 5.0, 1000, LOAD=False)  # working well -> the most value is 0.73 accuracy
    # run_by_k(Classifier.LINEAR_SVM, 5.0, 20, LOAD=False)  # unstable -> running between 0.5 to 0.9
    # run_by_c(Classifier.NN, 250.0, 1000, LOAD=False)    # still unstable for some values -> slow function of c -> need to try more values to reach the optimum
    # run_by_c(Classifier.LINEAR_SVM, 100.0, 5, LOAD=False) # unstable
