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
    TARGET_CLASS = "airplane"

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
        self.append_datasets(datasets)
        if FINAL_CLASSES:
            self.load_datasets_from_dir()


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


    """
        Return the defined target class number by its name
    """
    def get_target_class(self):
        return Database.ALLOWED_DIRS[Database.TARGET_CLASS]


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

    """
        [1] features -  a Feature class object
        [2] c        - margin flexible variable (the actual width)
    """
    def __init__(self, features, c=10):
        self._features = features
        self._c = c
        self._svm_instance = dlib.svm_c_trainer_linear()    # -> the linear case
        self._svm_instance.set_c(self._c)
        self._decision_function = None
        # Make it kind of binary problem => for multi-class confusion matrix, use: len(Database.ALLOWED_DIRS.values())
        classes_num = 2
        self._confusion_matrix = np.zeros((classes_num, classes_num, ), dtype=np.uint32)
        self._recognition_results = None



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
        Return labels as expected by dlib API
    """
    def __prepare_labels(self, labels, target_value):
        vals = dlib.array()
        for label in labels:
            if label == target_value:
                vals.append(+1)
            else:
                vals.append(-1)

        return vals


    """
        Training the classifier
    """
    def train(self):
        # prepare the data for being as type of dlib objects
        # -> dlib.vectors
        data = self.__prepare_data(self._features.get_bows())
        # -> dlib.array => [-1,1]
        labels = self.__prepare_labels(self._features.get_feature_vectors_labels_by_image(), self._features.get_target_value())
        # -> dlib._decision_function_radial_basis
        self._decision_function = self._svm_instance.train(data, labels)
        # print(self._decision_function(data[0]))
        return self._decision_function


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
            image_path = testImageDirName + "//" + image_name
            image = cv2.imread(image_path)
            image = cv2.resize(image, self._features.get_win_size(), interpolation=cv2.INTER_CUBIC)

            # Extracting features from an input new test image & computing the visual words
            feature_vectors = self._features.get_native_hog(image)

            # Build single BOW of dlib type
            dlib_bow = dlib.vector(self._features.generate_bows(feature_vectors, False))

            # Classifying a BOW using the classifier we have built in step 4
            prediction_activation = self.__activation(dlib_bow)         # -> [1, -1] by the activation function
            actual_activation = self.__pseudo_activation(image_name)    # -> [1, -1] by the activation function

            y_true.append(actual_activation)
            y_score.append(prediction_activation)

            # Given 2 classes determination -> need to determine for kind of binary problem
            self.__update_confusion_matrix(prediction_activation, actual_activation)

            #  A raw data image -> 1:1 [no another image with the same name on that test session]
            self._recognition_results[image_name] = prediction_activation

            # image = cv2.imread(image_path)
            # cv2.imshow("Predict: " + str(prediction_activation), image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        return y_true, y_score



    """
        Operating an activation function to determine the classification of dlib_bow
        [1] dlib_bow - bow conversion to dlib type
        Return [class, score]
    """
    def __activation(self, dlib_bow):
        float_value = self._decision_function(dlib_bow)
        if float_value > 0.0:
            return 1, float_value
        else:
            return -1, float_value


    """
        Pseudo activation operation - taking an image, split it than determine it's class from Database class hash
        [1] image name - input test image name
        Return [class, score]
    """
    def __pseudo_activation(self, image_name):
        actual_value = Database.ALLOWED_DIRS[image_name.split("_")[0]]
        if actual_value != 0:
            return -1, np.float32(actual_value)
        return 1, np.float32(actual_value)


    """
        Taking this problem as kind of binary problem and compare them
        [1] prediction  => [1,-1]
        [1] actual      => [1,-1]
    """
    def __update_confusion_matrix(self, prediction, actual):
        prediction = prediction[1]
        actual = actual[1]
        if prediction < 0:
            prediction = 1
        else:
            prediction = 0
        if actual < 0:
            actual = 1
        else:
            actual = 0
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
        return (self._confusion_matrix[0, 0] + self._confusion_matrix[1, 1]) / self._confusion_matrix.sum()


    """
        print the accuracy
    """
    def show_test_accuracy(self):
        print("Accuracy: (TP + TN)/#all_data = ({} + {})/{} = {} ".format(self._confusion_matrix[0, 0], self._confusion_matrix[1, 1], self._confusion_matrix.sum(), self.get_test_accuracy()))



    """
        get the precision
    """
    def get_test_precision(self):
        return self._confusion_matrix[0, 0] / (self._confusion_matrix[0, 0] + self._confusion_matrix[0, 1])



    """
        print the precision
    """
    def show_test_precision(self):
        print("Precision: TP/(TP + FP) = {}/({} + {}) = {} ".format(self._confusion_matrix[0, 0], self._confusion_matrix[0, 0], self._confusion_matrix[0, 1], self.get_test_precision()))


    """
        get the recall
    """
    def get_test_recall(self):
        return self._confusion_matrix[0, 0] / (self._confusion_matrix[0, 0] + self._confusion_matrix[1, 0])


    """
        print the recall
    """
    def show_test_recall(self):
        print("Recall: TP/(TP + FN) = {}/({} + {}) = {} ".format(self._confusion_matrix[0, 0], self._confusion_matrix[0, 0], self._confusion_matrix[1, 0], self.get_test_recall()))


    def show_test_c(self):
        print("Current C for the SVM margin width:", self._c)



    """
        Plot the desired Curve for choosing the optimal parameter
    """
    @staticmethod
    def ROC_Curve(y_true, y_score):
        # fpr, tpr, thresholds = roc_curve(y_true, y_score)
        # roc_auc = auc(fpr, tpr)
        # plt.figure()
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.title('SVM Classifier ROC Curve')
        # plt.plot(fpr, tpr, color='blue', lw=2, label='AUC = %0.2f)' % roc_auc)
        # plt.legend(loc="lower right")
        # plt.show()

        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('SVM Classifier Precision-Recall Curve')
        plt.plot(recall, precision, color='blue', lw=2, label='')
        plt.legend(loc="lower right")
        plt.show()

        # linearX = [0.0, 0.5, 1.0]
        # linearY = [1.0, 0.5, 0.0]
        # fig, axs = plt.subplots(2, 1, constrained_layout=True)
        # axs[0].plot()
        # axs[0].plot(recall, precision, '-')
        #
        # # axs.plot(t1, f(t1), 'o')
        # # axs.plot(t3, np.cos(2 * np.pi * t3), '--')
        #
        # axs[0].plot(linearX, linearY, '--')
        # axs[0].set_xlabel('Recall')
        # axs[0].set_ylabel('Precision')
        # fig.suptitle('ROC Curve (Dependent Variable: K)', fontsize=16)
        # axs[1].plot(dependent_variable, accuracy, '--')
        # axs[1].set_xlabel('K')
        # axs[1].set_ylabel('Accuracy')
        # plt.show()



"""
    Get all more optional datasets from the entry directory that is defined
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
def driver(datasets, k, c, SLEEP_TIME_OUT=3):
    recall = list()
    precision = list()
    accuracy = list()
    dependent_variable = list()
    db_instance = Database(trainImageDirName, FINAL_CLASSES=True, datasets=get_all_rest_datasets(datasets))
    db_instance.show_avaliable_datasets()
    feature_instance = Features(db_instance, k=k)
    feature_instance.generate_visual_word_dict(NEED_CLUSTERING=True)
    feature_instance.generate_bows(feature_instance.get_feature_vectors_by_image())
    feature_instance.save()
    feature_instance.load()
    classifier_instance = Classifier(feature_instance, c)
    classifier_instance.train()
    y_true, y_score = classifier_instance.recognizer()
    classifier_instance.save()
    classifier_instance.load()
    accuracy.append(classifier_instance.get_test_accuracy())
    precision.append(classifier_instance.get_test_precision())
    recall.append(classifier_instance.get_test_recall())
    dependent_variable.append(k)
    classifier_instance.show_test_accuracy()
    classifier_instance.show_test_precision()
    classifier_instance.show_test_recall()
    feature_instance.show_current_k()
    classifier_instance.show_test_c()
    time.sleep(SLEEP_TIME_OUT)
    return y_true, y_score, accuracy, precision, recall, dependent_variable


"""
    Determine the parameter to execute over
"""
def run(**kwargs):
    dataset_amount = 0
    k = 10
    c = 10.0
    for key, value in kwargs.items():
        if key.lower() == "c" and np.float32(value) > 0.0:
            c = np.float32(value)
        elif key.lower() == "k" and np.uint32(value) > 1:
            k = np.uint32(value)
        elif key.lower() == "dataset_amount" and np.uint32(value) > 0:
            dataset_amount = np.uint32(value)

    return driver(dataset_amount, k, c)



if __name__ == "__main__":

    y_true_glob = list()
    y_scores_glob = list()
    accuracy = list()
    precision = list()
    recall = list()

    # Configurable Section:
    # *********************
    loop_length = 10
    dataset_init = 0
    k_start = 5
    c_init = 10.0

    # [k_start, k_start + 1, k_start + 2, ..., k_start + loop_length + 1]
    K = range(k_start, k_start + loop_length + 1)
    # [0, 0, 0, ..., 0] => for loop_length size
    dataset_amounts = dataset_init * np.ones(loop_length, dtype=np.uint32)
    # [10.0, 10.0, 10.0, ..., 10.0] => for loop_length size
    C = c_init * np.ones(loop_length, dtype=np.float32)

    for ds_amt, k, c in zip(dataset_amounts, K, C):
        y_true, y_score, acc, prec, rec, dependent_variable = run(dataset_amount=ds_amt, k=k, c=c)

        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)

        for true, score in zip(y_true, y_score):
            y_true_glob.append(true)
            y_scores_glob.append(score)

        print("")

    precision.sort(reverse=True)
    recall.sort()
    # Classifier.ROC_Curve(y_true_glob, y_scores_glob)




    ######################################
    ## Changing K-means
    # recall = []
    # precision = []
    # accuracy = []
    # dependent_variable = []
    # name = 'K'
    #
    # db_instance = Database(trainImageDirName)
    # db_instance.show_avaliable_datasets()
    # # Must object to handle data as features
    # feature_instance = Features(db_instance)
    # ## Feature extraction process which is necessary while no pre-processing have been made yet
    # feature_instance.generate_visual_word_dict(NEED_CLUSTERING=False)
    # ## can get from cmd parameters or to determine through the main function
    # chunk = range(10, 12)
    # # iterating over 10 k values
    #
    # classifier_instance = None
    # for k in chunk:
    #     feature_instance.set_K(k)
    #     feature_instance.cluster_data()
    #     feature_instance.generate_bows(feature_instance.get_feature_vectors_by_image())
    #     feature_instance.save()
    #     feature_instance.load()
    #     classifier_instance = Classifier(feature_instance)
    #     classifier_instance.train()
    #     classifier_instance.recognizer()
    #     classifier_instance.save()
    #     classifier_instance.load()
    #
    #     accuracy.append(classifier_instance.get_test_accuracy())
    #     precision.append(classifier_instance.get_test_FPR())
    #     recall.append(classifier_instance.get_test_TPR())
    #     # precision.append(classifier_instance.get_test_precision())
    #     # recall.append(classifier_instance.get_test_recall())
    #     dependent_variable.append(k)
    #
    #     # classifier_instance.show_test_accuracy()
    #     # classifier_instance.show_test_precision()
    #     # classifier_instance.show_test_recall()
    #     feature_instance.show_current_k()
    #
    # print(classifier_instance._confusion_matrix)
    # precision.sort(reverse=True)
    # recall.sort()
    # Classifier.ROC_Curve(recall, precision, accuracy, dependent_variable, name)