## Students And Developers: Itay Guy, 305104184 & Elias Jadon, 207755737

import os
import cv2
import numpy as np
import argparse


trainImageDirName = "D://PycharmProjects//ComputerVision//Assignment_2//Datasets"
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
    def __init__(self, database, cell_size=8, bin_n=16):
        self._database = database
        # These are parameters which required from the program to get initially
        self._cell_size = cell_size # Descriptor cell computation on the image after resizing
        self._bin_n = bin_n         # Number of bins
        self._centers = None
        self._labels = None
        self._bow = None



    def __get_HOG_desctiptor(self, image_instance):
        # Gradients computations
        gx = cv2.Sobel(image_instance, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(image_instance, cv2.CV_32F, 0, 1)

        # Degrees computations [0 - 180]
        mag, ang = cv2.cartToPolar(gx, gy)

        # bin separation by _bin_n steps
        bin = np.int32(self._bin_n * ang / (2 * np.pi))

        bin_cells = []
        mag_cells = []
        # The cell_size can be separated to cell_x, cell_y which at this case they are the same for always
        for i in range(0, int(image_instance.shape[0] / self._cell_size)):
            for j in range(0, int(image_instance.shape[1] / self._cell_size)):
                bin_cells.append(bin[(i * self._cell_size): (i * self._cell_size + self._cell_size), (j * self._cell_size): (j * self._cell_size + self._cell_size)])
                mag_cells.append(mag[(i * self._cell_size): (i * self._cell_size + self._cell_size), (j * self._cell_size): (j * self._cell_size + self._cell_size)])

        hists = [np.bincount(b.ravel(), m.ravel(), self._bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # Transform to Hellinger kernel [Normalization issue]
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= cv2.norm(hist) + eps

        # print(hist)
        # return the descriptor which is fully describe the image features
        return hist



    def gen_visual_word_dict(self):
        data = self._database.get_data_hash()
        database_feature_vectors = list()
        feature_vectors_labels = list()
        for image_name, image_label in zip(data.keys(), data.values()):
            # Coloured image
            image_instance = cv2.imread(self._database.get_root_dir() + "//" + image_name)
            image_instance = cv2.resize(image_instance, (64, 128), interpolation=cv2.INTER_CUBIC)
            database_feature_vectors.append(np.array(self.__get_HOG_desctiptor(image_instance), dtype=np.float32))
            feature_vectors_labels.append(image_label)

        # Here - DB training-set is transferred to features vectors representation
        # Clustering into 3 groups at our problem
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, self._labels, self._centers = cv2.kmeans(np.asarray(database_feature_vectors), self._database.get_sets_amount(), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return self._centers, self._labels


    def gen_BOWs(self):
        pass


    def save(self):
        pass



db_instance = Database(trainImageDirName)
feature_instance = Features(db_instance)
feature_instance.gen_visual_word_dict()

# class Classifier:
#     def __init__(self):
#         pass
#
#     def train(self):
#         pass
#
#     def test(self):
#         pass