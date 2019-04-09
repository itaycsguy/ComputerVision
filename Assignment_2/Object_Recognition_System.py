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
    def __init__(self, database, hist_size=9):
        self._database = database
        self._hist_size = hist_size


    def __get_HOG_desctiptor(self, image_instance):
        image_instance = np.float32(image_instance) / 255.0
        for row in range(0, image_instance.shape[0], 8):
            for col in range(0, image_instance.shape[1], 8):
                sub_image = image_instance[row: (row + 8), col: (col + 8)]
                gx = cv2.Sobel(sub_image, cv2.CV_32F, 1, 0, ksize=1)
                gy = cv2.Sobel(sub_image, cv2.CV_32F, 0, 1, ksize=1)
                magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

                histograms = list()
                for x, y in zip(range(0, magnitude.shape[0]), range(0, magnitude.shape[1])):

                    # build a local histogram
                    hist_instance = {}
                    jump_chunk = 180.0 / self._hist_size
                    for i in np.arange(0, 180, jump_chunk):
                        hist_instance[i] = 0.0

                    grad_mag_value = magnitude[x, y]    # magnitude
                    grad_angle_value = angle[x, y]      # direction
                    if grad_angle_value in hist_instance.keys():
                        hist_instance[grad_angle_value] += grad_mag_value
                    else:
                        jump_half_size = 180.0 / (2.0 * self._hist_size)
                        sorted_hist = sorted(hist_instance.keys(), reverse=False)

                        # Can being into one location only
                        for i in range(0, len(sorted_hist) - 1):
                            if sorted_hist[i] < grad_angle_value and sorted_hist[i + 1] > grad_angle_value:
                                diff = grad_angle_value - sorted_hist[i]
                                if diff == jump_half_size:
                                    hist_instance[sorted_hist[i]] += grad_mag_value / 2.0
                                    hist_instance[sorted_hist[i + 1]] += grad_mag_value / 2.0
                                elif diff > jump_half_size:
                                    hist_instance[sorted_hist[i + 1]] += grad_mag_value
                                else:
                                    hist_instance[sorted_hist[i]] += grad_mag_value

                    # need to make it global
                    histograms.append(hist_instance.copy())






    def gen_visual_word_dict(self):
        data = self._database.get_data_hash()
        database_feature_vectors = list()
        feature_vectors_labels = list()
        for image_name, image_label in zip(data.keys(), data.values()):
            image_instance = cv2.imread(self._database.get_root_dir() + "//" + image_name, 0)
            image_instance = cv2.resize(image_instance, (64, 128), interpolation=cv2.INTER_CUBIC)
            database_feature_vectors.append(self.__get_HOG_desctiptor(image_instance))
            feature_vectors_labels.append(image_label)

        # how to make hash to trained vector?
        ## Clustering to K groups
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # ret, label, center = cv2.kmeans(database_feature_vectors, self._database.get_sets_amount(), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        #
        # return {
        #             0: database_feature_vectors[label.ravel() == 0],
        #             1: database_feature_vectors[label.ravel() == 1],
        #             2: database_feature_vectors[label.ravel() == 2]
        #         }


    def feature_vectors_to_BOWs(self):
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