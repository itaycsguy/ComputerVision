## Students And Developers: Itay Guy, 305104184 & Elias Jadon, 207755737

import cv2, argparse, os
import numpy as np

trainImageDirName = "D://PycharmProjects//ComputerVision//Assignment_2//Datasets"

## Positive class = Airplane
## Negative classes = Elephant & Motorbike
testImageDirName = "/Datasets/Testset"

class Feature_Extractor():
    def __init__(self):
        self._hog = cv2.HOGDescriptor()

    def build_visual_word_dictionary(self):
        descriptors = list()
        for gen_dir in os.listdir(trainImageDirName):
            dir_name = trainImageDirName + "//" + gen_dir
            for image in os.listdir(dir_name):
                image_path = dir_name + "//" + image
                descriptors.append(self._hog.compute(cv2.imread(image_path)))
        return descriptors


im = cv2.imread('D://PycharmProjects//ComputerVision//Assignment_2//Datasets//Airplane//image_0707.jpg')
fe = Feature_Extractor()
des = fe.build_visual_word_dictionary()
print(des)