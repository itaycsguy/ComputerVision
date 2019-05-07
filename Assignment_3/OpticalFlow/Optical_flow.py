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



def run_optimum(classifier):
    if classifier == Classifier.NN or classifier == Classifier.SVM:
        if classifier == Classifier.NN:
            print("Running NN Classifier..")
        else:
            print("Running SVM Classifier..")
        run(list(), classifier)
    else:
        print("A classifier name should be provided.")


if __name__ == "__main__":
    


