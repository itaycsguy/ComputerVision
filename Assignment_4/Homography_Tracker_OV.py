# Itay Guy, 305104184 & Elias Jadon, 207755737
import cv2, os, argparse
import numpy as np


# out data:
# directory where all input data should being resided - should be provided by the user
inputDirectoryPath = ".//Datasets//"
# directory where results should being saved - it is created if it doesn't exist
outputDirectoryPath = ".//Results//"

inputVideoName = "Soccer2.mp4"
MIN_MATCH_COUNT = 4
numberOfPoints = 15


class Homography_Tracker_OV:

    """
    Reference: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    run_tracking([save_out]) -> None
    .   @brief Computing the homography between a frame to some overhead view
    .   @param save_out - If true the video will be saved out to Results directory
    """
    def run_tracking(self, save_out=False):
        print("Running homography tracking process..")
        pass

if __name__ == "__main__":
    tracker = Homography_Tracker_OV()
    tracker.run_tracking()