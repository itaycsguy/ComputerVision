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


#Input Variables
inputVideoName = "ballet.mp4"
inputByVideo = True
im1 = "image001.jpg"
im2 = "image001.jpg"
frameNumber1 = 10   #range from zero to total_frames
frameNumber2 = 20   #range from zero to total_frames


def get_images():
    if(inputByVideo):
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        cap = cv2.VideoCapture(".//Datasets//" + inputVideoName)
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
            exit()

        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if(frameNumber1 < 0) | (frameNumber1 >= total_frame):
            print("Error! 'frameNumber1' is out of range")
            exit()
        if (frameNumber2 < 0) | (frameNumber2 >= total_frame):
            print("Error! 'frameNumber2' is out of range")
            exit()

        cap.set(1, frameNumber1)
        res1, image1 = cap.read()

        cap.set(1, frameNumber2)
        res2, image2 = cap.read()

        cap.release()
    else:
        image1 = cv2.imread(".//Datasets//" + im1)
        image2 = cv2.imread(".//Datasets//" + im2)
        height1, width1, channels1 = image1.shape
        height2, width2, channels2 = image2.shape
        if (height1 != height2) | (width1 != width2) | (channels1 != channels2):
            print("Error! the size of the two images is not equal!")
            exit()
    return image1,image2


if __name__ == "__main__":
    image1,image2 = get_images()

    cv2.imshow("first image",image1)
    cv2.imshow("second image",image2)
    cv2.waitKey(0)





