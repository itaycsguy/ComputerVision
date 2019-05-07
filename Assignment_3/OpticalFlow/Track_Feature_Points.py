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


Point_color = (0, 0, 255)
Point_size = 7
Line_color = (0, 255, 0)
Line_size = 2

#Input Variables
inputVideoName = "ballet.mp4"
selectPoints = True
numberOfPoints = 7






def functionvsd():
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(".//Datasets//" + inputVideoName)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        exit()

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()



# mouse callback function
def mouse_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        Points.append((x, y))
    paint_point(Points, Point_color)


# given a segment points and a color, paint in seg_image
def paint_point(segment, color):
    for center in segment:
        cv2.circle(point_img, center, Point_size , color, -1)


def paint_velocity(velocity_vector , point_vector , image):
    for i in range(0,numberOfPoints):
        p = point_vector[i]
        v = velocity_vector[i]
        to = (p[0] + v[0],p[1] + v[1])
        cv2.line(image, p, to, Line_color, Line_size)
    return image


#Get the first frame from the video
def GetFirstImage():
    cap = cv2.VideoCapture(".//Datasets//" + inputVideoName)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        exit()
    cap.set(1, 0) #zero is the index of the frame & one is a flag
    res, image = cap.read()
    if res == False:
        print("Error! Could not read image.")
        exit()
    cap.release()
    return image


#let the user choose the points on the image
def GetPointsFromUser():
    global orig_img, point_img
    global Points
    orig_img = GetFirstImage()
    point_img = GetFirstImage()
    cv2.namedWindow("Select Points")
    # mouse event listener
    cv2.setMouseCallback("Select Points", mouse_click)
    # lists to hold pixels in each segment
    Points = []
    while True:
        cv2.imshow("Select Points", point_img)
        k = cv2.waitKey(20)

        if (k == 27) | (Points.__len__() == numberOfPoints):  # escape
            break
    cv2.destroyAllWindows()
    return orig_img, point_img, Points


if __name__ == "__main__":
    if(selectPoints):
        orig_img, point_img, Points = GetPointsFromUser()
    else:
        #Find interested points by HOG
        #Hog will give us many points
        #We need to choose a number of them (like the parameter we have)
        print("TODO: Choose points from HOG")

    velocity = [(50,0)] * numberOfPoints
    velocity[0] = (0,100)

    point_img = paint_velocity(velocity, Points, point_img)



    cv2.imshow("origin", orig_img)
    cv2.imshow("added points", point_img)
    cv2.waitKey(0)






