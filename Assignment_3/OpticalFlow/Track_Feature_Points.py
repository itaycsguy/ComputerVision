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
Line_size = 3
Window_Size = 2

#Input Variables
inputVideoName = "bugs11.mp4"
selectPoints = True
numberOfPoints = 5






def show_video():
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
    paint_point(Points,point_img)


# given a segment points and a color, paint in seg_image
def paint_point(points , im):
    for center in points:
        cv2.circle(im, center, Point_size , Point_color, -1)
    return im


def paint_velocity(velocity , point , image):
    to = (int(point[0] + velocity[0]),int(point[1] + velocity[1]))
    cv2.line(image, p, to, Line_color, Line_size)
    return image



#Get a frame from the video
def GetFrameByIndex(index):
    cap_video = cv2.VideoCapture(".//Datasets//" + inputVideoName)
    # Check if camera opened successfully
    if not cap_video.isOpened():
        print("Error opening video stream or file")
        exit()
    cap_video.set(1, index) #zero is the index of the frame & one is a flag
    res, image = cap_video.read()
    if not res:
        print("Error! Could not read image.")
        exit()
    cap_video.release()
    return image


#let the user choose the points on the image
def GetPointsFromUser():
    global orig_img, point_img
    global Points
    orig_img = GetFrameByIndex(0)
    point_img = GetFrameByIndex(0)
    cv2.namedWindow("Select Points")
    # mouse event listener
    cv2.setMouseCallback("Select Points", mouse_click)
    # lists to hold pixels in each segment
    Points = []
    while True:
        cv2.imshow("Select Points", point_img)
        k = cv2.waitKey(20)

        if (k == 27) or (len(Points) == numberOfPoints):  # escape
            break
    cv2.destroyAllWindows()
    return orig_img, point_img, Points


def Getderivatives(frame1 , frame2 , centerX , centerY , windowSize):
    It = (np.average(frame2[centerX - windowSize:centerX + windowSize + 1, centerY - windowSize:centerY + windowSize + 1], axis=2) -
          np.average(frame1[centerX - windowSize:centerX + windowSize + 1, centerY - windowSize:centerY + windowSize + 1], axis=2)).reshape(np.power(2 * windowSize + 1, 2), 1)

    # It = np.average(fr2[x,y]) - np.average(fr1[x,y])

    Ix = (np.average(frame1[(centerX + 1) - windowSize:(centerX + 1) + windowSize + 1, centerY - windowSize:centerY + windowSize + 1], axis=2) - np.average(
        frame1[centerX - windowSize:centerX + windowSize + 1, centerY - windowSize:centerY + windowSize + 1], axis=2)).reshape(np.power(2 * windowSize + 1, 2), 1)
    # Ix = np.average(fr1[x+1, y]) - np.average(fr1[x, y])

    Iy = (np.average(frame1[centerX - windowSize:centerX + windowSize + 1, (centerY + 1) - windowSize:(centerY + 1) + windowSize + 1], axis=2) - np.average(
        frame1[centerX - windowSize:centerX + windowSize + 1, centerY - windowSize:centerY + windowSize + 1], axis=2)).reshape(np.power(2 * windowSize + 1, 2), 1)
    # Iy = np.average(fr1[x, y+1]) - np.average(fr1[x, y])
    return It,Ix,Iy

def Lucas_Kanade_system(Ix,Iy,It):
    A = np.concatenate((Ix, Iy), axis=1)
    b = -1 * It
    At = np.transpose(A)
    AtA = np.dot(At, A)
    Atb = np.dot(At, b)
    if (np.linalg.det(AtA) == 0):
        return [0.0,0.0]
    x = np.linalg.solve(AtA, Atb)
    return np.round(np.transpose(x)[0])

if __name__ == "__main__":
    if selectPoints:
        orig_img, point_img, Points = GetPointsFromUser()
    else:
        #Find interested points by HOG
        #Hog will give us many points
        #We need to choose a number of them (like the parameter we have)
        print("TODO: Choose points from HOG")




    cap = cv2.VideoCapture(".//Datasets//" + inputVideoName)
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    for indexFrame in range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1):
        print(">> " + str(indexFrame))
        fr1 = GetFrameByIndex(indexFrame).astype(float)
        fr2 = GetFrameByIndex(indexFrame+1).astype(float)
        draw_im = GetFrameByIndex(indexFrame)
        draw_im = paint_point(Points, draw_im)


        i=0
        print(Points)
        for (y, x) in Points:
            p = (x,y)

            print("x = " + str(x))
            print("y = " + str(y))

            It, Ix, Iy = Getderivatives(fr1 , fr2 , x , y , Window_Size)
            solution = Lucas_Kanade_system(Ix,Iy,It)
            print("solution = " + str(solution))




            height, width, channels = fr1.shape

            UpdateX = int(x + solution[0])
            UpdateY = int(y + solution[1])

            if (UpdateX >= (height-Window_Size-1)):
                UpdateX = height - Window_Size-2
            if (UpdateY >= (width-Window_Size-1)):
                UpdateY = width -Window_Size- 2
            if (UpdateX <= Window_Size):
                UpdateX = 1+Window_Size
            if (UpdateY <= Window_Size):
                UpdateY = 1+Window_Size

            Points[i] = (UpdateY , UpdateX)
            draw_im = paint_velocity(solution, [UpdateX,UpdateY], draw_im)
            cv2.imshow("added points", draw_im)
            cv2.waitKey(1)
            i=i+1

        print(Points)

