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
import os
import sys
sys.path.append(os.path.abspath('.'))
from KeyPointsFinder import *


Point_color = (0, 0, 255)
Point_size = 7
Line_color = (0, 255, 0)
Line_size = 3
Window_Size = 20
First_frame = 0

#Input Variables
inputVideoName = "highway.avi"  #"highway.avi" #""bugs11.mp4"
selectPoints = False
numberOfPoints = 200




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


def paint_velocity(point , velocity , image , Randomcolor = False , index = 0):
    if Randomcolor:
        rand = np.mod(np.power(index , 30) , 256)
        c = (int(rand),int(rand),int(rand))

    else:
        c = Line_color

    for i in range(len(point)):
        p = point[i]
        v = velocity[i]
        from_ = (p[0] , p[1])
        to_ = (int(p[0] + v[0]),int(p[1] + v[1]))
        cv2.line(image, from_, to_, c, Line_size)
        #print("from = " + str(from_) + " - to = " + to_)
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
        #exit()
    cap_video.release()
    return image


#let the user choose the points on the image
def GetPointsFromUser():
    global orig_img, point_img
    global Points
    orig_img = GetFrameByIndex(First_frame)
    point_img = GetFrameByIndex(First_frame)
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


def Getderivatives(frame1 , frame2 , centerX1 , centerY1 , centerX2 ,centerY2 , windowSize):
    It = (np.average(frame2[centerX2 - windowSize:centerX2 + windowSize + 1, centerY2 - windowSize:centerY2 + windowSize + 1], axis=2) -
          np.average(frame1[centerX1 - windowSize:centerX1 + windowSize + 1, centerY1 - windowSize:centerY1 + windowSize + 1], axis=2)).reshape(np.power(2 * windowSize + 1, 2), 1)

    # It = np.average(fr2[x,y]) - np.average(fr1[x,y])

    Ix = (np.average(frame1[(centerX1 + 1) - windowSize:(centerX1 + 1) + windowSize + 1, centerY1 - windowSize:centerY1 + windowSize + 1], axis=2) - np.average(
        frame1[centerX1 - windowSize:centerX1 + windowSize + 1, centerY1 - windowSize:centerY1 + windowSize + 1], axis=2)).reshape(np.power(2 * windowSize + 1, 2), 1)
    # Ix = np.average(fr1[x+1, y]) - np.average(fr1[x, y])

    Iy = (np.average(frame1[centerX1 - windowSize:centerX1 + windowSize + 1, (centerY1 + 1) - windowSize:(centerY1 + 1) + windowSize + 1], axis=2) - np.average(
        frame1[centerX1 - windowSize:centerX1 + windowSize + 1, centerY1 - windowSize:centerY1 + windowSize + 1], axis=2)).reshape(np.power(2 * windowSize + 1, 2), 1)
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

def Affine_Lucas_Kanade_system(Ix,Iy,It):
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
        orig_img = GetFrameByIndex(First_frame)
        point_img = GetFrameByIndex(First_frame)
        #Find interested points by HOG
        #Hog will give us many points
        #We need to choose a number of them (like the parameter we have)
        Points = KeyPointsFinder(orig_img).get_key_points(numberOfPoints)
        print("TODO: Choose points from HOG")

    height, width, channels = orig_img.shape

    cap = cv2.VideoCapture(".//Datasets//" + inputVideoName)
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(".//Results//Video" + inputVideoName, fourcc, 20.0, (width,height))
    LastFrame = GetFrameByIndex(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)


    for indexFrame in range(First_frame,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1):
        print(">> " + str(indexFrame))
        fr1 = GetFrameByIndex(indexFrame).astype(float)
        fr2 = GetFrameByIndex(indexFrame+1).astype(float)
        draw_im = GetFrameByIndex(indexFrame)
        draw_im = paint_point(Points, draw_im)


        i=0
        print(Points)
        velocity = []
        for (y, x) in Points: #Pay attention: the coordinates is reversed!
            p = (x,y)

            print("x = " + str(x))
            print("y = " + str(y))


            UpdateX = x
            UpdateY = y
            round = 0
            while round < 9:
                It, Ix, Iy = Getderivatives(fr1 , fr2 , x , y , UpdateX , UpdateY, Window_Size)
                solution = Lucas_Kanade_system(Ix,Iy,It)
                print("solution = " + str(solution))

                UpdateX = int(UpdateX + solution[0])
                UpdateY = int(UpdateY + solution[1])

                if (solution[0] == 0) and (solution[1] == 0):
                    break
                if (UpdateX >= (height-Window_Size-1)):
                    UpdateX = height - Window_Size-2
                    break
                if (UpdateY >= (width-Window_Size-1)):
                    UpdateY = width -Window_Size- 2
                    break
                if (UpdateX <= Window_Size):
                    UpdateX = 1+Window_Size
                    break
                if (UpdateY <= Window_Size):
                    UpdateY = 1+Window_Size
                    break
                round = round+1

            LastFrame = paint_velocity([[x,y]], [[UpdateX - x , UpdateY - y]], LastFrame , True , i)
            Points[i] = (UpdateY , UpdateX)
            velocity.append((UpdateX - x , UpdateY - y))
            i=i+1

        print(Points)
        draw_im = paint_velocity(Points, velocity, draw_im)
        cv2.imshow("added points", draw_im)
        out.write(draw_im)
        cv2.waitKey(1)


    cv2.imwrite(".//Results//Velocity" + inputVideoName+".jpg", LastFrame)
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


