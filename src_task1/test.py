import numpy as np
import dlib
import cv3

# np version should be at least 1.16.2
NUMPY_VER = '1.16.2'
if np.__version__ >= NUMPY_VER:
    print("Numpy version is ",np.__version__," ---> OK.")
else:
    print("Numpy version is ",np.__version__," ---> Failed. Should be",NUMPY_VER)

# OpenCV version should be 4.0.0.21
OPENCV_VER = '4.0.0.21'
if opencv.__version__ == OPENCV_VER:
    print("OpenCV version is ",opencv.__version__," ---> OK.")
else:
    print("OpenCV version is ",opencv.__version__," ---> Failed. Should be",OPENCV_VER)

# dlib version should be 19.16.0
DLIB_VER = '19.16.0'
if dlib.__version__ == DLIB_VER:
    print("dlib version is ",dlib.__version__," ---> OK.")
else:
    print("dlib version is ",dlib.__version__," ---> Failed. Should be",DLIB_VER)