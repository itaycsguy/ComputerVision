import numpy as np
import dlib
import cv2

IS_HOLD_VER_COND = False
# np version should be at least 1.16.2
NUMPY_VER = '1.16.2'
if np.__version__ >= NUMPY_VER:
    IS_HOLD_VER_COND = True | IS_HOLD_VER_COND
    print("Numpy version is ",np.__version__," ---> OK.")
else:
    IS_HOLD_VER_COND = False | IS_HOLD_VER_COND
    print("Numpy version is ",np.__version__," ---> Failed. Should be",NUMPY_VER)

# OpenCV version should be 4.0.0.21
OPENCV_VER = '4.0.0'
if cv2.__version__ == OPENCV_VER:
    IS_HOLD_VER_COND = True | IS_HOLD_VER_COND
    print("OpenCV version is ",cv2.__version__," ---> OK.")
else:
    IS_HOLD_VER_COND = False | IS_HOLD_VER_COND
    print("OpenCV version is ",cv2.__version__," ---> Failed. Should be",OPENCV_VER)

# dlib version should be 19.16.0
DLIB_VER = '19.16.0'
if dlib.__version__ == DLIB_VER:
    IS_HOLD_VER_COND = True | IS_HOLD_VER_COND
    print("dlib version is ",dlib.__version__," ---> OK.")
else:
    IS_HOLD_VER_COND = False | IS_HOLD_VER_COND
    print("dlib version is ",dlib.__version__," ---> Failed. Should be",DLIB_VER)

if IS_HOLD_VER_COND:
    print("All conditions are holding.")
else:
    print("Few conditions are disturbed.")