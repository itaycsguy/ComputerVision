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


class Homography_Tracker_FF:
    """
    get_frame_from_video(index=0) -> frame
    .   @brief Fetching a frame from some specific index
    .   @param index Frame index
    """
    def get_frame_from_video(self, index):
        cap, frame = self.get_video_capturer()
        if index > 0:
            cap.set(1, index)
            ret, frame = cap.read()
            if not ret:
                print("The frame is corrupted.")
                exit(-1)

        cap.release()
        return frame



    """
    get_video_capturer(key_points) -> cv2.VideoCapturer, first video frame
    .   @brief Accessing the video file and extracting the first frame
    """
    def get_video_capturer(self):
        if not os.path.exists(inputDirectoryPath):
            print(inputDirectoryPath, "does not exist.")
            exit(-1)
        video_name = inputDirectoryPath + inputVideoName
        cap = cv2.VideoCapture(video_name)
        if not cap.isOpened():
            print(video_name, "does not exist.")
            exit(-1)
        ret, frame = cap.read()
        if not ret:
            print("The frame is corrupted.")
            exit(-1)

        return cap, frame


    """
    fetch_key_points(image[, qualityLevel=0.0001[, minDistance=7[, blockSize=7]]]) -> array of key points as tuples
    .   @brief Extracting interesting points by harris corner algorithm
    .   @param image The original RGB image
    .   @param quality_level
    .   @param min_distance
    .   @param block_size
    """
    def fetch_key_points(self, image, quality_level=0.0001, min_distance=7, block_size=7):
        feature_params = dict(qualityLevel=quality_level,
                              minDistance=min_distance,
                              blockSize=block_size,
                              useHarrisDetector=True)
        dst = cv2.goodFeaturesToTrack(np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)),
                                      numberOfPoints,
                                      **feature_params)
        # handling the case harris corner doesn't bring enough key points
        # qualityLevel found to be effective parameter on the detected points amount
        prev_dst_amt = dst.shape[0]
        while dst.shape[0] != numberOfPoints:
            if dst.shape[0] < numberOfPoints:
                feature_params["qualityLevel"] = feature_params["qualityLevel"] / 10.0
            else:
                feature_params["qualityLevel"] = feature_params["qualityLevel"] * 10.0
            dst = cv2.goodFeaturesToTrack(np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)),
                                          numberOfPoints,
                                          **feature_params)
            if prev_dst_amt == dst.shape[0]:
                # out if there is no improvement - maybe there is no points to take
                break
            else:
                prev_dst_amt = dst.shape[0]
        return dst


    """
    calc_next_point(prev_img, next_img, prev_pts) -> computed next key points, success status
    .   @brief Extracting interesting points by corner harris algorithm
    .   @param prev_img The previous original frame
    .   @param next_img The current original frame
    .   @param prev_pts Key points of the previous frame
    """
    def calc_next_point(self, prev_img, next_img, prev_pts):
        lk_params = dict(winSize=(22, 22),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY),
                                                         cv2.cvtColor(next_img, cv2.COLOR_RGB2GRAY),
                                                         prev_pts,
                                                         None,
                                                         **lk_params)
        return next_pts, status.ravel()


    def blend_frames(self, A, B):
        # generate Gaussian pyramid for A
        G = A.copy()
        gpA = [G]
        for i in range(6):
            G = cv2.pyrDown(G)
            gpA.append(G)

        # generate Gaussian pyramid for B
        G = B.copy()
        gpB = [G]
        for i in range(6):
            G = cv2.pyrDown(G)
            gpB.append(G)

        # generate Laplacian Pyramid for A
        lpA = [gpA[5]]
        for i in range(5, 0, -1):
            size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
            GE = cv2.pyrUp(gpA[i], dstsize=size)
            L = cv2.subtract(gpA[i-1], GE)
            lpA.append(L)

        # generate Laplacian Pyramid for B
        lpB = [gpB[5]]
        for i in range(5, 0, -1):
            size = (gpB[i-1].shape[1], gpB[i-1].shape[0])
            GE = cv2.pyrUp(gpB[i], dstsize=size)
            L = cv2.subtract(gpB[i-1], GE)
            lpB.append(L)

        # Now add left and right halves of images in each level
        LS = []
        for la, lb in zip(lpA, lpB):
            rows, cols, dpt = la.shape
            ls = np.hstack((la[:, 0:int(cols/2)], lb[:, int(cols/2):]))
            LS.append(ls)

        # now reconstruct
        ls_ = LS[0]
        for i in range(1, 6):
            size = (LS[i].shape[1], LS[i].shape[0])
            ls_ = cv2.pyrUp(ls_, dstsize=size)
            ls_ = cv2.add(ls_, LS[i])

        return ls_


    """
    Reference: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    run_tracking([save_out]) -> None
    .   @brief Computing the homography between 2 frames into a video sequence
    .   @param save_out - If true the video will be saved out to Results directory
    """
    def run_tracking(self, save_out=False):
        print("Running homography tracking process..")
        global Points
        global point_img
        Points = list()
        cap, init_img = self.get_video_capturer()
        point_img = init_img.copy()
        video_instance = None
        video_name = ""
        if save_out:
            if not os.path.exists(outputDirectoryPath):
                os.mkdir(outputDirectoryPath)
            w, h, _ = init_img.shape
            video_name = outputDirectoryPath + "HCT_" + inputVideoName[0:-4] + "_" + str(numberOfPoints) + "_out.avi"
            video_instance = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (h, w))

        init_pts = self.fetch_key_points(init_img)
        while cap.isOpened():
            ret, next_img = cap.read()
            if not ret:
                break
            else:
                # computing the next points
                next_pts, status = self.calc_next_point(init_img, next_img, init_pts)
                good = next_pts[status == 1]
                # need to verify there is no less than 4 points due to projective requirements for at least 8 equations
                if len(good) >= MIN_MATCH_COUNT:
                    # try:
                    # computing homography to the initial frame
                    H, mask = cv2.findHomography(init_pts, good, cv2.RANSAC, 5.0)
                    # computing the transformation and paste it onto the initial frame
                    dst = cv2.perspectiveTransform(next_pts, H)
                    img = cv2.warpPerspective(next_img, H, (next_img.shape[1], next_img.shape[0]))
                    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
                    # img = cv2.add(np.zeros_like(img), cv2.polylines(np.zeros_like(init_img), [np.int32(dst)], True, 255, 2))
                    # img = self.blend_frames(init_img, img)
                    cv2.imshow('Processed Frame Out', img)
                    # cv2.waitKey(0)
                    if save_out:
                        video_instance.write(img)
                    k = cv2.waitKey(1) & 0xff
                    if k == 27:
                        break
                    # except Exception as _:
                    #     TODO: need to check the exception source
                    # print("Main loop exception!")
                    # pass
                else:
                    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))


        cv2.destroyAllWindows()
        if save_out:
            video_instance.release()
            print(video_name, "is saved.")
        cap.release()
        print("Done!")


if __name__ == "__main__":
    tracker = Homography_Tracker_FF()
    tracker.run_tracking()