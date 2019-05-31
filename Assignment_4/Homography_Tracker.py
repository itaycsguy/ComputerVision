# Itay Guy, 305104184 & Elias Jadon, 207755737
import cv2, os, argparse
import numpy as np

# out data:
# directory where all input data should being resided - should be provided by the user
inputDirectoryPath = ".//Datasets//"
# directory where results should being saved - it is created if it doesn't exist
outputDirectoryPath = ".//Results//"
MIN_MATCH_COUNT = 4
inputVideoName = "Soccer2.mp4"
selectPoints = False
numberOfPoints = 15
Point_color = (0, 0, 255)
Point_size = 7
Line_color = (0, 255, 0)
Line_size = 2


class VideoTracker:
    DIV_REFRESH = 4
    DEFAULT_REFRESH_THRESH = 100

    def __init__(self):
        self.total_frames = -1
        self.is_ar_least_half_kp = False
        self.refresh_thresh = VideoTracker.DEFAULT_REFRESH_THRESH
        self.last_frames_number = list()
        self.lfn_pointer = 0


    """
    mouse_click() -> None
    .   @brief Taking a position of some mouse click
    .   @param event Event object
    .   @param x An x location
    .   @param y An y location
    """
    def mouse_click(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            Points.append((x, y))
        self.paint_point(Points, point_img)


    """
    paint_point() -> painted image
    .   @brief Painting circles on an image
    .   @param points Points to draw
    .   @param im An image
    """
    def paint_point(self, points, im):
        for center in points:
            cv2.circle(im, center, Point_size, Point_color, -1)
        return im


    """
    get_points_from_user() -> Points array
    .   @brief Picking points from a frame interactively
    """
    def get_points_from_user(self):
        cv2.namedWindow("Select Points")
        # mouse event listener
        cv2.setMouseCallback("Select Points", self.mouse_click)

        # lists to hold pixels in each segment
        while True:
            cv2.imshow("Select Points", point_img)
            k = cv2.waitKey(20)
            if (k == 27) or (len(Points) == numberOfPoints):  # escape
                break
        cv2.destroyAllWindows()

        return np.asarray(Points, dtype=np.float32).reshape(-1, 1, 2)

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

        # handling session name
        if len(self.last_frames_number) == 2:
            self.last_frames_number[self.lfn_pointer] = index
            self.lfn_pointer = (self.lfn_pointer + 1) % 2
        else:
            self.last_frames_number.append(index)

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
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.refresh_thresh = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / VideoTracker.DIV_REFRESH)
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
        self.is_ar_least_half_kp = int(len(dst) / 2) >= self.total_frames
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


    """
    plot_peaks_of(image) -> None
    .   @brief Displaying the key point that were found to this specific image
    .   @param image
    """
    def plot_peaks_of(self, image):
        key_points = self.fetch_key_points(image)
        img = image.copy()
        for i, point in enumerate(key_points):
            a, b = point.ravel()
            img = cv2.circle(img, (a, b), 4, [0, 0, 255], -1)

        cv2.imshow('Key points displaying', img)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()


    """
    Reference: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    homography_tracking([overhead_view[, save_out]]) -> None
    .   @brief Computing the homography between 2 frames into a video sequence
    .   @param overhead_view [default=False] - If true it will be mapped to some overhead view
    .   @param save_out                      - If true the video will be saved out to Results directory
    """
    def homography_tracking(self, overhead_view=False, save_out=False):
        print("Running homography tracking process..")
        global Points
        global point_img
        Points = list()
        cap, prev_img = self.get_video_capturer()
        point_img = prev_img.copy()
        video_instance = None
        video_name = ""
        if save_out:
            if not os.path.exists(outputDirectoryPath):
                os.mkdir(outputDirectoryPath)
            w, h, _ = prev_img.shape
            video_name = outputDirectoryPath + "HCT_" + inputVideoName[0:-4] + "_" + str(numberOfPoints) + "_out.avi"
            video_instance = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (h, w))
        if selectPoints:

            prev_pts = self.get_points_from_user()
        else:
            prev_pts = self.fetch_key_points(prev_img)

        # Points of the initial frame where we should find the homography
        init_img = prev_img.copy()
        init_pts = prev_pts.copy()

        if overhead_view:
            # TODO: need to handle overhead case
            pass

        while cap.isOpened():
            ret, next_img = cap.read()
            if not ret:
                break
            else:
                # computing the next points
                next_pts, status = self.calc_next_point(init_img, next_img, prev_pts)
                good = next_pts[status == 1]
                # need to verify there is no less than 4 points due to projective requirements for at least 8 equations
                if len(good) >= MIN_MATCH_COUNT:
                    try:
                        # computing homography to the initial frame
                        H, mask = cv2.findHomography(init_pts, good, cv2.RANSAC, 5.0)
                        # computing the transformation and paste it onto the initial frame
                        h, w, _ = next_img.shape
                        dst = cv2.perspectiveTransform(init_pts, H)

                        # TODO: need to compute the object location and adjust it the first frame - mosic image
                        img = cv2.add(init_img, cv2.polylines(next_img, [np.int32(dst)], True, 255, 2, cv2.LINE_AA))
                        cv2.imshow('Processed Frame Out', img)
                        if save_out:
                            video_instance.write(img)
                        k = cv2.waitKey(1) & 0xff
                        if k == 27:
                            break
                    except Exception as _:
                        # TODO: need to check the exception source
                        pass
                else:
                    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))


        cv2.destroyAllWindows()
        if save_out:
            video_instance.release()
            print(video_name, "is saved.")
        cap.release()
        print("Done!")


if __name__ == "__main__":
    tracker = VideoTracker()
    tracker.homography_tracking()

    # Submission time - DO NOT REMOVE!
    # parser = argparse.ArgumentParser(description='Computer Vision - Assignment 4')
    # parser.add_argument("--homo_first_frame", help="Find an homography to the first frame.", action="store_true")
    # parser.add_argument("--homo_overhead_view", help="Find an homography to some overhead view.", action="store_true")
    # args = parser.parse_args()
    # tracker = VideoTracker()
    #
    # flag = False
    # if args.homo_first_frame:
    #     flag = True
    #     tracker.homography_tracking()
    # elif args.homo_overhead_view:
    #     flag = True
    #     tracker.homography_tracking(overhead_view=True)
    #
    # if not flag:
    #     print("Should pick any functionality to execute. for help use -h.")