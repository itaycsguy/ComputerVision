# Itay Guy, 305104184 & Elias Jadon, 207755737
import cv2, os, argparse
import numpy as np

# out data:
# directory where all input data should being resided - should be provided by the user
inputDirectoryPath = ".//Datasets//"
# directory where results should being saved - it is created if it doesn't exist
outputDirectoryPath = ".//Results//"

inputVideoName = "ParkingLot.mp4"
MIN_MATCH_COUNT = 4
numberOfPoints = 1000

Point_color = (0, 0, 255)
Point_size = 7
Line_color = (0, 255, 0)
Line_size = 2

class Homography_Tracker_FF:
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
        w, h, _ = prev_img.shape    # winSize=(22, 22)
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
    """
    def plot_ROI(self, image, points):
        img = image.copy()
        for _, point in enumerate(points):
            a, b = point
            img = cv2.circle(img, (int(a), int(b)), 5, [0, 0, 255], -1)

        cv2.imshow('Key points displaying', img)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()


    def grab_contours(self, cnts):
        # if the length the contours tuple returned by cv2.findContours
        # is '2' then we are using either OpenCV v2.4, v4-beta, or
        # v4-official
        if len(cnts) == 2:
            cnts = cnts[0]

        # if the length of the contours tuple is '3' then we are using
        # either OpenCV v3, v4-pre, or v4-alpha
        elif len(cnts) == 3:
            cnts = cnts[1]

        # otherwise OpenCV has changed their cv2.findContours return
        # signature yet again and I have no idea WTH is going on
        else:
            raise Exception(("Contours tuple must have length 2 or 3, "
                             "otherwise OpenCV changed their cv2.findContours return "
                             "signature yet again. Refer to OpenCV's documentation "
                             "in that case"))

        # return the actual contours array
        return cnts


    def cut_stitched(self, stitched_img):
        stitched = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = self.grab_contours(cnts)
        largest_cnt = max(cnts, key=cv2.contourArea)

        # allocate memory for the mask which will contain the rectangular bounding box of the stitched image region
        mask = np.zeros(thresh.shape, dtype=np.uint8)
        (x, y, w, h) = cv2.boundingRect(largest_cnt)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        min_rect = mask.copy()
        sub = mask.copy()

        while cv2.countNonZero(sub) > 0:
            min_rect = cv2.erode(min_rect, None)
            sub = cv2.subtract(min_rect, thresh)

        # find contours in the minimum rectangular mask and then extract the bounding box (x, y)-coordinates
        cnts = cv2.findContours(min_rect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = self.grab_contours(cnts)
        min_cnt = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(min_cnt)

        return stitched[y:y + h, x:x + w]


    def calc_homography_stitched(self, img0, pts0, img1, pts1):
        # Computing Homography transformation with the initial point's frame
        H, _ = cv2.findHomography(pts1, pts0, cv2.RANSAC, 5.0)
        # Warping the next frame with the Homography matrix
        warped_img = cv2.warpPerspective(img1, H, flags=cv2.WARP_INVERSE_MAP, dsize=(img1.shape[1], img1.shape[0]))
        # Stitching between the initial frame and the next warped frame to single new frame
        return cv2.Stitcher.create(cv2.Stitcher_SCANS).stitch([warped_img, img0])



    def overhead_mapping(self, img, overview_pts):
        img_pts = np.asarray([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]]).reshape(-1, 1, 2)
        points = overview_pts.reshape(len(overview_pts), 2)
        y_min = int(min(points[:, 0]))
        y_max = int(max(points[:, 0]))
        x_min = int(min(points[:, 1]))
        x_max = int(max(points[:, 1]))
        overview_img = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        status, stitched = self.calc_homography_stitched(img, img_pts, overview_img, overview_pts)
        if status == cv2.Stitcher_OK:
            cv2.imshow("series", stitched)
            # cv2.waitKey(0)
            return stitched




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
        # init_img = cv2.resize(init_img, (256, 256), interpolation=cv2.INTER_CUBIC)

        w, h, _ = init_img.shape
        video_instance = cv2.VideoWriter(outputDirectoryPath + "FF_" + inputVideoName[0:-4] + "_" + str(numberOfPoints) + "_out.avi",
                                         cv2.VideoWriter_fourcc(*'XVID'), 20.0, (h, w))

        point_img = init_img.copy()
        init_pts = self.fetch_key_points(init_img)  # self.get_points_from_user()
        # init_img = self.overhead_mapping(init_img, init_pts)
        while cap.isOpened():
            ret, next_img = cap.read()
            # next_img = cv2.resize(next_img, (256, 256), interpolation=cv2.INTER_CUBIC)
            if not ret:
                break
            else:
                # Optical flow: find correspondences
                next_pts, status = self.calc_next_point(init_img, next_img, init_pts)
                good = next_pts[status == 1]
                # Auto handling of the correspondences bugs
                if len(good) >= MIN_MATCH_COUNT:
                    status, stitched = self.calc_homography_stitched(init_img, init_pts, next_img, next_pts)
                    if status == cv2.Stitcher_OK:
                        stitched = cv2.resize(stitched, (init_img.shape[1], init_img.shape[0]), interpolation=cv2.INTER_CUBIC)
                        # cv2.imshow("series", stitched)
                        # cv2.waitKey(0)
                        video_instance.write(stitched)

                    k = cv2.waitKey(1) & 0xff
                    if k == 27:
                        break
                else:
                    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

        cv2.destroyAllWindows()
        if save_out:
            pass
        video_instance.release()
        cap.release()
        print("Done!")


if __name__ == "__main__":
    tracker = Homography_Tracker_FF()
    tracker.run_tracking()