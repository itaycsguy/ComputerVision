# Itay Guy, 305104184 & Elias Jadon, 207755737
import cv2, os, argparse
import numpy as np

# out data:
# directory where all input data should being resided - should be provided by the user
inputDirectoryPath = ".//Datasets//"
# directory where results should being saved - it is created if it doesn't exist
outputDirectoryPath = ".//Results//"

inputVideoName = "ParkingLot.mp4"
numberOfPoints = 20

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
        w, h, _ = prev_img.shape
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
    __align_images(im1, im2) -> projected im1 on im2, homography matrix
    .   @brief Computing the projected image of im1 on im2 using ORB object
    """
    def __align_images(self, im1, im2):
        good_match_percent = 0.15
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors
        orb = cv2.ORB_create(numberOfPoints)
        kp1, desc1 = orb.detectAndCompute(im1_gray, None)
        kp2, desc2 = orb.detectAndCompute(im2_gray, None)

        # Match features
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(desc1, desc2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        good_matches = int(len(matches) * good_match_percent)
        matches = matches[:good_matches]

        # Draw top matches
        # im_matches = cv2.drawMatches(im1, kp1, im2, kp2, matches, None)
        # cv2.imwrite("matches.jpg", im_matches)

        # Extract location of good matches
        p1 = np.zeros((len(matches), 2), dtype=np.float32)
        p2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            p1[i, :] = kp1[match.queryIdx].pt
            p2[i, :] = kp2[match.trainIdx].pt

        # Find homography
        H, _ = cv2.findHomography(p1, p2, cv2.RANSAC)

        # Use homography
        h, w, _ = im2.shape
        im_reg = cv2.warpPerspective(im1, H, (w, h))

        return im_reg, H

    """
    grab_contours(cnts) -> contours list
    .   @brief Implementing the imutils' function
    """
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


    """
    get_contour_mask(frame) -> mask of the outer rectangle
    .   @brief Computing the outer rectangle mask
    """
    def get_contour_mask(self, frame):
        # frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = self.grab_contours(cnts)
        largest_cnt = max(cnts, key=cv2.contourArea)

        # allocate memory for the mask which will contain the rectangular bounding box of the stitched image region
        mask = np.zeros(thresh.shape, dtype=np.uint8)
        (x, y, w, h) = cv2.boundingRect(largest_cnt)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        return mask


    """
    mark_ROI(image, points) -> marked point on the image
    .   @brief Marking the region of interest of an image
    """
    def mark_ROI(self, image, points):
        img = image.copy()
        for _, point in enumerate(points):
            a, b = point
            img = cv2.circle(img, (int(a), int(b)), 5, [0, 255, 0], +1)
        return img


    def overhead_mapping(self, img, overview_pts):
        pass


    """
    calc_homography(curr_frame, T, dsize, golden_frame, golden_pts) -> warped image, good point of the warped image
    .   @brief Computing an homography between 2 images
    """
    def calc_homography(self, curr_frame, T, dsize, golden_frame, golden_pts):
        # Computing the next points by optical flow
        curr_pts, status = self.calc_next_point(golden_frame, curr_frame, golden_pts)
        # Reaching the correspondences only
        good_curr_pts = curr_pts[status == 1]
        good_curr_pts = good_curr_pts.reshape(len(good_curr_pts), 2)
        good_golden_pts = golden_pts[status == 1]
        good_golden_pts = good_golden_pts.reshape(len(good_golden_pts), 2)
        # Marking the correspondences points on the current frame we are about the warping
        curr_frame = self.mark_ROI(curr_frame, good_curr_pts)
        # Computing the actual homography between the golden and the current frame
        H, _ = cv2.findHomography(srcPoints=good_curr_pts, dstPoints=good_golden_pts, method=cv2.RANSAC,
                                  ransacReprojThreshold=5.0, maxIters=1000, confidence=0.995)
        # Warping the current frame using the homography that has been found step ago
        curr_warped = cv2.warpPerspective(src=curr_frame, M=np.matmul(H, T), dsize=dsize,
                                          flags=cv2.INTER_NEAREST)
        return curr_warped, good_curr_pts


    """
    Reference: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    run_tracking() -> None
    .   @brief Running the tracking + mosaic process
    """
    def run_tracking(self):
        print("Processing..")
        global Points
        global point_img
        Points = list()
        cap, golden_frame = self.get_video_capturer()
        golden_pts = self.fetch_key_points(golden_frame)  # self.get_points_from_user()
        point_img = golden_frame.copy()
        w = golden_frame.shape[0]
        h = golden_frame.shape[1]
        c = golden_frame.shape[2]

        # Final accumulated mosaic
        mosaic = np.zeros((w * 2, h * 2, c), dtype=np.uint8)

        xt = w / 2
        yt = h / 2
        # Translation matrix to the mosaic center due to homography alignment property
        T = np.asmatrix([
            [1, 0, xt],
            [0, 1, yt],
            [0, 0, 1]], dtype=np.float32)

        while cap.isOpened():
            ret, curr_frame = cap.read()
            if not ret:
                break
            else:
                # Computing the homography and warping the current frame
                curr_warped, warped_pts = self.calc_homography(curr_frame, T, (h * 2, w * 2),
                                                               golden_frame, golden_pts)

                good_locations = np.where(curr_warped != [0, 0, 0])

                # Adding the pixel's which are not holding zero color to the final mosaic image
                mosaic[good_locations] = curr_warped[good_locations]

                resized_mosaic = cv2.resize(mosaic, (1400, 900), interpolation=cv2.INTER_CUBIC)
                cv2.imshow("mosaic", resized_mosaic)

                # Updating the previous frame (golden frame) to be the current one
                golden_frame = curr_frame.copy()
                golden_pts = warped_pts.copy()

                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break

        cv2.destroyAllWindows()
        cap.release()
        print("Done!")



if __name__ == "__main__":
    tracker = Homography_Tracker_FF()
    tracker.run_tracking()