# Itay Guy, 305104184 & Elias Jadon, 207755737
import cv2, os, argparse
import numpy as np

# out data:
# directory where all input data should being resided - should be provided by the user
input_dir_path = ".//Datasets//"
# directory where results should being saved - it is created if it doesn't exist
output_dir_path = ".//Results//"

input_video_name = "Billiard2.mp4" #"ParkingLot.mp4"
num_auto_key_points = 200
numb_manual_key_points = 10
is_manual_selection = True

# User key-points selection
Point_color = (0, 0, 255)
Point_size = 5
Line_color = (0, 255, 0)
Line_size = 2

class Homography_Tracker:
    """
    mouse_click() -> None
    .   @brief Taking a position of some mouse click
    """
    def mouse_click(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            Points.append((x, y))
        self.paint_point(Points, point_img)


    """
    paint_point() -> painted image
    .   @brief Painting circles on an image
    """
    def paint_point(self, points, im):
        for center in points:
            cv2.circle(im, center, Point_size, Point_color, -1)
        return im


    """
    select_points_from_user() -> Points array
    .   @brief Picking points from a frame interactively
    """
    def select_points_from_user(self):
        cv2.namedWindow("Select Points")
        # mouse event listener
        cv2.setMouseCallback("Select Points", self.mouse_click)

        # lists to hold pixels in each segment
        while True:
            cv2.imshow("Select Points", point_img)
            k = cv2.waitKey(20)
            if (k == 27) or (len(Points) == numb_manual_key_points):  # escape
                break
        cv2.destroyAllWindows()

        return np.asarray(Points, dtype=np.float32).reshape(-1, 1, 2)

    """
    get_frame_from_video(index=0) -> frame
    .   @brief Fetching a frame from some specific index
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
        if not os.path.exists(input_dir_path):
            print(input_dir_path, "does not exist.")
            exit(-1)
        video_name = input_dir_path + input_video_name
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
    extract_key_points(image[, qualityLevel=0.01[, minDistance=30[, blockSize=3[, max_corners=2000]]]]) -> array of key points as tuples
    .   @brief Extracting ROI points
    """
    def extract_key_points(self, image, quality_level=0.01, min_distance=30, block_size=3, max_corners=2000):
        if max_corners < 0:
            max_corners = num_auto_key_points
        return cv2.goodFeaturesToTrack(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                       qualityLevel=quality_level,
                                       minDistance=min_distance,
                                       blockSize=block_size,
                                       maxCorners=max_corners)


    """
    calc_next_points(prev_img, next_img, prev_pts) -> computed next key points, success status
    .   @brief Extracting interesting points by corner harris algorithm
    """
    def calc_next_points(self, prev_img, next_img, prev_pts):
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY),
                                                         cv2.cvtColor(next_img, cv2.COLOR_RGB2GRAY),
                                                         prev_pts,
                                                         None,
                                                         winSize=(22, 22),
                                                         maxLevel=3,
                                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                                   10, 0.03))
        return next_pts, status.ravel()


    """
    get_key_points(frame[, is_manual=False]) -> key-point from frame
    .   @brief Extracting or selection manually key-points from frame
    """
    def get_key_points(self, frame, is_manual=False):
        if is_manual:
            return self.select_points_from_user()

        return self.extract_key_points(frame)


    """
    mark_ROI(image, points) -> marked point on the image
    .   @brief Marking the region of interest of an image
    """
    def mark_ROI(self, image, points):
        img = image.copy()
        for _, point in enumerate(points):
            a, b = point
            img = cv2.circle(img, (int(a), int(b)), 5, [0, 0, 255], +1)

        return img



    """
    calc_homography(curr_frame, T, dsize, golden_frame, golden_pts) -> warped image, good point of the warped image
    .   @brief Computing an homography between 2 frames
    """
    def calc_homography(self, curr_frame, T, d_size, golden_frame, golden_pts):
        # Computing the next points by optical flow
        curr_pts, status = self.calc_next_points(golden_frame, curr_frame, golden_pts)

        # Reaching the correspondences only
        good_curr_pts = np.float32(curr_pts[status == 1])
        good_golden_pts = np.float32(golden_pts[status == 1])

        # Marking the correspondences points on the current frame we are about the warping
        # curr_frame = self.mark_ROI(curr_frame, good_curr_pts.reshape(len(good_curr_pts), 2))

        # Computing the actual homography between the golden and the current frame
        H, _ = cv2.findHomography(good_curr_pts, good_golden_pts, cv2.RANSAC, 5.0)

        T_new = np.matmul(T, H)
        # Warping the current frame using the homography that has been found step ago
        curr_warped = cv2.warpPerspective(curr_frame,
                                          T_new, d_size,
                                          flags=cv2.INTER_NEAREST)

        return curr_warped, T_new


    """
    Reference: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    run_tracking([is_manual=False]) -> None
    .   @brief Running the tracking + mosaic process
    """
    def run_tracking(self, is_manual=False):
        print("Processing..")
        global Points
        global point_img
        Points = list()

        cap, golden_frame = self.get_video_capturer()
        point_img = golden_frame.copy()

        golden_pts = self.get_key_points(golden_frame, is_manual)
        w, h, c = golden_frame.shape

        # Final accumulated mosaic
        mosaic = np.zeros((w * 2, h * 2, c), dtype=np.uint8)

        # Translation matrix to the mosaic center due to homography alignment property
        x_translate = 1
        y_translate = 0
        scale = 1
        T = np.asmatrix([
            [scale, 0, w*x_translate],
            [0, scale, h*y_translate],
            [0, 0, scale]], dtype=np.float32)

        while cap.isOpened():
            ret, curr_frame = cap.read()
            if not ret:
                break
            else:
                # Computing the homography and warping the current frame
                curr_warped, T = self.calc_homography(curr_frame,
                                                      T, (h*2, w*2),
                                                      golden_frame,
                                                      golden_pts)

                # Adding the pixel's which are not holding zeros to the final mosaic image
                good_locations = np.where(curr_warped != [0, 0, 0])
                mosaic[good_locations] = curr_warped[good_locations]

                # Update the first frame to be the current
                golden_frame = curr_frame.copy()
                golden_pts = self.get_key_points(golden_frame, is_manual)

                cv2.imshow("mosaic", cv2.resize(mosaic, (1200, 700)))
                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break

        cv2.destroyAllWindows()
        cap.release()
        print("Done!")

        cv2.imshow("mosaic", mosaic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    tracker = Homography_Tracker()
    tracker.run_tracking(is_manual=is_manual_selection)