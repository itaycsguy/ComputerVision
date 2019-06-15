# Itay Guy, 305104184 & Elias Jadon, 207755737
import cv2, os
import numpy as np

# out data:
# directory where all input data should being resided - should be provided by the user
input_dir_path = ".//Datasets//"
# directory where results should being saved - it is created if it doesn't exist
output_dir_path = ".//Results//"

input_video_name = "HexBugs.mp4"    # "ParkingLot.mp4"
num_auto_key_points = 2000
num_manual_key_points = 20
moving_scene = False
is_manual_selection = False
save_out = False

# User key-points selection
Point_color = (0, 0, 255)
Point_size = 4
Line_color = (0, 255, 0)
Line_size = 3

# Action with the user
Action_Rect = "Select 4 Rectangle Points.."
Action_Track = "Select " + str(num_manual_key_points) + " Points To Track.."

class Rectification_Tracker:
    VISUAL_DEVIATION = 3/4
    SCALE = 1/6
    BLACK = [0, 0, 0]
    RANSAC_THRESH = 5.0
    ACTION_NAME = ""
    JPEG_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

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
    def select_points_from_user(self, is_rect=False):
        cv2.namedWindow(Rectification_Tracker.ACTION_NAME)
        # mouse event listener
        cv2.setMouseCallback(Rectification_Tracker.ACTION_NAME, self.mouse_click)

        num_points = num_manual_key_points
        if is_rect:
            num_points = 4
        # lists to hold pixels in each segment
        while True:
            cv2.imshow(Rectification_Tracker.ACTION_NAME, point_img)
            k = cv2.waitKey(20)
            if (k == 27) or (len(Points) == num_points):  # escape
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
    def extract_key_points(self, image, quality_level=0.01, min_distance=30, block_size=3, max_corners=200):
        if is_manual_selection:
            max_corners = num_manual_key_points
        elif num_auto_key_points > 0:
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
        if status is not None:
            status = status.ravel()
        else:
            status = list()
        return next_pts, status.ravel()


    """
    get_key_points(frame[, is_manual=False]) -> key-point from frame
    .   @brief Extracting or selection manually key-points from frame
    """
    def get_key_points(self, frame, is_manual=False, is_rect=False):
        if is_manual:
            points = self.select_points_from_user(is_rect)
            Points.clear()
            return points

        points = self.extract_key_points(frame)
        Points.clear()
        return points


    """
    mark_ROI(image, points) -> marked point on the frame
    .   @brief Marking the region of interest of an frame
    """
    def mark_ROI(self, frame, points):
        frm = frame.copy()
        for _, point in enumerate(points):
            a, b = point
            frm = cv2.circle(frm, (int(a), int(b)), Point_size, Point_color, +1)

        return frm


    """
    mark_velocity(frame, new_pts, old_pts) -> marked velocity on the frame
    .   @brief Marking the velocity on the frame
    """
    def mark_velocity(self, frame, new_pts, old_pts):
        frm = frame.copy()
        mask = np.zeros_like(frm)
        for i, (n, o) in enumerate(zip(new_pts, old_pts)):
            a, b = n.ravel()
            c, d = o.ravel()

            # velocity computation
            mask = cv2.line(mask, (a, b), (c, d), Line_color, Line_size)
            # frm = cv2.circle(frm, (a, b), 3, [0, 0, 255], -1)

        # cv2.add(frm, mask)
        return mask


    """
    calc_homography(curr_frame, T, dsize, golden_frame, golden_pts) -> warped image, good point of the warped image
    .   @brief Computing an homography between 2 frames
    """
    def calc_homography(self, curr_frame, T, d_size, golden_frame, golden_pts, visual_pts, visual_deviation):
        # Computing the next points by optical flow
        curr_pts, status = self.calc_next_points(golden_frame, curr_frame, golden_pts)
        next_visual_pts, visual_status = self.calc_next_points(golden_frame, curr_frame, visual_pts)

        # Reaching the correspondences only
        good_curr_pts = np.float32(curr_pts[status == 1])
        good_golden_pts = np.float32(golden_pts[status == 1])

        # Computing the actual homography between the golden and the current frame
        H, _ = cv2.findHomography(good_curr_pts, good_golden_pts, cv2.RANSAC, Rectification_Tracker.RANSAC_THRESH)

        T_new = np.matmul(T, H)
        # Warping the current frame using the homography that has been found step ago
        curr_warped = cv2.warpPerspective(curr_frame,
                                          T_new, d_size,
                                          flags=cv2.INTER_NEAREST)
        velocity_mask = None
        if len(visual_pts):
            # Keeping the points bulk from being removed
            if len(next_visual_pts) < (visual_deviation * len(visual_pts)):
                next_visual_pts = visual_pts.copy()

            # Marking the velocity on the current frame than warp it
            visual_pts = visual_pts[visual_status == 1]
            next_visual_pts = next_visual_pts[visual_status == 1]
            velocity_mask = cv2.warpPerspective(self.mark_velocity(curr_frame,
                                                                   next_visual_pts.reshape(len(next_visual_pts), 2),
                                                                   visual_pts.reshape(len(visual_pts), 2)),
                                                T_new, d_size,
                                                flags=cv2.INTER_NEAREST)

        return curr_warped, T_new, velocity_mask, next_visual_pts, visual_status


    """
    reorder_points(pts) -> reorganized points as defined
    .   @brief Ordering 4 points as predefined by the rectangle
    """
    def reorder_points(self, pts):
        reshaped_pts = pts.reshape(len(pts), 2)
        min_x = int(min(reshaped_pts, key=lambda x: x[0])[0])
        max_x = int(max(reshaped_pts, key=lambda x: x[0])[0])
        min_y = int(min(reshaped_pts, key=lambda x: x[1])[1])
        max_y = int(max(reshaped_pts, key=lambda x: x[1])[1])
        pts = np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]).reshape(-1, 1, 2)
        return pts


    """
    get_overview_coordinates(h, w) -> return reshaped 4 points of a rectangle
    .   @brief Building a rectangle by height and width of the input image
    """
    def get_overview_coordinates(self, h, w):
        return np.asarray([[0., 0.],
                           [np.float32(h), 0.],
                           [0., np.float32(w)],
                           [np.float32(h), np.float32(w)]]).reshape(-1, 1, 2)


    """
    Reference: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    run_tracking([is_manual=False[, save_out=False]]) -> None
    .   @brief Running the tracking + mosaic process
    """
    def run_tracking(self, moving_scene=False, is_manual=False, save_out=False):
        print("Processing..")
        global Points
        global point_img
        Points = list()

        cap, golden_frame = self.get_video_capturer()
        point_img = golden_frame.copy()
        w, h, c = golden_frame.shape

        Rectification_Tracker.ACTION_NAME = Action_Rect
        golden_pts = self.reorder_points(self.get_key_points(golden_frame, is_manual=True, is_rect=True))
        rect_coord = self.get_overview_coordinates(w, h)
        H, _ = cv2.findHomography(golden_pts, rect_coord, cv2.RANSAC, Rectification_Tracker.RANSAC_THRESH)
        golden_pts = self.get_key_points(golden_frame, is_manual=False)

        # Make the decision to be on the projected frame
        # visual_frame = cv2.warpPerspective(golden_frame,
        #                                    T,
        #                                    (h, w),
        #                                    flags=cv2.INTER_NEAREST)
        # point_img = visual_frame.copy()

        Rectification_Tracker.ACTION_NAME = Action_Track
        visual_pts = self.get_key_points(golden_frame, is_manual=True)

        T = H.copy()
        if moving_scene:
            # Translation matrix to the mosaic center due to homography alignment property
            x_translate = (1-Rectification_Tracker.SCALE)/2
            y_translate = (1-Rectification_Tracker.SCALE)/2
            T = np.matmul(
                np.asmatrix([
                    [Rectification_Tracker.SCALE, 0, w*x_translate],
                    [0, Rectification_Tracker.SCALE, h*y_translate],
                    [0, 0, 1]], dtype=np.float32), H)

        # Final accumulated mosaic
        mosaic = np.zeros((h, w, c), dtype=np.uint8)
        velocity_mask = np.zeros_like(mosaic)

        while cap.isOpened():
            ret, curr_frame = cap.read()
            if not ret:
                break
            else:
                # Computing the homography and warping the current frame
                curr_warped, T, mask, pts, status = self.calc_homography(curr_frame,
                                                                         T, (w, h),
                                                                         golden_frame,
                                                                         golden_pts,
                                                                         visual_pts,
                                                                         Rectification_Tracker.VISUAL_DEVIATION)
                # Adding the pixel's which are not holding zeros to the final mosaic image
                good_locations = np.where(curr_warped != Rectification_Tracker.BLACK)
                mosaic[good_locations] = curr_warped[good_locations]

                # Update the first frame to be the current
                golden_frame = curr_frame.copy()
                golden_pts = self.get_key_points(golden_frame, is_manual)

                if len(status) > 0:
                    visual_pts = pts
                if mask is not None:
                    velocity_mask = cv2.add(velocity_mask, mask)
                    mosaic = cv2.add(mosaic, velocity_mask)

                cv2.imshow("Stable Mosaic Scene In Progress..", np.rot90(np.fliplr(mosaic)))
                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break

        cv2.destroyAllWindows()
        cap.release()
        print("Done!")

        final_mosaic = np.rot90(np.fliplr(mosaic))
        if save_out:
            if not os.path.exists(output_dir_path):
                os.mkdir(output_dir_path)
            out_name = "Stable_Scene_" + input_video_name[0:-4] + "_out.jpg"
            cv2.imwrite(output_dir_path + out_name, final_mosaic, Rectification_Tracker.JPEG_PARAM)
            print(out_name, "is saved.")

        cv2.imshow("Final Stable Mosaic Scene", final_mosaic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = Rectification_Tracker()
    tracker.run_tracking(moving_scene=moving_scene, is_manual=is_manual_selection, save_out=save_out)