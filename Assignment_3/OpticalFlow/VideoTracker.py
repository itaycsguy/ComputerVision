import cv2
import numpy as np

# Key points:
# out data:
refreshThresh = 100
inputDirectoryPath = ".//Datasets//"

# task data:
inputVideoName = "ballet.mp4"  # "highway.avi" # "bugs11.mp4" # "rushHour.mp4"
selectPoints = True
numberOfPoints = 10

# Segmentation:
# out data:
SEGMENTATION = 1
SEGMENT_ZERO = 0
SEGMENT_ONE = 1
SEGMENT_TWO = 2
SEGMENT_THREE = 3
SEG_ZERO_COLOR = (0, 0, 255)
SEG_ONE_COLOR = (0, 255, 0)
SEG_TWO_COLOR = (255, 0, 0)
SEG_THREE_COLOR = (0, 255, 255)
INIT_REQUIRED_COUNTER = 2
INIT_CLUSTERS = 4


Point_color = (0, 0, 255)
Point_size = 7
Line_color = (0, 255, 0)
Line_size = 3


class SegmentDetector:

    def __init__(self, image, seg_counter=INIT_CLUSTERS):
        self.img = image
        self.segments_counter = seg_counter


    """
    calc_bin_grabcut(f_seg_indices[, b_seg_indices[, iterations=20]]) -> mask
    .   @brief Calculating the binary graph cut
    .   @param f_seg_indices
    .   @param b_seg_indices
    .   @param iterations[default=20]
    """
    def calc_bin_grabcut(self, f_seg_indices, b_seg_indices, iterations=20):
        mask = np.ones(self.img.shape[:2], dtype=np.uint8) * cv2.GC_PR_BGD

        for (x, y) in f_seg_indices:
            # (x, y) -> (y, x) due to image input and numpy conversion
            mask[y][x] = cv2.GC_FGD

        for (x, y) in b_seg_indices:
            mask[y][x] = cv2.GC_BGD

        # algorithm MUST parameters:
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        mask, _, _ = cv2.grabCut(self.img, mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
        mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype(np.uint8)

        return mask


    """
    calc_grabcut_combinations(seg0[, seg1[, seg2[, seg3]]]) -> mask, mask complement
    .   @brief Calculating the graph cut combinations
    .   @param seg0[global variable]
    .   @param seg1[global variable]
    .   @param seg2[global variable]
    .   @param seg3[global variable]
    """
    def calc_grabcut_combinations(self, seg0, seg1, seg2, seg3):
        f0_123 = self.calc_bin_grabcut(seg0, seg1 + seg2 + seg3)
        f0_12 = self.calc_bin_grabcut(seg0, seg1 + seg2)
        f0_13 = self.calc_bin_grabcut(seg0, seg1 + seg3)
        f0_23 = self.calc_bin_grabcut(seg0, seg2 + seg3)
        f0_1 = self.calc_bin_grabcut(seg0, seg1)
        f0_2 = self.calc_bin_grabcut(seg0, seg2)
        f0_3 = self.calc_bin_grabcut(seg0, seg3)
        f0_complement = np.ones(self.img.shape[:2], dtype=np.uint8) - self.calc_bin_grabcut(seg1 + seg2 + seg3, seg0)

        # voting per segment
        f0 = f0_1 + f0_2 + f0_3 + f0_complement + f0_12 + f0_13 + f0_23 + (2 * f0_123)
        return f0, f0_complement


    """
    segmenting_isolation(f0[, f0_complement[, f1[, f1_complement[, f2[, f2_complement[, f3[, f3_complement]]]]]]]) -> segmentation complement mask
    .   @brief Calculating the segmentation complement segments decisions
    .   @param f0
    .   @param f0_complement
    .   @param f1
    .   @param f1_complement
    .   @param f2
    .   @param f2_complement
    .   @param f3
    .   @param f3_complement
    """
    def segmenting_isolation(self, f0, f0_complement, f1, f1_complement, f2, f2_complement, f3, f3_complement):
        # make majority voting:
        f0_middle = ((f0 > f1) & (f0 > f2) & (f0 > f3))
        f1_middle = ((f1 > f0) & (f1 > f2) & (f1 > f3))
        f2_middle = ((f2 > f0) & (f2 > f1) & (f2 > f3))
        f3_middle = ((f3 > f0) & (f3 > f1) & (f3 > f2))

        empty_px = np.ones(self.img.shape[:2], dtype=np.uint8) - (f0_middle + f1_middle + f2_middle + f3_middle)
        # getting masks intersects and detect conflicts
        f0_px_amt = ((f0 >= f1) & (f0 >= f2) & (f0 >= f3))
        f1_px_amt = ((f1 >= f0) & (f1 >= f2) & (f1 >= f3))
        f2_px_amt = ((f2 >= f0) & (f2 >= f1) & (f2 >= f3))
        f3_px_amt = ((f3 >= f0) & (f3 >= f1) & (f3 >= f2))
        f01_conflict = (f0_px_amt == f1_px_amt)
        f02_conflict = (f0_px_amt == f2_px_amt)
        f03_conflict = (f0_px_amt == f3_px_amt)
        f12_conflict = (f1_px_amt == f2_px_amt)
        f13_conflict = (f1_px_amt == f3_px_amt)
        f23_conflict = (f2_px_amt == f3_px_amt)

        # added more pixels to the final mask segment by empty mask comparison the conflicts
        irrelevant_added_px0 = (empty_px & f01_conflict & f0_complement) | (empty_px & f02_conflict & f0_complement) | (empty_px & f03_conflict & f0_complement)
        f0_middle = f0_middle + irrelevant_added_px0

        empty_px = empty_px - irrelevant_added_px0
        irrelevant_added_px1 = (empty_px & f01_conflict & f1_complement) | (empty_px & f12_conflict & f1_complement) | (empty_px & f13_conflict & f1_complement)
        f1_middle = f1_middle + irrelevant_added_px1

        if self.segments_counter == 3:
            empty_px = empty_px - irrelevant_added_px1
            f2_middle = f2_middle + empty_px

        else:
            empty_px = empty_px - irrelevant_added_px1
            irrelevant_added_px2 = (empty_px & f02_conflict & f2_complement) | (empty_px & f12_conflict & f2_complement) | (empty_px & f23_conflict & f2_complement)
            f2_middle = f2_middle + irrelevant_added_px2

            empty_px = empty_px - irrelevant_added_px2
            f3_middle = f3_middle + empty_px

        return f0_middle, f1_middle, f2_middle, f3_middle


    """
    segmenting_isolation() -> final mask
    .   @brief Calculating the final mask
    """
    def calc_multivoting_grabcut(self):
        print("\nProcessing..")
        f0 = f0_complement = f1 = f1_complement = f2 = f2_complement = f3 = f3_complement = np.zeros(self.img.shape[:2], dtype=np.uint8)
        if self.segments_counter == 2:
            f0 = self.calc_bin_grabcut(seg0, seg1)
            f1 = np.ones(self.img.shape[:2], dtype=np.uint8) - f0
        elif self.segments_counter == 3:
            f0, f0_complement = self.calc_grabcut_combinations(seg0, seg1, seg2, seg3)
            f1, f1_complement = self.calc_grabcut_combinations(seg1, seg0, seg2, seg3)
            f2, f2_complement = self.calc_grabcut_combinations(seg2, seg1, seg0, seg3)
        elif self.segments_counter == 4:
            f0, f0_complement = self.calc_grabcut_combinations(seg0, seg1, seg2, seg3)
            f1, f1_complement = self.calc_grabcut_combinations(seg1, seg0, seg2, seg3)
            f2, f2_complement = self.calc_grabcut_combinations(seg2, seg1, seg0, seg3)
            f3, f3_complement = self.calc_grabcut_combinations(seg3, seg1, seg2, seg0)

        print("Done grab-cut computations..")

        ## extract empty pixels:
        f0_final = np.zeros(self.img.shape[:2], dtype=np.uint8)
        f1_final = np.zeros(self.img.shape[:2], dtype=np.uint8)
        f2_final = np.zeros(self.img.shape[:2], dtype=np.uint8)
        f3_final = np.zeros(self.img.shape[:2], dtype=np.uint8)
        if self.segments_counter > 2:
            f0_final, f1_final, f2_final, f3_final = self.segmenting_isolation(f0, f0_complement, f1, f1_complement, f2, f2_complement, f3, f3_complement)
        else:
            f0_final, f1_final = (f0, f1)
        print("Done masks multi-voting isolation and global mask segments enhancing..")

        f0_mask = f0_final[:, :, np.newaxis] * SEG_ZERO_COLOR
        f1_mask = f1_final[:, :, np.newaxis] * SEG_ONE_COLOR
        f2_mask = f2_final[:, :, np.newaxis] * SEG_TWO_COLOR
        f3_mask = f3_final[:, :, np.newaxis] * SEG_THREE_COLOR
        print("The process is finished.")

        return f0_mask + f1_mask + f2_mask + f3_mask


class SegmentationWrapper:

    """
    prepare_seg(curr_label, kmeans_curr_center, kmeans_labels, optical_flow_float) -> seg array with numpy pair to tuple pairs into
    .   @brief Converting point from numpy array to tuple.
    .   @param curr_label The current cluster where the segment is belonging to.
    .   @param kmeans_labels All labels.
    .   @param kmeans_curr_center Kmeans output centers.
    .   @param optical_flow_float All key-points array.
    """
    @staticmethod
    def prepare_seg(curr_label, kmeans_curr_center, kmeans_labels, optical_flow_float):
        locations = list()
        min_values = list()
        for i in range(optical_flow_float.shape[0]):
            for j in range(optical_flow_float.shape[1]):
                if kmeans_labels[i, j] == curr_label:
                    locations.append((j, i))
                    min_values.append(np.linalg.norm(kmeans_curr_center - optical_flow_float[i, j]))

        min_indices = np.argsort(min_values)
        final_locations = list()
        for loc in min_indices:
            final_locations.append(locations[loc])
        return final_locations


    """
    segmentation(im, optical_flow) -> a segmented image
    .   @brief Calculating the segmentation image
    .   @param im An image to make the segmentation
    .   @param optical_flow Key-points from HOG peak points
    """
    @staticmethod
    def segmentation(im, optical_flow):
        global orig_img, seg_img
        global seg0, seg1, seg2, seg3
        orig_img = im.copy()
        seg_img = im.copy()

        # auto separating to clusters without the ordinary way - user's picking
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        optical_flow_float = np.asarray(optical_flow, dtype=np.float32)

        ROW_SIZE = 4
        optical_flow_float_by_idx = np.zeros((optical_flow.shape[0], optical_flow.shape[1], ROW_SIZE), dtype=np.float32)
        for i in range(optical_flow_float_by_idx.shape[0]):
            for j in range(optical_flow_float_by_idx.shape[1]):
                optical_flow_float_by_idx[i, j] = np.concatenate([[i, j], optical_flow_float[i, j]])
        optical_flow_float = optical_flow_float_by_idx

        _, labels, centers = cv2.kmeans(optical_flow_float.reshape(-1, ROW_SIZE), INIT_CLUSTERS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.reshape(optical_flow.shape[:2])

        # lists to hold pixels in each segment
        seg0 = SegmentationWrapper.prepare_seg(0, centers[0], labels, optical_flow_float)
        seg1 = SegmentationWrapper.prepare_seg(1, centers[1], labels, optical_flow_float)
        seg2 = SegmentationWrapper.prepare_seg(2, centers[2], labels, optical_flow_float)
        seg3 = SegmentationWrapper.prepare_seg(3, centers[3], labels, optical_flow_float)

        ig = SegmentDetector(orig_img)
        final_mask = ig.calc_multivoting_grabcut()
        trans_img = cv2.addWeighted(orig_img.copy().astype(np.uint8), 0.5, final_mask.copy().astype(np.uint8), 0.5, 0)

        return trans_img


class VideoTracker:

    """
    get_frame_from_video(index=0) -> frame
    .   @brief Fetching a frame from some specific index
    .   @param index Frame index
    """
    def get_frame_from_video(self, index):
        cap, frame = self.get_video_capturer()
        if index == 0:
            return frame
        cap.set(1, index)
        ret, frame = cap.read()
        if not ret:
            print("Frame index is corrupted.")
            exit(-1)
        cap.release()
        return frame



    """
    get_video_capturer(key_points) -> cv2.VideoCapturer, first video frame
    .   @brief Accessing the video file and extracting the first frame
    """
    def get_video_capturer(self):
        video_name = inputDirectoryPath + inputVideoName
        cap = cv2.VideoCapture(video_name)
        if not cap.isOpened():
            print(video_name, "does not exist.")
            exit(-1)
        _, frame = cap.read()

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
        print("Found", dst.shape[0], "key points.")
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



    # mouse callback function
    def mouse_click(self,event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            Points.append((x, y))
        self.paint_point(Points, point_img)


    # given a segment points and a color, paint in seg_image
    def paint_point(self,points, im):
        for center in points:
            cv2.circle(im, center, Point_size, Point_color, -1)
        return im

    # let the user choose the points on the image
    def GetPointsFromUser(self):
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

        arr = []
        for (y,x) in Points:
            arr.__add__([[x ,y]])

        return Points , arr


    """
    video_processing(key_points) -> None
    .   @brief Executing the video file frame by frame and processing the optical flow algorithm by key points
    """
    def video_processing(self):
        global Points
        global point_img
        Points = []
        cap, prev_img = self.get_video_capturer()
        point_img = prev_img
        height, width, channels = prev_img.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(".//Results//Video" + inputVideoName[0:-4] + ".avi", fourcc, 20.0, (width, height))
        mask = np.zeros_like(prev_img)
        if selectPoints:
            # prev_pts , arr = self.GetPointsFromUser()
            prev_pts = self.fetch_key_points(prev_img)
        else:
            prev_pts = self.fetch_key_points(prev_img)
        iterate_num = 1
        count = 0
        while cap.isOpened():
            ret, next_img = cap.read()
            if ret:
                # computing the next points
                next_pts, status = self.calc_next_point(prev_img, next_img, prev_pts)
                new = next_pts[status == 1]
                old = prev_pts[status == 1]

                # draw the tracks
                img = next_img.copy()
                for i, (n, o) in enumerate(zip(new, old)):
                    a, b = n.ravel()
                    c, d = o.ravel()
                    # velocity computation
                    mask = cv2.line(mask, (a, b), (c, d), [0, 255, 0], 2)
                    img = cv2.circle(img, (a, b), 3, [0, 0, 255], -1)
                img = cv2.add(img, mask)

                # print('processed frame #' + str(iterate_num))
                cv2.imshow('Processed Frame Out', img)
                out.write(img)
                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break

                # updating the next iteration
                mask = np.zeros_like(mask)
                prev_img = next_img.copy()
                if count == refreshThresh:
                    print("Points refreshing..")
                    prev_pts = self.fetch_key_points(prev_img)
                    count = 0
                else:
                    prev_pts = new.copy()

                count += 1
                iterate_num += 1
            else:
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()



    """
    plot_peaks_of(image, key_points) -> None
    .   @brief Displaying the key point that were found to this specific image
    .   @param image
    .   @param key_points
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

    def flow_to_hsv(self, next_img, flow):
        hsv = np.zeros(next_img.shape)
        # set saturation
        hsv[:, :, 1] = cv2.cvtColor(next_img, cv2.COLOR_RGB2HSV)[:, :, 1]
        # convert from cartesian to polar
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # hue corresponds to direction
        hsv[..., 0] = ang * (180 / np.pi / 2)
        # value corresponds to magnitude
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # convert HSV to int32's
        hsv = np.asarray(hsv, dtype=np.float32)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        gray_flow = cv2.cvtColor(rgb_flow, cv2.COLOR_RGB2GRAY)
        return rgb_flow, gray_flow


    def segment_flow(self, im0, im1, show_out=True):
        if np.isscalar(im0) and np.isscalar(im1):
            im0 = tracker.get_frame_from_video(index=im0)
            im1 = tracker.get_frame_from_video(index=im1)


        # obtain dense optical flow parameters
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(im0, cv2.COLOR_RGB2GRAY),
                                            cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY),
                                            flow=None,
                                            pyr_scale=0.5, levels=1, winsize=15,
                                            iterations=2, poly_n=5, poly_sigma=1.1, flags=0)

        # kind of rough segmentation
        rgb, gray = self.flow_to_hsv(im1, flow)
        # cv2.imshow('hsv_out', rgb.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        trans_img = SegmentationWrapper.segmentation(im1, flow)
        if not show_out:
            cv2.imshow('trans_img', trans_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return trans_img


if __name__ == "__main__":
    tracker = VideoTracker()

    # debugging:
    # frame = tracker.get_frame_from_video(index=0)
    # tracker.plot_peaks_of(frame)

    # task 1:
    tracker.video_processing()

    # task 2:
    # first_index = 0
    # second_index = 5

    # option A:
    # prev_img = tracker.get_frame_from_video(index=first_index)
    # next_img = tracker.get_frame_from_video(index=second_index)
    # tracker.segment_flow(prev_img, next_img)

    # option B:
    # tracker.segment_flow(first_index, second_index, show_out=True)

    # task 2 by loop:
    # for i in range(0, 20, 5):
    #     trans_img = tracker.segment_flow(i, i + 1, show_out=False)
    #     cv2.imwrite("./Results/Images/" + str(i) + ".jpg", trans_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    trans_img = tracker.segment_flow(10, 12, show_out=False)
    cv2.imwrite("./Results/Images/" + str(10) + ".jpg", trans_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])