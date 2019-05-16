import cv2
import numpy as np

# Key points
inputDirectoryPath = ".//Datasets//"
inputVideoName = "bugs11.mp4"  # "highway.avi" # "bugs11.mp4"
selectPoints = False
numberOfPoints = 150
refreshThresh = 100

# Segmentation
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
    prepare_seg(seg[, curr_label[, key_points_float[, labels]]]) -> seg array with numpy pair to tuple pairs into
    .   @brief Converting point from numpy array to tuple
    .   @param seg A segment of points from the image.
    .   @param curr_label The current cluster where the segment is belonging to.
    .   @param key_points_float All key-points array.
    .   @param labels All labels.
    """
    @staticmethod
    def prepare_seg(seg, curr_label, key_points_float, labels):
        seg_t = np.asarray(key_points_float[labels.ravel() == curr_label], dtype=np.uint32)
        for i, _ in enumerate(seg_t):
            seg.append(tuple(seg_t[i]))

        return seg


    """
    segmentation(inputImage, key_points) -> a segmented image
    .   @brief Calculating the segmentation image
    .   @param inputImage An image to make the segmentation
    .   @param key_points Key-points from HOG peak points
    """
    @staticmethod
    def segmentation(im, flow):
        global orig_img, seg_img
        global seg0, seg1, seg2, seg3
        orig_img = im.copy()
        seg_img = im.copy()

        # auto separating to clusters without the ordinary way - user's picking
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        key_points_float = np.asarray(flow, dtype=np.float32)
        _, labels, _ = cv2.kmeans(key_points_float, INIT_CLUSTERS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # lists to hold pixels in each segment
        seg0 = SegmentationWrapper.prepare_seg(list(), 0, key_points_float, labels)
        seg1 = SegmentationWrapper.prepare_seg(list(), 1, key_points_float, labels)
        seg2 = SegmentationWrapper.prepare_seg(list(), 2, key_points_float, labels)
        seg3 = SegmentationWrapper.prepare_seg(list(), 3, key_points_float, labels)

        ig = SegmentDetector(orig_img)
        seg_img = ig.calc_multivoting_grabcut().astype(np.uint8)

        return seg_img


class VideoTracker:

    def __init__(self):
        pass


    """
    get_frame_from_video(index=0) -> frame
    .   @brief Fetching a frame from some specific index
    .   @param index Frame index
    """
    def get_frame_from_video(self, index=0):
        cap, _ = self.get_video_capturer()
        ret, frame = cap.retrieve(flag=index)
        if not ret:
            print("Frame index is missing.")
            exit(-1)

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



    """
    video_processing(key_points) -> None
    .   @brief Executing the video file frame by frame and processing the optical flow algorithm by key points
    """
    def video_processing(self):
        cap, prev_img = self.get_video_capturer()
        mask = np.zeros_like(prev_img)
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
                k = cv2.waitKey(30) & 0xff
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

        cap.release()
        cv2.destroyAllWindows()



    """
    plot_peaks_of(image, key_points) -> None
    .   @brief Displaying the key point that were found to this specific image
    .   @param image
    .   @param key_points
    """
    def plot_peaks_of(self, image, key_points):
        img = image.copy()
        for i, point in enumerate(key_points):
            a, b = point.ravel()
            img = cv2.circle(img, (a, b), 4, [0, 0, 255], -1)

        cv2.imshow('Key points displaying', img)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()



if __name__ == "__main__":
    tracker = VideoTracker()
    # frame = tracker.get_frame_from_video()
    # dst = tracker.fetch_key_points(frame)
    # tracker.plot_peaks_of(frame, dst)
    # tracker.video_processing()

    cap, prev_img = tracker.get_video_capturer()
    mask = np.zeros_like(prev_img)
    prev_pts = tracker.fetch_key_points(prev_img)
    while cap.isOpened():
        ret, next_img = cap.read()
        if ret:
            flow = cv2.calcOpticalFlowFarneback(prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            seg_img = SegmentationWrapper.segmentation(prev_img, flow)
            cv2.imshow('seg_img', seg_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break