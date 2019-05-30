# Itay Guy, 305104184 & Elias Jadon, 207755737
import cv2, os, argparse
import numpy as np

# out data:
# directory where all input data should being resided - should be provided by the user
inputDirectoryPath = ".//Datasets//"
# directory where results should being saved - it is created if it doesn't exist
outputDirectoryPath = ".//Results//"

# task data:
inputVideoName = "Soccer1.mp4"
selectPoints = False
numberOfPoints = 500

# out data
Point_color = (0, 0, 255)
Point_size = 7
Line_color = (0, 255, 0)
Line_size = 2

class VideoTracker:
    SF_INITIAL = 0
    SF_GAUSSIAN = 1
    DIV_REFRESH = 4
    DEFAULT_REFRESH_THRESH = 100
    JPEG_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

    def __init__(self):
        self.total_frames = -1
        self.is_ar_least_half_kp = False
        self.refresh_thresh = VideoTracker.DEFAULT_REFRESH_THRESH
        self.last_frames_number = list()
        self.lfn_pointer = 0


    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    def print_progress_bar(self, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()


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
    video_processing(save_out=False) -> None
    .   @brief Executing the video file frame by frame and processing the optical flow algorithm by key points
    .   @param save_out
    """
    def video_processing(self, save_out=False):
        print("Running video processing..")
        global Points
        global point_img
        Points = list()
        cap, prev_img = self.get_video_capturer()
        point_img = prev_img
        velocity_image = prev_img.copy()
        video_instance = None
        if save_out:
            if not os.path.exists(outputDirectoryPath):
                os.mkdir(outputDirectoryPath)
            w, h, _ = prev_img.shape
            video_instance = cv2.VideoWriter(outputDirectoryPath + "OF_" + inputVideoName[0:-4] + "_" + str(numberOfPoints) + "_out.avi",
                                             cv2.VideoWriter_fourcc(*'XVID'), 20.0, (h, w))
        mask = np.zeros_like(prev_img)
        if selectPoints:
            prev_pts = self.get_points_from_user()
        else:
            prev_pts = self.fetch_key_points(prev_img)

        count = 0
        while cap.isOpened():
            ret, next_img = cap.read()
            if not ret:
                break
            else:
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
                velocity_image = cv2.add(velocity_image, mask)

                cv2.imshow('Processed Frame Out', img)
                if save_out:
                    video_instance.write(img)
                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break

                # updating the next iteration
                mask = np.zeros_like(mask)
                prev_img = next_img.copy()
                if count == self.refresh_thresh:
                    print("Points refreshing..")
                    prev_pts = self.fetch_key_points(prev_img)
                    count = 0
                else:
                    prev_pts = new.copy()

                count += 1

        cv2.destroyAllWindows()
        if save_out:
            out_name = outputDirectoryPath + "Velocity_" + inputVideoName[0:-4] + "_" + str(numberOfPoints) + "_out.jpg"
            cv2.imwrite(out_name, velocity_image, VideoTracker.JPEG_PARAM)
            print(out_name, "is saved.")
            video_instance.release()
        cap.release()
        print("Done!")



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


if __name__ == "__main__":
    tracker = VideoTracker()
    tracker.video_processing()

    # Next Section is made for the submission time - DO NOT REMOVE!
    """
    parser = argparse.ArgumentParser(description='Computer Vision - Assignment 4')
    parser.add_argument("--optical_flow", help="Executing an optical flow.", action="store_true")
    args = parser.parse_args()
    tracker = VideoTracker()

    flag = False
    if args.optical_flow:
        # debugging + example:
        # frame = tracker.get_frame_from_video(index=frameNumber1)
        # out_name = inputDirectoryPath + inputVideoName[0:-4] + "_" + str(frameNumber1) + "_out.jpg"
        # cv2.imwrite(out_name, frame, VideoTracker.JPEG_PARAM)
        # tracker.plot_peaks_of(frame)

        # task 1:
        # debugging:
        # frame = tracker.get_frame_from_video(index=frameNumber2)
        # tracker.plot_peaks_of(frame)
        # practice example with all parameters:
        # tracker.video_processing(save_out=False)

        flag = True
        tracker.video_processing()

    if not flag:
        print("Should pick any functionality to execute. for help use -h.")
    """