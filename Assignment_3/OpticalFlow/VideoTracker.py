import cv2
import numpy as np

inputDirectoryPath = ".//Datasets//"
inputVideoName = "highway.avi"  # "highway.avi" # "bugs11.mp4"
selectPoints = False
numberOfPoints = 150

class VideoTracker:

    def __init__(self):
        pass


    """
    get_frame_from_video_at(index) -> frame
    .   @brief Fetching a frame from some specific index
    .   @param index Frame index
    """
    def get_frame_from_video_at(self, index):
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
    fetch_key_points(image, n_points[, blockSize=2[, ksize=3[, k=0.04]]]) -> array of key points as tuples
    .   @brief Extracting interesting points by corner harris algorithm
    .   @param image The original RGB image
    .   @param n_points Number of points that's wanted to be returned
    """
    def fetch_key_points(self, image, blockSize=10, ksize=3, k=0.2):
        gray_image = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        init_dst = cv2.cornerHarris(gray_image, blockSize, ksize, k)
        dst = np.zeros_like(init_dst)
        sums = list()
        for row in init_dst:
            sums.append(row.sum())

        max_idx = np.flip(np.argsort(sums))[:numberOfPoints]
        for idx in max_idx:
            dst[idx] = init_dst[idx].copy()

        return dst



    """
    calc_next_point(prev_img, next_img, prev_pts) -> computed next key points
    .   @brief Extracting interesting points by corner harris algorithm
    .   @param prev_img The previous original frame
    .   @param next_img The current original frame
    .   @param prev_pts Key points of the previous frame
    """
    def calc_next_point(self, prev_img, next_img, prev_pts):
        lk_params = dict(winSize=(20, 20),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY),
                                                         cv2.cvtColor(next_img, cv2.COLOR_RGB2GRAY),
                                                         prev_pts,
                                                         None,
                                                         **lk_params)
        return next_pts, status


    """
    video_processing(key_points) -> None
    .   @brief Executing the video file frame by frame and processing the optical flow algorithm
    """
    def video_processing(self):
        cap, prev_img = self.get_video_capturer()
        mask = np.zeros_like(prev_img)
        prev_pts = np.float32(self.fetch_key_points(prev_img, numberOfPoints)).reshape(-1, 1, 2)
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
                    mask = cv2.line(mask, (a, b), (c, d), [0, 255, 0], 2)
                    img = cv2.circle(img, (a, b), 5, [0, 0, 255], -1)
                img = cv2.add(img, mask)
                cv2.imshow('frame', img)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

                # updating the next iteration
                mask = np.zeros_like(mask)
                prev_pts = next_pts.copy()
                prev_img = next_img.copy()

        cap.release()
        cv2.destroyAllWindows()


    """
    plot_peaks_of(image, key_points) -> None
    .   @brief Displaying the key point that were found to this specific image
    .   @param image
    .   @param key_points
    """
    def plot_peaks_of(self, image, key_points):
        #result is dilated for marking the corners, not important
        key_points = cv2.dilate(key_points, None)
        # Threshold for an optimal value, it may vary depending on the image.
        image[key_points > 0.0] = [0, 0, 255]

        cv2.imshow('key_points', image)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()



if __name__ == "__main__":
    tracker = VideoTracker()
    frame = tracker.get_frame_from_video_at(0)
    dst = tracker.fetch_key_points(frame)
    print(frame.shape)
    print(dst.shape)
    # tracker.plot_peaks_of(frame, dst)
    tracker.video_processing()
