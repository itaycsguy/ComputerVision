## Students And Developers: Itay Guy, 305104184 & Elias Jadon, 207755737

import cv2, argparse, os
import numpy as np

# directory handling:
save_directory = "results"
my_examples_directory = "my_examples"

# CAN TOUCH THESE PARAMETERS WITH FULL OF ATTENTION:
# **************************************************
# Stay like this is meaning to open an image which stays beside the script (at the same directory)
open_directory = "images"

# An example to open an image from another directory:
# open_directory = "C://Users//itgu1//OneDrive//Desktop//ComputerVision//Assignment_1//Submission_305104184_207755737//"

# image handing:
Image = "man.jpg"
inputImage = open_directory + Image

# where to save handling:
segmentedImage = "Seg_" + Image
segmaskImage = "SegMask_" + Image


# DO NOT TOUCH HERE - PROGRAM'S PARAMETERS
SEGMENT_ZERO = 0
SEGMENT_ONE = 1
SEGMENT_TWO = 2
SEGMENT_THREE = 3

SEG_ZERO_COLOR = (0, 0, 255)
SEG_ONE_COLOR = (0, 255, 0)
SEG_TWO_COLOR = (255, 0, 0)
SEG_THREE_COLOR = (0, 255, 255)

INIT_REQUIRED_COUNTER = 2

"""
    Computation unit which calculates the grabcut to multi-classes objects
"""
class ImGraph:
    def __init__(self, image, seg_counter):
        self.img = image
        self.segments_counter = seg_counter

    """
        seg2mask - mapping from segments to mask colors
        f_idx    - foreground index between 4 we have got and the rest are the background immediately 
        Return binary grabcut foreground
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
        segX - each is the selected segment which is the critical area of grab-cut algorithm
        This method is calculating all the 2 combinations of binary grab-cut algorithm
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
        fX = binary grab-cut output mask, fX_complement = the complement mask
        These inputs are came from 'calc_grabcut_combinations' method
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
        This method is calculating the final mask result to the image segmentations all together by multivoting technique
    """
    def calc_multivoting_grabcut(self):
        print("\nProcessing", inputImage, "..")
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
        f0_final = f1_final = f2_final = f3_final = np.zeros(self.img.shape[:2], dtype=np.uint8)
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



"""
    Interface for interactive selection of segment points

    Interface instruction:
    Image opens to the user once the program starts to run.
    The user then selects segments with mouse:
    - Right click: segments a point
    - Left click: start a line
    - Line is from the previous point selected to the one clicked.
    - All points in the line belong to the current segment
    User uses `Space-Bar` to switch between segments, the message board shows on which segment the user is on at a given time.
    There are 4 segments in total, each has a color: RED, GREEN, BLUE, YELLOW respectively.
    Once you finish segmenting, press `ESC`.

    Once manual segmentation is finished:
    The user will have four lists: seg0, seg1, seg2, seg3. Each is a list with all the points belonging to the segment.
"""
class Interactive:
    """
        mouse callback function
    """
    def mouse_click(self, event, x, y, flags, params):
        # if left button is pressed, draw line
        if event == cv2.EVENT_LBUTTONDOWN:
            if current_segment == SEGMENT_ZERO:
                if len(seg0) == 0:
                    seg0.append((x, y))
                else:
                    points = self.add_line_point(seg0[-1], (x, y))
                    seg0.extend(points)
            if current_segment == SEGMENT_ONE:
                if len(seg1) == 0:
                    seg1.append((x, y))
                else:
                    points = self.add_line_point(seg1[-1], (x, y))
                    seg1.extend(points)
            if current_segment == SEGMENT_TWO:
                if len(seg2) == 0:
                    seg2.append((x, y))
                else:
                    points = self.add_line_point(seg2[-1], (x, y))
                    seg2.extend(points)
            if current_segment == SEGMENT_THREE:
                if len(seg3) == 0:
                    seg3.append((x, y))
                else:
                    points = self.add_line_point(seg3[-1], (x, y))
                    seg3.extend(points)

        # right mouse click adds single point
        if event == cv2.EVENT_RBUTTONDOWN:
            if current_segment == SEGMENT_ZERO:
                seg0.append((x, y))
            if current_segment == SEGMENT_ONE:
                seg1.append((x, y))
            if current_segment == SEGMENT_TWO:
                seg2.append((x, y))
            if current_segment == SEGMENT_THREE:
                seg3.append((x, y))

        # show on seg_img with colors
        self.paint_segment(seg0, SEG_ZERO_COLOR)
        self.paint_segment(seg1, SEG_ONE_COLOR)
        self.paint_segment(seg2, SEG_TWO_COLOR)
        self.paint_segment(seg3, SEG_THREE_COLOR)

    """
        given two points, this function returns all the points on line between.
        this is used when user selects lines on segments
    """
    def add_line_point(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        points = []
        is_steep = abs(y2 - y1) > abs(x2 - x1)
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        rev = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            rev = True
        delta_x = x2 - x1
        delta_y = abs(y2 - y1)
        error = int(delta_x / 2)
        y = y1
        y_step = None
        if y1 < y2:
            y_step = 1
        else:
            y_step = -1
        for x in range(x1, x2 + 1):
            if is_steep:
                points.append((y, x))
            else:
                points.append((x, y))
            error -= delta_y
            if error < 0:
                y += y_step
                error += delta_x
        # Reverse the list if the coordinates were reversed
        if rev:
            points.reverse()
        return points

    """
        given a segment points and a color, paint in seg_image
    """
    def paint_segment(self, segment, color):
        for center in segment:
            cv2.circle(seg_img, center, 2, color, -1)


    """
        given segmented image and segmented mask image (transparency)
        and save them in ".jpg" format to "results" directory
    """
    def save_results(self, seg_image, trans_image):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        cv2.imwrite(save_directory + '//' + segmentedImage, seg_image)
        cv2.imwrite(save_directory + '//' + segmaskImage, trans_image)
        print("Results have been saved to", save_directory)


    def save_seq_images(self, orig_img, seg_img, final_mask, trans_img):
        total_seq_image = np.concatenate((orig_img, np.concatenate((seg_img, np.concatenate((final_mask, trans_img), axis=1)), axis=1)), axis=1)
        if not os.path.exists(my_examples_directory):
            os.makedirs(my_examples_directory)
        cv2.imwrite(my_examples_directory + '//' + inputImage, total_seq_image)
        print("Results have been saved to", my_examples_directory)

    def main_loop(self):
        global orig_img, seg_img, current_segment
        global seg0, seg1, seg2, seg3
        orig_img = cv2.imread(inputImage)
        seg_img = cv2.imread(inputImage)
        cv2.namedWindow("Select segments")

        # mouse event listener
        cv2.setMouseCallback("Select segments", self.mouse_click)
        # lists to hold pixels in each segment
        seg0 = []
        seg1 = []
        seg2 = []
        seg3 = []
        # segment you're on
        current_segment = 0

        # 4-colouring is MUST
        while True:
            cv2.imshow("Select segments", seg_img)
            k = cv2.waitKey(20)

            # space bar to switch between segments
            if k == 32:
                current_segment = (current_segment + 1) % 4
                print('current segment =', str(current_segment))
            # ESC to finish
            if k == 27:
                break


        if not (seg0 and seg1):
            print("At least 2 segments are required.")
            exit(-1)


        """
            graph cut implementation for 4 segments
            add functions and code as you wish
        """

        seg_counter = INIT_REQUIRED_COUNTER
        if seg2:
            seg_counter += 1
        if seg3:
            seg_counter += 1

        # keeping important data as in object creation
        ig = ImGraph(orig_img, seg_counter)

        final_mask = ig.calc_multivoting_grabcut()

        print("Displaying results..")
        cv2.imshow('Original_image', orig_img.astype(np.uint8))

        # show segmentation image
        cv2.imshow('Segmented_image', final_mask.astype(np.uint8))

        # show transparency image
        trans_img = cv2.addWeighted(orig_img.copy().astype(np.uint8), 0.5, final_mask.copy().astype(np.uint8), 0.5, 0)
        cv2.imshow('Transparency_image', trans_img.astype(np.uint8))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # save results as asked to
        self.save_results(final_mask, trans_img)

        # self.save_seq_images(orig_img, seg_img, final_mask, trans_img)

        # destroy all windows
        cv2.destroyAllWindows()


"""
    Run the program from here with full attention to the description
    Do not forget to put the right image name with exclusive output names
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assign names to: inputImage, segmentedImage, segmaskImage at the TOP of the script.')
    args = parser.parse_args()
    Interactive().main_loop()