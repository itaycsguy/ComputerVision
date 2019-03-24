import cv2, argparse, os, sys, dlib
import numpy as np
import matplotlib.pyplot as plt


save_directory = "results"
open_directory = "images"
Image = "guard.jpg"
inputImage = open_directory + "//" + Image
segmentedImage = "Seg" + Image
segmaskImage = "SegMask" + Image


SEGMENT_ZERO = 0
SEGMENT_ONE = 1
SEGMENT_TWO = 2
SEGMENT_THREE = 3

SEG_ZERO_COLOR = (0, 0, 255)
SEG_ONE_COLOR = (0, 255, 0)
SEG_TWO_COLOR = (255, 0, 0)
SEG_THREE_COLOR = (0, 255, 255)

"""
    Computation unit which calculates the grabcut to multi-classes objects
"""
class ImGraph:
    def __init__(self, image):
        self.img = image

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


    def elias(self):
        print("cal...")

        f0_123 = self.calc_bin_grabcut(seg0, seg1 + seg2 + seg3)
        print(">> 1")
        f0_12 = self.calc_bin_grabcut(seg0, seg1 + seg2)
        print(">> 2")
        f0_13 = self.calc_bin_grabcut(seg0, seg1 + seg3)
        print(">> 3")
        f0_23 = self.calc_bin_grabcut(seg0, seg2 + seg3)
        print(">> 4")
        f0_1 = self.calc_bin_grabcut(seg0, seg1)
        print("5")
        f0_2 = self.calc_bin_grabcut(seg0, seg2)
        print("6")
        f0_3 = self.calc_bin_grabcut(seg0, seg3)
        print("7")
        f0_not = np.ones(self.img.shape[:2], dtype=np.uint8) - self.calc_bin_grabcut(seg1 + seg2 + seg3, seg0)

        # Voting per segment
        f0 = (2 * f0_123) + f0_12 + f0_13 + f0_23 + f0_1 + f0_2 + f0_3 + f0_not

        '''
        cv2.imshow('f0_123', (f0_123[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f0_not', (f0_not[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f0_12', (f0_12[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f0_13', (f0_13[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f0_23', (f0_23[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f0_1', (f0_1[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f0_2', (f0_2[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f0_3', (f0_3[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        ###################################################################
        ###################################################################

        f1_023 = self.calc_bin_grabcut(seg1, seg0 + seg2 + seg3)
        print("1")
        f1_02 = self.calc_bin_grabcut(seg1, seg0 + seg2)
        print("2")
        f1_03 = self.calc_bin_grabcut(seg1, seg0 + seg3)
        print("3")
        f1_23 = self.calc_bin_grabcut(seg1, seg2 + seg3)
        print("4")
        f1_0 = self.calc_bin_grabcut(seg1, seg0)
        print("5")
        f1_2 = self.calc_bin_grabcut(seg1, seg2)
        print("6")
        f1_3 = self.calc_bin_grabcut(seg1, seg3)
        print("7")
        f1_not = np.ones(self.img.shape[:2], dtype=np.uint8) - self.calc_bin_grabcut(seg0 + seg2 + seg3, seg1)

        # Voting per segment
        f1 = (2 * f1_023) + f1_02 + f1_03 + f1_23 + f1_0 + f1_2 + f1_3 + f1_not

        '''
        cv2.imshow('f1_023', (f1_023[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f1_not', (f1_not[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f1_02', (f1_02[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f1_03', (f1_03[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f1_23', (f1_23[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f1_0', (f1_0[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f1_2', (f1_2[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f1_3', (f1_3[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        ###################################################################
        ###################################################################


        f2_013 = self.calc_bin_grabcut(seg2, seg1 + seg0 + seg3)
        print("1")
        f2_01 = self.calc_bin_grabcut(seg2, seg1 + seg0)
        print("2")
        f2_03 = self.calc_bin_grabcut(seg2, seg3 + seg0)
        print("3")
        f2_13 = self.calc_bin_grabcut(seg2, seg1 + seg3)
        print("4")
        f2_0 = self.calc_bin_grabcut(seg2, seg0)
        print("5")
        f2_1 = self.calc_bin_grabcut(seg2, seg1)
        print("6")
        f2_3 = self.calc_bin_grabcut(seg2, seg3)
        print("7")
        f2_not = np.ones(self.img.shape[:2], dtype=np.uint8) - self.calc_bin_grabcut(seg1 + seg0 + seg3, seg2)

        # Voting per segment
        f2 = (2 * f2_013) + f2_01 + f2_03 + f2_13 + f2_0 + f2_1 + f2_3 + f2_not

        '''
        cv2.imshow('f2_013', (f2_013[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f2_not', (f2_not[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f2_01', (f2_01[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f2_03', (f2_03[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f2_13', (f2_13[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f2_0', (f2_0[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f2_1', (f2_1[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f2_3', (f2_3[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        ###################################################################
        ###################################################################


        f3_012 = self.calc_bin_grabcut(seg3, seg1 + seg2 + seg0)
        print("1")
        f3_01 = self.calc_bin_grabcut(seg3, seg1 + seg0)
        print("2")
        f3_02 = self.calc_bin_grabcut(seg3, seg0 + seg2)
        print("3")
        f3_12 = self.calc_bin_grabcut(seg3, seg2 + seg1)
        print("4")
        f3_0 = self.calc_bin_grabcut(seg3, seg0)
        print("5")
        f3_1 = self.calc_bin_grabcut(seg3, seg1)
        print("6")
        f3_2 = self.calc_bin_grabcut(seg3, seg2)
        print("7")
        f3_not = np.ones(self.img.shape[:2], dtype=np.uint8) - self.calc_bin_grabcut(seg1 + seg2 + seg0, seg3)

        # Voting per segment
        f3 = (2 * f3_012) + f3_01 + f3_02 + f3_12 + f3_0 + f3_1 + f3_2 + f3_not

        '''
        cv2.imshow('f3_012', (f3_012[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f3_not', (f3_not[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f3_01', (f3_01[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f3_02', (f3_02[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f3_12', (f3_12[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f3_0', (f3_0[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f3_1', (f3_1[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.imshow('f3_2', (f3_2[:, :, np.newaxis] * (255, 255, 255)).astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        # multi-voting to all segments together

        # get the majority voting for each pixel
        f0_final = ((f0 > f1) & (f0 > f2) & (f0 > f3))
        f1_final = ((f1 > f0) & (f1 > f2) & (f1 > f3))
        f2_final = ((f2 > f0) & (f2 > f1) & (f2 > f3))
        f3_final = ((f3 > f0) & (f3 > f1) & (f3 > f2))

        ##################################################### checkpoint
        f0_mask = f0_final[:, :, np.newaxis] * SEG_ZERO_COLOR
        f1_mask = f1_final[:, :, np.newaxis] * SEG_ONE_COLOR
        f2_mask = f2_final[:, :, np.newaxis] * SEG_TWO_COLOR
        f3_mask = f3_final[:, :, np.newaxis] * SEG_THREE_COLOR
        cv2.imshow('before', (f0_mask + f1_mask + f2_mask + f3_mask).astype(np.uint8))
        #####################################################


        # pixels that have not majority
        blackpixels = np.ones(self.img.shape[:2], dtype=np.uint8) - (f0_final + f1_final + f2_final + f3_final)
        f0_help = ((f0 >= f1) & (f0 >= f2) & (f0 >= f3))
        f1_help = ((f1 >= f0) & (f1 >= f2) & (f1 >= f3))
        f2_help = ((f2 >= f0) & (f2 >= f1) & (f2 >= f3))
        f3_help = ((f3 >= f0) & (f3 >= f1) & (f3 >= f2))
        f01_conflict = f0_help == f1_help
        f02_conflict = f0_help == f2_help
        f03_conflict = f0_help == f3_help
        f12_conflict = f1_help == f2_help
        f13_conflict = f1_help == f3_help
        f23_conflict = f2_help == f3_help

        plus0 = (blackpixels & f01_conflict & f0_not) | (blackpixels & f02_conflict & f0_not) | (blackpixels & f03_conflict & f0_not)
        blackpixels = blackpixels - plus0
        plus1 = (blackpixels & f01_conflict & f1_not) | (blackpixels & f12_conflict & f1_not) | (blackpixels & f13_conflict & f1_not)
        blackpixels = blackpixels - plus1
        plus2 = (blackpixels & f02_conflict & f2_not) | (blackpixels & f12_conflict & f2_not) | (blackpixels & f23_conflict & f2_not)
        blackpixels = blackpixels - plus2
        plus3 = blackpixels

        f0_final = f0_final + plus0
        f1_final = f1_final + plus1
        f2_final = f2_final + plus2
        f3_final = f3_final + plus3


        # f3_mask = np.ones(self.img.shape[:2], dtype=np.uint8) - (f0_mask + f1_mask + f2_mask)
        # f3_mask = self.calc_bin_grabcut(seg3, seg0 + seg1 + seg2)

        '''
        f0_mask = self.find_segment(f0_mask, seg0)
        f1_mask = self.find_segment(f1_mask, seg1)
        f2_mask = self.find_segment(f2_mask, seg2)

        f0_mask = f0_mask - np.logical_and(f0_mask, f1_mask)
        f0_mask = f0_mask - np.logical_and(f0_mask, f2_mask)
        f1_mask = f1_mask - np.logical_and(f1_mask, f2_mask)

        f3_mask = np.ones(self.img.shape[:2], dtype=np.uint8) - (f0_mask + f1_mask + f2_mask)
        f3_mask = f3_mask - np.logical_and(f3_mask, f0_mask)
        f3_mask = f3_mask - np.logical_and(f3_mask, f1_mask)
        f3_mask = f3_mask - np.logical_and(f3_mask, f2_mask)
        '''
        f0_mask = f0_final[:, :, np.newaxis] * SEG_ZERO_COLOR
        f1_mask = f1_final[:, :, np.newaxis] * SEG_ONE_COLOR
        f2_mask = f2_final[:, :, np.newaxis] * SEG_TWO_COLOR
        f3_mask = f3_final[:, :, np.newaxis] * SEG_THREE_COLOR

        return f0_mask + f1_mask + f2_mask + f3_mask



    def multivoting_by_grabcut(self):
        print("\nProcessing the image..")
        masks0 = self.calc_multi_grabcut(seg0, np.array(SEG_ZERO_COLOR), seg1, np.array(SEG_ONE_COLOR), seg2, np.array(SEG_TWO_COLOR), seg3, np.array(SEG_THREE_COLOR))
        masks1 = self.calc_multi_grabcut(seg1, np.array(SEG_ONE_COLOR), seg2, np.array(SEG_TWO_COLOR), seg3, np.array(SEG_THREE_COLOR), seg0, np.array(SEG_ZERO_COLOR))
        masks2 = self.calc_multi_grabcut(seg2, np.array(SEG_TWO_COLOR), seg3, np.array(SEG_THREE_COLOR), seg0, np.array(SEG_ZERO_COLOR), seg1, np.array(SEG_ONE_COLOR))
        masks3 = self.calc_multi_grabcut(seg3, np.array(SEG_THREE_COLOR), seg0, np.array(SEG_ZERO_COLOR), seg1, np.array(SEG_ONE_COLOR), seg2, np.array(SEG_TWO_COLOR))

        cv2.imshow('mask0', masks0.astype(np.uint8))
        cv2.imshow('mask1', masks1.astype(np.uint8))
        cv2.imshow('mask2', masks2.astype(np.uint8))
        cv2.imshow('mask3', masks3.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Done!\n")
        return self.multivoting_decision_by_color([masks0, masks1, masks2, masks3], seg0, SEG_ZERO_COLOR, seg1, SEG_ONE_COLOR, seg2, SEG_TWO_COLOR, seg3, SEG_THREE_COLOR)



    def multivoting_decision_by_color(self, masks, seg0, seg0_color, seg1, seg1_color, seg2, seg2_color, seg3, seg3_color):
        for mask in masks:
            counter = 0
            exist = False
            for (x, y) in seg0:
                exist = True if all(np.equal(mask[y][x], seg0_color)) else False
            if exist:
                counter += 1
            for (x, y) in seg1:
                exist = True if all(np.equal(mask[y][x], seg1_color)) else False
            if exist:
                counter += 1
            for (x, y) in seg2:
                exist = True if all(np.equal(mask[y][x], seg2_color)) else False
            if exist:
                counter += 1
            for (x, y) in seg3:
                exist = True if all(np.equal(mask[y][x], seg3_color)) else False
            if exist:
                counter += 1

            if counter == 4:
                return mask

        return masks[0]



    """
        seg2mask - mapping from segments to mask colors
        Return multi-grabcut image total result
    """
    def calc_multi_grabcut(self, seg0, seg0_color, seg1, seg1_color, seg2, seg2_color, seg3, seg3_color):
        f0_mask = self.calc_bin_grabcut(seg0, seg1 + seg2 + seg3)
        f1_mask = self.calc_bin_grabcut(seg1, seg0 + seg2 + seg3)
        f2_mask = self.calc_bin_grabcut(seg2, seg0 + seg1 + seg3)

        # f3_mask = np.ones(self.img.shape[:2], dtype=np.uint8) - (f0_mask + f1_mask + f2_mask)
        # f3_mask = self.calc_bin_grabcut(seg3, seg0 + seg1 + seg2)

        f0_mask = self.find_segment(f0_mask, seg0)
        f1_mask = self.find_segment(f1_mask, seg1)
        f2_mask = self.find_segment(f2_mask, seg2)

        f0_mask = f0_mask - np.logical_and(f0_mask, f1_mask)
        f0_mask = f0_mask - np.logical_and(f0_mask, f2_mask)
        f1_mask = f1_mask - np.logical_and(f1_mask, f2_mask)

        f3_mask = np.ones(self.img.shape[:2], dtype=np.uint8) - (f0_mask + f1_mask + f2_mask)
        f3_mask = f3_mask - np.logical_and(f3_mask, f0_mask)
        f3_mask = f3_mask - np.logical_and(f3_mask, f1_mask)
        f3_mask = f3_mask - np.logical_and(f3_mask, f2_mask)

        f0_mask = f0_mask[:, :, np.newaxis] * seg0_color
        f1_mask = f1_mask[:, :, np.newaxis] * seg1_color
        f2_mask = f2_mask[:, :, np.newaxis] * seg2_color
        f3_mask = f3_mask[:, :, np.newaxis] * seg3_color

        return f0_mask + f1_mask + f2_mask + f3_mask


    """
        By multi-voting between areas that have found we can detect the correct one to draw as the target segment
    """
    def multivoting_area_desicion(self, real_mask, total_areas, segment, threshold):
        for area in total_areas:
            for (x, y) in segment:
                if area[y][x] == threshold:
                    return area
        return real_mask


    """
        A method to determine which color matches to which pixel where overlapping is up
    """
    def find_segment(self, mask, segment):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        total_areas = list()
        for cnt in contours:
            new_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            cv2.fillPoly(new_mask, pts=[cnt], color=(255, 255, 255))
            total_areas.append(new_mask)

        mask = self.multivoting_area_desicion(mask, total_areas, segment, 255)

        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return mask / 255



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
        print("Images are save to", save_directory)


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

        """
            graph cut implementation for 4 segments
            add functions and code as you wish
        """
        # keeping important data as in object creation
        ig = ImGraph(orig_img)
        #final_mask = ig.multivoting_by_grabcut()
        final_mask = ig.elias()

        # show segmentation image
        cv2.imshow('seg_image', final_mask.astype(np.uint8))

        # show transparency image
        # The same equation: alpha * image + (1.0 - alpha) * output
        trans_img = cv2.addWeighted(orig_img.copy().astype(np.uint8), 0.5, final_mask.copy().astype(np.uint8), 0.5, 0)
        cv2.imshow('trans_image', trans_img.astype(np.uint8))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # save results as asked to
        self.save_results(seg_img, trans_img)

        # destroy all windows
        cv2.destroyAllWindows()


"""
    Left work:
    ==========
    1. Save 2 results: seg_img + trans_img
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assign name to: inputImage, segmentedImage, segmaskImage at the TOP of the script.')
    args = parser.parse_args()
    Interactive().main_loop()