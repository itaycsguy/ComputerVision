import cv2, sys, dlib
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, image, mask):
        self.img = image
        self.mask = mask


    """
    seg - list of interactively selected pixels on the original image
    Return the pixel value which is the most frequent with the seg indices at the mask image
    """
    def get_most_freq_value(self, seg):
        u, indices = np.unique(self.mask[seg], return_inverse=True)
        return u[np.argmax(np.bincount(indices))]

    """
    Return a mapping from selected segments to mask color 
    """
    def seg2mask_color(self):
        voted_seg0 = self.get_most_freq_value(seg0)
        voted_seg1 = self.get_most_freq_value(seg1)
        voted_seg2 = self.get_most_freq_value(seg2)
        voted_seg3 = self.get_most_freq_value(seg3)
        return {SEGMENT_ZERO: voted_seg0, SEGMENT_ONE: voted_seg1, SEGMENT_TWO: voted_seg2, SEGMENT_THREE: voted_seg3}


    """
    seg2mask - mapping from segments to mask colors
    f_idx    - foreground index between 4 we have got and the rest are the background immediately 
    Return binary grabcut foreground
    """
    def calc_bin_grabcut(self, seg2mask_map, f_idx):
        new_mask = np.zeros(self.mask.shape[:2], dtype=np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (0, 0, self.mask.shape[0], self.mask.shape[1])
        iterations = 10
        mode = cv2.GC_INIT_WITH_RECT
        mask, bgd_model, fgd_model = cv2.grabCut(self.img, new_mask, rect, bgd_model, fgd_model, iterations, mode)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        middle_img = self.img * mask[:, :, np.newaxis]

        f_mask_color = seg2mask_map.get(f_idx)
        new_mask[new_mask == f_mask_color] = cv2.GC_BGD
        new_mask[new_mask != f_mask_color] = cv2.GC_BGD
        mode = cv2.GC_INIT_WITH_MASK
        mask, _, _ = cv2.grabCut(middle_img, new_mask, None, bgd_model, fgd_model, iterations, mode)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

        return mask


    """
    seg2mask - mapping from segments to mask colors
    Return multi-grabcut image total result
    """
    def calc_multi_grabcut(self, seg2mask_map):
        image_result = self.img
        f0_mask = self.calc_bin_grabcut(seg2mask_map, SEGMENT_ZERO)
        image_result[f0_mask == 1] = SEG_ZERO_COLOR
        f1_mask = self.calc_bin_grabcut(seg2mask_map, SEGMENT_ONE)
        image_result[f1_mask == 1] = SEG_ONE_COLOR
        f2_mask = self.calc_bin_grabcut(seg2mask_map, SEGMENT_TWO)
        image_result[f2_mask == 1] = SEG_TWO_COLOR
        f3_mask = self.calc_bin_grabcut(seg2mask_map, SEGMENT_THREE)
        image_result[f3_mask == 1] = SEG_THREE_COLOR

        plt.imshow(image_result), plt.colorbar(), plt.show()

        return image_result


"""
Interface for interactive selection of segement points

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
                    seg3.append((x,y))
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


    def main_loop(self):
        global orig_img, seg_img, current_segment
        global seg0, seg1, seg2, seg3
        input_image = "images\\boat.jpg"
        input_mask = "images\\boatMask.tif"
        orig_img = cv2.imread(input_image)
        seg_img = cv2.imread(input_image)
        orig_mask = cv2.imread(input_mask, 0)
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
        ig = ImGraph(orig_img, orig_mask)

        # making a map between marked segments and the given mask
        seg2mask_map = ig.seg2mask_color()

        # making one-against-all binary grabcut computations and return the union foreground images
        concat_image = ig.calc_multi_grabcut(seg2mask_map)

        # TODO
        # last step: show the results

        # destroy all windows
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Interactive().main_loop()


