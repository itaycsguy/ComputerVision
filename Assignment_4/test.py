import numpy as np
import cv2

im1 = cv2.imread("C:/Users/itgu1/OneDrive/Desktop/m1.jpg")
im2 = cv2.imread("C:/Users/itgu1/OneDrive/Desktop/m2.jpg")
im1 = cv2.resize(im1, (256, 256), interpolation=cv2.INTER_CUBIC)
im2 = cv2.resize(im2, (256, 256), interpolation=cv2.INTER_CUBIC)
numberOfPoints = 4

def grab_contours(cnts):
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

def get_stitched(images):
    stitcher = cv2.Stitcher.create(cv2.STITCHER_SCANS)
    (status, stitched) = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)
        largest_cnt = max(cnts, key=cv2.contourArea)

        # allocate memory for the mask which will contain the rectangular bounding box of the stitched image region
        mask = np.zeros(thresh.shape, dtype=np.uint8)
        (x, y, w, h) = cv2.boundingRect(largest_cnt)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        min_rect = mask.copy()
        sub = mask.copy()

        while cv2.countNonZero(sub) > 0:
            min_rect = cv2.erode(min_rect, None)
            sub = cv2.subtract(min_rect, thresh)

        # find contours in the minimum rectangular mask and then extract the bounding box (x, y)-coordinates
        cnts = cv2.findContours(min_rect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)
        min_cnt = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(min_cnt)

        return stitched[y:y + h, x:x + w]
    else:
        print("[INFO] image stitching failed ({})".format(status))
        return None


def calc_next_point(prev_img, next_img, prev_pts):
    w, h, _ = prev_img.shape    # winSize=(22, 22)
    lk_params = dict(winSize=(22, 22),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY),
                                                     cv2.cvtColor(next_img, cv2.COLOR_RGB2GRAY),
                                                     prev_pts,
                                                     None,
                                                     **lk_params)
    return next_pts, status.ravel()



def fetch_key_points(image, quality_level=0.0001, min_distance=7, block_size=7):
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


kp1 = fetch_key_points(im1)
next_pts, status = calc_next_point(im1, im2, kp1)
good = next_pts[status == 1]

H, mask = cv2.findHomography(kp1, good, cv2.RANSAC, 5.0)
img = cv2.warpPerspective(im2, H, (im2.shape[1], im2.shape[0]))

stitched = get_stitched([im2, im1])
cv2.imshow("stitched", stitched)
cv2.waitKey(0)
cv2.destroyAllWindows()