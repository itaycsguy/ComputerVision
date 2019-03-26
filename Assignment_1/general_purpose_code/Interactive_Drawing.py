import cv2 as cv


"""
    Marking a point each iteration of the same color
"""
def draw_points():
    def mouse_drawing(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            print("Left click")
            circles.append((x, y))
    cap = cv.VideoCapture(0)

    cv.namedWindow("Frame")
    cv.setMouseCallback("Frame", mouse_drawing)

    circles = []
    while True:
        _, frame = cap.read()

        for center_position in circles:
            cv.circle(frame, center_position, 5, (0, 0, 255), -1)

        cv.imshow("Frame", frame)

        key = cv.waitKey(1)
        if key == 27:
            break
        elif key == ord("d"):
            circles = []

    cap.release()
    cv.destroyAllWindows()


"""
Building a rectangle at a time - single each
"""
drawing = False
point1 = ()
point2 = ()
def draw_rectangle():
    def mouse_drawing(event, x, y, flags, params):
        global point1, point2, drawing
        if event == cv.EVENT_LBUTTONDOWN:
            if drawing is False:
                drawing = True
                point1 = (x, y)
            else:
                drawing = False

        elif event == cv.EVENT_MOUSEMOVE:
            if drawing is True:
                point2 = (x, y)
    cap = cv.VideoCapture(0)

    cv.namedWindow("Frame")
    cv.setMouseCallback("Frame", mouse_drawing)

    while True:
        _, frame = cap.read()

        if point1 and point2:
            cv.rectangle(frame, point1, point2, (0, 255, 0))

        cv.imshow("Frame", frame)

        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()


"""
    TODO: need to generalize those functions up-stairs for next assignment execution interactively
"""
if __name__ == "__main__":
    draw_rectangle()
    draw_points()