import cv2
import numpy as np
import imutils

def detect_sudoku(image):
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    image = imutils.resize(image, height = 500)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurry = cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)

    contours, _ = cv2.findContours(blurry, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    cnt = contours[1] #We don't want the outer contour.

    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if len(approx) == 4:
        image = four_point_transform(orig, approx.reshape(4, 2) * ratio)
        return image

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return image

def order_points(pts):
    """Returns a list of coordinates where the upper left corner is the first value,
    second value is top-right, third is bottom right and fourth is bottom left."""
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

if __name__ == '__main__':
    image = cv2.imread('../example_images/example1.jpg')
    cv2.imshow('Detected sudoku: ', detect_sudoku(image))
    k = cv2.waitKey(0)
