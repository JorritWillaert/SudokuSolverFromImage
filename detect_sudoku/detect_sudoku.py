import cv2
import numpy as np

def detect_sudoku(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurry = cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)

    contours, _ = cv2.findContours(blurry ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[1] #We don't want the outer contour.

    epsilon = 0.01*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    if len(approx) == 4:
        rect = cv2.minAreaRect(cnt)
        angle = rect[2]
        if angle < -45:
            angle = 90 + angle
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        #Now that the image is rotated, find contours again and use it to cut out sudoku
        blurry = cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)
        contours, _ = cv2.findContours(blurry ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cnt = contours[1]
        black = np.zeros(image.shape).astype(image.dtype)
        cv2.fillPoly(black, [cnt], (255, 255, 255))
        image = cv2.bitwise_and(image, black)

        x, y, w, h = cv2.boundingRect(cnt)
        image = image[y: y + h, x: x + w]
        image = cv2.resize(image, (min(image.shape), min(image.shape)))
        return image

if __name__ == '__main__':
    image = cv2.imread('../example_images/example1.jpg')
    cv2.imshow('Detected sudoku: ', detect_sudoku(image))
    k = cv2.waitKey(0)
