import cv2
import numpy as np

def digit_classification(digit_image, model):
    image = cv2.cvtColor(digit_image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)
    # Some testing with erosion and dilation
    """
    kernel = np.ones((2,2), np.uint8)

    img_erosion = cv2.erode(thresh, kernel, iterations=1)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    cv2.imshow('Input', image)
    cv2.imshow('Erosion', img_erosion)
    cv2.imshow('Dilation', img_dilation)

    cv2.imshow('thresh', thresh)
    k = cv2.waitKey(0)
    """

    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    if len(contours) < 2:
        return 0
    cnt = contours[1] #We don't want the outer contour.
    x, y, w, h = cv2.boundingRect(cnt)
    if w < 5 or h < 5:
        return 0

    thresh = thresh[y: y + h, x: x + w]

    #Resize it to (20, 20) so we can add black borders
    thresh = cv2.resize(thresh, (20, 20))

    thresh = 255 - thresh
    thresh = cv2.copyMakeBorder(thresh, 4 , 4, 4, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0)) #Add borders

    thresh = thresh.astype('float32')
    thresh /= 255.0

    #cv2.imshow('Input image', thresh)
    #k = cv2.waitKey(0)

    image = np.reshape(thresh, (1, 28, 28, 1)) #Reshape the image to 4-dims, needed for Keras.

    number_prob = model.predict(image)
    if max(number_prob[0]) > 0.5:
        return np.argmax(max(number_prob)) # Means the model found a number
    else:
        return 0 # Means there was no number in digit_image
