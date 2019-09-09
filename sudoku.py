import numpy as np
import cv2
from tensorflow.keras.models import load_model

from detect_sudoku import detect_sudoku
from digits_classification import digit_classification
from sudoku_solver.sudoku_solver import SudokuSolver

def sudoku(image):
    image = detect_sudoku.detect_sudoku(image)
    y_digit_shape = (image.shape[0] / 9)
    x_digit_shape = (image.shape[1] / 9)
    model = load_model('digits_classification/testing_model_digits.h5')
    sudoku = np.zeros((9, 9))
    for y in range(9):
        for x in range(9):
            digit_image = image[int(y * y_digit_shape): int((y + 1) * y_digit_shape), int(x * x_digit_shape): int((x + 1) * x_digit_shape)]
            detected_digit = digit_classification.digit_classification(digit_image, model)
            sudoku[y, x] = detected_digit
    solver = SudokuSolver(sudoku)
    solver.solve_sudoku()
    solved_sudoku = solver.sudoku_array
    return solved_sudoku

if __name__ == '__main__':
    image = cv2.imread('example_images/example1.jpg')
    print(sudoku(image))
