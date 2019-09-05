import numpy as np

class SudokuSolver():
    """Simple class that solves sudoku with backtracking algorithm."""
    def __init__(self, sudoku_array):
        self.sudoku_array = sudoku_array

    def solve_sudoku(self):
        l = [0,0]
        if not self.get_unsigned_location(l):
            return True
        row = l[0]
        col = l[1]

        for number in range(1,10):
            if self.is_safe(row, col, number):
                self.sudoku_array[row, col] = number
                if self.solve_sudoku():
                    return True
                else:
                    self.sudoku_array[row, col] = 0
        return False

    def used_in_row(self, row, number):
        for col in range(9):
            if self.sudoku_array[row, col] == number:
                return True
        return False

    def used_in_col(self, col, number):
        for row in range(9):
            if self.sudoku_array[row, col] == number:
                return True
        return False

    def used_in_box(self, box_start_row, box_start_col, number):
        for row in range(3):
            for col in range(3):
                if self.sudoku_array[box_start_row + row, box_start_col + col] == number:
                    return True
        return False

    def is_safe(self, row, col, number):
        if (not self.used_in_col(col, number)) and (not self.used_in_row(row, number) and (not self.used_in_box(row - row % 3, col - col % 3, number))):
            return True
        else:
            return False

    def get_unsigned_location(self, l):
        for row in range(9):
            for col in range(9):
                if self.sudoku_array[row, col] == 0:
                    l[0] = row
                    l[1] = col
                    return True
        return False
