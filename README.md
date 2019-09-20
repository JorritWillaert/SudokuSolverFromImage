# Sudoku Solver From Image

Extract a sudoku from an image and solve it.

## Installation

This program relies on external packages. To install them you can type in the
following command on your terminal:
```
pip install -r requirements.txt
```
or alternatively:
```
pip3 install -r requirements.txt
```

## Usage

```
$ python sudoku.py
```

## Pipeline explained
This is the input image.

![Input image](https://raw.githubusercontent.com/Randpy/SudokuSolverFromImage/improve_README_and_requirements/example_images/example_pipeline/1_start.jpg)

This input image is first cut out.

![After cut-out](https://raw.githubusercontent.com/Randpy/SudokuSolverFromImage/improve_README_and_requirements/example_images/example_pipeline/2_cut.JPG)

This is followed by a loop over the 81 fields of the board. This is for example a detected digit.

![Detected digit](https://raw.githubusercontent.com/Randpy/SudokuSolverFromImage/improve_README_and_requirements/example_images/example_pipeline/3_digit.JPG)

After which a so-called mask is placed over it, so that it looks the same as the examples of the 'MNIST' training set.

![Masked digit](https://raw.githubusercontent.com/Randpy/SudokuSolverFromImage/improve_README_and_requirements/example_images/example_pipeline/4_threshold_digit.JPG)

Now the trained model can be used to detect which digit is here.
So, the image can been converted to an array that looks the same as the sudoku. After this, a simple back-tracking algorithm can solve the sudoku.
