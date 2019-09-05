import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

import matplotlib.pyplot as plt

def train_model():
    """Use 60.000 digit images provides by mnist"""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape the array to 4-dims, so it can work with the Keras API.
    # The images are 28 x 28
    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_train.astype(np.float32)

    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))
    X_test.astype(np.float32)

    input_shape = (28, 28, 1)

    # Normalize the RGB codes
    X_train = np.true_divide(X_train, 255)
    X_test = np.true_divide(X_test, 255)

    # Creating sequential model and adding layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size = (3, 3), input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(50, kernel_size = (3, 3), input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten()) # Create fully connected layers
    model.add(Dense(128, activation = tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation = tf.nn.softmax)) # We need to classify digits ranging from 0 - 10

    model.summary()

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.fit(x = X_train, y = y_train, epochs = 10)

    print("\nResults on test set: ")
    model.evaluate(X_test, y_test)

    check = input("Do want to save this new model? (This will overwrite the previous weights!) ")
    # Save model and weights to HDF5
    if check.lower() in ['y', 'yes']:
        model.save('testing_model_digits.h5')
        print('\nSaved model and weights.')

if __name__ == '__main__':
    train_model()
