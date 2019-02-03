import tensorflow as tf
import process_images
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import process_images


def collect_training_data_lines():
    x_train, y_train = process_images.convert_resized_images_to_numpy("lines",0,200)
    x_test, y_test = process_images.convert_resized_images_to_numpy("lines",201,400)
    return (x_train, y_train), (x_test ,y_test)

def construct_model():
    (x_train, y_train), (x_test, y_test) = collect_training_data_lines()
    model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print('Test accuracy:', test_acc)

if __name__ == "__main__":
    construct_model()