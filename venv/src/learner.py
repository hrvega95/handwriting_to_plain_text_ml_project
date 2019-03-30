from tensorflow import keras
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import process_images
import tensorflow as tf
import process_images



def collect_training_data_words():
    print("Getting training data")
    x_train, y_train = process_images.get_words()
    print("Getting test data")
    x_test, y_test = process_images.get_words(1,300)
    label_encoder = preprocessing.LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)
    return (x_train, y_train), (x_test ,y_test)


def construct_model():
    (x_train, y_train), (x_test, y_test) = collect_training_data_words()
    model = keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu,
                               input_shape=(32, 128,1)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Flatten(),
        keras.layers.Dense(4096, activation=tf.nn.relu),
        keras.layers.Dense(y_train.max() + 1, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(x_train.shape)
    model.fit(x_train, y_train, epochs=5)
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print('Test accuracy:', test_acc)

if __name__ == "__main__":
    construct_model()