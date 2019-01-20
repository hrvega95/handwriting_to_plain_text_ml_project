import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def collect_training_data():
    training_data = keras.datasets.fashion_mnist
    (tr_x,tr_y), (ts_x,ts_l) = training_data.load_data()
    print(tf.__version__)



if __name__ == "__main__":
    collect_training_data()