import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot
import tensorflow as tf


def rand_condition(n_conditions, parameters = 2):
    arr = np.zeros(n_conditions)
    arr[:parameters] = 1
    np.random.shuffle(arr)
    return arr

def get_condition_names(arr, names):
    categories = []
    for i in range(len(arr)):
        if arr[i] == 1:
            categories.append(names[i])
    return categories

def save_plot(examples, n):
    examples = (examples + 1) / 2.0
    for i in range(n ** 2):
        pyplot.subplot(n, n, i + 1)
        pyplot.axis("off")
        pyplot.imshow(examples[i])
    filename = "dcganimg.png"
    pyplot.savefig(filename)
    pyplot.close()

if __name__== "__main__":
    model = load_model("saved_model/regg_model.h5")
    n_samples = 25
    latent_dim = 128

    latent_points = np.random.normal(size=(n_samples, latent_dim))
    examples = model.predict(latent_points)
    save_plot(examples, int(np.sqrt(n_samples)))

