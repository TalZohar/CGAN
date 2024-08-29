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

def save_plot(examples, condition_names,  n):
    examples = (examples + 1) / 2.0
    for i in range(n ** 2):
        pyplot.subplot(n, n, i + 1)
        pyplot.axis("off")
        pyplot.imshow(examples[i])
        pyplot.title(' '.join(condition_names[i]), fontdict={'fontsize': 5})
    filename = f"testImages/_{condition_names[1]}"
    pyplot.savefig(filename)
    pyplot.close()


def saveAllTypes(categories, n_samples, latent_dim):
    for i in range(len(categories)):
        conditions = [np.zeros(len(categories)) for i in range(n_samples)]
        for condition in conditions:
            condition[i] = 1
        latent_points = np.random.normal(size=(n_samples, latent_dim))
        condition_names = [get_condition_names(condition, categories) for condition in conditions]
        examples = model.predict([latent_points, tf.stack(conditions)])
        save_plot(examples, condition_names, int(np.sqrt(n_samples)))

if __name__== "__main__":
    model = load_model("saved_model/g_model30.h5")
    n_samples = 25
    latent_dim = 128
    categories = ['Blue Mountain', 'Chino', 'Chiya', 'Cocoa', 'Maya', 'Megumi', 'Mocha', 'Rize', 'Sharo']
    latent_points = np.random.normal(size=(n_samples, latent_dim))
    saveAllTypes(categories, n_samples, latent_dim)



























