import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# this is done to ensure that Keras won't use my non-functional gpu
import numpy as np  # easy array manipulation
from matplotlib import pyplot  # draw images and graphs
import tensorflow as tf  # necessary for Keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

HEIGHT = 64  # Height of fake images
WIDTH = 64  # Width of fake images
CHANNELS = 3  # number of channels per pixel

w_init = tf.keras.initializers.random_normal(mean=0.0, stddev=0.02)  # will be used later in model build


def deconv_block(inputs, num_filters, kernel_size, strides, bn=True):
    # returns custom made deconv layer that is added to inputs argument
    x = Conv2DTranspose(filters=num_filters, kernel_size=kernel_size, kernel_initializer=w_init, padding="same",
                        strides=strides, use_bias=False)(inputs)
    if bn:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

    return x


def conv_block(inputs, num_filters, kernel_size, padding="same", strides=2, activation=True):
    # returns custom made conv layer that is added to inputs argument
    x = Conv2D(filters=num_filters, kernel_size=kernel_size, kernel_initializer=w_init, padding=padding,
               strides=strides)(inputs)
    if activation:
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)

    return x


def build_generator(latent_dim, n_classes=9):
    # f*filters represents num of filters per layer
    f = [2 ** i for i in range(5)][::-1]  # [16,8,4,2,1]
    filters = 32  # baseline filters per layer
    output_strides = 16  # total number of strides - the amount the axis will be expanded
    h_output = HEIGHT // output_strides  # initial layer height
    w_output = WIDTH // output_strides  # initial layer width

    # label input
    label_input = Input(shape=(n_classes,), name="condition")
    y = Dense(h_output * w_output)(label_input)
    y = Reshape((h_output, w_output, 1))(y)

    # image input
    noise = Input(shape=(latent_dim,), name="gen_noise_input")
    x = Dense(f[0] * filters * h_output * w_output, use_bias=False)(noise)
    x = Reshape((h_output, w_output, f[0] * filters))(x)  # desired initial proportions

    x = Concatenate()([x, y])  # concatenate inputs
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # five deconv layers, expanding in size and shrinking in filters
    for i in range(1, 5):
        x = deconv_block(x, num_filters=f[i] * filters, kernel_size=5, strides=2, bn=True)

    # conv layer in order to convert filters to 3
    x = conv_block(x, num_filters=3, kernel_size=5, strides=1, activation=False)
    fake_output = Activation("tanh")(x)
    return Model([noise, label_input], fake_output, name="generator")


def build_discriminator(n_classes=9):
    # f*filters represents num of filters per layer
    f = [2 ** i for i in range(4)]  # [1, 2, 4, 8]
    filters = 64  # base number of filters
    output_strides = 16  # total number of strides - the amount the axis will shrink
    h_output = HEIGHT // output_strides  # initial layer height
    w_output = WIDTH // output_strides  # initial layer width

    # label input
    label_input = Input(shape=(n_classes,), name="condition")
    x = Dense(HEIGHT * WIDTH)(label_input)
    x = Reshape((HEIGHT, WIDTH, 1))(x)  # reformat to initial size

    # image input
    image_input = Input(shape=(HEIGHT, WIDTH, CHANNELS), name="images")

    x = Concatenate()([image_input, x])  # concatenate inputs
    # 4 conv layers, expanding in filters and shrinking in size
    for i in range(0, 4):
        x = conv_block(x, num_filters=f[i] * filters, kernel_size=5, strides=2)

    # reduce to a single neuron
    x = Flatten()(x)
    x = Dense(1)(x)
    return Model([image_input, label_input], x, name="discriminator")


class GAN(Model):  # the aggregation of the discriminator and the generator
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real):
        global categories

        real_images, real_conditions = real  # real conditions are categorical
        # dynamic batch_size and n_condition, not sure why I did this, it was a pain in the ass because the sizes are
        # only known during training sessions
        batch_size = tf.shape(real_images)[0]
        n_conditions = tf.shape(real_conditions)[1]
        # train discriminator
        # fake images
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_conditions = tf.one_hot(tf.random.uniform(shape=(batch_size,), dtype=tf.int32, maxval=n_conditions),
                                          depth=n_conditions)  # create random y as onehot
        generated_images = self.generator([random_latent_vectors, generated_conditions])  # fake images
        generated_labels = tf.zeros((batch_size, 1)) # desired discriminator prediction

        with tf.GradientTape() as tape: # convention of writing in case of failure
            predictions = self.discriminator([generated_images, generated_conditions]) # D(G(z/y))
            d1_loss = self.loss_fn(generated_labels, predictions)

        grads = tape.gradient(d1_loss, self.discriminator.trainable_weights) # get gradients on discriminator
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))   # back propagate using the gradients

        # real images
        labels = tf.ones((batch_size, 1)) # desired discriminator prediction
        with tf.GradientTape() as tape:
            predictions = self.discriminator([(real_images - 127.5) / 127.5, real_conditions]) # D(x/y)
            d2_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d2_loss, self.discriminator.trainable_weights) # get gradients on discriminator
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights)) # back propagate using the gradients

        # train generator

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_conditions = tf.one_hot(tf.random.uniform(shape=(batch_size,), dtype=tf.int32, maxval=n_conditions),
                                          depth=n_conditions) # create random y as onehot
        misleading_labels = tf.ones((batch_size, 1)) # desired discriminator prediction

        with tf.GradientTape() as tape:
            predictions = self.discriminator(
                [self.generator([random_latent_vectors, generated_conditions]), generated_conditions]) # D(G(z/y))
            g_loss = self.loss_fn(misleading_labels, predictions)

        # get gradients on generator
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights)) # back propagate using the gradients

        return {"d1_loss": d1_loss, "d2 loss": d2_loss, "g_loss": g_loss} # the return value is the displayed string on console during training


def rand_condition(n_conditions, parameters=2): #returns random vector condition y
    arr = np.zeros(n_conditions)
    arr[:parameters] = 1
    np.random.shuffle(arr)
    return arr


def get_condition_names(arr, names): #return character name from random vector condition y
    categories = []
    for i in range(len(arr)):
        if arr[i] == 1:
            categories.append(names[i])
    return categories


def save_plot(examples, condition_names, epoch, n):
    examples = (examples + 1) / 2.0 # transform range from -1 -1 to 0 1
    for i in range(n ** 2): # draw images
        pyplot.subplot(n, n, i + 1)  # n by n plot
        pyplot.axis("off")  # dont need markers for pixels
        pyplot.imshow(examples[i])
        pyplot.title(' '.join(condition_names[i]), fontdict={'fontsize': 5})
    filename = f"samples/fake_image_epoch_{epoch + 1}.png"
    pyplot.savefig(filename) # save file
    pyplot.close()


if __name__ == "__main__":
    # parameters
    batch_size = 128
    latent_dim = 128
    num_epochs = 100

    # Dataset
    # fetch real images
    dataset = tf.keras.preprocessing.image_dataset_from_directory("dataset/images/main_dataset/main_dataset",
                                                                  label_mode="categorical", batch_size=batch_size,
                                                                  image_size=(HEIGHT, WIDTH), shuffle=True)
    categories = dataset.class_names

    d_model = build_discriminator()
    g_model = build_generator(latent_dim)
    # summarize model architecture
    d_model.summary()
    g_model.summary()

    gan = GAN(d_model, g_model, latent_dim)
    # gan compilation
    bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)# from logits because the output has no range, label smoothing is recommended
    d_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    gan.compile(d_optimizer, g_optimizer, bce_loss_fn)

    # train loop
    for epoch in range(num_epochs):
        gan.fit(dataset, epochs=1)
        g_model.save("saved_model/g_model.h5")
        d_model.save("saved_model/d_model.h5")
        # save samples
        n_samples = 25
        noise = np.random.normal(size=(n_samples, latent_dim))
        conditions = [rand_condition(len(categories), 2) for i in range(n_samples)] #get random condition y (2 parameters)
        condition_names = [get_condition_names(condition, categories) for condition in conditions] #get names
        examples = g_model.predict([noise, tf.stack(conditions)])
        save_plot(examples, condition_names, epoch, int(np.sqrt(n_samples)))
