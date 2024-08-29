import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

HEIGHT = 64
WIDTH = 64
CHANNELS = 3

w_init = tf.keras.initializers.random_normal(mean=0.0, stddev=0.02)


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, (HEIGHT, WIDTH))
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 127.5
    return image


def tf_dataset(images_path, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def deconv_block(inputs, num_filters, kernel_size, strides, bn=True):
    x = Conv2DTranspose(filters=num_filters, kernel_size=kernel_size, kernel_initializer=w_init, padding="same",
                        strides=strides, use_bias=False)(inputs)
    if bn:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

    return x


def conv_block(inputs, num_filters, kernel_size, padding="same", strides=2, activation=True):
    x = Conv2D(filters=num_filters, kernel_size=kernel_size, kernel_initializer=w_init, padding=padding,
               strides=strides)(inputs)
    if activation:
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)

    return x


def build_generator(latent_dim):
    f = [2 ** i for i in range(5)][::-1]
    filters = 32
    output_strides = 16
    h_output = HEIGHT // output_strides
    w_output = WIDTH // output_strides

    noise = Input(shape=(latent_dim,), name="gen_noise_input")

    x = Dense(f[0] * filters * h_output * w_output, use_bias=False)(noise)

    x = Reshape((h_output, w_output, f[0] * filters))(x)

    for i in range(1, 5):
        x = deconv_block(x, num_filters=f[i] * filters, kernel_size=5, strides=2, bn=True)

    x = conv_block(x, num_filters=3, kernel_size=5, strides=1, activation=False)
    fake_output = Activation("tanh")(x)
    return Model(noise, fake_output, name="generator")


def build_discriminator():
    f = [2 ** i for i in range(4)]
    filters = 64
    output_strides = 16
    h_output = HEIGHT // output_strides
    w_output = WIDTH // output_strides

    image_input = Input(shape=(HEIGHT, WIDTH, CHANNELS), name="images")
    x = image_input
    for i in range(0, 4):
        x = conv_block(x, num_filters=f[i] * filters, kernel_size=5, strides=2)

    x = Flatten()(x)
    x = Dense(1)(x)
    return Model(image_input, x, name="discriminator")


class GAN(Model):
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

    def train_step(self, real_images):
        real_images, _ = real_images
        batch_size = tf.shape(real_images)[0]
        # train discriminator
        # fake images
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(random_latent_vectors)
        generated_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(generated_images)
            d1_loss = self.loss_fn(generated_labels, predictions)

        grads = tape.gradient(d1_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # real images
        labels = tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator((real_images-127.5)/127.5)
            d2_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d2_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # train generator

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d1_loss": d1_loss, "d2 loss": d2_loss, "g_loss": g_loss}


def save_plot(examples, epoch, n):
    examples = (examples + 1) / 2.0
    for i in range(n ** 2):
        pyplot.subplot(n, n, i + 1)
        pyplot.axis("off")
        pyplot.imshow(examples[i])
    filename = f"regsamples/fake_image_epoch_{epoch+1}.png"
    pyplot.savefig(filename)
    pyplot.close()


if __name__ == "__main__":
    # parameters
    batch_size = 128
    latent_dim = 128
    num_epochs = 100

    # Dataset
    dataset = tf.keras.preprocessing.image_dataset_from_directory("dataset/images/main_dataset/main_dataset",
                                                                  label_mode="categorical", batch_size=batch_size,
                                                                  image_size=(HEIGHT, WIDTH), shuffle=True)

    d_model = build_discriminator()
    g_model = build_generator(latent_dim)

    d_model.summary()
    g_model.summary()

    gan = GAN(d_model, g_model, latent_dim)

    bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    d_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    gan.compile(d_optimizer, g_optimizer, bce_loss_fn)


    for epoch in range(num_epochs):
        gan.fit(dataset, epochs=1)
        g_model.save("saved_model/regg_model.h5")
        d_model.save("saved_model/regd_model.h5")

        n_samples = 25
        noise = np.random.normal(size=(n_samples, latent_dim))
        examples = g_model.predict(noise)
        save_plot(examples, epoch, int(np.sqrt(n_samples)))