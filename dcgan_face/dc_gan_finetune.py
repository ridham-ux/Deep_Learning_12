import time

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from keras.layers import SpectralNormalization
import os

base_dir = '/kaggle/input/facerexpressions/dataset/'
data_path = '/kaggle/input/facerexpressions/data.csv'

data = pd.read_csv(data_path)

# Define image size and batch size
IMG_SIZE = 64
BATCH_SIZE = 32
NOISE_DIM = 150


def preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img.set_shape([None, None, 3])  # Explicitly set the shape after decoding
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = (img - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return img

def load_and_preprocess_from_path_label(path):
    # Use tf.strings.join instead of os.path.join
    full_img_path = tf.strings.join([base_dir, path], separator=os.sep)
    return preprocess_image(full_img_path)

# Create a TensorFlow Dataset from the DataFrame
paths = data['path'].values
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset = dataset.map(lambda x: load_and_preprocess_from_path_label(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*512, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((8, 8, 512)))
    assert model.output_shape == (None, 8, 8, 512)

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 16, 16, 256)
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 32, 32, 128)

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 64, 64, 64)

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)
    
    return model

def build_discriminator():
    model = tf.keras.Sequential()
    model.add(SpectralNormalization(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3])))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(SpectralNormalization(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(SpectralNormalization(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = build_generator()
discriminator = build_discriminator()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Set constants
EPOCHS = 500
NUM_EXAMPLES_TO_GENERATE = 5
GENERATOR_UPDATES = 2
EPOCHS_TO_SAVE = 50

seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    # Train the discriminator
    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Train the generator more frequently
    gen_loss = None
    for _ in range(GENERATOR_UPDATES):
        with tf.GradientTape() as gen_tape:
            noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
            generated_images = generator(noise, training=True)
            fake_output = discriminator(generated_images, training=True)
            gen_loss = generator_loss(fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig, axes = plt.subplots(1, NUM_EXAMPLES_TO_GENERATE, figsize=(15, 15))

    for i in range(predictions.shape[0]):
        axes[i].imshow((predictions[i] * 127.5 + 127.5).numpy().astype(np.uint8))
        axes[i].axis('off')

    plt.show()

def log_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    with summary_writer.as_default():
        for i in range(predictions.shape[0]):
            tf.summary.image(f"Generated Images at Epoch {epoch}", predictions, step=epoch)

# Initialize lists to store loss values
gen_losses = []
disc_losses = []  

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        gen_loss_epoch = 0
        disc_loss_epoch = 0
        num_batches = 0

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            gen_loss_epoch += gen_loss
            disc_loss_epoch += disc_loss
            num_batches += 1

        gen_loss_epoch /= num_batches
        disc_loss_epoch /= num_batches


        # Append the average losses to the lists
        gen_losses.append(gen_loss_epoch)
        disc_losses.append(disc_loss_epoch)
        
        # Produce images for the GIF as we go
        generate_and_save_images(generator, epoch + 1, seed)

        # Display the loss chart and save the model every 'x' epochs
        if (epoch + 1) % EPOCHS_TO_SAVE == 0:
            generator.save(f'/kaggle/working/generator_epoch_{epoch + 1}.h5')
            discriminator.save(f'/kaggle/working/discriminator_epoch_{epoch + 1}.h5')
            # Plot losses
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, epoch + 2), gen_losses, label='Generator Loss')
            plt.plot(range(1, epoch + 2), disc_losses, label='Discriminator Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('GAN Training Losses')
            plt.show()
            
        print(f'Epoch {epoch+1}, Generator Loss: {gen_loss_epoch:.4f}, Discriminator Loss: {disc_loss_epoch:.4f}')
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        
    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)
    log_images(generator, epochs, seed)
    
# Start training
train(dataset, EPOCHS)
