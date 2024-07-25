import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Define the CVAE model class
class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(128, 128, 1)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='leaky_relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='leaky_relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=32*32*128, activation=tf.nn.leaky_relu),
            tf.keras.layers.Reshape(target_shape=(32, 32, 128)),
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='leaky_relu'),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='leaky_relu'),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='same', activation='leaky_relu'),
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', activation='sigmoid'),
        ])
    
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits
    
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        return x_logit, mean, logvar

# Load the trained model
latent_dim = 300  # Ensure this matches the latent dimension used during training
model_1_1 = CVAE(latent_dim)
model_1_1.build((None, 128, 128, 1))  # Adjust if necessary
model_1_1.load_weights('model_weights.weights.h5')  # Assuming weights are saved separately

# # Define a function to generate images from latent vectors
# def generate_images(latent_dim, num_images=16):
#     random_vector_for_generation = tf.random.normal(shape=[num_images, latent_dim])
    
#     # Generate images
#     predictions = model_1_1.sample(random_vector_for_generation)
    
#     fig = plt.figure(figsize=(4, 4))
#     for i in range(num_images):
#         plt.subplot(4, 4, i + 1)
#         plt.imshow(predictions[i, :, :, 0], cmap='gray')
#         plt.axis('off')
#     st.pyplot(fig)

def generate_images(latent_dim, num_images=16):
    # Create an instance of CVAE with the specified latent dimension
    model_1_1 = CVAE(latent_dim)
    
    # Load weights into the model
    model_1_1.load_weights('model_weights.weights.h5')

    # Generate random latent vectors
    random_vector_for_generation = tf.random.normal(shape=[num_images, latent_dim])

    # Use the model's sample method to generate images
    predictions = model_1_1.sample(random_vector_for_generation)

    # Plot and display the generated images
    fig = plt.figure(figsize=(4, 4))
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
    st.pyplot(fig)


# Streamlit app
st.title('CVAE Image Generation')

latent_dim = st.sidebar.slider('Latent Dimension', 2, 512, 300)
num_images = st.sidebar.slider('Number of Images', 1, 16, 16)

if st.button('Generate Images'):
    st.write('Generating images...')
    generate_images(latent_dim, num_images)
