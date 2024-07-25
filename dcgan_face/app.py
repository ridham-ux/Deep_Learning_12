import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('./generate_faces.h5')
    return model

generator = load_model()

NOISE_DIM = 100

# Function to generate faces
def generate_faces(num_faces):
    random_latent_vectors = tf.random.normal([num_faces, NOISE_DIM])
    generated_faces = generator(random_latent_vectors, training=False)
    generated_faces = (generated_faces * 127.5 + 127.5).numpy().astype(np.uint8)
    return generated_faces

# Streamlit UI
st.title("Face Generator using DCGAN")
num_faces = st.selectbox("Select number of faces to generate:", [1, 5, 10, 20])

if st.button("Generate"):
    faces = generate_faces(num_faces)
    st.write(f"Generated {num_faces} face(s)")

    # Display faces in rows of 5
    num_rows = (num_faces + 4) // 5  # Calculate the number of rows needed
    for row in range(num_rows):
        cols = st.columns(5)
        for col_idx in range(5):
            face_idx = row * 5 + col_idx
            if face_idx < num_faces:
                cols[col_idx].image(faces[face_idx], width=128, caption=f"Face {face_idx + 1}")
