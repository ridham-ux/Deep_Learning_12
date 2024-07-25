import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Function to load the trained model
def load_model():
    try:
        model = tf.keras.models.load_model('nis_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Class names
class_names = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Preprocessing function
def preprocess_image(image):
    processed_image = cv2.GaussianBlur(image, (3, 3), 0)
    return processed_image

def custom_preprocessing_function(image):
    image = np.array(image, dtype=np.uint8)
    image = preprocess_image(image)
    image = image.astype(np.float32)
    return image

def prepare_image(image):
    image = custom_preprocessing_function(image)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
st.title("Facial Expression Recognition")
st.write("Upload an image and the model will predict the facial expression.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # OK button
        if st.button("OK"):
            if model is not None:
                # Preprocess image
                img = prepare_image(image)
                
                # Make prediction
                prediction = model.predict(img)
                decoded_prediction = class_names[np.argmax(prediction)]
                
                st.write(f'Predicted Expression: {decoded_prediction}')
    
                # Display probabilities
                st.write("Prediction Probabilities:")
                probabilities = {class_name: round(float(prob), 4) for class_name, prob in zip(class_names, prediction[0])}
                st.write(probabilities)
            else:
                st.error("Model is not loaded. Check the error logs for details.")
    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.error(f"Exception details: {e}")
