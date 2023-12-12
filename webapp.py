import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model architecture from JSON file
json_file_path = "model.json"
with open(json_file_path, 'r') as json_file:
    model_json = json_file.read()

# Create the model based on the loaded architecture
loaded_model = tf.keras.models.model_from_json(model_json)

# Load model weights from HDF5 file
loaded_model.load_weights("model.h5")

# Function to preprocess and predict an image
def predict_pneumonia(image):
    # Ensure the input is a NumPy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif not isinstance(image, np.ndarray):
        raise ValueError("Unsupported image type. Expecting a NumPy array or PIL Image.")

    # Convert image to float32 if it's not already
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # Normalize the image
    image = image / 255.0

    # Preprocess the image for the model
    img = tf.image.resize(image, [200, 200])
    img = tf.expand_dims(img, 0)

    # Make predictions
    prediction = loaded_model.predict(img)
    return prediction

# Streamlit app
st.title("Pneumonia Detection Web App")
st.sidebar.title("Upload Image")

# Upload image through sidebar
uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Make prediction
    if st.button("Predict"):
        prediction = predict_pneumonia(image)

        # Interpret the prediction
        threshold = 0.5  # Adjust the threshold as needed
        pneumonia_likelihood = prediction[0, 0]

        if pneumonia_likelihood > threshold:
            result = "The Patient has Pneumonia"
            color = "red"
        else:
            result = "The Patient is Normal"
            color = "green"

        custom_css = f"""
            <style>
                .custom-text {{
                    color: {color};
                    font-family: 'Pacifico', cursive;
                    font-size: 20px;
                }}
            </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)

        # Display the result with the custom style
        st.markdown(f"<p class='custom-text'>Prediction: {result} (Likelihood: {pneumonia_likelihood:.4f})</p>", unsafe_allow_html=True)