import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model("your_model.keras")

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# Function to classify emotion
# Function to classify emotion
from PIL import ImageOps


# Function to classify emotion
def classify_emotion(image):
    # Convert image to grayscale
    image = ImageOps.grayscale(image)
    # Resize image to match model input size
    image = image.resize((48, 48))
    # Normalize pixel values
    image = np.array(image) / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Print the shape of the image
    print("Image shape:", image.shape)

    result = model.predict(image)
    emotion_index = np.argmax(result)
    return emotion_labels[emotion_index]


# Main function
def main():
    st.title("Facial Expression Recognition")

    st.write("Upload an image to classify its emotion: Happy, Surprise, Fear, Angry, Neutral, Disgust or Sad")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify Emotion"):
            emotion = classify_emotion(image)
            st.title(f"Predicted Emotion: {emotion}")


if __name__ == "__main__":
    main()
