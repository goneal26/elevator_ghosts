# app.py
import streamlit as st
import cv2
import numpy as np

def get_subject_mask(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection (you can customize this based on your needs)
    edges = cv2.Canny(gray_image, 30, 100)

    # Find contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black mask
    mask = np.zeros_like(gray_image)

    # Fill the mask with white for the identified subject
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    return mask

def pointillism_filter(image, subject_mask):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce the number of colors only in the subject region
    _, reduced_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # Apply the subject mask
    reduced_image = cv2.bitwise_and(reduced_image, subject_mask)

    # Add small dots for the pixel art effect
    dot_size = st.slider("Dot Size", 1, 20, 5)
    kernel = np.ones((dot_size, dot_size), np.uint8)
    pointillism_image = cv2.dilate(reduced_image, kernel, iterations=1)

    # Convert back to BGR for display
    pointillism_image = cv2.cvtColor(pointillism_image, cv2.COLOR_GRAY2BGR)

    return pointillism_image

def main():
    st.title("Pointillism Filter App")

    # Upload an image through the Streamlit interface
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Convert the uploaded image to a NumPy array
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)

        # Get the subject mask
        subject_mask = get_subject_mask(image)

        # Apply the pointillism filter only to the subject region
        pointillism_image = pointillism_filter(image, subject_mask)

        # Display the original and filtered images
        st.image([image, pointillism_image], caption=["Original Image", "Pointillism Filter"], width=300)

if __name__ == "__main__":
    main()
