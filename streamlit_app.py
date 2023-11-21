import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageFilter

model = load_model("newMobileNet.h5")

st.title('Image Forgery Detection')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Preprocess Image
def preprocess_image(img):
    img = img.resize((256, 256))

    img_gray = img.convert('L')
    img_eq = Image.fromarray(np.array(img_gray), 'L')  # Equalize histogram for grayscale

    img_denoised = img_eq.filter(ImageFilter.DENOISE)
    img_color_corrected = img_denoised.convert('HSV')
    
    img_array = np.array(img_color_corrected)
    img_array[:, :, 1] = img_array[:, :, 1] * 1.2
    img_array[:, :, 2] = img_array[:, :, 2] * 0.8
    img_color_corrected = Image.fromarray(img_array, 'HSV').convert('RGB')

    return img_color_corrected

# Predict
def predict(img):
    img = preprocess_image(img)
    img_array = np.array(img)
    img = img_array.reshape((1, 256, 256, 3))
    prediction = model.predict(img)
    pred = np.argmax(prediction, axis=1)
    pred = pred[0]
    if pred == 0:
        st.image(img_array, caption="Uploaded Image", use_column_width=True)
        st.success("Forged Image")
        st.success(f'{prediction[0][0]*100:.2f} % :thumbsup:')
    else:
        st.image(img_array, caption="Uploaded Image", use_column_width=True)
        st.info("Image is not Forged")
        st.info(f'{prediction[0][1]*100:.2f} % :thumbsup:')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    trigger = st.button('Predict', on_click=lambda: predict(image))