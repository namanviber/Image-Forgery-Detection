import streamlit as st
import numpy as np
import cv2
from keras.models import load_model

model = load_model("newMobileNet.h5")

st.title('Image Forgery Detection')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Preprocess Image
def preprocess_image(img):
    img = cv2.resize(img, (256, 256))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_gray)

    img_denoised = cv2.fastNlMeansDenoising(img_eq, None, h=10, templateWindowSize=7, searchWindowSize=21)
    img_color_corrected = cv2.cvtColor(img_denoised, cv2.COLOR_GRAY2BGR)
    img_color_corrected = cv2.cvtColor(img_color_corrected, cv2.COLOR_BGR2HSV)
    img_color_corrected[:, :, 1] = img_color_corrected[:, :, 1] * 1.2
    img_color_corrected[:, :, 2] = img_color_corrected[:, :, 2] * 0.8
    img_color_corrected = cv2.cvtColor(img_color_corrected, cv2.COLOR_HSV2BGR)

    return img_color_corrected

# Predict
def predict(img):
    img = preprocess_image(img)
    img = img.reshape((1, 256, 256, 3))
    prediction = model.predict(img)
    pred = np.argmax(prediction, axis=1)
    pred = pred[0]
    if pred == 0:
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.success("Forged Image")
        st.success(f'{prediction[0][0]*100:.2f} % :thumbsup:')
    else:
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.info("Image is not Forged")
        st.info(f'{prediction[0][1]*100:.2f} % :thumbsup:')

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    trigger = st.button('Predict', on_click=lambda: predict(image))
