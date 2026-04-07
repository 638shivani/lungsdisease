import streamlit as st
import keras
import numpy as np
from PIL import Image
import gdown

st.title("🫁 Lung Disease Detection")



# Download models
vgg_url = "https://drive.google.com/file/d/1ZdUs5kj-TfiH58HnnnSAFm7sM1BPnnoo/view?usp=drive_link"
resnet_url = "https://drive.google.com/file/d/1CrjfNCG-J349QBKrbQ59nnm8hhL3c0nq/view?usp=drive_link"

gdown.download(vgg_url, "vgg_model.h5", quiet=False)
gdown.download(resnet_url, "resnet_model.h5", quiet=False)

# Load models
from keras.models import load_model

vgg_model = load_model("vgg_model.h5")
resnet_model = load_model("resnet_model.h5")

# Classes
xray_classes = ["Normal", "Pneumonia"]

ct_classes = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Squamous Cell Carcinoma",
    "Covid",
    "Normal"
]

# Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

image_type = st.selectbox("Select Image Type", ["xray", "ct"])

def preprocess(image):
    image = image.resize((224,224))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = preprocess(image)

    if image_type == "xray":
        pred = vgg_model.predict(img)
        result = xray_classes[np.argmax(pred)]
    else:
        pred = resnet_model.predict(img)
        result = ct_classes[np.argmax(pred)]

    confidence = float(np.max(pred))

    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {round(confidence*100,2)}%")
