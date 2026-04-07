import os
import io
import gdown
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- Model Download Logic ---
# Google Drive IDs from your links

gdown.download("https://drive.google.com/file/d/1ZdUs5kj-TfiH58HnnnSAFm7sM1BPnnoo/view?usp=drive_link", "vgg_model.h5", quiet=False)
gdown.download("https://drive.google.com/file/d/1CrjfNCG-J349QBKrbQ59nnm8hhL3c0nq/view?usp=drive_link", "resnet_model.h5", quiet=False)


VGG_ID = "https://drive.google.com/file/d/1ZdUs5kj-TfiH58HnnnSAFm7sM1BPnnoo/view?usp=drive_link"
RESNET_ID ="https://drive.google.com/file/d/1CrjfNCG-J349QBKrbQ59nnm8hhL3c0nq/view?usp=drive_link"

def download_models():
    if not os.path.exists("vgg_model.h5"):
        print("Downloading VGG model...")
        gdown.download(id=VGG_ID, output="vgg_model.h5", quiet=False)
    
    if not os.path.exists("resnet_model.h5"):
        print("Downloading ResNet model...")
        gdown.download(id=RESNET_ID, output="resnet_model.h5", quiet=False)

# Run download before loading
download_models()

# Load models
vgg_model = tf.keras.models.load_model("vgg_model.h5")
resnet_model = tf.keras.models.load_model("resnet_model.h5")

# Classes
xray_classes = ["Normal", "Pneumonia"]
ct_classes = ["Adenocarcinoma", "Large Cell Carcinoma", "Squamous Cell Carcinoma", "Covid", "Normal"]

def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...), image_type: str = Form(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img = preprocess(image)

    if image_type == "xray":
        pred = vgg_model.predict(img)
        result = xray_classes[np.argmax(pred)]
    else:
        pred = resnet_model.predict(img)
        result = ct_classes[np.argmax(pred)]

    confidence = float(np.max(pred))

    return {
        "prediction": result,
        "confidence": round(confidence * 100, 2)
    }
