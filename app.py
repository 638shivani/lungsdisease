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

# --- 1. SET PROPER IDS (Just the string, not the URL) ---
VGG_ID = "https://drive.google.com/file/d/1ZdUs5kj-TfiH58HnnnSAFm7sM1BPnnoo/view?usp=drive_link"
RESNET_ID = "https://drive.google.com/file/d/1CrjfNCG-J349QBKrbQ59nnm8hhL3c0nq/view?usp=drive_link"

# --- 2. UNIFIED DOWNLOAD FUNCTION ---
def download_models():
    models = {
        "vgg_model.h5": VGG_ID,
        "resnet_model.h5": RESNET_ID
    }
    
    for filename, file_id in models.items():
        # Delete the file if it exists but is too small (broken download)
        if os.path.exists(filename) and os.path.getsize(filename) < 10000000:
            print(f"Detected corrupted file for {filename}. Deleting...")
            os.remove(filename)

        # Download only if the file doesn't exist
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            # Using id= ensures we get the raw file data
            gdown.download(id=file_id, output=filename, quiet=False)

# Run the download logic
download_models()

# --- 3. LOAD MODELS ---
print("Loading models into memory...")
vgg_model = tf.keras.models.load_model("vgg_model.h5")
resnet_model = tf.keras.models.load_model("resnet_model.h5")

# Classes and Preprocessing
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
