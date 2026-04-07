from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load models
vgg_model = tf.keras.models.load_model("vgg_model.h5")
resnet_model = tf.keras.models.load_model("resnet_model.h5")

# Classes
xray_classes = ["Normal", "Pneumonia"]

ct_classes = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Squamous Cell Carcinoma",
    "Covid",
    "Normal"
]

# Preprocess
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction API
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
