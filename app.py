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

# --- 1. CLEAN IDs (No slashes, no 'view', no URLs) ---
VGG_ID = "1ZdUs5kj-TfiH58HnnnSAFm7sM1BPnnoo"
RESNET_ID = "1CrjfNCG-J349QBKrbQ59nnm8hhL3c0nq"

def download_models():
    models = {
        "vgg_model.h5": VGG_ID,
        "resnet_model.h5": RESNET_ID
    }
    
    for filename, file_id in models.items():
        # Delete broken 8KB files
        if os.path.exists(filename) and os.path.getsize(filename) < 10000000:
            print(f"Deleting broken file: {filename}")
            os.remove(filename)

        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            # Using only the ID string fixes the retrieval error
            gdown.download(id=file_id, output=filename, quiet=False)

# Run download
download_models()

# Load models
print("Loading models into memory...")
vgg_model = tf.keras.models.load_model("vgg_model.h5")
resnet_model = tf.keras.models.load_model("resnet_model.h5")

# ... (rest of your prediction and route code) ...
