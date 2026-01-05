from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import pandas as pd
import random
import string
import base64
from captcha.image import ImageCaptcha 

# Import relatif supposant l'exécution via 'uvicorn backend.main:app'
try:
    from .architecture import CRNN
except ImportError:
    # Fallback pour exécution directe ou debug
    from architecture import CRNN

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-True-Label", "X-Prediction"],
)

# Configuration des chemins robustes
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

MODEL_PATH = os.path.join(BACKEND_DIR, "model.pth")
DATA_DIR = os.path.join(PROJECT_ROOT, "data/images")
VAL_CSV = os.path.join(PROJECT_ROOT, "data/val.csv")

IMG_WIDTH = 400
IMG_HEIGHT = 80
ALPHABET = string.ascii_uppercase + string.digits
IDX2CHAR = {idx + 1: char for idx, char in enumerate(ALPHABET)}

# Device Selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"API Device: {device}")

# Load Model
model = CRNN(num_chars=len(ALPHABET))
if os.path.exists(MODEL_PATH):
    try:
        # map_location is important if trained on MPS/GPU but loaded on CPU or vice versa
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        print(f"Modèle chargé avec succès depuis {MODEL_PATH}")
    except Exception as e:
        print(f"Erreur chargement modèle: {e}")
else:
    print(f"ATTENTION: Modèle non trouvé à {MODEL_PATH}")

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def decode_prediction(preds):
    # preds: [1, TimeSteps, NumClasses]
    preds = preds.argmax(dim=2).detach().cpu().numpy()
    decoded_texts = []
    for p in preds:
        text = ""
        for i in range(len(p)):
            char_idx = int(p[i])
            if char_idx != 0 and (i == 0 or char_idx != int(p[i-1])):
                text += IDX2CHAR.get(char_idx, "")
        decoded_texts.append(text)
    return decoded_texts

@app.get("/test-batch")
def test_batch(n: int = 5):
    if not os.path.exists(VAL_CSV):
        return {"error": "Validation set not found"}
    
    # Check paths
    actual_val_csv = VAL_CSV
    actual_data_dir = DATA_DIR
    
    if not os.path.exists(actual_val_csv):
         # Try fallback paths relative to file
         actual_val_csv = "../data/val.csv"
         actual_data_dir = "../data/images"

    try:
        df = pd.read_csv(actual_val_csv)
    except:
        return {"error": "Could not read CSV"}

    if len(df) == 0:
        return {"error": "Validation set empty"}
    
    # Sample n rows
    n = min(n, len(df))
    random_rows = df.sample(n)
    
    results = []
    
    for _, row in random_rows.iterrows():
        img_name = row['filename']
        true_label = row['Label']
        img_path = os.path.join(actual_data_dir, img_name)
        
        if not os.path.exists(img_path):
            continue
            
        try:
            image = Image.open(img_path).convert("L")
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
            prediction = decode_prediction(output)[0]
            
            # Encode image to base64
            with open(img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                
            results.append({
                "image": f"data:image/png;base64,{encoded_string}",
                "true_label": true_label,
                "prediction": prediction
            })
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

    return results

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    print(f"Received file: {file.filename}, Size: {len(image_data)} bytes")
    print(f"First 10 bytes: {image_data[:10]}")
    image = Image.open(io.BytesIO(image_data)).convert("L")
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
    
    text = decode_prediction(output)[0]
    return {"prediction": text}

@app.get("/test-sample")
def get_test_sample():
    if not os.path.exists(VAL_CSV):
        return {"error": "Validation set not found"}
    
    # Check paths
    actual_val_csv = VAL_CSV
    actual_data_dir = DATA_DIR
    
    if not os.path.exists(actual_val_csv):
         # Try fallback paths relative to file
         actual_val_csv = "../data/val.csv"
         actual_data_dir = "../data/images"

    try:
        df = pd.read_csv(actual_val_csv)
    except:
        return {"error": "Could not read CSV"}

    if len(df) == 0:
        return {"error": "Validation set empty"}
    
    random_row = df.sample(1).iloc[0]
    img_name = random_row['filename']
    true_label = random_row['Label']
    img_path = os.path.join(actual_data_dir, img_name)
    
    if not os.path.exists(img_path):
        return {"error": f"Image {img_name} not found at {img_path}"}
    
    # Predict on this sample
    try:
        image = Image.open(img_path).convert("L")
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
        prediction = decode_prediction(output)[0]
    except Exception as e:
        prediction = f"Error: {e}"

    return FileResponse(img_path, headers={
        "X-True-Label": true_label,
        "X-Prediction": prediction
    })

@app.post("/generate-custom")
async def generate_custom(text: str = Form(...)):
    text = text.upper()
    text = ''.join([c for c in text if c in ALPHABET])
    if not text: return {"error": "Invalid text"}

    image_generator = ImageCaptcha(width=IMG_WIDTH, height=IMG_HEIGHT)
    data = image_generator.generate(text)
    image = Image.open(data)
    
    img_gray = image.convert("L")
    img_tensor = transform(img_gray).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
    prediction = decode_prediction(output)[0]
    
    data.seek(0)
    return StreamingResponse(data, media_type="image/png", headers={
        "X-True-Label": text,
        "X-Prediction": prediction
    })

@app.get("/")
def read_root():
    return {"message": "Captcha Solver API Ready (CRNN + CTC)"}