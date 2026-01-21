from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import timm
import librosa
import numpy as np
import cv2
import os
import tempfile
from typing import Dict
import logging

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Cat Meow Emotion Prediction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ À restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
DEVICE = torch.device("cpu")
logger.info(f"Using device: {DEVICE}")

# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------
CATEGORIES = [
    "affection",
    "angry",
    "defensive",
    "feed_me",
    "happy",
    "hunt",
    "in_heat",
    "mother_call",
    "pain",
    "wants_attention",
    "back_off",
]

CLASS_MAPPING_INFO = {
    "wants_attention": ["wants_out", "wants_play", "greeting"],
    "feed_me": ["hungry"],
    "back_off": ["leave_me_alone", "warning"],
    "affection": ["affection"],
    "angry": ["angry"],
    "defensive": ["defensive"],
    "happy": ["happy"],
    "hunt": ["hunt"],
    "in_heat": ["in_heat"],
    "mother_call": ["mother_call"],
    "pain": ["pain"],
}

# -----------------------------------------------------------------------------
# Models container
# -----------------------------------------------------------------------------
audio_models: Dict[str, torch.nn.Module] = {}

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_models() -> None:
    logger.info("Loading ML models...")

    # ---- Master Specialist ----------------------------------------------------
    logger.info("Loading Master Specialist model...")
    m_spec = timm.create_model(
        "efficientformerv2_s0",
        num_classes=len(CATEGORIES)
    ).to(DEVICE)

    for name, module in m_spec.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            new_conv = torch.nn.Conv2d(
                4,
                module.out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ).to(DEVICE)

            parts = name.split(".")
            parent = m_spec
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_conv)
            break

    # Assurez-vous que le fichier .pth est présent
    m_spec.load_state_dict(
        torch.load("best_specialist_modle_eff.pth", map_location=DEVICE),
        strict=False,
    )
    audio_models["master"] = m_spec.eval()
    logger.info("✓ Master Specialist loaded")

    # ---- Student --------------------------------------------------------------
    logger.info("Loading Student model...")
    m_student = timm.create_model(
        "efficientformerv2_s0",
        num_classes=len(CATEGORIES)
    ).to(DEVICE)

    checkpoint = torch.load("best_student.pth", map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    m_student.load_state_dict(state_dict)

    audio_models["student"] = m_student.eval()
    logger.info("✓ Student loaded")

    logger.info("All models loaded successfully")

# -----------------------------------------------------------------------------
# FastAPI lifecycle
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    load_models()

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": len(audio_models) == 2,
        "device": str(DEVICE),
    }

# -----------------------------------------------------------------------------
# Info endpoints
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Cat Meow Emotion Prediction API",
        "version": "1.0.0",
    }

@app.get("/categories")
async def categories():
    return {
        "categories": CATEGORIES,
        "count": len(CATEGORIES),
        "mapping_info": CLASS_MAPPING_INFO,
    }

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min() + 1e-6)

# -----------------------------------------------------------------------------
# Audio processing & inference
# -----------------------------------------------------------------------------
def process_audio(audio_path: str) -> Dict:
    # Chargement audio
    waveform, _ = librosa.load(audio_path, sr=16000, duration=5.0)

    target_length = 80000
    if len(waveform) < target_length:
        waveform = np.pad(waveform, (0, target_length - len(waveform)))
    else:
        waveform = waveform[:target_length]

    # ---- Master (Multi-channel Input) -----------------------------------------
    # Correction : y=waveform pour melspectrogram
    mel_128 = librosa.power_to_db(
        librosa.feature.melspectrogram(y=waveform, sr=16000, n_mels=128),
        ref=np.max,
    )
    mel_img_128 = cv2.resize((mel_128 + 40) / 40, (224, 224))

    # Correction : y=waveform pour zero_crossing_rate
    zcr = cv2.resize(librosa.feature.zero_crossing_rate(y=waveform), (224, 224))
    
    # Déjà correct dans votre snippet, mais vérifié
    centroid = cv2.resize(
        librosa.feature.spectral_centroid(y=waveform, sr=16000), (224, 224)
    )
    
    # Calcul du delta sur le centroïde
    delta_centroid = cv2.resize(
        librosa.feature.delta(centroid, axis=-1), (224, 224)
    )

    x_master = torch.from_numpy(
        np.stack(
            [
                mel_img_128,
                normalize(zcr),
                normalize(centroid),
                normalize(delta_centroid),
            ],
            axis=0,
        )
    ).float().unsqueeze(0).to(DEVICE)

    # ---- Student (RGB-like Spectrogram) ---------------------------------------
    # Correction : y=waveform pour melspectrogram
    mel_192 = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=waveform, sr=16000, n_mels=192, fmax=8000
        ),
        ref=np.max,
    )
    mel_img_192 = cv2.resize((mel_192 + 40) / 40, (224, 224))

    # Le modèle attend 3 canaux (RGB), on stack l'image 3 fois
    x_student = torch.from_numpy(
        np.stack([mel_img_192] * 3, axis=0)
    ).float().unsqueeze(0).to(DEVICE)

    # Inférence
    with torch.no_grad():
        p_master = F.softmax(audio_models["master"](x_master), dim=1).cpu().numpy()[0]
        p_student = F.softmax(audio_models["student"](x_student), dim=1).cpu().numpy()[0]

    # Fusion des probabilités (Ensemble)
    eps = 1e-7
    log_probs = 0.6 * np.log(p_master + eps) + 0.4 * np.log(p_student + eps)
    probs = np.exp(log_probs)
    probs /= probs.sum()

    top_idx = probs.argsort()[-3:][::-1]

    return {
        "success": True,
        "predictions": {CATEGORIES[i]: float(probs[i]) for i in range(len(CATEGORIES))},
        "top_3": [
            {
                "class": CATEGORIES[i],
                "score": float(probs[i]),
                "mapped_from": CLASS_MAPPING_INFO[CATEGORIES[i]],
            }
            for i in top_idx
        ],
        "top_prediction": {
            "class": CATEGORIES[top_idx[0]],
            "score": float(probs[top_idx[0]]),
        },
    }

# -----------------------------------------------------------------------------
# Prediction endpoint
# -----------------------------------------------------------------------------
@app.post("/predict")
async def predict(audio_file: UploadFile = File(...)):
    if len(audio_models) != 2:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Gestion de l'extension de fichier temporaire
    suffix = os.path.splitext(audio_file.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio_file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = process_audio(tmp_path)
        return result
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
