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
import time
from pydub import AudioSegment

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cat Meow Emotion Prediction API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cpu")
CATEGORIES = ["affection", "angry", "defensive", "feed_me", "happy", "hunt", "in_heat", "mother_call", "pain", "wants_attention", "back_off"]

CLASS_MAPPING_INFO = {
    "wants_attention": ["wants_out", "wants_play", "greeting"],
    "feed_me": ["hungry"],
    "back_off": ["leave_me_alone", "warning"],
    "affection": ["affection"], "angry": ["angry"], "defensive": ["defensive"],
    "happy": ["happy"], "hunt": ["hunt"], "in_heat": ["in_heat"],
    "mother_call": ["mother_call"], "pain": ["pain"],
}

audio_models: Dict[str, torch.nn.Module] = {}

def load_models() -> None:
    logger.info("Loading ML models...")
    try:
        # Master Specialist
        m_spec = timm.create_model("efficientformerv2_s0", num_classes=len(CATEGORIES)).to(DEVICE)
        # Modification du premier bloc de conv pour accepter 4 canaux
        for name, module in m_spec.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                new_conv = torch.nn.Conv2d(4, module.out_channels, kernel_size=3, stride=2, padding=1, bias=True).to(DEVICE)
                parts = name.split(".")
                parent = m_spec
                for part in parts[:-1]: parent = getattr(parent, part)
                setattr(parent, parts[-1], new_conv)
                break
        
        m_spec.load_state_dict(torch.load("best_specialist_modle_eff.pth", map_location=DEVICE), strict=False)
        audio_models["master"] = m_spec.eval()

        # Student
        m_student = timm.create_model("efficientformerv2_s0", num_classes=len(CATEGORIES)).to(DEVICE)
        checkpoint = torch.load("best_student.pth", map_location=DEVICE)
        m_student.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        audio_models["student"] = m_student.eval()
        
        logger.info("✓ All models loaded successfully")
    except Exception as e:
        logger.error(f"FATAL: Model loading failed: {e}")

@app.on_event("startup")
async def startup_event():
    load_models()

def normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min() + 1e-6)

def process_audio(audio_path: str) -> Dict:
    start_time = time.time()
    
    # Chargement (sr=16000 match avec la conversion Pydub)
    waveform, _ = librosa.load(audio_path, sr=16000)
    
    target_length = 80000
    waveform = np.pad(waveform, (0, max(0, target_length - len(waveform))))[:target_length]

    # ---- Master Specialist Path ----
    mel_128 = librosa.power_to_db(librosa.feature.melspectrogram(y=waveform, sr=16000, n_mels=128), ref=np.max)
    mel_img_128 = cv2.resize((mel_128 + 40) / 40, (224, 224))
    
    zcr = cv2.resize(librosa.feature.zero_crossing_rate(y=waveform), (224, 224))
    centroid = cv2.resize(librosa.feature.spectral_centroid(y=waveform, sr=16000), (224, 224))
    delta_centroid = cv2.resize(librosa.feature.delta(centroid, axis=-1), (224, 224))

    x_master = torch.from_numpy(np.stack([mel_img_128, normalize(zcr), normalize(centroid), normalize(delta_centroid)], axis=0)).float().unsqueeze(0).to(DEVICE)

    # ---- Student Path ----
    mel_192 = librosa.power_to_db(librosa.feature.melspectrogram(y=waveform, sr=16000, n_mels=192, fmax=8000), ref=np.max)
    mel_img_192 = cv2.resize((mel_192 + 40) / 40, (224, 224))
    x_student = torch.from_numpy(np.stack([mel_img_192] * 3, axis=0)).float().unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        p_master = F.softmax(audio_models["master"](x_master), dim=1).cpu().numpy()[0]
        p_student = F.softmax(audio_models["student"](x_student), dim=1).cpu().numpy()[0]

    probs = np.exp(0.6 * np.log(p_master + 1e-7) + 0.4 * np.log(p_student + 1e-7))
    probs /= probs.sum()

    top_idx = probs.argsort()[-3:][::-1]
    
    logger.info(f"Inference completed in {time.time() - start_time:.2f}s")
    
    return {
        "success": True,
        "top_3": [{"class": CATEGORIES[i], "score": float(probs[i]), "mapped_from": CLASS_MAPPING_INFO[CATEGORIES[i]]} for i in top_idx],
        "top_prediction": {"class": CATEGORIES[top_idx[0]], "score": float(probs[top_idx[0]])}
    }

@app.post("/predict")
async def predict(audio_file: UploadFile = File(...)):
    if not audio_models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    tmp_path = None
    wav_path = None
    
    try:
        # 1. Sauvegarde du fichier uploadé
        ext = os.path.splitext(audio_file.filename)[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await audio_file.read())
            tmp_path = tmp.name

        # 2. Conversion forcée en WAV 16kHz Mono via Pydub
        # Cela règle définitivement les warnings Librosa et accélère le chargement
        wav_path = tmp_path + ".wav"
        audio = AudioSegment.from_file(tmp_path)
        audio.set_frame_rate(16000).set_channels(1).export(wav_path, format="wav")

        return process_audio(wav_path)

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return {"success": False, "error": str(e)}
    finally:
        # Nettoyage
        for p in [tmp_path, wav_path]:
            if p and os.path.exists(p): os.remove(p)
