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
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Cat Meow Emotion Prediction API", version="1.0.0")

# CORS configuration - Allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device configuration
DEVICE = torch.device("cpu")  # Use CPU for Docker deployment
logger.info(f"Using device: {DEVICE}")

# 11 fused categories (after mapping)
CATEGORIES = [
    'affection',
    'angry', 
    'defensive',
    'feed_me',        # hungry + greeting merged
    'happy',
    'hunt',
    'in_heat',
    'mother_call',
    'pain',
    'wants_attention',  # wants_out + wants_play + greeting merged
    'back_off'        # leave_me_alone + warning merged
]

# Mapping information for reference (11 -> 14 classes context)
CLASS_MAPPING_INFO = {
    'wants_attention': ['wants_out', 'wants_play', 'greeting'],
    'feed_me': ['hungry'],
    'back_off': ['leave_me_alone', 'warning'],
    # Unmapped classes stay the same
    'affection': ['affection'],
    'angry': ['angry'],
    'defensive': ['defensive'],
    'happy': ['happy'],
    'hunt': ['hunt'],
    'in_heat': ['in_heat'],
    'mother_call': ['mother_call'],
    'pain': ['pain']
}

# Global variable to store loaded models
audio_models = {}

def load_models():
    """Load both ML models at startup"""
    try:
        logger.info("Loading ML models...")
        
        # 1. Master Specialist (Expert 4ch - 128 Mels)
        logger.info("Loading Master Specialist model...")
        m_spec = timm.create_model("efficientformerv2_s0", num_classes=len(CATEGORIES)).to(DEVICE)
        
        # Modify first conv layer to accept 4 channels
        for name, module in m_spec.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                new_conv = torch.nn.Conv2d(4, module.out_channels, 3, 2, 1, bias=True).to(DEVICE)
                parts = name.split('.')
                parent = m_spec
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_conv)
                break
        
        m_spec.load_state_dict(torch.load("best_specialist_modle_eff.pth", map_location=DEVICE), strict=False)
        audio_models['master'] = m_spec.eval()
        logger.info("✓ Master Specialist model loaded")
        
        # 2. Student V3 (3ch - 192 Mels)
        logger.info("Loading Student model...")
        m_student = timm.create_model("efficientformerv2_s0", num_classes=len(CATEGORIES)).to(DEVICE)
        checkpoint = torch.load("best_student.pth", map_location=DEVICE)
        
        # Support both checkpoint formats
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        m_student.load_state_dict(state_dict)
        audio_models['student'] = m_student.eval()
        logger.info("✓ Student model loaded")
        
        logger.info("All models loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load models when the API starts"""
    load_models()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(audio_models) == 2,
        "device": str(DEVICE)
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cat Meow Emotion Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict cat emotion from audio file",
            "/health": "GET - Health check",
            "/categories": "GET - List of emotion categories"
        }
    }

@app.get("/categories")
async def get_categories():
    """Get list of emotion categories"""
    return {
        "categories": CATEGORIES,
        "count": len(CATEGORIES),
        "mapping_info": CLASS_MAPPING_INFO
    }

def normalize(x):
    """Normalize array to [0, 1] range"""
    return (x - x.min()) / (x.max() - x.min() + 1e-6)

def process_audio(audio_path: str) -> Dict:
    """
    Process audio file and run inference with both models
    
    Args:
        audio_path: Path to audio file (.webm, .wav, etc.)
        
    Returns:
        Dictionary with predictions
    """
    try:
        # Load audio file (max 5 seconds)
        waveform, sr = librosa.load(audio_path, sr=16000, duration=5.0)
        
        # Pad or trim to exactly 5 seconds (80000 samples at 16kHz)
        target_length = 80000
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        else:
            waveform = waveform[:target_length]
        
        # ===== MASTER MODEL (4 channels: Mel + ZCR + Centroid + Delta) =====
        # 1. Mel spectrogram (128 mels)
        mel_128 = librosa.power_to_db(
            librosa.feature.melspectrogram(y=waveform, sr=16000, n_mels=128),
            ref=np.max
        )
        mel_img_128 = cv2.resize((mel_128 + 40) / 40, (224, 224))
        
        # 2. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(waveform)
        zcr_resized = cv2.resize(zcr, (224, 224))
        
        # 3. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=waveform, sr=16000)
        centroid_resized = cv2.resize(centroid, (224, 224))
        
        # 4. Delta of Centroid
        delta_centroid = librosa.feature.delta(centroid)
        delta_resized = cv2.resize(delta_centroid, (224, 224))
        
        # Stack 4 channels and create tensor
        x_master = torch.from_numpy(
            np.stack([
                mel_img_128,
                normalize(zcr_resized),
                normalize(centroid_resized),
                normalize(delta_resized)
            ], axis=0)
        ).float().unsqueeze(0).to(DEVICE)
        
        # ===== STUDENT MODEL (3 channels: Mel-192 x3) =====
        # Mel spectrogram (192 mels, max freq 8kHz)
        mel_192 = librosa.power_to_db(
            librosa.feature.melspectrogram(y=waveform, sr=16000, n_mels=192, fmax=8000),
            ref=np.max
        )
        mel_img_192 = cv2.resize((mel_192 + 40) / 40, (224, 224))
        
        # Stack 3 identical channels (RGB-like)
        x_student = torch.from_numpy(
            np.stack([mel_img_192] * 3, axis=0)
        ).float().unsqueeze(0).to(DEVICE)
        
        # ===== INFERENCE =====
        with torch.no_grad():
            # Get predictions from both models
            logits_master = audio_models['master'](x_master)
            logits_student = audio_models['student'](x_student)
            
            # Convert to probabilities
            probs_master = F.softmax(logits_master, dim=1).cpu().numpy()[0]
            probs_student = F.softmax(logits_student, dim=1).cpu().numpy()[0]
        
        # ===== ENSEMBLE (60% Master, 40% Student via log-space fusion) =====
        eps = 1e-7
        log_probs_ensemble = (0.60 * np.log(probs_master + eps)) + (0.40 * np.log(probs_student + eps))
        probs_ensemble = np.exp(log_probs_ensemble)
        probs_ensemble = probs_ensemble / probs_ensemble.sum()  # Renormalize
        
        # Create predictions dictionary
        predictions = {
            CATEGORIES[i]: float(probs_ensemble[i])
            for i in range(len(CATEGORIES))
        }
        
        # Get top 3 predictions
        top_indices = np.argsort(probs_ensemble)[-3:][::-1]
        top_3 = [
            {
                "class": CATEGORIES[idx],
                "score": float(probs_ensemble[idx]),
                "mapped_from": CLASS_MAPPING_INFO[CATEGORIES[idx]]
            }
            for idx in top_indices
        ]
        
        return {
            "success": True,
            "predictions": predictions,
            "top_3": top_3,
            "top_prediction": {
                "class": CATEGORIES[top_indices[0]],
                "score": float(probs_ensemble[top_indices[0]])
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise

@app.post("/predict")
async def predict_emotion(audio_file: UploadFile = File(...)):
    """
    Predict cat emotion from audio file
    
    Args:
        audio_file: Audio file (.webm, .wav, .mp3, etc.)
        
    Returns:
        JSON with predictions and top 3 emotions
    """
    if not audio_models:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    # Create temporary file to store uploaded audio
    temp_audio = None
    try:
        # Create temp file with same extension as uploaded file
        suffix = os.path.splitext(audio_file.filename)[1] or '.webm'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_audio = tmp.name
            content = await audio_file.read()
            tmp.write(content)
        
        logger.info(f"Processing audio file: {audio_file.filename}")
        
        # Process audio and get predictions
        result = process_audio(temp_audio)
        
        logger.info(f"Prediction complete: {result['top_prediction']['class']} ({result['top_prediction']['score']:.2%})")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
    finally:
        # Clean up temporary file
        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)