#!/usr/bin/env python3
"""
Script to download and cache SpeechBrain ECAPA-TDNN model locally
This ensures the application can run without internet access after initial setup
"""

import os
import sys
import torch
from pathlib import Path
from speechbrain.inference import EncoderClassifier

def download_ecapa_model():
    """Download and cache ECAPA-TDNN model"""
    
    # Add parent directory to path to import from src
    sys.path.append(str(Path(__file__).parent.parent))
    
    model_dir = Path("models/speechbrain_ecapa")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading ECAPA-TDNN model from SpeechBrain...")
    
    try:
        # Download the model - this will cache it locally
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(model_dir),
            run_opts={"device": "cpu"}  # Download on CPU first
        )
        
        print(f"✓ Model downloaded successfully to {model_dir}")
        print(f"✓ Model files: {list(model_dir.rglob('*'))}")
        
        # Test the model to ensure it works
        print("Testing model...")
        dummy_input = torch.randn(1, 16000)  # 1 second of audio at 16kHz
        embeddings = model.encode_batch(dummy_input)
        print(f"✓ Model test successful! Embedding shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False

if __name__ == "__main__":
    success = download_ecapa_model()
    sys.exit(0 if success else 1) 