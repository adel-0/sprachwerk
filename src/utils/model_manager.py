"""
Model Manager for centralized model download and initialization
Handles Whisper and SpeechBrain ECAPA-TDNN model setup
"""

import logging
import torch
from pathlib import Path
from speechbrain.inference import EncoderClassifier
from faster_whisper import WhisperModel
from colorama import Fore, Style

from src.core.config import CONFIG

logger = logging.getLogger(__name__)

class ModelManager:
    """Centralized model download and initialization manager"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def _create_model_dir(self, model_name: str) -> Path:
        """Create and return model directory path"""
        model_dir = self.models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def _handle_cuda_fallback(self, device: str, compute_type: str = None):
        """Handle CUDA fallback to CPU if not available"""
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
            if compute_type:
                compute_type = 'float32'
        return device, compute_type
    
    def ensure_speechbrain_ecapa_download(self):
        """Download and test SpeechBrain ECAPA-TDNN model"""
        model_dir = self._create_model_dir("speechbrain_ecapa")
        
        try:
            print(f"{Fore.YELLOW}Ensuring SpeechBrain ECAPA-TDNN model is available...{Style.RESET_ALL}")
            
            # Download the model if not already cached
            model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(model_dir),
                run_opts={"device": "cpu"}
            )
            
            # Test the model to ensure it works
            dummy_input = torch.randn(1, 48000)
            _ = model.encode_batch(dummy_input)
            del model
            
            print(f"{Fore.GREEN}✓ SpeechBrain ECAPA-TDNN model ready{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}✗ Failed to download or test SpeechBrain ECAPA-TDNN model: {e}{Style.RESET_ALL}")
            logger.error(f"SpeechBrain ECAPA download failed: {e}")
            return False
    
    def ensure_whisper_model_download(self):
        """Download and test Whisper model"""
        model_dir = self._create_model_dir("whisper")
        
        try:
            print(f"{Fore.YELLOW}Ensuring Whisper model ({CONFIG['whisper_model']}) is available...{Style.RESET_ALL}")
            
            device = CONFIG['whisper_device']
            compute_type = CONFIG['whisper_compute_type']
            
            # Fall back to CPU if CUDA not available
            device, compute_type = self._handle_cuda_fallback(device, compute_type)
            
            model = WhisperModel(
                CONFIG['whisper_model'],
                device=device,
                compute_type=compute_type,
                download_root=str(model_dir)
            )
            
            # Test the model to ensure it works
            del model
            
            print(f"{Fore.GREEN}✓ Whisper model ready{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}✗ Failed to download or test Whisper model: {e}{Style.RESET_ALL}")
            logger.error(f"Whisper model download failed: {e}")
            return False
    
    def ensure_all_models(self):
        """Download and verify all required models"""
        print(f"{Fore.CYAN}Initializing models...{Style.RESET_ALL}")
        
        speechbrain_ok = self.ensure_speechbrain_ecapa_download()
        whisper_ok = self.ensure_whisper_model_download()
        
        if speechbrain_ok and whisper_ok:
            print(f"{Fore.GREEN}✓ All models initialized successfully!{Style.RESET_ALL}")
            return True
        else:
            print(f"{Fore.RED}✗ Some models failed to initialize{Style.RESET_ALL}")
            return False 