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
from src.core.config import get_models_dir

logger = logging.getLogger(__name__)

class ModelManager:
    """Centralized model download and initialization manager"""
    
    def __init__(self):
        self.models_dir = Path(get_models_dir())
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_device_config(self):
        """Get device configuration with automatic CUDA fallback"""
        device = CONFIG['whisper_device']
        compute_type = CONFIG['whisper_compute_type']
        
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return 'cpu', 'float32'
        return device, compute_type
    
    def ensure_speechbrain_ecapa_download(self):
        """Download SpeechBrain ECAPA-TDNN model"""
        try:
            print(f"{Fore.YELLOW}Downloading SpeechBrain ECAPA-TDNN model...{Style.RESET_ALL}")
            
            model_dir = self.models_dir / "speechbrain_ecapa"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(model_dir),
                run_opts={"device": "cpu"}
            )
            
            print(f"{Fore.GREEN}✓ SpeechBrain ECAPA-TDNN model ready{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}✗ Failed to download SpeechBrain ECAPA-TDNN model: {e}{Style.RESET_ALL}")
            logger.error(f"SpeechBrain ECAPA download failed: {e}")
            return False
    
    def ensure_whisper_model_download(self):
        """Download Whisper model"""
        try:
            print(f"{Fore.YELLOW}Downloading Whisper model ({CONFIG['whisper_model']})...{Style.RESET_ALL}")
            
            model_dir = self.models_dir / "whisper"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            device, compute_type = self._get_device_config()
            
            WhisperModel(
                CONFIG['whisper_model'],
                device=device,
                compute_type=compute_type,
                download_root=str(model_dir)
            )
            
            print(f"{Fore.GREEN}✓ Whisper model ready{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}✗ Failed to download Whisper model: {e}{Style.RESET_ALL}")
            logger.error(f"Whisper model download failed: {e}")
            return False
    
    def ensure_all_models(self):
        """Download all required models"""
        print(f"{Fore.CYAN}Initializing models...{Style.RESET_ALL}")
        
        success = (
            self.ensure_speechbrain_ecapa_download() and 
            self.ensure_whisper_model_download()
        )
        
        if success:
            print(f"{Fore.GREEN}✓ All models initialized successfully!{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}✗ Some models failed to initialize{Style.RESET_ALL}")
        
        return success 