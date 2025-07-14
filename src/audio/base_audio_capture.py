"""
Base audio capture class with shared functionality
Provides common functionality for both microphone and system audio capture
"""

import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List
from pathlib import Path
import numpy as np

from src.core.config import get_typed_config
from src.audio.preprocessing import AudioPreprocessor
from src.utils.audio_device_manager import AudioDeviceManager

logger = logging.getLogger(__name__)


class BaseAudioCapture(ABC):
    """Base class for audio capture with shared functionality and context manager support"""
    
    def __init__(self, sample_rate: Optional[int] = None):
        # Get typed configuration
        config = get_typed_config()
        
        # Use provided sample rate or default from config
        self.sample_rate = sample_rate or config.audio.sample_rate
        self.channels = config.audio.channels
        
        # State management
        self.is_recording = False
        self.audio_queue = queue.Queue(maxsize=config.processing.max_queue_size)
        self.recording_thread = None
        self.audio_buffer = []
        
        # Audio preprocessing
        self.preprocessor = AudioPreprocessor(self.sample_rate)
        self.enable_preprocessing = config.audio_preprocessing.enable_audio_preprocessing
        
        # Audio device manager
        self.device_manager = AudioDeviceManager(self.sample_rate)
        
        # Cleanup tracking
        self._cleanup_called = False
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup"""
        self.cleanup()
    
    def list_audio_devices(self) -> List[Dict[str, Any]]:
        """List available audio input devices"""
        return self.device_manager.list_all_devices()
    
    def set_preprocessing_enabled(self, enabled: bool):
        """Enable or disable audio preprocessing"""
        self.enable_preprocessing = enabled
        mode = "enabled" if enabled else "disabled"
        logger.info(f"Audio preprocessing {mode}")
    
    def configure_preprocessing(self, **kwargs):
        """Configure preprocessing parameters"""
        if self.preprocessor:
            self.preprocessor.set_parameters(**kwargs)
    
    def reset_noise_profile(self):
        """Reset the noise profile for noise reduction"""
        if self.preprocessor:
            self.preprocessor.reset_noise_profile()
    
    def update_noise_profile(self, noise_audio):
        """Update noise profile with new noise sample"""
        if self.preprocessor:
            self.preprocessor.update_noise_profile(noise_audio)
    
    def use_minimal_processing(self, show_info=True):
        """Configure minimal processing mode for better real-world performance"""
        if show_info:
            print("🎵 Audio Processing: Minimal pre-processing enabled")
            print("   • Light noise reduction and normalization")
            print("   • Optimized for real-world audio conditions")
        self.enable_preprocessing = True
    
    def use_raw_audio_mode(self, show_info=True):
        """Disable all audio processing for raw audio"""
        if show_info:
            print("🎵 Audio Processing: Raw audio mode (no processing)")
            print("   • Direct audio input without any modifications")
        self.enable_preprocessing = False
    
    def get_processing_mode(self) -> str:
        """Get current processing mode description"""
        return "Raw Audio (No Processing)" if not self.enable_preprocessing else "Minimal Pre-processing"
    
    def _process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Process audio data through preprocessing pipeline"""
        if not self.enable_preprocessing or not self.preprocessor:
            return audio_data
        
        try:
            return self.preprocessor.enhance_for_whisper(audio_data)
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}")
            return audio_data
    
    def save_audio(self, audio_data: np.ndarray, filename: str) -> Optional[Path]:
        """Save audio data to file"""
        try:
            from src.core.config import get_typed_config
            import scipy.io.wavfile as wavfile
            
            config = get_typed_config()
            
            # Ensure filename has .wav extension
            if not filename.endswith('.wav'):
                filename += '.wav'
            
            filepath = Path(config.output.output_directory) / filename
            
            # Convert to int16 for saving
            if audio_data.dtype != np.int16:
                if audio_data.dtype in [np.float32, np.float64]:
                    # Convert from float [-1, 1] to int16
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)
            
            wavfile.write(filepath, self.sample_rate, audio_data)
            logger.info(f"Audio saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return None
    
    def save_dual_audio(self, processed_audio: np.ndarray, raw_audio: np.ndarray, base_filename: str) -> tuple[Optional[Path], Optional[Path]]:
        """Save both processed and raw audio versions"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        processed_path = self.save_audio(processed_audio, f"{base_filename}_processed_{timestamp}")
        raw_path = self.save_audio(raw_audio, f"{base_filename}_raw_{timestamp}")
        
        return processed_path, raw_path
    
    def cleanup(self):
        """Clean up resources"""
        if self._cleanup_called:
            return
        
        self._cleanup_called = True
        
        # Stop recording if active
        if self.is_recording:
            self.is_recording = False
        
        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        # Clear audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clear buffers
        self.audio_buffer.clear()
        
        logger.info(f"{self.__class__.__name__} cleanup completed")
    
    # Abstract methods that subclasses must implement
    @abstractmethod
    def start_real_time_recording(self):
        """Start real-time audio recording"""
        pass
    
    @abstractmethod
    def stop_real_time_recording(self) -> Optional[np.ndarray]:
        """Stop real-time recording and return accumulated audio"""
        pass
    
    @abstractmethod
    def record_batch(self, duration_seconds: float) -> Optional[np.ndarray]:
        """Record audio for a specified duration"""
        pass
    
    @abstractmethod
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get next audio chunk from queue"""
        pass 