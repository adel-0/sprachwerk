"""
Base audio capture class with shared functionality for microphone and system audio capture.
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
    def __init__(self, sample_rate: Optional[int] = None):
        config = get_typed_config()
        self.sample_rate = sample_rate or config.audio.sample_rate
        self.channels = config.audio.channels
        self.is_recording = False
        self.audio_queue = queue.Queue(maxsize=config.processing.max_queue_size)
        self.recording_thread = None
        self.audio_buffer = []
        self.preprocessor = AudioPreprocessor(self.sample_rate)
        self.enable_preprocessing = config.audio_preprocessing.enable_audio_preprocessing
        self.device_manager = AudioDeviceManager(self.sample_rate)
        self._cleanup_called = False
        logger.info(f"{self.__class__.__name__} initialized")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def list_audio_devices(self) -> List[Dict[str, Any]]:
        return self.device_manager.list_all_devices()

    def set_preprocessing_enabled(self, enabled: bool):
        self.enable_preprocessing = enabled
        logger.info(f"Audio preprocessing {'enabled' if enabled else 'disabled'}")

    def configure_preprocessing(self, **kwargs):
        self.preprocessor.set_parameters(**kwargs)

    def reset_noise_profile(self):
        self.preprocessor.reset_noise_profile()

    def update_noise_profile(self, noise_audio):
        self.preprocessor.update_noise_profile(noise_audio)

    def use_minimal_processing(self, show_info=True):
        if show_info:
            print("🎵 Audio Processing: Minimal pre-processing enabled\n   • Light noise reduction and normalization\n   • Optimized for real-world audio conditions")
        self.enable_preprocessing = True

    def use_raw_audio_mode(self, show_info=True):
        if show_info:
            print("🎵 Audio Processing: Raw audio mode (no processing)\n   • Direct audio input without any modifications")
        self.enable_preprocessing = False

    def get_processing_mode(self) -> str:
        return "Raw Audio (No Processing)" if not self.enable_preprocessing else "Minimal Pre-processing"

    def _process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        if not self.enable_preprocessing:
            return audio_data
        try:
            return self.preprocessor.enhance_for_whisper(audio_data)
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}")
            return audio_data

    def save_audio(self, audio_data: np.ndarray, filename: str) -> Optional[Path]:
        try:
            import scipy.io.wavfile as wavfile
            from src.core.config import get_outputs_dir
            config = get_typed_config()
            if not filename.endswith('.wav'):
                filename += '.wav'
            # Resolve the output directory path
            output_dir = get_outputs_dir() if config.output.output_directory == 'outputs' else config.output.output_directory
            filepath = Path(output_dir) / filename
            if audio_data.dtype != np.int16:
                if audio_data.dtype in [np.float32, np.float64]:
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
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (
            self.save_audio(processed_audio, f"{base_filename}_processed_{timestamp}"),
            self.save_audio(raw_audio, f"{base_filename}_raw_{timestamp}")
        )

    def cleanup(self):
        if self._cleanup_called:
            return
        self._cleanup_called = True
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.audio_buffer.clear()
        logger.info(f"{self.__class__.__name__} cleanup completed")

    @abstractmethod
    def start_real_time_recording(self):
        pass

    @abstractmethod
    def stop_real_time_recording(self) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def record_batch(self, duration_seconds: float) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        pass 