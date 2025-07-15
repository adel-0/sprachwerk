"""
Main application class for Offline Whisper + Speaker Diarization
Coordinates all modules and handles mode selection
"""

import logging
import sys
import time
import signal
import threading
from pathlib import Path
from datetime import datetime
from colorama import init, Fore, Style
import numpy as np

# Suppress warnings and logging from speechbrain as early as possible
from src.utils.warning_suppressor import setup_logging_suppressions
setup_logging_suppressions()

# Initialize colorama for colored output
init()

from src.core.config import CONFIG, LOGGING_CONFIG, set_mode
from src.audio.capture import AudioCapture
from src.audio.system_audio_capture import SystemAudioCapture
from src.processing.transcription import WhisperTranscriber
from src.processing.diarization import SpeakerDiarizer
from src.processing.alignment import TranscriptionAligner
from src.utils.output_formatter import OutputFormatter
from src.utils.model_manager import ModelManager
from src.core.processing_modes import BatchProcessingMode, RealTimeProcessingMode

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class TranscriptionApp:
    def __init__(self, enable_signal_handlers=True):
        # Initialize model manager and ensure models are available
        self.model_manager = ModelManager()
        self.model_manager.ensure_all_models()
        
        # Initialize core components
        self.audio_capture = AudioCapture()
        self.system_audio_capture = SystemAudioCapture()
        self.transcriber = WhisperTranscriber()
        self.diarizer = SpeakerDiarizer()
        self.aligner = TranscriptionAligner()
        self.formatter = OutputFormatter()
        
        # Initialize processing modes
        self.batch_mode = BatchProcessingMode(
            self.transcriber, self.diarizer, self.aligner, self.formatter
        )
        self.realtime_mode = RealTimeProcessingMode(
            self.transcriber, self.diarizer, self.aligner, self.formatter
        )
        
        # State management
        self.is_running = False
        self.stop_requested = False
        self._cleanup_called = False
        
        if enable_signal_handlers:
            self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.debug("Signal handlers set up successfully")
        except (ValueError, Exception) as e:
            logger.debug(f"Could not set up signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_processing()
        sys.exit(0)
    
    def models_loaded(self):
        """Check if models are already loaded"""
        return (hasattr(self.transcriber, 'is_loaded') and self.transcriber.is_loaded and
                hasattr(self.diarizer, 'is_loaded') and self.diarizer.is_loaded)
    
    def initialize_models(self):
        """Load all required models"""
        print(f"{Fore.CYAN}Initializing models...{Style.RESET_ALL}")
        
        try:
            print(f"{Fore.YELLOW}Loading Whisper model ({CONFIG['whisper_model']})...{Style.RESET_ALL}")
            self.transcriber.load_model()
            
            print(f"{Fore.YELLOW}Loading speaker diarization model...{Style.RESET_ALL}")
            self.diarizer.load_model()
            
            print(f"{Fore.GREEN}✓ All models loaded successfully!{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}✗ Failed to load models: {e}{Style.RESET_ALL}")
            logger.error(f"Model loading failed: {e}")
            return False
    

    
    def run_real_time_mode(self):
        """Run real-time transcription mode"""
        return self.realtime_mode.run(self.audio_capture, self.system_audio_capture)
    


    def run_batch_mode(self, duration=None, input_file=None):
        """Run batch processing mode"""
        return self.batch_mode.run(self.audio_capture, self.system_audio_capture, duration, input_file)
    
    def stop_processing(self):
        """Stop all processing and cleanup"""
        if self._cleanup_called:
            return
        
        self._cleanup_called = True
        logger.info("Stopping all processing...")
        self.is_running = False
        
        # Stop processing threads
        if hasattr(self.transcriber, 'stop_real_time_processing'):
            self.transcriber.stop_real_time_processing()
        
        if hasattr(self.diarizer, 'stop_real_time_processing'):
            self.diarizer.stop_real_time_processing()
        
        # Cleanup resources
        for component in [self.transcriber, self.diarizer, self.audio_capture, self.system_audio_capture]:
            if component:
                component.cleanup()
        
        print(f"{Fore.GREEN}✓ Processing stopped and resources cleaned up{Style.RESET_ALL}") 