"""
Immutable configuration system for sprachwerk
Replaces global mutable CONFIG dict with typed, immutable configuration objects
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class DiarizationBackend(Enum):
    """Diarization backend options"""
    PYANNOTE = "pyannote"
    SPEECHBRAIN = "speechbrain"


@dataclass(frozen=True)
class ModelConfig:
    """Model-related configuration"""
    whisper_model: str = 'large-v3-turbo'
    whisper_device: str = 'cuda'
    whisper_compute_type: str = 'float16'
    diarization_device: str = 'cuda'


@dataclass(frozen=True)
class AudioConfig:
    """Audio-related configuration"""
    sample_rate: int = 48000
    channels: int = 1
    audio_format: str = 'wav'
    audio_device_index: Optional[int] = None


@dataclass(frozen=True)
class SpeakerConfig:
    """Speaker diarization configuration"""
    min_speakers: int = 1
    max_speakers: int = 2
    enable_adaptive_speaker_thresholds: bool = False
    base_speaker_similarity_threshold: float = 0.5
    single_speaker_similarity_boost: float = 0.1
    single_speaker_clustering_boost: float = 0.1
    
    # Diarization backend selection
    diarization_backend: DiarizationBackend = DiarizationBackend.SPEECHBRAIN
    
    # SpeechBrain diarization parameters
    window_length: float = 1.5  # seconds
    hop_length: float = 0.75  # seconds
    cluster_threshold: float = 0.3
    min_cluster_size: int = 2
    
    # Clustering algorithm parameters
    clustering_algorithm: str = 'agglomerative'  # 'agglomerative' or 'dbscan'
    clustering_linkage: str = 'ward'  # 'ward', 'complete', 'average', 'single'
    clustering_affinity: str = 'euclidean'  # 'euclidean', 'cosine'
    
    # DBSCAN specific parameters
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2
    
    # Adaptive clustering
    enable_adaptive_clustering: bool = True
    silhouette_threshold: float = 0.3


@dataclass(frozen=True)
class WhisperConfig:
    """Whisper transcription configuration"""
    beam_size: int = 2
    temperature: float = 0.1
    vad_threshold: float = 0.05  # Very low threshold for more permissive VAD
    no_speech_threshold: float = 0.6  # Higher threshold to reduce false positives
    language: Optional[str] = None
    multilingual_segments: bool = True
    language_constraints: Optional[List[str]] = None
    vad_filter: bool = True  # Re-enabled with better parameters
    condition_on_previous_text: bool = False
    compression_ratio_threshold: float = 3.0
    word_timestamps: bool = True
    hallucination_silence_threshold: Optional[float] = None
    patience: Optional[float] = None
    length_penalty: Optional[float] = None


@dataclass(frozen=True)
class ProcessingConfig:
    """Processing-related configuration"""
    chunk_duration: float = 4
    overlap_duration: float = 0.5
    buffer_size: int = 1024
    max_queue_size: int = 10


@dataclass(frozen=True)
class AudioPreprocessingConfig:
    """Audio preprocessing configuration"""
    enable_audio_preprocessing: bool = True
    max_gain_boost_db: float = 2.0
    target_rms_level: float = 0.05
    noise_gate_threshold: float = 0.0001
    highpass_frequency: int = 40
    lowpass_frequency: int = 20000


@dataclass(frozen=True)
class SystemAudioConfig:
    """System audio recording configuration"""
    system_audio_recording_mode: str = 'mic'  # 'system', 'mic', or 'both'
    system_audio_device_index: Optional[int] = None
    system_audio_mic_device_index: Optional[int] = None
    system_audio_gain: float = 0.7
    system_audio_mic_gain: float = 1.0
    system_audio_auto_normalize: bool = True
    system_audio_target_level: float = 0.1
    system_audio_save_separate: bool = False


@dataclass(frozen=True)
class OutputConfig:
    """Output-related configuration"""
    output_format: str = 'txt'
    timestamp_format: str = '%H:%M:%S.%f'
    output_directory: str = 'outputs'
    temp_directory: str = 'temp'


@dataclass(frozen=True)
class UserPreferencesConfig:
    """User preferences configuration"""
    preferred_mode: str = 'realtime'


@dataclass(frozen=True)
class TranscriptionConfig:
    """Complete transcription configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    audio_preprocessing: AudioPreprocessingConfig = field(default_factory=AudioPreprocessingConfig)
    system_audio: SystemAudioConfig = field(default_factory=SystemAudioConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    user_preferences: UserPreferencesConfig = field(default_factory=UserPreferencesConfig)
    
    @property
    def chunk_samples(self) -> int:
        """Computed property for chunk samples"""
        return int(self.processing.chunk_duration * self.audio.sample_rate)
    
    @property
    def overlap_samples(self) -> int:
        """Computed property for overlap samples"""
        return int(self.processing.overlap_duration * self.audio.sample_rate)
    
    def with_mode(self, mode: str) -> 'TranscriptionConfig':
        """Create a new configuration with mode-specific optimizations"""
        from dataclasses import replace
        
        return replace(
            self,
            processing=replace(
                self.processing,
                chunk_duration=4,
                overlap_duration=0.5,
                buffer_size=1024,
                max_queue_size=10
            ),
            whisper=replace(
                self.whisper,
                beam_size=2,
                temperature=0.1,
                vad_threshold=0.25,
                no_speech_threshold=0.3,
                condition_on_previous_text=False,
                compression_ratio_threshold=3.0
            )
        )
    
    def _with_batch_mode(self) -> 'TranscriptionConfig':
        """Create configuration optimized for batch mode"""
        from dataclasses import replace
        
        return replace(
            self,
            processing=replace(
                self.processing,
                chunk_duration=30,
                overlap_duration=3,
                buffer_size=4096,
                max_queue_size=5
            ),
            whisper=replace(
                self.whisper,
                beam_size=8,
                temperature=0.2,
                vad_threshold=0.25,
                no_speech_threshold=0.5,
                condition_on_previous_text=True,
                compression_ratio_threshold=2.0
            )
        )
    
    def with_user_settings(self, user_settings: Dict[str, Any]) -> 'TranscriptionConfig':
        """Create a new configuration with user settings applied"""
        from dataclasses import replace
        
        # Map flat user settings to nested configuration structure
        updates = {}
        
        # Model settings
        model_updates = {}
        if 'whisper_model' in user_settings:
            model_updates['whisper_model'] = user_settings['whisper_model']
        if 'whisper_device' in user_settings:
            model_updates['whisper_device'] = user_settings['whisper_device']
        if 'whisper_compute_type' in user_settings:
            model_updates['whisper_compute_type'] = user_settings['whisper_compute_type']
        if 'diarization_device' in user_settings:
            model_updates['diarization_device'] = user_settings['diarization_device']
        if model_updates:
            updates['model'] = replace(self.model, **model_updates)
        
        # Audio settings
        audio_updates = {}
        if 'sample_rate' in user_settings:
            audio_updates['sample_rate'] = user_settings['sample_rate']
        if 'channels' in user_settings:
            audio_updates['channels'] = user_settings['channels']
        if 'audio_device_index' in user_settings:
            audio_updates['audio_device_index'] = user_settings['audio_device_index']
        if audio_updates:
            updates['audio'] = replace(self.audio, **audio_updates)
        
        # Speaker settings
        speaker_updates = {}
        if 'min_speakers' in user_settings:
            speaker_updates['min_speakers'] = user_settings['min_speakers']
        if 'max_speakers' in user_settings:
            speaker_updates['max_speakers'] = user_settings['max_speakers']
        if 'enable_adaptive_speaker_thresholds' in user_settings:
            speaker_updates['enable_adaptive_speaker_thresholds'] = user_settings['enable_adaptive_speaker_thresholds']
        if 'base_speaker_similarity_threshold' in user_settings:
            speaker_updates['base_speaker_similarity_threshold'] = user_settings['base_speaker_similarity_threshold']
        if 'single_speaker_similarity_boost' in user_settings:
            speaker_updates['single_speaker_similarity_boost'] = user_settings['single_speaker_similarity_boost']
        if 'single_speaker_clustering_boost' in user_settings:
            speaker_updates['single_speaker_clustering_boost'] = user_settings['single_speaker_clustering_boost']
        if speaker_updates:
            updates['speaker'] = replace(self.speaker, **speaker_updates)
        
        # Whisper settings
        whisper_updates = {}
        if 'whisper_language' in user_settings:
            whisper_updates['language'] = user_settings['whisper_language']
        if 'whisper_multilingual_segments' in user_settings:
            whisper_updates['multilingual_segments'] = user_settings['whisper_multilingual_segments']
        if 'whisper_language_constraints' in user_settings:
            whisper_updates['language_constraints'] = user_settings['whisper_language_constraints']
        if whisper_updates:
            updates['whisper'] = replace(self.whisper, **whisper_updates)
        
        # Audio preprocessing settings
        preprocessing_updates = {}
        if 'enable_audio_preprocessing' in user_settings:
            preprocessing_updates['enable_audio_preprocessing'] = user_settings['enable_audio_preprocessing']
        if preprocessing_updates:
            updates['audio_preprocessing'] = replace(self.audio_preprocessing, **preprocessing_updates)
        
        # System audio settings
        system_audio_updates = {}
        for key in ['system_audio_recording_mode', 'system_audio_device_index', 'system_audio_mic_device_index',
                   'system_audio_gain', 'system_audio_mic_gain', 'system_audio_auto_normalize',
                   'system_audio_target_level', 'system_audio_save_separate']:
            if key in user_settings:
                system_audio_updates[key] = user_settings[key]
        if system_audio_updates:
            updates['system_audio'] = replace(self.system_audio, **system_audio_updates)
        
        # User preferences
        prefs_updates = {}
        if 'preferred_mode' in user_settings:
            prefs_updates['preferred_mode'] = user_settings['preferred_mode']
        if prefs_updates:
            updates['user_preferences'] = replace(self.user_preferences, **prefs_updates)
        
        return replace(self, **updates)
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to legacy dictionary format for backward compatibility"""
        result = {}
        
        # Model settings
        result['whisper_model'] = self.model.whisper_model
        result['whisper_device'] = self.model.whisper_device
        result['whisper_compute_type'] = self.model.whisper_compute_type
        result['diarization_device'] = self.model.diarization_device
        
        # Audio settings
        result['sample_rate'] = self.audio.sample_rate
        result['channels'] = self.audio.channels
        result['audio_format'] = self.audio.audio_format
        result['audio_device_index'] = self.audio.audio_device_index
        
        # Speaker settings
        result['min_speakers'] = self.speaker.min_speakers
        result['max_speakers'] = self.speaker.max_speakers
        result['enable_adaptive_speaker_thresholds'] = self.speaker.enable_adaptive_speaker_thresholds
        result['base_speaker_similarity_threshold'] = self.speaker.base_speaker_similarity_threshold
        result['single_speaker_similarity_boost'] = self.speaker.single_speaker_similarity_boost
        result['single_speaker_clustering_boost'] = self.speaker.single_speaker_clustering_boost
        
        # Whisper settings
        result['whisper_beam_size'] = self.whisper.beam_size
        result['whisper_temperature'] = self.whisper.temperature
        result['whisper_vad_threshold'] = self.whisper.vad_threshold
        result['whisper_no_speech_threshold'] = self.whisper.no_speech_threshold
        result['whisper_language'] = self.whisper.language
        result['whisper_multilingual_segments'] = self.whisper.multilingual_segments
        result['whisper_language_constraints'] = self.whisper.language_constraints
        result['whisper_vad_filter'] = self.whisper.vad_filter
        result['whisper_condition_on_previous_text'] = self.whisper.condition_on_previous_text
        result['whisper_compression_ratio_threshold'] = self.whisper.compression_ratio_threshold
        result['whisper_word_timestamps'] = self.whisper.word_timestamps
        result['whisper_hallucination_silence_threshold'] = self.whisper.hallucination_silence_threshold
        result['whisper_patience'] = self.whisper.patience
        result['whisper_length_penalty'] = self.whisper.length_penalty
        
        # Processing settings
        result['chunk_duration'] = self.processing.chunk_duration
        result['overlap_duration'] = self.processing.overlap_duration
        result['buffer_size'] = self.processing.buffer_size
        result['max_queue_size'] = self.processing.max_queue_size
        
        # Audio preprocessing settings
        result['enable_audio_preprocessing'] = self.audio_preprocessing.enable_audio_preprocessing
        result['max_gain_boost_db'] = self.audio_preprocessing.max_gain_boost_db
        result['target_rms_level'] = self.audio_preprocessing.target_rms_level
        result['noise_gate_threshold'] = self.audio_preprocessing.noise_gate_threshold
        result['highpass_frequency'] = self.audio_preprocessing.highpass_frequency
        result['lowpass_frequency'] = self.audio_preprocessing.lowpass_frequency
        
        # System audio settings
        result['system_audio_recording_mode'] = self.system_audio.system_audio_recording_mode
        result['system_audio_device_index'] = self.system_audio.system_audio_device_index
        result['system_audio_mic_device_index'] = self.system_audio.system_audio_mic_device_index
        result['system_audio_gain'] = self.system_audio.system_audio_gain
        result['system_audio_mic_gain'] = self.system_audio.system_audio_mic_gain
        result['system_audio_auto_normalize'] = self.system_audio.system_audio_auto_normalize
        result['system_audio_target_level'] = self.system_audio.system_audio_target_level
        result['system_audio_save_separate'] = self.system_audio.system_audio_save_separate
        
        # Output settings
        result['output_format'] = self.output.output_format
        result['timestamp_format'] = self.output.timestamp_format
        result['output_directory'] = self.output.output_directory
        result['temp_directory'] = self.output.temp_directory
        
        # User preferences
        result['preferred_mode'] = self.user_preferences.preferred_mode
        
        # Computed properties
        result['CHUNK_SAMPLES'] = self.chunk_samples
        result['OVERLAP_SAMPLES'] = self.overlap_samples
        
        return result


class ConfigManager:
    """Manages configuration loading, saving, and mode switching"""
    
    def __init__(self):
        self._user_settings_file = Path('user_settings.json')
        self._persisted_settings = {
            'whisper_language', 'whisper_multilingual_segments', 'whisper_language_constraints',
            'min_speakers', 'max_speakers', 'enable_adaptive_speaker_thresholds',
            'base_speaker_similarity_threshold', 'single_speaker_similarity_boost',
            'single_speaker_clustering_boost', 'audio_device_index', 'enable_audio_preprocessing',
            'preferred_mode', 'system_audio_recording_mode', 'system_audio_device_index',
            'system_audio_mic_device_index', 'system_audio_gain', 'system_audio_mic_gain',
            'system_audio_auto_normalize', 'system_audio_target_level', 'system_audio_save_separate'
        }
    
    def load_config(self, mode: str = 'realtime') -> TranscriptionConfig:
        """Load configuration for specified mode with user settings applied"""
        config = TranscriptionConfig().with_mode(mode)
        user_settings = self._load_user_settings()
        return config.with_user_settings(user_settings) if user_settings else config
    
    def save_user_settings(self, config: TranscriptionConfig) -> bool:
        """Save user settings from configuration"""
        try:
            # Convert to legacy dict and extract persisted settings
            legacy_dict = config.to_legacy_dict()
            user_settings = {k: legacy_dict[k] for k in self._persisted_settings if k in legacy_dict}
            
            # Save to file with nice formatting
            with open(self._user_settings_file, 'w', encoding='utf-8') as f:
                json.dump(user_settings, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"⚠️  Warning: Could not save user settings to {self._user_settings_file}: {e}")
            return False
    
    def _load_user_settings(self) -> Optional[Dict[str, Any]]:
        """Load user settings from file"""
        if not self._user_settings_file.exists():
            return None
        
        try:
            with open(self._user_settings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None


# Global configuration manager instance
config_manager = ConfigManager()

# Create directories
OUTPUT_DIR = Path('outputs')
TEMP_DIR = Path('temp')
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Model paths
PYANNOTE_MODEL_PATH = "pyannote/speaker-diarization-3.1"
HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# Backward compatibility: provide a global CONFIG that behaves like the old system
# This will be populated by the first call to get_config()
CONFIG = None


def get_config(mode: str = 'realtime') -> Dict[str, Any]:
    """Get configuration for specified mode (backward compatibility)"""
    global CONFIG
    config = config_manager.load_config(mode)
    CONFIG = config.to_legacy_dict()
    return CONFIG


def set_mode(mode: str = 'batch') -> Dict[str, Any]:
    """Switch configuration mode (backward compatibility)"""
    global CONFIG
    config = config_manager.load_config(mode)
    CONFIG = config.to_legacy_dict()
    return CONFIG


def get_typed_config(mode: str = 'realtime') -> TranscriptionConfig:
    """Get typed configuration object (new API)"""
    return config_manager.load_config(mode)


def save_config(config: TranscriptionConfig) -> bool:
    """Save typed configuration (new API)"""
    return config_manager.save_user_settings(config)


# Backward compatibility functions for individual setting updates
def save_user_setting(key: str, value: Any) -> None:
    """Save a single user setting (backward compatibility)"""
    global CONFIG
    # Update the global CONFIG dict
    CONFIG[key] = value


def save_user_settings_to_file() -> bool:
    """Save user settings to file (backward compatibility)"""
    global CONFIG
    if CONFIG is None:
        return False
    
    try:
        # Create a temporary TranscriptionConfig from the current CONFIG dict
        # and save it using the new system
        current_config = config_manager.load_config()
        current_config = current_config.with_user_settings(CONFIG)
        return config_manager.save_user_settings(current_config)
    except Exception:
        return False


def update_setting(key: str, value: Any) -> None:
    """Update a single configuration setting (backward compatibility)"""
    global CONFIG
    CONFIG[key] = value


def bulk_update(**kwargs) -> None:
    """Helper to bulk update CONFIG and save settings (backward compatibility)"""
    global CONFIG
    CONFIG.update(kwargs)
    for key, value in kwargs.items():
        CONFIG[key] = value
    save_user_settings_to_file()


# Initialize with default configuration
CONFIG = get_config('realtime') 