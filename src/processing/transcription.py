"""
Transcription module using faster-whisper for GPU-accelerated speech-to-text
Provides word-level timestamps for precise alignment with speaker diarization
"""

import logging
import numpy as np
import torch
from faster_whisper import WhisperModel
import threading
import queue
import time
from pathlib import Path

from src.core.config import CONFIG, TEMP_DIR
from src.utils.warning_suppressor import configure_torch_tf32

# Configure TF32 settings
configure_torch_tf32()

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self):
        self.model_size = CONFIG['whisper_model']
        self.device = CONFIG['whisper_device']
        self.compute_type = CONFIG['whisper_compute_type']
        self.sample_rate = CONFIG['sample_rate']
        
        # Performance optimization settings with defaults
        self.mode = CONFIG.get('whisper_mode', 'balanced')
        self.beam_size = self._get_beam_size()
        self.temperature = self._get_temperature()
        self.language = CONFIG.get('whisper_language', None)  # None for auto-detection
        self.language_constraints = CONFIG.get('whisper_language_constraints', None)  # List of allowed languages
        self.vad_filter = CONFIG.get('whisper_vad_filter', True)
        self.vad_threshold = CONFIG.get('whisper_vad_threshold', 0.5)
        
        # Advanced settings for better multilingual performance
        self.condition_on_previous_text = CONFIG.get('whisper_condition_on_previous_text', True)
        self.compression_ratio_threshold = CONFIG.get('whisper_compression_ratio_threshold', 2.4)
        self.no_speech_threshold = CONFIG.get('whisper_no_speech_threshold', 0.6)
        self.multilingual_segments = CONFIG.get('whisper_multilingual_segments', False)
        
        # New optimization parameters
        self.word_timestamps = CONFIG.get('whisper_word_timestamps', True)
        self.hallucination_silence_threshold = CONFIG.get('whisper_hallucination_silence_threshold', None)
        self.patience = CONFIG.get('whisper_patience', None)  # Batch mode only
        self.length_penalty = CONFIG.get('whisper_length_penalty', None)  # Batch mode only
        
        self.model = None
        self.is_loaded = False
        
        # Processing queues for real-time mode
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
        
    def _get_config_value(self, param_name, mode_settings=None):
        """Unified method to get configuration values with mode-based defaults"""
        config_value = CONFIG.get(f'whisper_{param_name}', 'auto')
        if config_value != 'auto':
            return config_value
        
        if mode_settings is None:
            mode_settings = {
                'speed': {'beam_size': 1, 'temperature': 0.0},
                'balanced': {'beam_size': 3, 'temperature': 0.2},
                'accuracy': {'beam_size': 5, 'temperature': 0.0}
            }
        
        return mode_settings.get(self.mode, mode_settings['balanced']).get(param_name, mode_settings['balanced'][param_name])
    
    def _get_beam_size(self):
        """Get optimal beam size based on mode"""
        return self._get_config_value('beam_size')
    
    def _get_temperature(self):
        """Get optimal temperature based on mode"""
        return self._get_config_value('temperature')
    
    def _create_language_prompt(self):
        """Create a simple initial prompt to guide language detection"""
        if not self.language_constraints or not self.multilingual_segments:
            return None
        
        # Use a simple, neutral prompt that doesn't influence content
        return "Hello. "
    
    def load_model(self):
        """Load the Whisper model"""
        if self.is_loaded:
            logger.info("Whisper model already loaded")
            return
        
        logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
        
        try:
            # Check CUDA availability
            if self.device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = 'cpu'
                self.compute_type = 'float32'
            
            # Load the model
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=None  # Use default cache
            )
            
            self.is_loaded = True
            logger.info(f"Whisper model loaded successfully on {self.device}")
            
            # Log GPU memory usage if CUDA
            if self.device == 'cuda':
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_chunk(self, audio_chunk, chunk_timestamp=0.0):
        """Transcribe a single audio chunk with word-level timestamps"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Ensure audio is float32 and properly shaped
            if isinstance(audio_chunk, np.ndarray):
                if audio_chunk.dtype != np.float32:
                    audio_chunk = audio_chunk.astype(np.float32)
                # Ensure 1D array
                if len(audio_chunk.shape) > 1:
                    audio_chunk = audio_chunk.flatten()
            
            # Validate audio chunk quality
            if len(audio_chunk) == 0:
                logger.warning(f"Empty audio chunk at {chunk_timestamp:.2f}s")
                return self._create_empty_transcription_result(chunk_timestamp, "Empty audio chunk")
            
            # Check audio levels
            max_amplitude = np.max(np.abs(audio_chunk))
            rms_level = np.sqrt(np.mean(audio_chunk ** 2))
            duration = len(audio_chunk) / self.sample_rate
            
            logger.debug(f"Chunk quality at {chunk_timestamp:.2f}s: duration={duration:.2f}s, max={max_amplitude:.6f}, rms={rms_level:.6f}")
            
            # Check if audio chunk has sufficient content
            if max_amplitude < 1e-6:
                logger.warning(f"Very low audio levels in chunk at {chunk_timestamp:.2f}s (max: {max_amplitude:.6f})")
                return self._create_empty_transcription_result(chunk_timestamp, "Very low audio levels")
            
            if duration < 0.5:
                logger.warning(f"Very short audio chunk at {chunk_timestamp:.2f}s ({duration:.2f}s)")
                # Still try to process it but with a warning
            
            # Determine language setting for multilingual support
            # If user explicitly set a language, always use it regardless of multilingual_segments setting
            if self.language and self.language.lower() != 'auto':
                language_setting = self.language
                condition_on_previous = self.condition_on_previous_text
                logger.debug(f"Using explicit language setting: {self.language}")
            else:
                # Only use auto-detection if no explicit language was set
                language_setting = None if self.multilingual_segments else self.language
                condition_on_previous = False if self.multilingual_segments else self.condition_on_previous_text
                logger.debug(f"Using auto-detection, multilingual_segments: {self.multilingual_segments}")
            
            # Create initial prompt for constrained multilingual mode
            initial_prompt = self._create_language_prompt()
            
            # Prepare transcription parameters
            transcribe_params = {
                'beam_size': self.beam_size,
                'word_timestamps': self.word_timestamps,
                'vad_filter': self.vad_filter,
                'language': language_setting,
                'task': "transcribe",
                'temperature': self.temperature,
                'initial_prompt': initial_prompt,
                'condition_on_previous_text': condition_on_previous,
                'compression_ratio_threshold': self.compression_ratio_threshold,
                'no_speech_threshold': self.no_speech_threshold
            }
            
            # Add VAD parameters if enabled
            if self.vad_filter:
                transcribe_params['vad_parameters'] = dict(
                    threshold=self.vad_threshold,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=100
                )
            
            # Add optimization parameters if available
            if self.hallucination_silence_threshold is not None:
                transcribe_params['hallucination_silence_threshold'] = self.hallucination_silence_threshold
            
            if self.patience is not None:
                transcribe_params['patience'] = self.patience
                
            if self.length_penalty is not None:
                transcribe_params['length_penalty'] = self.length_penalty
            
            # Transcribe with optimized parameters
            segments, info = self.model.transcribe(audio_chunk, **transcribe_params)
            
            # Process segments and extract words with timestamps
            words_with_timestamps = []
            full_text = ""
            
            for segment in segments:
                # Add chunk timestamp offset to segment timestamps
                segment_start = segment.start + chunk_timestamp
                segment_end = segment.end + chunk_timestamp
                
                segment_text = segment.text.strip()
                full_text += segment_text + " "
                
                # Extract word-level timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_start = word.start + chunk_timestamp
                        word_end = word.end + chunk_timestamp
                        words_with_timestamps.append({
                            'word': word.word.strip(),
                            'start': word_start,
                            'end': word_end,
                            'confidence': getattr(word, 'probability', 0.0)
                        })
                else:
                    # Fallback: use segment timestamps for the whole text
                    if segment_text:
                        words_with_timestamps.append({
                            'word': segment_text,
                            'start': segment_start,
                            'end': segment_end,
                            'confidence': 0.0
                        })
            
            # Validate language constraints if specified
            detected_language = info.language
            language_confidence = info.language_probability
            
            # Check if detected language is within constraints
            if (self.language_constraints and 
                detected_language not in self.language_constraints and 
                language_confidence < 0.8):  # Only override if confidence is low
                logger.debug(f"Detected language '{detected_language}' not in constraints {self.language_constraints}, confidence: {language_confidence:.3f}")
                # For now, we keep the result but log the constraint violation
                # In the future, we could retry with different parameters
            
            # Log language detection for user awareness but keep all transcriptions
            if self.language and detected_language != self.language:
                logger.info(f"Language mismatch at {chunk_timestamp:.2f}s: expected '{self.language}', detected '{detected_language}' (confidence: {language_confidence:.2f})")
            
            result = {
                'text': full_text.strip(),
                'words': words_with_timestamps,
                'language': detected_language,
                'language_probability': language_confidence,
                'language_constraints': self.language_constraints,
                'chunk_timestamp': chunk_timestamp,
                'audio_quality': {
                    'max_amplitude': max_amplitude,
                    'rms_level': rms_level,
                    'duration': duration
                }
            }
            
            logger.debug(f"Transcribed chunk at {chunk_timestamp:.2f}s: '{result['text'][:50]}...', language: {detected_language} ({language_confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed for chunk at {chunk_timestamp:.2f}s: {e}")
            return self._create_empty_transcription_result(chunk_timestamp, str(e))
    
    def _create_empty_transcription_result(self, chunk_timestamp, error_reason):
        """Create an empty transcription result with error information"""
        return {
            'text': '',
            'words': [],
            'language': 'unknown',
            'language_probability': 0.0,
            'chunk_timestamp': chunk_timestamp,
            'error': error_reason
        }
    
    def transcribe_file(self, audio_file_path):
        """Transcribe a complete audio file"""
        logger.info(f"Transcribing audio file: {audio_file_path}")
        
        if not self.is_loaded:
            self.load_model()
        
        # Check if multilingual segment processing is enabled
        if self.multilingual_segments:
            return self._transcribe_file_multilingual(audio_file_path)
        
        try:
            # Transcribe the entire file with optimized settings
            segments, info = self.model.transcribe(
                str(audio_file_path),
                beam_size=self.beam_size,
                word_timestamps=True,
                vad_filter=self.vad_filter,
                vad_parameters=dict(
                    threshold=self.vad_threshold,
                    min_speech_duration_ms=100,  # Reduced from 250ms for shorter utterances
                    min_silence_duration_ms=200  # Reduced from 500ms for less aggressive filtering
                ) if self.vad_filter else None,
                language=self.language,  # Auto-detect or use specified language
                task="transcribe",
                temperature=self.temperature,
                condition_on_previous_text=self.condition_on_previous_text,
                compression_ratio_threshold=self.compression_ratio_threshold,
                no_speech_threshold=self.no_speech_threshold,
                initial_prompt=None
            )
            
            # Process all segments
            words_with_timestamps = []
            full_text = ""
            
            for segment in segments:
                segment_text = segment.text.strip()
                full_text += segment_text + " "
                
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        words_with_timestamps.append({
                            'word': word.word.strip(),
                            'start': word.start,
                            'end': word.end,
                            'confidence': getattr(word, 'probability', 0.0)
                        })
                else:
                    if segment_text:
                        words_with_timestamps.append({
                            'word': segment_text,
                            'start': segment.start,
                            'end': segment.end,
                            'confidence': 0.0
                        })
            
            result = {
                'text': full_text.strip(),
                'words': words_with_timestamps,
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration
            }
            
            logger.info(f"File transcription completed. Language: {info.language}, Duration: {info.duration:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            raise
    
    def _transcribe_file_multilingual(self, audio_file_path):
        """Transcribe a file with chunk-based language detection for multilingual content"""
        logger.info(f"Transcribing multilingual audio file using chunk-based approach: {audio_file_path}")
        
        try:
            import librosa
            
            # Load audio file
            audio, sr = librosa.load(str(audio_file_path), sr=self.sample_rate)
            
            # Split audio into chunks (10 seconds each for better language detection)
            chunk_duration = 10.0  # seconds
            chunk_samples = int(chunk_duration * self.sample_rate)
            overlap_samples = int(1.0 * self.sample_rate)  # 1 second overlap
            
            chunks = []
            for start in range(0, len(audio), chunk_samples - overlap_samples):
                end = min(start + chunk_samples, len(audio))
                chunk = audio[start:end]
                chunk_start_time = start / self.sample_rate
                chunks.append((chunk, chunk_start_time))
            
            logger.info(f"Split audio into {len(chunks)} chunks for multilingual processing")
            
            # Process each chunk separately
            all_words = []
            detected_languages = {}
            full_text = ""
            
            for i, (chunk, chunk_start_time) in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} (starts at {chunk_start_time:.1f}s)")
                
                # Transcribe this chunk with auto language detection
                segments, info = self.model.transcribe(
                    chunk,
                    beam_size=self.beam_size,
                    word_timestamps=True,
                    vad_filter=self.vad_filter,
                    vad_parameters=dict(
                        threshold=self.vad_threshold,
                        min_speech_duration_ms=100,  # Reduced from 250ms for shorter utterances
                        min_silence_duration_ms=200  # Reduced from 500ms for less aggressive filtering
                    ) if self.vad_filter else None,
                    language=None,  # Force auto-detection for each chunk
                    task="transcribe",
                    temperature=self.temperature,
                    condition_on_previous_text=False,  # Disable to allow language switching
                    compression_ratio_threshold=self.compression_ratio_threshold,
                    no_speech_threshold=self.no_speech_threshold,
                    initial_prompt=None
                )
                
                chunk_language = info.language
                chunk_confidence = info.language_probability
                
                logger.info(f"Chunk {i+1} detected language: {chunk_language} (confidence: {chunk_confidence:.2f})")
                
                # Track detected languages
                if chunk_language not in detected_languages:
                    detected_languages[chunk_language] = {'count': 0, 'duration': 0}
                detected_languages[chunk_language]['count'] += 1
                
                # Process segments in this chunk
                for segment in segments:
                    segment_text = segment.text.strip()
                    if not segment_text:
                        continue
                        
                    full_text += segment_text + " "
                    
                    # Add chunk start time offset to segment timestamps
                    segment_start = segment.start + chunk_start_time
                    segment_end = segment.end + chunk_start_time
                    detected_languages[chunk_language]['duration'] += segment_end - segment_start
                    
                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            word_start = word.start + chunk_start_time
                            word_end = word.end + chunk_start_time
                            all_words.append({
                                'word': word.word.strip(),
                                'start': word_start,
                                'end': word_end,
                                'confidence': getattr(word, 'probability', 0.0),
                                'language': chunk_language
                            })
                    else:
                        if segment_text:
                            all_words.append({
                                'word': segment_text,
                                'start': segment_start,
                                'end': segment_end,
                                'confidence': 0.0,
                                'language': chunk_language
                            })
            
            # Determine primary language based on duration
            if detected_languages:
                primary_language = max(detected_languages.keys(), 
                                     key=lambda x: detected_languages[x]['duration'])
            else:
                primary_language = 'unknown'
            
            # Calculate total duration
            total_duration = len(audio) / self.sample_rate
            
            result = {
                'text': full_text.strip(),
                'words': all_words,
                'language': primary_language,
                'language_probability': sum(detected_languages.get(primary_language, {}).get('duration', 0) for _ in [1]) / total_duration if total_duration > 0 else 0.0,
                'duration': total_duration,
                'detected_languages': {lang: data['count'] for lang, data in detected_languages.items()},
                'language_durations': {lang: data['duration'] for lang, data in detected_languages.items()},
                'multilingual': len(detected_languages) > 1
            }
            
            logger.info(f"Multilingual transcription completed. Primary: {primary_language}")
            logger.info(f"Languages detected: {list(detected_languages.keys())}")
            for lang, data in detected_languages.items():
                logger.info(f"  - {lang.upper()}: {data['count']} chunks, {data['duration']:.1f}s duration")
            
            return result
            
        except Exception as e:
            logger.error(f"Multilingual transcription failed: {e}")
            # Fallback to regular transcription
            logger.info("Falling back to regular transcription")
            return self._transcribe_file_fallback(audio_file_path)
    
    def _transcribe_file_fallback(self, audio_file_path):
        """Fallback transcription method"""
        segments, info = self.model.transcribe(
            str(audio_file_path),
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=self.vad_filter,
            language=None,  # Auto-detect
            task="transcribe",
            temperature=self.temperature,
            condition_on_previous_text=False,
            compression_ratio_threshold=self.compression_ratio_threshold,
            no_speech_threshold=self.no_speech_threshold
        )
        
        words_with_timestamps = []
        full_text = ""
        
        for segment in segments:
            segment_text = segment.text.strip()
            full_text += segment_text + " "
            
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    words_with_timestamps.append({
                        'word': word.word.strip(),
                        'start': word.start,
                        'end': word.end,
                        'confidence': getattr(word, 'probability', 0.0)
                    })
        
        return {
            'text': full_text.strip(),
            'words': words_with_timestamps,
            'language': info.language,
            'language_probability': info.language_probability,
            'duration': info.duration
        }
    
    def start_real_time_processing(self):
        """Start real-time transcription processing thread"""
        if self.is_processing:
            logger.warning("Real-time processing already started")
            return
        
        if not self.is_loaded:
            self.load_model()
        
        logger.info("Starting real-time transcription processing")
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
    
    def stop_real_time_processing(self):
        """Stop real-time transcription processing"""
        if not self.is_processing:
            return
        
        logger.info("Stopping real-time transcription processing")
        self.is_processing = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
    
    def _processing_loop(self):
        """Main processing loop for real-time transcription"""
        while self.is_processing:
            try:
                # Get audio chunk from queue (with timeout)
                chunk_data = self.input_queue.get(timeout=1)
                if chunk_data is None:
                    continue
                
                audio_chunk, chunk_timestamp = chunk_data
                
                # Transcribe the chunk
                start_time = time.time()
                result = self.transcribe_chunk(audio_chunk, chunk_timestamp)
                processing_time = time.time() - start_time
                
                # Add processing time to result
                result['processing_time'] = processing_time
                
                # Put result in output queue
                try:
                    self.output_queue.put(result, block=False)
                except queue.Full:
                    logger.warning("Transcription output queue full, dropping result")
                
                # Log performance
                chunk_duration = len(audio_chunk) / self.sample_rate
                real_time_factor = processing_time / chunk_duration
                logger.debug(f"Transcription RTF: {real_time_factor:.2f}x")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in transcription processing loop: {e}")
    
    def add_audio_chunk(self, audio_chunk, chunk_timestamp):
        """Add audio chunk to processing queue"""
        try:
            self.input_queue.put((audio_chunk, chunk_timestamp), block=False)
        except queue.Full:
            logger.warning("Transcription input queue full, dropping chunk")
    
    def get_transcription_result(self, timeout=1):
        """Get next transcription result"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def transcribe_chunks(self, audio_chunks):
        """Transcribe multiple audio chunks (for batch processing)"""
        results = []
        
        logger.info(f"Transcribing {len(audio_chunks)} audio chunks")
        
        for i, (chunk, timestamp) in enumerate(audio_chunks):
            logger.info(f"Processing chunk {i+1}/{len(audio_chunks)}")
            result = self.transcribe_chunk(chunk, timestamp)
            results.append(result)
        
        return results
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_real_time_processing()
        
        if self.model is not None:
            # faster-whisper doesn't need explicit cleanup
            self.model = None
            self.is_loaded = False
        
        # Clear GPU cache if using CUDA
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")
