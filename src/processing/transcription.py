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

from src.core.config import CONFIG, OUTPUT_DIR
from src.utils.warning_suppressor import configure_torch_tf32

configure_torch_tf32()
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self):
        # Static configuration that doesn't change during runtime
        self.model_size = CONFIG['whisper_model']
        self.device = CONFIG['whisper_device']
        self.compute_type = CONFIG['whisper_compute_type']
        self.sample_rate = CONFIG['sample_rate']
        self.mode = CONFIG.get('whisper_mode', 'balanced')
        
        # Dynamic configuration that can change during runtime
        self.vad_filter = CONFIG.get('whisper_vad_filter', True)
        self.vad_threshold = CONFIG.get('whisper_vad_threshold', 0.5)
        self.condition_on_previous_text = CONFIG.get('whisper_condition_on_previous_text', True)
        self.compression_ratio_threshold = CONFIG.get('whisper_compression_ratio_threshold', 2.4)
        self.no_speech_threshold = CONFIG.get('whisper_no_speech_threshold', 0.6)
        self.multilingual_segments = CONFIG.get('whisper_multilingual_segments', False)
        self.word_timestamps = CONFIG.get('whisper_word_timestamps', True)
        
        self.model = None
        self.is_loaded = False
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
        self.beam_size, self.temperature = self._get_mode_defaults()

    def _get_mode_defaults(self):
        mode_defaults = {
            'speed': (1, 0.0),
            'balanced': (3, 0.2),
            'accuracy': (5, 0.0)
        }
        beam = CONFIG.get('whisper_beam_size', 'auto')
        temp = CONFIG.get('whisper_temperature', 'auto')
        if beam != 'auto' and temp != 'auto':
            return beam, temp
        b, t = mode_defaults.get(self.mode, mode_defaults['balanced'])
        return (beam if beam != 'auto' else b, temp if temp != 'auto' else t)

    def load_model(self):
        if self.is_loaded:
            logger.info("Whisper model already loaded")
            return
        logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'
            self.compute_type = 'float32'
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            download_root=None
        )
        self.is_loaded = True
        logger.info(f"Whisper model loaded successfully on {self.device}")
        if self.device == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

    def transcribe_chunk(self, audio_chunk, chunk_timestamp=0.0):
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        try:
            if isinstance(audio_chunk, np.ndarray):
                audio_chunk = audio_chunk.astype(np.float32).flatten()
            if len(audio_chunk) == 0:
                logger.warning(f"Empty audio chunk at {chunk_timestamp:.2f}s")
                return self._empty_result(chunk_timestamp, "Empty audio chunk")
            max_amplitude = np.max(np.abs(audio_chunk))
            rms_level = np.sqrt(np.mean(audio_chunk ** 2))
            duration = len(audio_chunk) / self.sample_rate
            if max_amplitude < 1e-6:
                logger.warning(f"Very low audio levels in chunk at {chunk_timestamp:.2f}s (max: {max_amplitude:.6f})")
                return self._empty_result(chunk_timestamp, "Very low audio levels")
            if duration < 0.5:
                logger.warning(f"Very short audio chunk at {chunk_timestamp:.2f}s ({duration:.2f}s)")
            language_setting = self._get_language_setting() if self._get_language_setting() and self._get_language_setting().lower() != 'auto' else (None if self.multilingual_segments else self._get_language_setting())
            condition_on_previous = self.condition_on_previous_text if (self._get_language_setting() and self._get_language_setting().lower() != 'auto') else (False if self.multilingual_segments else self.condition_on_previous_text)
            transcribe_params = {
                'beam_size': self.beam_size,
                'word_timestamps': self.word_timestamps,
                'vad_filter': self.vad_filter,
                'language': language_setting,
                'task': "transcribe",
                'temperature': self.temperature,
                'initial_prompt': None,
                'condition_on_previous_text': condition_on_previous,
                'compression_ratio_threshold': self.compression_ratio_threshold,
                'no_speech_threshold': self.no_speech_threshold
            }
            if self.vad_filter:
                transcribe_params['vad_parameters'] = dict(threshold=self.vad_threshold, min_speech_duration_ms=250, min_silence_duration_ms=100)
            segments, info = self.model.transcribe(audio_chunk, **transcribe_params)
            words, text = self._extract_words(segments, chunk_timestamp)
            result = {
                'text': text.strip(),
                'words': words,
                'language': info.language,
                'language_probability': info.language_probability,
                'language_constraints': self._get_language_constraints(),
                'chunk_timestamp': chunk_timestamp,
                'audio_quality': {
                    'max_amplitude': max_amplitude,
                    'rms_level': rms_level,
                    'duration': duration
                }
            }
            if self._get_language_setting() and info.language != self._get_language_setting():
                logger.info(f"Language mismatch at {chunk_timestamp:.2f}s: expected '{self._get_language_setting()}', detected '{info.language}' (confidence: {info.language_probability:.2f})")
            return result
        except Exception as e:
            logger.error(f"Transcription failed for chunk at {chunk_timestamp:.2f}s: {e}")
            return self._empty_result(chunk_timestamp, str(e))

    def _extract_words(self, segments, chunk_timestamp=0.0):
        words = []
        full_text = ""
        for segment in segments:
            segment_start = segment.start + chunk_timestamp
            segment_end = segment.end + chunk_timestamp
            segment_text = segment.text.strip()
            full_text += segment_text + " "
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    word_start = word.start + chunk_timestamp
                    word_end = word.end + chunk_timestamp
                    words.append({
                        'word': word.word.strip(),
                        'start': word_start,
                        'end': word_end,
                        'confidence': getattr(word, 'probability', 0.0)
                    })
            elif segment_text:
                words.append({
                    'word': segment_text,
                    'start': segment_start,
                    'end': segment_end,
                    'confidence': 0.0
                })
        return words, full_text

    def _empty_result(self, chunk_timestamp, error_reason):
        return {
            'text': '',
            'words': [],
            'language': 'unknown',
            'language_probability': 0.0,
            'chunk_timestamp': chunk_timestamp,
            'error': error_reason
        }

    def transcribe_file(self, audio_file_path):
        logger.info(f"Transcribing audio file: {audio_file_path}")
        if not self.is_loaded:
            self.load_model()
        if self.multilingual_segments:
            return self._transcribe_file_multilingual(audio_file_path)
        try:
            segments, info = self.model.transcribe(
                str(audio_file_path),
                beam_size=self.beam_size,
                word_timestamps=True,
                vad_filter=self.vad_filter,
                vad_parameters=dict(
                    threshold=self.vad_threshold,
                    min_speech_duration_ms=100,
                    min_silence_duration_ms=200
                ) if self.vad_filter else None,
                language=self._get_language_setting(),
                task="transcribe",
                temperature=self.temperature,
                condition_on_previous_text=self.condition_on_previous_text,
                compression_ratio_threshold=self.compression_ratio_threshold,
                no_speech_threshold=self.no_speech_threshold,
                initial_prompt=None
            )
            words, text = self._extract_words(segments)
            return {
                'text': text.strip(),
                'words': words,
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration
            }
        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            raise

    def _transcribe_file_multilingual(self, audio_file_path):
        logger.info(f"Transcribing multilingual audio file using chunk-based approach: {audio_file_path}")
        try:
            import librosa
            audio, sr = librosa.load(str(audio_file_path), sr=self.sample_rate)
            chunk_duration = 10.0
            chunk_samples = int(chunk_duration * self.sample_rate)
            overlap_samples = int(1.0 * self.sample_rate)
            chunks = [(audio[start:min(start + chunk_samples, len(audio))], start / self.sample_rate)
                      for start in range(0, len(audio), chunk_samples - overlap_samples)]
            logger.info(f"Split audio into {len(chunks)} chunks for multilingual processing")
            all_words = []
            detected_languages = {}
            full_text = ""
            for i, (chunk, chunk_start_time) in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} (starts at {chunk_start_time:.1f}s)")
                segments, info = self.model.transcribe(
                    chunk,
                    beam_size=self.beam_size,
                    word_timestamps=True,
                    vad_filter=self.vad_filter,
                    vad_parameters=dict(
                        threshold=self.vad_threshold,
                        min_speech_duration_ms=100,
                        min_silence_duration_ms=200
                    ) if self.vad_filter else None,
                    language=None,
                    task="transcribe",
                    temperature=self.temperature,
                    condition_on_previous_text=False,
                    compression_ratio_threshold=self.compression_ratio_threshold,
                    no_speech_threshold=self.no_speech_threshold,
                    initial_prompt=None
                )
                chunk_language = info.language
                chunk_confidence = info.language_probability
                logger.info(f"Chunk {i+1} detected language: {chunk_language} (confidence: {chunk_confidence:.2f})")
                if chunk_language not in detected_languages:
                    detected_languages[chunk_language] = {'count': 0, 'duration': 0}
                detected_languages[chunk_language]['count'] += 1
                for segment in segments:
                    segment_text = segment.text.strip()
                    if not segment_text:
                        continue
                    full_text += segment_text + " "
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
                    elif segment_text:
                        all_words.append({
                            'word': segment_text,
                            'start': segment_start,
                            'end': segment_end,
                            'confidence': 0.0,
                            'language': chunk_language
                        })
            primary_language = max(detected_languages.keys(), key=lambda x: detected_languages[x]['duration']) if detected_languages else 'unknown'
            total_duration = len(audio) / self.sample_rate
            return {
                'text': full_text.strip(),
                'words': all_words,
                'language': primary_language,
                'language_probability': detected_languages.get(primary_language, {}).get('duration', 0) / total_duration if total_duration > 0 else 0.0,
                'duration': total_duration,
                'detected_languages': {lang: data['count'] for lang, data in detected_languages.items()},
                'language_durations': {lang: data['duration'] for lang, data in detected_languages.items()},
                'multilingual': len(detected_languages) > 1
            }
        except Exception as e:
            logger.error(f"Multilingual transcription failed: {e}")
            logger.info("Falling back to regular transcription")
            return self._transcribe_file_fallback(audio_file_path)

    def _transcribe_file_fallback(self, audio_file_path):
        segments, info = self.model.transcribe(
            str(audio_file_path),
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=self.vad_filter,
            language=None,
            task="transcribe",
            temperature=self.temperature,
            condition_on_previous_text=False,
            compression_ratio_threshold=self.compression_ratio_threshold,
            no_speech_threshold=self.no_speech_threshold
        )
        words, text = self._extract_words(segments)
        return {
            'text': text.strip(),
            'words': words,
            'language': info.language,
            'language_probability': info.language_probability,
            'duration': info.duration
        }

    def start_real_time_processing(self):
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
        if not self.is_processing:
            return
        logger.info("Stopping real-time transcription processing")
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break

    def _processing_loop(self):
        while self.is_processing:
            try:
                chunk_data = self.input_queue.get(timeout=1)
                if chunk_data is None:
                    continue
                audio_chunk, chunk_timestamp = chunk_data
                start_time = time.time()
                result = self.transcribe_chunk(audio_chunk, chunk_timestamp)
                processing_time = time.time() - start_time
                result['processing_time'] = processing_time
                try:
                    self.output_queue.put(result, block=False)
                except queue.Full:
                    logger.warning("Transcription output queue full, dropping result")
                chunk_duration = len(audio_chunk) / self.sample_rate
                real_time_factor = processing_time / chunk_duration
                logger.debug(f"Transcription RTF: {real_time_factor:.2f}x")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in transcription processing loop: {e}")

    def add_audio_chunk(self, audio_chunk, chunk_timestamp):
        try:
            self.input_queue.put((audio_chunk, chunk_timestamp), block=False)
        except queue.Full:
            logger.warning("Transcription input queue full, dropping chunk")

    def get_transcription_result(self, timeout=1):
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def transcribe_chunks(self, audio_chunks):
        logger.info(f"Transcribing {len(audio_chunks)} audio chunks")
        results = []
        for i, (chunk, timestamp) in enumerate(audio_chunks):
            logger.info(f"Processing chunk {i+1}/{len(audio_chunks)}")
            result = self.transcribe_chunk(chunk, timestamp)
            results.append(result)
        return results

    def cleanup(self):
        self.stop_real_time_processing()
        if self.model is not None:
            self.model = None
            self.is_loaded = False
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")
