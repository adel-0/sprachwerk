"""
Audio capture module for recording microphone input
Supports both real-time streaming and batch recording modes
"""

import sounddevice as sd
import numpy as np
import queue
import time
import logging

from src.core.config import CONFIG
from src.audio.base_audio_capture import BaseAudioCapture

logger = logging.getLogger(__name__)

class AudioCapture(BaseAudioCapture):
    def __init__(self):
        super().__init__()
        
        # AudioCapture-specific configuration
        self.chunk_duration = CONFIG['chunk_duration']
        self.overlap_duration = CONFIG['overlap_duration']
        self.buffer_size = CONFIG['buffer_size']
        
        # For real-time recording accumulation
        self.realtime_recording = []
        self.realtime_recording_raw = []
        self.recording_start_time = None
    
    def select_device(self, device_index=None):
        """Select audio input device"""
        device_index = device_index or CONFIG.get('audio_device_index')
        
        # Try specified device first
        if device_index and self._test_device(device_index):
            sd.default.device[0] = device_index
            return True
        
        # Auto-detect best device
        logger.info("Auto-detecting audio input device...")
        best_device = self.device_manager.find_best_device()
        if best_device and self._test_device(best_device['index']):
            CONFIG['audio_device_index'] = best_device['index']
            sd.default.device[0] = best_device['index']
            return True
        
        # Fall back to system default
        logger.warning("No working devices found, using system default")
        sd.default.device[0] = None
        return False
    
    def _test_device(self, device_index, test_duration=0.5):
        """Test if device is working"""
        original_device = sd.default.device[0]
        try:
            sd.default.device[0] = device_index
            device_info = sd.query_devices(device_index)
            if device_info['max_input_channels'] <= 0:
                return False
            
            # Quick test recording
            recording = sd.rec(
                int(test_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                device=device_index
            )
            sd.wait()
            
            return recording is not None and len(recording) > 0
            
        except Exception:
            return False
        finally:
            if sd.default.device[0] == device_index:
                sd.default.device[0] = original_device
    
    def test_audio_input(self, duration=1):
        """Quick audio input test"""
        try:
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                device=sd.default.device[0]
            )
            sd.wait()
            
            return recording is not None and len(recording) > 0
            
        except Exception as e:
            logger.error(f"Audio test failed: {e}")
            return False
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback function for real-time audio capture"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert to mono if stereo
        audio_data = np.mean(indata, axis=1) if len(indata.shape) > 1 else indata.flatten()
        
        if len(audio_data) == 0:
            return
        
        # Store raw audio data
        raw_audio_data = audio_data.copy()
        
        # Apply preprocessing if enabled
        if self.enable_preprocessing:
            try:
                audio_data = self._process_audio(audio_data)
            except Exception:
                audio_data = raw_audio_data
        
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        # Accumulate for complete recording
        if self.is_recording:
            self.realtime_recording_raw.extend(raw_audio_data)
            self.realtime_recording.extend(audio_data)
        
        # Process chunks
        chunk_samples = CONFIG.get('CHUNK_SAMPLES', int(4 * self.sample_rate))
        overlap_samples = CONFIG.get('OVERLAP_SAMPLES', int(0.5 * self.sample_rate))
        
        while len(self.audio_buffer) >= chunk_samples:
            chunk = np.array(self.audio_buffer[:chunk_samples])
            samples_to_remove = chunk_samples - overlap_samples
            self.audio_buffer = self.audio_buffer[samples_to_remove:]
            
            try:
                self.audio_queue.put((chunk, time.time()), block=False)
            except queue.Full:
                logger.warning("Audio queue full, dropping chunk")
    
    def start_real_time_recording(self):
        """Start real-time audio recording"""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return
        
        # Ensure device is selected
        if sd.default.device[0] is None:
            self.select_device()
        
        self.is_recording = True
        self.audio_buffer = []
        self.realtime_recording = []
        self.realtime_recording_raw = []
        self.recording_start_time = time.time()
        
        try:
            self.stream = sd.InputStream(
                callback=self._audio_callback,
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.buffer_size,
                dtype=np.float32,
                device=sd.default.device[0]
            )
            self.stream.start()
            logger.info("Real-time recording started")
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
            raise
    
    def stop_real_time_recording(self):
        """Stop real-time audio recording"""
        if not self.is_recording:
            return np.array(self.realtime_recording) if self.realtime_recording else None
        
        # Stop stream
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            time.sleep(0.1)  # Brief delay for final processing
        
        self.is_recording = False
        
        # Add remaining buffer to recording
        if self.audio_buffer:
            remaining_chunk = np.array(self.audio_buffer)
            self.realtime_recording.extend(remaining_chunk)
            self.realtime_recording_raw.extend(remaining_chunk)
            
            try:
                self.audio_queue.put((remaining_chunk, time.time()), block=False)
            except queue.Full:
                pass
        
        # Return recording
        if self.realtime_recording:
            recording_duration = len(self.realtime_recording) / self.sample_rate
            logger.info(f"Recording completed: {recording_duration:.1f} seconds")
            return np.array(self.realtime_recording)
        
        logger.warning("No audio data recorded")
        return None
    
    def get_realtime_recording(self):
        """Get the accumulated real-time recording"""
        return np.array(self.realtime_recording) if self.realtime_recording else None
    
    def get_audio_chunk(self, timeout=1):
        """Get the next audio chunk from the queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def record_batch(self, duration_seconds):
        """Record audio for a specific duration (batch mode)"""
        logger.info(f"Starting batch recording for {duration_seconds} seconds...")
        
        # Ensure device is selected
        if sd.default.device[0] is None:
            self.select_device()
        
        try:
            total_samples = int(duration_seconds * self.sample_rate)
            
            recording = sd.rec(
                total_samples,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                device=sd.default.device[0]
            )
            
            logger.info("Recording in progress... speak now!")
            sd.wait()
            
            # Convert to mono if needed
            recording = np.mean(recording, axis=1) if len(recording.shape) > 1 else recording.flatten()
            
            if len(recording) == 0:
                raise RuntimeError("No audio data recorded")
            
            # Apply preprocessing if enabled
            if self.enable_preprocessing:
                try:
                    recording = self.preprocessor.enhance_for_whisper(recording)
                except Exception as e:
                    logger.error(f"Audio preprocessing failed: {e}")
            
            return recording
            
        except Exception as e:
            logger.error(f"Batch recording failed: {e}")
            raise
    
    def chunk_audio_file(self, audio_data):
        """Split audio data into overlapping chunks"""
        chunks = []
        chunk_samples = CONFIG.get('CHUNK_SAMPLES', int(4 * self.sample_rate))
        overlap_samples = CONFIG.get('OVERLAP_SAMPLES', int(0.5 * self.sample_rate))
        
        start = 0
        while start < len(audio_data):
            end = min(start + chunk_samples, len(audio_data))
            chunk = audio_data[start:end]
            
            if len(chunk) > chunk_samples // 4:  # At least 25% of chunk size
                timestamp = start / self.sample_rate
                chunks.append((chunk, timestamp))
            
            start += (chunk_samples - overlap_samples)
        
        logger.info(f"Audio split into {len(chunks)} chunks")
        return chunks


 