"""
Audio capture module for recording microphone input
Supports both real-time streaming and batch recording modes
"""

import sounddevice as sd
import numpy as np
import queue
import time
import logging
from datetime import datetime
import scipy.io.wavfile as wavfile

from src.core.config import CONFIG, TEMP_DIR, OUTPUT_DIR
from src.audio.base_audio_capture import BaseAudioCapture

logger = logging.getLogger(__name__)

class AudioCapture(BaseAudioCapture):
    def __init__(self):
        super().__init__()
        
        # AudioCapture-specific configuration
        self.chunk_duration = CONFIG['chunk_duration']
        self.overlap_duration = CONFIG['overlap_duration']
        self.buffer_size = CONFIG['buffer_size']
        
        # For batch mode
        self.batch_audio = []
        
        # For real-time recording accumulation
        self.realtime_recording = []
        self.realtime_recording_raw = []  # Store raw unprocessed audio
        self.recording_start_time = None
        

    
    def select_device(self, device_index=None):
        """Select audio input device with improved fallback logic and Windows compatibility"""
        # Check if device index is specified in config
        if device_index is None and CONFIG.get('audio_device_index') is not None:
            device_index = CONFIG['audio_device_index']
            logger.info(f"Using audio device from config: {device_index}")
        
        # If we have a specific device index, try to use it
        if device_index is not None:
            if self._test_and_select_device(device_index):
                return True
            else:
                logger.warning(f"Configured device {device_index} failed, will auto-detect")
        
        # Auto-detect best available input device with Windows compatibility
        logger.info("Auto-detecting best audio input device...")
        return self._auto_detect_best_device()
    
    def _test_and_select_device(self, device_index):
        """Test and select a specific device with comprehensive error handling"""
        try:
            # Check if device exists and is valid
            device_info = sd.query_devices(device_index)
            if device_info['max_input_channels'] <= 0:
                logger.warning(f"Device {device_index} has no input channels")
                return False
            
            logger.info(f"Testing device {device_index}: {device_info['name']}")
            
            # Check for known problematic device patterns on Windows
            device_name = device_info['name'].lower()
            if any(term in device_name for term in ['output', 'speaker', 'headphone']):
                logger.warning(f"Device {device_index} appears to be an output device: {device_info['name']}")
                return False
            
            # Set the device temporarily for testing
            original_device = sd.default.device[0]
            sd.default.device[0] = device_index
            
            # Perform device test
            success = self._perform_device_test(device_index)
            
            if success:
                logger.info(f"Selected audio device: {device_info['name']}")
                return True
            else:
                # Restore original device on failure
                sd.default.device[0] = original_device
                logger.warning(f"Device {device_index} failed testing")
                return False
                
        except Exception as e:
            logger.error(f"Failed to test device {device_index}: {e}")
            return False
    
    def _perform_device_test(self, device_index, test_duration=0.5):
        """Unified device testing method with comprehensive validation"""
        try:
            # Record test audio
            recording = sd.rec(
                int(test_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,  
                dtype=np.float32,
                device=device_index
            )
            sd.wait()
            
            # Validate recording data
            if recording is None or len(recording) == 0:
                logger.debug(f"Device {device_index}: No data captured")
                return False
            
            # Check for data corruption
            if np.any(np.isnan(recording)) or np.any(np.isinf(recording)):
                logger.debug(f"Device {device_index}: Invalid data (NaN/Inf)")
                return False
            
            logger.debug(f"Device {device_index}: Test passed")
            return True
            
        except Exception as e:
            logger.debug(f"Device {device_index}: Test failed - {e}")
            return False
    
    def _auto_detect_best_device(self):
        """Auto-detect best available input device using AudioDeviceManager"""
        try:
            best_device = self.device_manager.find_best_device()
            if best_device:
                device_index = best_device['index']
                if self._test_and_select_device(device_index):
                    # Update config for future use
                    CONFIG['audio_device_index'] = device_index
                    logger.info(f"Updated config to use device {device_index} for future sessions")
                    return True
            
            # If no device worked, try system default as last resort
            logger.warning("No working devices found, falling back to system default")
            sd.default.device[0] = None
            return False
            
        except Exception as e:
            logger.error(f"Failed to auto-detect audio device: {e}")
            logger.info("Falling back to system default device")
            sd.default.device[0] = None
            return False
    

    
    def test_audio_input(self, duration=2):
        """Test audio input - checks if device is responsive and can capture audio"""
        logger.info(f"Testing audio input for {duration} seconds...")
        
        # Get current device info for better error reporting
        current_device = sd.default.device[0]
        try:
            if current_device is not None:
                device_info = sd.query_devices(current_device)
                logger.info(f"Testing device: {device_info['name']}")
        except:
            pass
        
        try:
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                device=current_device
            )
            sd.wait()
            
            # Check if we got any audio data at all
            if recording is None or len(recording) == 0:
                logger.error("No audio data captured")
                return False
            
            # Check for obvious data corruption issues
            if np.any(np.isnan(recording)) or np.any(np.isinf(recording)):
                logger.error("Audio data contains invalid values (NaN/Inf)")
                return False
            
            # Check for any non-zero values (even very small ambient noise)
            max_amplitude = np.max(np.abs(recording))
            mean_amplitude = np.mean(np.abs(recording))
            
            # Very permissive test - just check that we're getting some signal
            # Even electronic noise should register above this threshold
            if max_amplitude > 0.00001:
                logger.info(f"Audio test successful! Device is responsive (max: {max_amplitude:.6f}, mean: {mean_amplitude:.6f})")
                return True
            else:
                logger.warning(f"Audio device may not be working - no signal detected (max: {max_amplitude:.6f})")
                # Still return True for very quiet environments - let user decide
                logger.info("Proceeding anyway - device appears to be connected")
                return True
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Audio test failed: {error_msg}")
            
            # Provide specific guidance for common Windows audio errors
            if "PaErrorCode -9999" in error_msg or "WDM-KS" in error_msg:
                logger.error("Windows WDM-KS driver error detected!")
                logger.error("This usually means the selected audio device is incompatible or already in use.")
                logger.error("Try running: python tools/audio_device_helper.py")
                logger.error("Or check if another application is using the microphone.")
            elif "Invalid number of channels" in error_msg:
                logger.error("Channel configuration error - device may not support mono recording")
            elif "Device unavailable" in error_msg or "No such device" in error_msg:
                logger.error("Audio device is not available - it may be disconnected or in use")
            
            return False
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback function for real-time audio capture"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert to mono if stereo
        if len(indata.shape) > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata.flatten()
        
        # Check for valid audio data
        if len(audio_data) == 0:
            logger.warning("Empty audio data in callback")
            return
        
        # Log audio data quality for debugging
        max_amplitude = np.max(np.abs(audio_data))
        rms_level = np.sqrt(np.mean(audio_data ** 2))
        logger.debug(f"Audio callback: {len(audio_data)} samples, max={max_amplitude:.6f}, rms={rms_level:.6f}, is_recording={self.is_recording}")
        
        # Apply preprocessing using base class method
        if len(audio_data) > 0:
            audio_data = self._process_audio(audio_data)
        
        # Add to buffer for processing
        self.audio_buffer.extend(audio_data)
        
        # Also accumulate for complete recording
        if self.is_recording:
            prev_length = len(self.realtime_recording)
            
            # Store raw audio before any preprocessing (original input)
            if len(indata.shape) > 1:
                raw_data = np.mean(indata, axis=1)  # Convert to mono like we do for processing
            else:
                raw_data = indata.flatten()
            self.realtime_recording_raw.extend(raw_data)
            
            # Store processed audio
            self.realtime_recording.extend(audio_data)
            
            logger.debug(f"Recording accumulated: {prev_length} -> {len(self.realtime_recording)} samples (+{len(audio_data)})")
            # Debug logging every few seconds to track accumulation
            if len(self.realtime_recording) % (self.sample_rate * 5) < len(audio_data):
                duration = len(self.realtime_recording) / self.sample_rate
                logger.info(f"Real-time recording accumulated: {duration:.1f} seconds, {len(self.realtime_recording)} samples")
        else:
            logger.debug(f"Not recording: is_recording={self.is_recording}")
        
        # Check if we have enough for a chunk
        while len(self.audio_buffer) >= CONFIG['CHUNK_SAMPLES']:
            # Extract chunk with overlap
            chunk = np.array(self.audio_buffer[:CONFIG['CHUNK_SAMPLES']])
            
            # Remove processed samples (keeping overlap)
            samples_to_remove = CONFIG['CHUNK_SAMPLES'] - CONFIG['OVERLAP_SAMPLES']
            self.audio_buffer = self.audio_buffer[samples_to_remove:]
            
            # Add to queue if not full
            try:
                # Use current time as timestamp
                import time as time_module
                timestamp = time_module.time()
                self.audio_queue.put((chunk, timestamp), block=False)
                logger.debug(f"Audio chunk queued: {len(chunk)} samples at {timestamp:.2f}s")
            except queue.Full:
                logger.warning("Audio queue full, dropping chunk")
    
    def start_real_time_recording(self):
        """Start real-time audio recording"""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return
        
        logger.info("Starting real-time audio recording...")
        
        # Ensure device is properly selected
        if sd.default.device[0] is None:
            logger.info("No device selected, auto-detecting...")
            self.select_device()
        
        # Test the selected device before starting
        current_device = sd.default.device[0]
        if current_device is not None:
            logger.info(f"Testing device {current_device} before starting recording...")
            if not self.test_audio_input(duration=0.5):
                logger.warning("Device test failed, attempting auto-detection...")
                self.select_device()
        
        self.is_recording = True
        self.audio_buffer = []
        self.realtime_recording = []  # Reset recording accumulation
        self.realtime_recording_raw = []  # Reset raw recording accumulation
        self.recording_start_time = time.time()
        logger.info(f"Real-time recording initialized: buffer={len(self.audio_buffer)}, recording={len(self.realtime_recording)}")
        logger.info(f"Using audio device: {sd.default.device[0]}")
        logger.info(f"Recording state: is_recording={self.is_recording}")
        
        try:
            # Start the audio stream
            self.stream = sd.InputStream(
                callback=self._audio_callback,
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.buffer_size,
                dtype=np.float32,
                device=sd.default.device[0]
            )
            self.stream.start()
            logger.info("Real-time recording started successfully")
            
            # Log initial callback for verification
            time.sleep(0.1)  # Give it time for first callback
            logger.info(f"After initial setup: buffer={len(self.audio_buffer)}, recording={len(self.realtime_recording)}")
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
            raise
    
    def stop_real_time_recording(self):
        """Stop real-time audio recording"""
        if not self.is_recording:
            logger.debug("Stop recording called but no recording in progress")
            # Fallback: return any accumulated recording if available
            if len(self.realtime_recording) > 0:
                logger.info(f"Returning accumulated recording of {len(self.realtime_recording)} samples despite recording flag off")
                return np.array(self.realtime_recording)
            return None
        
        logger.info("Stopping real-time audio recording...")
        
        # Log current state before stopping
        logger.info(f"Before stopping: buffer={len(self.audio_buffer)}, recording={len(self.realtime_recording)}")
        
        # Stop the stream first but keep is_recording=True temporarily
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            # Give a small delay to ensure all buffered audio is processed
            time.sleep(0.2)  # Increased delay to ensure completion
            logger.info("Audio stream stopped")
        
        # Now set recording to false to stop accumulation
        self.is_recording = False
        
        # Process any remaining buffer and add to realtime_recording
        if len(self.audio_buffer) > 0:
            remaining_chunk = np.array(self.audio_buffer)
            # Add remaining buffer to realtime recording
            self.realtime_recording.extend(remaining_chunk)
            # For raw, we need to store the unprocessed version - but since this is from buffer, it's already processed
            # So we'll just use the same data (this is a small portion at the end anyway)
            self.realtime_recording_raw.extend(remaining_chunk)
            logger.info(f"Added remaining buffer to recording: {len(remaining_chunk)} samples")
            
            try:
                self.audio_queue.put((remaining_chunk, time.time()), block=False)
            except queue.Full:
                logger.warning("Queue full, lost final chunk")
        
        logger.info(f"Final recording state: {len(self.realtime_recording)} samples")
        
        # Return the complete recording for saving
        if len(self.realtime_recording) > 0:
            recording_duration = len(self.realtime_recording) / self.sample_rate
            logger.info(f"Real-time recording completed: {recording_duration:.1f} seconds, {len(self.realtime_recording)} samples")
            
            # Check audio quality
            recording_array = np.array(self.realtime_recording)
            max_amplitude = np.max(np.abs(recording_array))
            rms_level = np.sqrt(np.mean(recording_array ** 2))
            logger.info(f"Recording quality: max={max_amplitude:.6f}, rms={rms_level:.6f}")
            
            return recording_array
        else:
            logger.warning("No audio data accumulated during real-time recording")
            logger.warning("Possible causes: audio device not working, no sound input, or processing error")
            return None
    
    def get_realtime_recording(self):
        """Get the accumulated real-time recording"""
        if len(self.realtime_recording) > 0:
            return np.array(self.realtime_recording)
        return None
    
    def get_audio_chunk(self, timeout=1):
        """Get the next audio chunk from the queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def record_batch(self, duration_seconds):
        """Record audio for a specific duration (batch mode) - improved version"""
        logger.info(f"Starting batch recording for {duration_seconds} seconds...")
        
        try:
            # Ensure device is selected
            if sd.default.device[0] is None:
                self.select_device()
            
            total_samples = int(duration_seconds * self.sample_rate)
            
            # Record with explicit device specification
            recording = sd.rec(
                total_samples,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                device=sd.default.device[0]  # Explicitly use selected device
            )
            
            # Wait for recording to complete
            logger.info("Recording in progress... speak now!")
            sd.wait()
            
            # Convert to mono if needed
            if len(recording.shape) > 1:
                recording = np.mean(recording, axis=1)
            else:
                recording = recording.flatten()
            
            # Validate recording
            if len(recording) == 0:
                raise RuntimeError("No audio data recorded")
            
            # Check audio levels before preprocessing
            rms_level_before = np.sqrt(np.mean(recording**2))
            max_level_before = np.max(np.abs(recording))
            
            logger.info(f"Raw recording: {len(recording)} samples, RMS: {rms_level_before:.4f}, Max: {max_level_before:.4f}")
            
            if rms_level_before < 0.0001:
                logger.warning("Very low audio level detected - preprocessing will boost gain")
            
            # Apply preprocessing for distant microphones and real-world conditions
            if self.enable_preprocessing:
                try:
                    logger.info("Applying audio preprocessing for distant microphones...")
                    recording = self.preprocessor.enhance_for_whisper(recording)
                    
                    # Check levels after preprocessing
                    rms_level_after = np.sqrt(np.mean(recording**2))
                    max_level_after = np.max(np.abs(recording))
                    
                    gain_db = 20 * np.log10(rms_level_after / max(rms_level_before, 1e-8))
                    logger.info(f"Preprocessed recording: RMS: {rms_level_after:.4f}, Max: {max_level_after:.4f}, Gain: {gain_db:.1f}dB")
                    
                except Exception as e:
                    logger.error(f"Audio preprocessing failed: {e}")
                    logger.info("Continuing with unprocessed audio")
            
            return recording
            
        except Exception as e:
            logger.error(f"Batch recording failed: {e}")
            raise
    

    
    def chunk_audio_file(self, audio_data):
        """Split audio data into overlapping chunks"""
        chunks = []
        chunk_size = CONFIG['CHUNK_SAMPLES']
        overlap = CONFIG['OVERLAP_SAMPLES']
        
        start = 0
        while start < len(audio_data):
            end = min(start + chunk_size, len(audio_data))
            chunk = audio_data[start:end]
            
            # Only add chunk if it's substantial enough
            if len(chunk) > chunk_size // 4:  # At least 25% of chunk size
                timestamp = start / self.sample_rate
                chunks.append((chunk, timestamp))
            
            start += (chunk_size - overlap)
        
        logger.info(f"Audio split into {len(chunks)} chunks")
        return chunks
    

 