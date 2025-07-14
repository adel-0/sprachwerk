"""
System Audio Capture module for recording system audio and microphone simultaneously
Uses pyaudiowpatch for Windows loopback audio recording
"""

import pyaudiowpatch as pyaudio
import numpy as np
import sys
import threading
import queue
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import time

from src.core.config import CONFIG, OUTPUT_DIR
from src.audio.base_audio_capture import BaseAudioCapture

logger = logging.getLogger(__name__)

class SystemAudioCapture(BaseAudioCapture):
    """System audio capture using pyaudiowpatch for loopback recording"""
    
    def __init__(self):
        super().__init__()
        
        # System audio specific configuration
        self.recording_threads = []
        
        # Recording configuration
        self.system_device = None
        self.mic_device = None
        self.recording_mode = 'both'  # 'system', 'mic', or 'both'
        self.system_gain = 0.7
        self.mic_gain = 1.0
        self.auto_normalize = True
        self.target_level = 0.1
        
        # Recording data
        self.system_audio_data = None
        self.mic_audio_data = None
        self.system_sample_rate = 44100
        self.mic_sample_rate = 44100
    
    def list_devices(self) -> Dict[str, List[Dict]]:
        """List all available audio devices with loopback and microphone detection"""
        devices = {'loopback': [], 'microphone': [], 'all': []}
        
        try:
            # Get all devices from AudioDeviceManager
            all_devices = self.device_manager.list_all_devices()
            devices['all'] = all_devices
            
            # Get specialized device lists
            devices['loopback'] = self.device_manager.get_loopback_devices()
            devices['microphone'] = self.device_manager.get_microphone_devices()
            
        except Exception as e:
            logger.error(f"Error listing devices: {e}")
            
        return devices
    
    def get_recording_device(self, device_index: Optional[int] = None) -> Optional[Dict]:
        """Get a suitable device for recording system audio (loopback)"""
        if device_index is not None:
            # Use specific device index
            try:
                all_devices = self.device_manager.list_all_devices()
                for device in all_devices:
                    if device['index'] == device_index and device['max_input_channels'] > 0:
                        logger.info(f"Using system audio device [{device_index}]: {device['name']} ({device['hostapi']})")
                        return device

                # If we reach here, no matching device was found
                logger.warning(f"Device index {device_index} not found or has no input channels")
                return None
            except Exception as e:
                logger.error(f"Error getting device {device_index}: {e}")
                return None
            
        # Auto-detect best loopback device
        try:
            device = self.device_manager.get_best_loopback_device()
            if device:
                logger.info(f"Auto-selected loopback device: [{device['index']}] {device['name']}")
                return device
            else:
                logger.warning("No loopback device found")
                return None
        except Exception as e:
            logger.error(f"Error getting recording device: {e}")
            return None
    
    def get_microphone_device(self, device_index: Optional[int] = None) -> Optional[Dict]:
        """Get a suitable microphone device"""
        if device_index is not None:
            # Use specific device index
            try:
                all_devices = self.device_manager.list_all_devices()
                for device in all_devices:
                    if device['index'] == device_index and device['max_input_channels'] > 0:
                        logger.info(f"Using microphone device [{device_index}]: {device['name']} ({device['hostapi']})")
                        return device

                logger.warning(f"Device index {device_index} not found or has no input channels")
                return None
            except Exception as e:
                logger.error(f"Error getting device {device_index}: {e}")
                return None
            
        # Auto-detect best microphone device
        try:
            device = self.device_manager.get_best_microphone_device()
            if device:
                logger.info(f"Auto-selected microphone device: [{device['index']}] {device['name']}")
                return device
            else:
                logger.warning("No microphone device found")
                return None
        except Exception as e:
            logger.error(f"Error getting microphone device: {e}")
            return None
    
    def set_recording_mode(self, mode: str, system_device_index: Optional[int] = None, 
                          mic_device_index: Optional[int] = None):
        """Set recording mode and devices"""
        self.recording_mode = mode
        
        if mode in ['system', 'both']:
            self.system_device = self.get_recording_device(system_device_index)
            if not self.system_device:
                logger.error("Failed to get system audio device")
                return False
        
        if mode in ['mic', 'both']:
            self.mic_device = self.get_microphone_device(mic_device_index)
            if not self.mic_device:
                logger.error("Failed to get microphone device")
                return False
        
        return True
    
    def _record_audio_stream(self, p: pyaudio.PyAudio, device: Dict, duration: float, 
                            audio_queue: queue.Queue, stream_name: str):
        """Record audio from a single device"""
        stream = None
        try:
            # Find supported sample rate
            supported_rates = [48000, 44100, 32000, 22050, 16000]
            sample_rate = None
            channels = min(device['max_input_channels'], 2)
            
            for rate in supported_rates:
                try:
                    if p.is_format_supported(rate, 
                                              input_device=device['index'], 
                                              input_channels=channels, 
                                              input_format=pyaudio.paInt16):
                        sample_rate = rate
                        break
                except ValueError:
                    continue

            if sample_rate is None:
                raise RuntimeError(f"No supported sample rate found for {device['name']}")

            stream = p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=2048,
                input_device_index=device['index']
            )
            
            frames = []
            total_chunks = int(duration * sample_rate / 2048)
            
            for _ in range(total_chunks):
                data = stream.read(2048, exception_on_overflow=False)
                frames.append(data)

            if frames:
                audio_data = b''.join(frames)
                recording = np.frombuffer(audio_data, dtype=np.int16)
                
                # Convert to mono if stereo
                if channels > 1:
                    recording = recording.reshape(-1, channels).mean(axis=1).astype(np.int16)
                
                audio_queue.put((stream_name, recording, sample_rate))
            else:
                audio_queue.put((stream_name, None, sample_rate))
                
        except Exception as e:
            logger.error(f"Error recording {stream_name}: {e}")
            audio_queue.put((stream_name, None, 44100))
        finally:
            if stream:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
    
    def _resample_audio(self, audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """Simple resampling using linear interpolation"""
        if original_rate == target_rate:
            return audio
        
        ratio = target_rate / original_rate
        new_length = int(len(audio) * ratio)
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(new_indices, old_indices, audio.astype(np.float32))
        return resampled.astype(np.int16)
    
    def _normalize_audio_levels(self, system_audio: np.ndarray, mic_audio: np.ndarray, 
                               target_rms: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize audio levels to have similar RMS values"""
        if system_audio is None or mic_audio is None:
            return system_audio, mic_audio
        
        # Calculate RMS for each stream
        system_rms = np.sqrt(np.mean((system_audio.astype(np.float32) / 32768.0) ** 2))
        mic_rms = np.sqrt(np.mean((mic_audio.astype(np.float32) / 32768.0) ** 2))
        
        # Avoid division by zero
        if system_rms < 1e-6:
            system_rms = 1e-6
        if mic_rms < 1e-6:
            mic_rms = 1e-6
        
        # Calculate normalization factors
        system_factor = target_rms / system_rms
        mic_factor = target_rms / mic_rms
        
        # Apply normalization with safety limits
        system_factor = min(system_factor, 10.0)  # Max 10x amplification
        mic_factor = min(mic_factor, 10.0)
        
        # Apply normalization
        system_normalized = (system_audio.astype(np.float32) * system_factor).astype(np.int16)
        mic_normalized = (mic_audio.astype(np.float32) * mic_factor).astype(np.int16)
        
        logger.info(f"Level normalization: System x{system_factor:.2f}, Mic x{mic_factor:.2f}")
        
        return system_normalized, mic_normalized
    
    def _mix_audio_streams(self, system_audio: np.ndarray, mic_audio: np.ndarray, 
                          system_rate: int, mic_rate: int) -> Tuple[np.ndarray, int]:
        """Mix system audio and microphone audio with adjustable gains"""
        if system_audio is None and mic_audio is None:
            return None, 44100
        
        if system_audio is None:
            return (mic_audio * self.mic_gain).astype(np.int16) if mic_audio is not None else None, mic_rate
        
        if mic_audio is None:
            return (system_audio * self.system_gain).astype(np.int16), system_rate
        
        # Use the higher sample rate
        target_rate = max(system_rate, mic_rate)
        
        # Resample if necessary
        if system_rate != target_rate:
            system_audio = self._resample_audio(system_audio, system_rate, target_rate)
        
        if mic_rate != target_rate:
            mic_audio = self._resample_audio(mic_audio, mic_rate, target_rate)
        
        # Ensure same length
        min_length = min(len(system_audio), len(mic_audio))
        system_audio = system_audio[:min_length]
        mic_audio = mic_audio[:min_length]
        
        # Auto-normalize levels if enabled
        if self.auto_normalize:
            system_audio, mic_audio = self._normalize_audio_levels(system_audio, mic_audio, 
                                                                  target_rms=self.target_level)
        
        # Mix with gains
        system_float = system_audio.astype(np.float32) * self.system_gain
        mic_float = mic_audio.astype(np.float32) * self.mic_gain
        mixed = system_float + mic_float
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 32767:
            mixed = mixed * (32767 / max_val)
        
        return mixed.astype(np.int16), target_rate
    
    def record_batch(self, duration: float) -> Optional[np.ndarray]:
        """Record system audio and/or microphone for a specific duration"""
        if self.recording_mode not in ['system', 'mic', 'both']:
            logger.error(f"Invalid recording mode: {self.recording_mode}")
            return None
        
        logger.info(f"Starting system audio recording for {duration}s in mode: {self.recording_mode}")
        
        p = None
        try:
            p = pyaudio.PyAudio()
            audio_queue = queue.Queue()
            threads = []
            
            # Start recording threads
            if self.recording_mode in ['system', 'both'] and self.system_device:
                threads.append(threading.Thread(
                    target=self._record_audio_stream,
                    args=(p, self.system_device, duration, audio_queue, "system audio")
                ))
            
            if self.recording_mode in ['mic', 'both'] and self.mic_device:
                threads.append(threading.Thread(
                    target=self._record_audio_stream,
                    args=(p, self.mic_device, duration, audio_queue, "microphone")
                ))
            
            if not threads:
                logger.error("No recording threads started")
                return None
            
            # Start all threads
            for thread in threads:
                thread.start()
            
            logger.info(f"Recording for {duration}s...")
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Collect results
            system_audio, mic_audio = None, None
            system_rate, mic_rate = 44100, 44100
            
            while not audio_queue.empty():
                stream_name, audio_data, rate = audio_queue.get()
                if stream_name == "system audio":
                    system_audio, system_rate = audio_data, rate
                elif stream_name == "microphone":
                    mic_audio, mic_rate = audio_data, rate
            
            # Store for later use
            self.system_audio_data = system_audio
            self.mic_audio_data = mic_audio
            self.system_sample_rate = system_rate
            self.mic_sample_rate = mic_rate
            
            # Mix audio streams
            mixed_audio, final_sample_rate = self._mix_audio_streams(
                system_audio, mic_audio, system_rate, mic_rate
            )
            
            if mixed_audio is None:
                logger.error("No audio recorded from any source")
                return None
            
            # Convert to float32 for further processing
            mixed_audio_float = mixed_audio.astype(np.float32) / 32768.0
            
            # Apply preprocessing if enabled
            if self.enable_preprocessing:
                try:
                    mixed_audio_float = self.preprocessor.enhance_for_whisper(mixed_audio_float)
                except Exception as e:
                    logger.error(f"Audio preprocessing failed: {e}")
            
            # Resample to target sample rate if needed
            if final_sample_rate != self.sample_rate:
                mixed_audio_float = self._resample_audio(mixed_audio_float, final_sample_rate, self.sample_rate)
            
            logger.info(f"System audio recording completed: {len(mixed_audio_float)} samples")
            return mixed_audio_float
            
        except Exception as e:
            logger.error(f"System audio recording failed: {e}")
            return None
            
        finally:
            if p:
                p.terminate()
    
    def save_mixed_audio(self, audio_data: np.ndarray, filename: Optional[str] = None) -> Optional[Path]:
        """Save mixed audio data to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_audio_recording_{timestamp}.wav"
        
        try:
            filepath = OUTPUT_DIR / filename
            
            # Ensure proper type/range for saving
            if audio_data.dtype in (np.float32, np.float64):
                # Clip to valid range and convert to int16 PCM
                audio_clipped = np.clip(audio_data, -1.0, 1.0)
                audio_int16 = (audio_clipped * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16, copy=False)
            
            import scipy.io.wavfile as wavfile
            wavfile.write(filepath, self.sample_rate, audio_int16)
            
            file_size = filepath.stat().st_size
            logger.info(f"Mixed audio saved: {filepath} ({file_size/1024/1024:.2f} MB)")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving mixed audio: {e}")
            return None
    
    def save_separate_audio_files(self, base_filename: Optional[str] = None) -> Dict[str, Optional[Path]]:
        """Save system and microphone audio as separate files"""
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"recording_{timestamp}"
        
        saved_files = {}
        
        # Save system audio
        if self.system_audio_data is not None:
            system_filename = f"{base_filename}_system.wav"
            try:
                filepath = OUTPUT_DIR / system_filename
                import scipy.io.wavfile as wavfile
                wavfile.write(filepath, self.system_sample_rate, self.system_audio_data)
                saved_files['system'] = filepath
                logger.info(f"System audio saved: {filepath}")
            except Exception as e:
                logger.error(f"Error saving system audio: {e}")
                saved_files['system'] = None
        
        # Save microphone audio
        if self.mic_audio_data is not None:
            mic_filename = f"{base_filename}_microphone.wav"
            try:
                filepath = OUTPUT_DIR / mic_filename
                import scipy.io.wavfile as wavfile
                wavfile.write(filepath, self.mic_sample_rate, self.mic_audio_data)
                saved_files['microphone'] = filepath
                logger.info(f"Microphone audio saved: {filepath}")
            except Exception as e:
                logger.error(f"Error saving microphone audio: {e}")
                saved_files['microphone'] = None
        
        return saved_files
    
    def set_gains(self, system_gain: float = 0.7, mic_gain: float = 1.0):
        """Set audio gains for mixing"""
        self.system_gain = system_gain
        self.mic_gain = mic_gain
        logger.info(f"Audio gains set: System={system_gain}, Mic={mic_gain}")
    
    def set_normalization(self, enable: bool = True, target_level: float = 0.1):
        """Configure audio normalization"""
        self.auto_normalize = enable
        self.target_level = target_level
        logger.info(f"Normalization: {'enabled' if enable else 'disabled'}, target={target_level}")
    
    def cleanup(self):
        """Clean up system audio specific resources"""
        # Stop recording
        self.is_recording = False
        
        # Wait for recording threads to finish
        for thread in self.recording_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        self.recording_threads.clear()
        
        # Clear any remaining data
        self.system_audio_data = None
        self.mic_audio_data = None
        
        # Call parent cleanup
        super().cleanup()
    
    # Abstract method implementations required by BaseAudioCapture
    def start_real_time_recording(self):
        """Start real-time audio recording using pyaudiowpatch"""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return
        
        if self.recording_mode not in ['system', 'mic', 'both']:
            logger.error(f"Invalid recording mode: {self.recording_mode}")
            return
        
        logger.info(f"Starting real-time system audio recording in mode: {self.recording_mode}")
        
        # Reset recording state
        self.is_recording = True
        self.audio_buffer = []
        self.realtime_recording = []
        self.realtime_recording_raw = []
        self.recording_start_time = time.time()
        
        # Clear any existing data
        self.system_audio_data = None
        self.mic_audio_data = None
        
        # Start recording threads based on mode
        self.recording_threads = []
        
        try:
            p = pyaudio.PyAudio()
            
            # Start system audio recording if needed
            if self.recording_mode in ['system', 'both'] and self.system_device:
                system_thread = threading.Thread(
                    target=self._real_time_record_stream,
                    args=(p, self.system_device, "system_audio"),
                    daemon=True
                )
                system_thread.start()
                self.recording_threads.append(system_thread)
                logger.info("Started system audio recording thread")
            
            # Start microphone recording if needed
            if self.recording_mode in ['mic', 'both'] and self.mic_device:
                mic_thread = threading.Thread(
                    target=self._real_time_record_stream,
                    args=(p, self.mic_device, "microphone"),
                    daemon=True
                )
                mic_thread.start()
                self.recording_threads.append(mic_thread)
                logger.info("Started microphone recording thread")
            
            if not self.recording_threads:
                logger.error("No recording threads started")
                self.is_recording = False
                p.terminate()
                return
            
            # Start the mixing thread
            mixing_thread = threading.Thread(target=self._real_time_mixing_loop, daemon=True)
            mixing_thread.start()
            self.recording_threads.append(mixing_thread)
            
            logger.info("Real-time system audio recording started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start real-time recording: {e}")
            self.is_recording = False
            raise
    
    def stop_real_time_recording(self) -> Optional[np.ndarray]:
        """Stop real-time recording and return accumulated audio"""
        if not self.is_recording:
            logger.debug("Stop recording called but no recording in progress")
            # Return any accumulated recording if available
            if len(self.realtime_recording_raw) > 0: # Return raw recording
                logger.info(f"Returning accumulated raw recording of {len(self.realtime_recording_raw)} samples")
                return np.array(self.realtime_recording_raw)
            return None
        
        logger.info("Stopping real-time system audio recording...")
        
        # Stop recording flag
        self.is_recording = False
        
        # Wait for all recording threads to finish
        for thread in self.recording_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        self.recording_threads.clear()
        
        # Return the complete recording
        if len(self.realtime_recording_raw) > 0: # Return raw recording
            recording_duration = len(self.realtime_recording_raw) / self.sample_rate
            logger.info(f"Real-time recording completed: {recording_duration:.1f} seconds, {len(self.realtime_recording_raw)} samples")
            
            # Check audio quality
            recording_array = np.array(self.realtime_recording_raw)
            max_amplitude = np.max(np.abs(recording_array))
            rms_level = np.sqrt(np.mean(recording_array ** 2))
            logger.info(f"Recording quality: max={max_amplitude:.6f}, rms={rms_level:.6f}")
            
            return recording_array
        else:
            logger.warning("No audio data accumulated during real-time recording")
            return None
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get next audio chunk from queue for real-time processing"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _real_time_record_stream(self, p: pyaudio.PyAudio, device: Dict, stream_name: str):
        """Record audio from a single device in real-time"""
        stream = None
        try:
            # Find supported sample rate
            supported_rates = [48000, 44100, 32000, 22050, 16000]
            sample_rate = None
            channels = min(device['max_input_channels'], 2)
            
            for rate in supported_rates:
                try:
                    if p.is_format_supported(rate, 
                                              input_device=device['index'], 
                                              input_channels=channels, 
                                              input_format=pyaudio.paInt16):
                        sample_rate = rate
                        break
                except ValueError:
                    continue

            if sample_rate is None:
                logger.error(f"No supported sample rate found for {device['name']}")
                return

            # Open stream
            stream = p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=2048,
                input_device_index=device['index']
            )
            
            logger.info(f"Started real-time recording for {stream_name} at {sample_rate}Hz")
            
            # Record continuously until stopped
            while self.is_recording:
                try:
                    data = stream.read(2048, exception_on_overflow=False)
                    if data:
                        # Convert to numpy array
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        
                        # Convert to mono if stereo
                        if channels > 1:
                            audio_data = audio_data.reshape(-1, channels).mean(axis=1).astype(np.int16)

                        # Resample to target rate for real-time processing if needed
                        if sample_rate != self.sample_rate:
                            try:
                                audio_data = self._resample_audio(audio_data, sample_rate, self.sample_rate)
                                sample_rate = self.sample_rate
                            except Exception as e:
                                logger.error(f"Resampling failed in real-time stream ({stream_name}): {e}")
                                # Proceed with original data even if resampling fails
 
                        # Store raw data for this stream
                        if stream_name == "system_audio":
                            if not hasattr(self, '_system_audio_buffer'):
                                self._system_audio_buffer = []
                            self._system_audio_buffer.extend(audio_data)
                        elif stream_name == "microphone":
                            if not hasattr(self, '_mic_audio_buffer'):
                                self._mic_audio_buffer = []
                            self._mic_audio_buffer.extend(audio_data)
                        
                except Exception as e:
                    if self.is_recording:  # Only log if we're still supposed to be recording
                        logger.error(f"Error reading from {stream_name}: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Error in real-time recording for {stream_name}: {e}")
        finally:
            if stream:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            logger.info(f"Real-time recording stopped for {stream_name}")
    
    def _real_time_mixing_loop(self):
        """Mix audio streams in real-time and feed to audio queue"""
        chunk_size = int(self.sample_rate * 0.5)  # 0.5 second chunks
        last_process_time = time.time()
        
        while self.is_recording:
            try:
                time.sleep(0.1)  # Process every 100ms
                
                current_time = time.time()
                if current_time - last_process_time < 0.5:  # Process every 0.5 seconds
                    continue
                
                last_process_time = current_time
                
                # Get accumulated audio data
                system_audio = None
                mic_audio = None
                
                if hasattr(self, '_system_audio_buffer') and len(self._system_audio_buffer) >= chunk_size:
                    system_audio = np.array(self._system_audio_buffer[:chunk_size])
                    self._system_audio_buffer = self._system_audio_buffer[chunk_size:]
                
                if hasattr(self, '_mic_audio_buffer') and len(self._mic_audio_buffer) >= chunk_size:
                    mic_audio = np.array(self._mic_audio_buffer[:chunk_size])
                    self._mic_audio_buffer = self._mic_audio_buffer[chunk_size:]
                
                # Mix audio if we have data
                if system_audio is not None or mic_audio is not None:
                    # Handle different recording modes
                    if self.recording_mode == 'system' and system_audio is not None:
                        mixed_audio = system_audio
                    elif self.recording_mode == 'mic' and mic_audio is not None:
                        mixed_audio = mic_audio
                    elif self.recording_mode == 'both' and system_audio is not None and mic_audio is not None:
                        # Mix both streams
                        min_length = min(len(system_audio), len(mic_audio))
                        system_audio = system_audio[:min_length]
                        mic_audio = mic_audio[:min_length]
                        
                        # Apply gains and mix
                        system_float = system_audio.astype(np.float32) * self.system_gain
                        mic_float = mic_audio.astype(np.float32) * self.mic_gain
                        mixed = system_float + mic_float
                        
                        # Normalize to prevent clipping
                        max_val = np.max(np.abs(mixed))
                        if max_val > 32767:
                            mixed = mixed * (32767 / max_val)
                        
                        mixed_audio = mixed.astype(np.int16)
                    else:
                        continue  # No data to process yet
                    
                    # Convert to float32 for processing
                    mixed_audio_float_raw = mixed_audio.astype(np.float32) / 32768.0  # Preserve raw audio

                    # Apply preprocessing on a copy to avoid altering the raw version
                    mixed_audio_float_proc = mixed_audio_float_raw
                    if self.enable_preprocessing:
                        try:
                            mixed_audio_float_proc = self.preprocessor.enhance_for_whisper(mixed_audio_float_proc.copy())
                        except Exception as e:
                            logger.error(f"Audio preprocessing failed: {e}")

                    if len(mixed_audio_float_proc) > 0:
                        # Accumulate recordings
                        self.realtime_recording.extend(mixed_audio_float_proc)       # Processed (for ASR)
                        self.realtime_recording_raw.extend(mixed_audio_float_raw)    # Raw (for saving/playback)

                        # Push processed chunk for downstream real-time processing
                        try:
                            self.audio_queue.put((mixed_audio_float_proc, current_time), block=False)
                        except queue.Full:
                            logger.warning("Audio queue full, dropping chunk")
                
            except Exception as e:
                if self.is_recording:  # Only log if we're still supposed to be recording
                    logger.error(f"Error in mixing loop: {e}")
                time.sleep(0.1)
        
        logger.info("Real-time mixing loop stopped") 