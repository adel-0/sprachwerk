"""
Audio Device Manager - Centralized audio device testing and selection
Uses pyaudiowpatch for Windows loopback audio recording
"""

import pyaudiowpatch as pyaudio
import numpy as np
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    from colorama import Fore, Style
except ImportError:
    # Fallback if colorama is not available
    class DummyColor:
        def __getattr__(self, name):
            return ""
    Fore = Style = DummyColor()

logger = logging.getLogger(__name__)

class AudioDeviceManager:
    """Centralized audio device management and testing using pyaudiowpatch"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.test_duration = 2.0  # Default test duration

    def _device_info_dict(self, p, i):
        device = p.get_device_info_by_index(i)
        host_api = p.get_host_api_info_by_index(device['hostApi'])
        return {
            'index': i,
            'name': device['name'],
            'max_input_channels': device['maxInputChannels'],
            'default_samplerate': device['defaultSampleRate'],
            'hostapi': host_api['name']
        }

    def _is_loopback(self, name: str) -> bool:
        name = name.lower()
        return '[loopback]' in name or 'loopback' in name

    def _is_microphone(self, name: str) -> bool:
        name = name.lower()
        return (
            '[loopback]' not in name and
            'stereo mix' not in name and
            ('microphone' in name or 'mic' in name)
        )

    def list_all_devices(self) -> List[Dict]:
        """List all available audio devices with detailed information"""
        devices = []
        try:
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                device = p.get_device_info_by_index(i)
                if device['maxInputChannels'] > 0:
                    devices.append(self._device_info_dict(p, i))
            p.terminate()
        except Exception as e:
            logger.error(f"Error listing audio devices: {e}")
        return devices

    def get_loopback_devices(self) -> List[Dict]:
        """Get list of loopback devices for system audio recording"""
        devices = []
        try:
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                device = p.get_device_info_by_index(i)
                if device['maxInputChannels'] > 0 and self._is_loopback(device['name']):
                    devices.append(self._device_info_dict(p, i))
            p.terminate()
        except Exception as e:
            logger.error(f"Error getting loopback devices: {e}")
        return devices

    def get_microphone_devices(self) -> List[Dict]:
        """Get list of microphone devices"""
        devices = []
        try:
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                device = p.get_device_info_by_index(i)
                if device['maxInputChannels'] > 0 and self._is_microphone(device['name']):
                    devices.append(self._device_info_dict(p, i))
            p.terminate()
        except Exception as e:
            logger.error(f"Error getting microphone devices: {e}")
        return devices

    def get_best_loopback_device(self) -> Optional[Dict]:
        """Get the best loopback device for system audio recording (prefer WASAPI)"""
        best = None
        try:
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                device = p.get_device_info_by_index(i)
                if device['maxInputChannels'] > 0 and self._is_loopback(device['name']):
                    info = self._device_info_dict(p, i)
                    if info['hostapi'] == 'Windows WASAPI':
                        logger.info(f"Found WASAPI loopback device: [{i}] {device['name']}")
                        p.terminate()
                        return info
                    if not best:
                        best = info
            p.terminate()
        except Exception as e:
            logger.error(f"Error getting best loopback device: {e}")
        if not best:
            logger.warning("No loopback device found")
        return best

    def get_best_microphone_device(self) -> Optional[Dict]:
        """Get the best microphone device (prefer WASAPI)"""
        best = None
        try:
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                device = p.get_device_info_by_index(i)
                if device['maxInputChannels'] > 0 and self._is_microphone(device['name']):
                    info = self._device_info_dict(p, i)
                    if info['hostapi'] == 'Windows WASAPI':
                        logger.info(f"Found WASAPI microphone device: [{i}] {device['name']}")
                        p.terminate()
                        return info
                    if not best:
                        best = info
            p.terminate()
        except Exception as e:
            logger.error(f"Error getting best microphone device: {e}")
        if not best:
            logger.warning("No microphone device found")
        return best

    def test_device(self, device_index: int, duration: float = None) -> Tuple[bool, Dict]:
        """
        Test a specific audio input device comprehensively
        
        Returns:
            Tuple of (is_working, test_results)
        """
        if duration is None:
            duration = self.test_duration
            
        test_results = {
            'device_index': device_index,
            'duration_tested': duration,
            'max_amplitude': 0.0,
            'rms_amplitude': 0.0,
            'error': None,
            'working': False
        }
        
        p = None
        stream = None
        try:
            p = pyaudio.PyAudio()
            
            # Get device info
            device_info = p.get_device_info_by_index(device_index)
            test_results['device_name'] = device_info['name']
            test_results['max_channels'] = device_info['maxInputChannels']
            
            # Check if device has input capabilities
            if device_info['maxInputChannels'] <= 0:
                test_results['error'] = "Device has no input channels"
                return False, test_results
            
            # Find supported sample rate - prioritize 48000 Hz for better audio quality
            supported_rates = [48000, 44100, 32000, 22050, 16000]
            sample_rate = None
            channels = min(device_info['maxInputChannels'], 2)
            
            for rate in supported_rates:
                try:
                    if p.is_format_supported(rate, 
                                              input_device=device_index, 
                                              input_channels=channels, 
                                              input_format=pyaudio.paInt16):
                        sample_rate = rate
                        break
                except ValueError:
                    continue

            if sample_rate is None:
                test_results['error'] = "No supported sample rate found"
                return False, test_results
                
            # Test recording
            stream = p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=2048,
                input_device_index=device_index
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
                
                # Convert to float for analysis
                recording_float = recording.astype(np.float32) / 32768.0
                
                # Analyze the recording
                max_amplitude = np.max(np.abs(recording_float))
                rms_amplitude = np.sqrt(np.mean(recording_float**2))
                
                test_results['max_amplitude'] = float(max_amplitude)
                test_results['rms_amplitude'] = float(rms_amplitude)
                
                # Determine if device is working based on signal strength
                if max_amplitude > 0.001:
                    test_results['working'] = True
                    test_results['signal_quality'] = 'good'
                elif max_amplitude > 0.0001:
                    test_results['working'] = True
                    test_results['signal_quality'] = 'weak'
                else:
                    test_results['working'] = False
                    test_results['signal_quality'] = 'none'
                    test_results['error'] = "No signal detected"
            else:
                test_results['error'] = "No audio data recorded"
                test_results['working'] = False
                
        except Exception as e:
            error_msg = str(e)
            test_results['error'] = error_msg
            test_results['working'] = False
            
            # Provide specific guidance for common errors
            if "PaErrorCode -9999" in error_msg or "WDM-KS" in error_msg:
                test_results['error_type'] = 'driver_error'
                test_results['suggestion'] = "Windows WDM-KS driver error - try different device"
            elif "Invalid number of channels" in error_msg:
                test_results['error_type'] = 'channel_error'
                test_results['suggestion'] = "Channel configuration error"
            elif "Device unavailable" in error_msg:
                test_results['error_type'] = 'unavailable'
                test_results['suggestion'] = "Device may be in use by another application"
            else:
                test_results['error_type'] = 'unknown'
                test_results['suggestion'] = "Try restarting audio service or using different device"
                
        finally:
            if stream:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            if p:
                p.terminate()
        
        return test_results['working'], test_results
    
    def find_best_device(self) -> Optional[Dict]:
        """Test all input devices and find the best working one"""
        input_devices = self.list_all_devices()
        working_devices = []
        
        logger.info(f"Testing {len(input_devices)} input devices...")
        
        for device in input_devices:
            is_working, test_results = self.test_device(device['index'], duration=1.0)  # Shorter test for auto-selection
            if is_working:
                device['test_results'] = test_results
                working_devices.append(device)
                logger.info(f"✓ Device {device['index']}: {device['name']} - {test_results.get('signal_quality', 'unknown')}")
            else:
                logger.debug(f"✗ Device {device['index']}: {device['name']} - {test_results.get('error', 'failed')}")
        
        if working_devices:
            # Sort by signal quality (good > weak) and then by max amplitude
            def device_score(device):
                test_results = device['test_results']
                quality_score = {'good': 3, 'weak': 2}.get(test_results.get('signal_quality'), 1)
                amplitude_score = test_results.get('max_amplitude', 0)
                return (quality_score, amplitude_score)
            
            best_device = max(working_devices, key=device_score)
            logger.info(f"Best device: {best_device['index']} - {best_device['name']}")
            return best_device
        else:
            logger.warning("No working input devices found")
            return None
    
    def auto_select_device(self) -> Optional[int]:
        """Automatically select the best working device"""
        best_device = self.find_best_device()
        return best_device['index'] if best_device else None
    
    def print_device_list(self):
        """Print a formatted list of all available devices"""
        print("\n--- Available Audio Devices ---")
        
        all_devices = self.list_all_devices()
        loopback_devices = self.get_loopback_devices()
        microphone_devices = self.get_microphone_devices()
        
        print("Input Devices:")
        for device in all_devices:
            print(f"  [{device['index']}] {device['name']}")
            print(f"      Channels: {device['max_input_channels']}, Rate: {device['default_samplerate']}, API: {device['hostapi']}")
            
            if device in loopback_devices:
                print("      *** LOOPBACK DEVICE (System Audio) ***")
            elif device in microphone_devices:
                print("      *** MICROPHONE DEVICE ***")
            print()
        
        if loopback_devices:
            print("*** LOOPBACK DEVICES FOUND (for system audio) ***")
            for device in loopback_devices:
                print(f"  [{device['index']}] {device['name']}")
        
        if microphone_devices:
            print("*** MICROPHONE DEVICES FOUND ***")
            for device in microphone_devices:
                print(f"  [{device['index']}] {device['name']}")