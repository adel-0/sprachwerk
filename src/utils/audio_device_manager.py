"""
Audio Device Manager - Centralized audio device testing and selection
Uses pyaudiowpatch for Windows loopback audio recording
"""

import pyaudiowpatch as pyaudio
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
import sounddevice as sd

logger = logging.getLogger(__name__)

class AudioDeviceManager:
    """Centralized audio device management and testing using pyaudiowpatch"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.test_duration = 2.0

    def _get_device_info(self, device_index: int, p: pyaudio.PyAudio) -> Dict:
        """Get device information as dictionary"""
        device = p.get_device_info_by_index(device_index)
        host_api = p.get_host_api_info_by_index(device['hostApi'])
        return {
            'index': device_index,
            'name': device['name'],
            'max_input_channels': device['maxInputChannels'],
            'default_samplerate': device['defaultSampleRate'],
            'hostapi': host_api['name']
        }

    def _is_loopback(self, name: str) -> bool:
        """Check if device is a loopback device"""
        return 'loopback' in name.lower()

    def _is_microphone(self, name: str) -> bool:
        """Check if device is a microphone"""
        name_lower = name.lower()
        return (
            'loopback' not in name_lower and
            'stereo mix' not in name_lower and
            ('microphone' in name_lower or 'mic' in name_lower)
        )

    def list_all_devices(self) -> List[Dict]:
        """List all available audio devices"""
        devices = []
        try:
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                device = p.get_device_info_by_index(i)
                if device['maxInputChannels'] > 0:
                    devices.append(self._get_device_info(i, p))
            p.terminate()
        except Exception as e:
            logger.error(f"Error listing audio devices: {e}")
        return devices

    def get_loopback_devices(self) -> List[Dict]:
        """Get list of loopback devices"""
        devices = []
        try:
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                device = p.get_device_info_by_index(i)
                if device['maxInputChannels'] > 0 and self._is_loopback(device['name']):
                    devices.append(self._get_device_info(i, p))
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
                    devices.append(self._get_device_info(i, p))
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
                    info = self._get_device_info(i, p)
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
                    info = self._get_device_info(i, p)
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
        """Test a specific audio input device"""
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
            device_info = p.get_device_info_by_index(device_index)
            test_results['device_name'] = device_info['name']
            test_results['max_channels'] = device_info['maxInputChannels']
            
            if device_info['maxInputChannels'] <= 0:
                test_results['error'] = "Device has no input channels"
                return False, test_results
            
            # Find supported sample rate
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
                
                # Determine if device is working
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
            test_results['error'] = str(e)
            test_results['working'] = False
                
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
            is_working, test_results = self.test_device(device['index'], duration=1.0)
            if is_working:
                device['test_results'] = test_results
                working_devices.append(device)
                logger.info(f"✓ Device {device['index']}: {device['name']} - {test_results.get('signal_quality', 'unknown')}")
            else:
                logger.debug(f"✗ Device {device['index']}: {device['name']} - {test_results.get('error', 'failed')}")
        
        if working_devices:
            # Sort by signal quality and amplitude
            def device_score(device):
                test_results = device['test_results']
                quality_score = {'good': 3, 'weak': 2}.get(test_results.get('signal_quality'), 1)
                amplitude_score = test_results.get('max_amplitude', 0)
                return (quality_score, amplitude_score)
            
            best_device = max(working_devices, key=device_score)
            logger.info(f"Best device: {best_device['index']} - {best_device['name']}")
            return best_device
        
        logger.warning("No working input devices found")
        return None
    
    def auto_select_device(self) -> Optional[int]:
        """Automatically select the best working device"""
        best_device = self.find_best_device()
        return best_device['index'] if best_device else None
    
    def auto_select_output_device(self) -> Optional[int]:
        """Automatically select the best available output device and set sd.default.device[1]."""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    sd.default.device = (sd.default.device[0], i)
                    logger.info(f"Auto-selected output device: [{i}] {device['name']}")
                    return i
            logger.warning("No output device found")
        except Exception as e:
            logger.error(f"Error auto-selecting output device: {e}")
        return None
    
    def auto_select_default_output_loopback(self) -> Optional[Dict]:
        """Auto-select the loopback device corresponding to the current default output device (Windows WASAPI)."""
        import sounddevice as sd
        try:
            default_output = sd.query_devices(None, 'output')
            output_name = default_output['name'].lower()
            # Remove common suffixes/prefixes for robust matching
            def clean(name):
                name = name.lower()
                for suffix in [" (loopback)", " (wasapi)", " (windows wasapi)"]:
                    name = name.replace(suffix, "")
                return name.strip()
            cleaned_output = clean(output_name)
            best_match = None
            best_score = 0
            for i, device in enumerate(sd.query_devices()):
                if device['max_input_channels'] > 0 and 'loopback' in device['name'].lower():
                    cleaned_loopback = clean(device['name'])
                    # Heuristic: prefer exact, then partial, then any loopback
                    if cleaned_output == cleaned_loopback:
                        return {**device, 'index': i}
                    if cleaned_output in cleaned_loopback or cleaned_loopback in cleaned_output:
                        score = min(len(cleaned_output), len(cleaned_loopback))
                        if score > best_score:
                            best_score = score
                            best_match = {**device, 'index': i}
            if best_match:
                return best_match
            # Fallback: return any loopback device
            for i, device in enumerate(sd.query_devices()):
                if device['max_input_channels'] > 0 and 'loopback' in device['name'].lower():
                    return {**device, 'index': i}
        except Exception as e:
            logger.error(f"Error auto-selecting default output loopback: {e}")
        return None
    
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