"""
Audio Device Manager - Centralized audio device testing and selection
"""

import sounddevice as sd
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
    """Centralized audio device management and testing"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.test_duration = 2.0  # Default test duration
        
    def list_all_devices(self) -> List[Dict]:
        """List all available audio devices with detailed information"""
        devices = []
        try:
            sd_devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            
            for idx, device in enumerate(sd_devices):
                if device['max_input_channels'] > 0:  # Only input devices
                    hostapi_name = hostapis[device['hostapi']]['name']
                    devices.append({
                        'index': idx,
                        'name': device['name'],
                        'max_input_channels': device['max_input_channels'],
                        'default_samplerate': device['default_samplerate'],
                        'hostapi': hostapi_name
                    })
        except Exception as e:
            logger.error(f"Error listing audio devices: {e}")
            
        return devices
    
    def get_input_devices(self) -> List[Dict]:
        """Get list of input-capable devices"""
        return [device for device in self.list_all_devices() if device['max_input_channels'] > 0]
    
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
        
        try:
            # Get device info
            device_info = sd.query_devices(device_index)
            test_results['device_name'] = device_info['name']
            test_results['max_channels'] = device_info['max_input_channels']
            
            # Check if device has input capabilities
            if device_info['max_input_channels'] <= 0:
                test_results['error'] = "Device has no input channels"
                return False, test_results
            
            # Store original device and set test device
            original_device = sd.default.device[0]
            sd.default.device[0] = device_index
            
            try:
                # Quick connectivity test first
                quick_recording = sd.rec(
                    int(0.5 * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype=np.float32,
                    device=device_index
                )
                sd.wait()
                
                # Check for data corruption
                if np.any(np.isnan(quick_recording)) or np.any(np.isinf(quick_recording)):
                    test_results['error'] = "Device returned corrupted data (NaN/Inf values)"
                    return False, test_results
                
                # Full test recording
                recording = sd.rec(
                    int(duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype=np.float32,
                    device=device_index
                )
                sd.wait()
                
                # Analyze the recording
                max_amplitude = np.max(np.abs(recording))
                rms_amplitude = np.sqrt(np.mean(recording**2))
                
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
                
            finally:
                # Restore original device
                sd.default.device[0] = original_device
                
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
        
        return test_results['working'], test_results
    
    def find_best_device(self) -> Optional[Dict]:
        """Test all input devices and find the best working one"""
        input_devices = self.get_input_devices()
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