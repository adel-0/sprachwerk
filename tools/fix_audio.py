#!/usr/bin/env python3
"""
Quick Audio Fix - Automatically detect and resolve common audio issues
Run this script when you encounter audio device problems
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import sounddevice as sd
import numpy as np
from src.audio.capture import AudioCapture
from src.core.config import CONFIG
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("🔧 AUDIO QUICK FIX TOOL")
    print("=" * 50)
    print("This tool will automatically detect and fix common audio issues.\n")
    
    # Step 1: List all devices
    print("📋 Step 1: Scanning audio devices...")
    try:
        devices = sd.query_devices()
        input_devices = []
        
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'index': idx,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
        
        if not input_devices:
            print("❌ No input devices found!")
            print("   Check that your microphone is connected.")
            return False
        
        print(f"✅ Found {len(input_devices)} input device(s)")
        
    except Exception as e:
        print(f"❌ Failed to scan devices: {e}")
        return False
    
    # Step 2: Test current configuration
    print("\n🧪 Step 2: Testing current audio configuration...")
    try:
        audio_capture = AudioCapture()
        
        # Reset device selection to force re-detection
        CONFIG['audio_device_index'] = None
        sd.default.device[0] = None
        
        # Try to select and test a device
        if audio_capture.select_device():
            print("✅ Device selection successful")
            
            if audio_capture.test_audio_input(duration=1):
                print("✅ Audio test successful!")
                current_device = sd.default.device[0]
                if current_device is not None:
                    device_info = sd.query_devices(current_device)
                    print(f"   Using device: {device_info['name']}")
                    
                    # Update config with working device
                    CONFIG['audio_device_index'] = current_device
                    print(f"   Updated config to use device {current_device}")
                
                print("\n🎉 Audio is working correctly!")
                print("   You can now run your transcription application.")
                return True
            else:
                print("⚠️  Device selected but audio test failed")
        else:
            print("❌ Device selection failed")
            
    except Exception as e:
        print(f"❌ Audio test failed: {e}")
    
    # Step 3: Try to find a working device manually
    print("\n🔍 Step 3: Searching for working audio device...")
    
    working_devices = []
    for device in input_devices:
        print(f"   Testing device {device['index']}: {device['name'][:50]}...")
        
        try:
            # Set device
            sd.default.device[0] = device['index']
            
            # Quick test
            recording = sd.rec(
                int(0.5 * 16000),  # 0.5 second test
                samplerate=16000,
                channels=1,
                dtype=np.float32,
                device=device['index']
            )
            sd.wait()
            
            # Check if it worked
            if recording is not None and len(recording) > 0:
                if not (np.any(np.isnan(recording)) or np.any(np.isinf(recording))):
                    working_devices.append(device)
                    print(f"   ✅ Device {device['index']} works!")
                else:
                    print(f"   ❌ Device {device['index']} returned corrupted data")
            else:
                print(f"   ❌ Device {device['index']} returned no data")
                
        except Exception as e:
            error_msg = str(e)
            if "PaErrorCode -9999" in error_msg:
                print(f"   ❌ Device {device['index']} - Windows driver error")
            elif "Invalid number of channels" in error_msg:
                print(f"   ❌ Device {device['index']} - Channel error")
            else:
                print(f"   ❌ Device {device['index']} - {error_msg}")
    
    if working_devices:
        print(f"\n✅ Found {len(working_devices)} working device(s):")
        for i, device in enumerate(working_devices):
            print(f"   {i+1}. Device {device['index']}: {device['name']}")
        
        # Use the first working device
        best_device = working_devices[0]
        CONFIG['audio_device_index'] = best_device['index']
        sd.default.device[0] = best_device['index']
        
        print(f"\n🎯 Selected device {best_device['index']}: {best_device['name']}")
        print("   Configuration updated!")
        
        # Final test
        print("\n🧪 Final test...")
        audio_capture = AudioCapture()
        if audio_capture.test_audio_input(duration=1):
            print("🎉 SUCCESS! Audio is now working correctly.")
            return True
        else:
            print("⚠️  Device works but final test failed - you may need to adjust volume")
            return True
    else:
        print("\n❌ No working audio devices found.")
        print("\nTroubleshooting suggestions:")
        print("1. Check that your microphone is connected and powered on")
        print("2. Make sure no other applications are using the microphone")
        print("3. Try restarting the Windows Audio service:")
        print("   - Press Win+R, type 'services.msc', press Enter")
        print("   - Find 'Windows Audio', right-click, select 'Restart'")
        print("4. Check Windows Sound settings:")
        print("   - Right-click speaker icon in system tray")
        print("   - Select 'Open Sound settings'")
        print("   - Check 'Input' section and test your microphone")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Audio fix completed successfully!")
        else:
            print("\n❌ Could not automatically fix audio issues.")
            print("   Please run 'python tools/audio_device_helper.py' for manual configuration.")
    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("   Please run 'python tools/audio_device_helper.py' for manual configuration.") 