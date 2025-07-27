#!/usr/bin/env python3
"""
Test to verify audio capture selection
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.app import TranscriptionApp
from src.core.processing_modes import RealTimeProcessingMode
from src.core.config import CONFIG

def main():
    print("=== Testing Audio Capture Selection ===")
    
    # Set configuration to microphone only
    CONFIG['system_audio_recording_mode'] = 'mic'
    
    # Create app
    app = TranscriptionApp()
    
    # Create processing mode
    processing_mode = RealTimeProcessingMode(
        app.transcriber, 
        app.diarizer, 
        app.aligner, 
        app.formatter
    )
    
    print(f"Configuration: system_audio_recording_mode = {CONFIG.get('system_audio_recording_mode')}")
    print(f"ProcessingMode.use_system_audio = {processing_mode.use_system_audio}")
    
    # Test _get_audio_capture
    audio_capture = processing_mode._get_audio_capture(app.audio_capture, app.system_audio_capture)
    print(f"Selected audio capture type: {type(audio_capture).__name__}")
    
    # Test _setup_audio_capture
    setup_result = processing_mode._setup_audio_capture(app.audio_capture, app.system_audio_capture)
    print(f"Setup result: {setup_result}")
    
    # Now test what happens when we call start_real_time_recording
    print(f"\nTesting start_real_time_recording...")
    try:
        # This should call AudioCapture.start_real_time_recording(), not SystemAudioCapture.start_real_time_recording()
        audio_capture.start_real_time_recording()
        print(f"✓ start_real_time_recording() called successfully on {type(audio_capture).__name__}")
        
        # Stop recording
        audio_capture.stop_real_time_recording()
        print(f"✓ stop_real_time_recording() called successfully")
        
    except Exception as e:
        print(f"✗ Error calling start_real_time_recording(): {e}")

if __name__ == "__main__":
    main() 