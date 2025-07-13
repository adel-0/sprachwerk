"""
sprachwerk - Offline Whisper + Speaker Diarization
Main entry point for the application
"""

import logging

# Apply logging suppressions first
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.utils.warning_suppressor import setup_logging_suppressions
setup_logging_suppressions()

# Configure logging after suppressions
logging.basicConfig(level=logging.WARNING)

import argparse

from src.core.app import TranscriptionApp
from src.core.config import set_mode, CONFIG, bulk_update
from src.utils.cli_helpers import apply_language_setting, apply_speaker_setting

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="sprachwerk - Offline Whisper + Speaker Diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive mode selection
  python main.py --mode batch --duration 30        # Batch mode, 30 seconds
  python main.py --mode realtime                    # Real-time mode
  python main.py --mode batch --file audio.wav     # Process existing file
  python main.py -l "en de" -s "2-3" --mode batch  # Constrained multilingual, 2-3 speakers
  python main.py -i 1 --mode realtime              # Use specific audio device
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['batch', 'realtime', 'interactive'],
        default='interactive',
        help='Processing mode: batch (record then process), realtime (live transcription), or interactive (menu-driven) (default: interactive)'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=int,
        help='Recording duration in seconds (batch mode only)'
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Audio file to process (batch mode only)'
    )
    
    parser.add_argument(
        '--language', '-l',
        type=str,
        help='Language: single code (e.g., "en"), multiple codes for constrained multilingual (e.g., "en de fr"), or "auto" for detection'
    )
    
    parser.add_argument(
        '--speakers', '-s',
        type=str,
        help='Number of speakers: exact number (e.g., "2"), range (e.g., "1-3"), or "auto" for auto-detect'
    )
    
    parser.add_argument(
        '--device-index', '-i',
        type=int,
        help='Audio input device index (use interactive mode to see available devices, or omit for auto-detection)'
    )
    
    parser.add_argument(
        '--help-devices',
        action='store_true',
        help='Show available audio devices and exit'
    )
    
    args = parser.parse_args()
    
    # Handle help-devices flag
    if args.help_devices:
        show_audio_devices()
        return 0
    
    # Set configuration mode based on processing mode
    if args.mode in ['batch', 'realtime']:
        set_mode(args.mode)
    
    # Apply command-line settings
    if args.language:
        apply_language_setting(args.language)
    if args.speakers:
        apply_speaker_setting(args.speakers)
    if args.device_index is not None:
        apply_device_setting(args.device_index)
    
    # Create and run the application
    app = TranscriptionApp()
    
    try:
        # Run based on mode
        if args.mode == 'interactive':
            run_interactive_mode(app, args)
        elif args.mode == 'batch':
            # Initialize models and setup audio before batch processing
            if not app.initialize_models():
                print("Failed to initialize models. Exiting.")
                return 1
            if not app.setup_audio():
                print("Failed to setup audio. Exiting.")
                return 1
                
            app.run_batch_mode(duration=args.duration, input_file=args.file)
        elif args.mode == 'realtime':
            # Initialize models and setup audio before real-time processing
            if not app.initialize_models():
                print("Failed to initialize models. Exiting.")
                return 1
            if not app.setup_audio():
                print("Failed to setup audio. Exiting.")
                return 1
                
            app.run_real_time_mode()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        app.stop_processing()

def run_interactive_mode(app, args):
    """Run interactive mode with comprehensive settings menu"""
    from colorama import Fore, Style
    from src.utils.interactive_menu import InteractiveMenu
    
    # Create and run the interactive menu
    menu = InteractiveMenu(app, args)
    menu.run()

def show_audio_devices():
    """Show available audio devices"""
    from colorama import Fore, Style
    from src.utils.audio_device_manager import AudioDeviceManager
    
    print(f"{Fore.CYAN}Available Audio Input Devices:{Style.RESET_ALL}")
    print("=" * 60)
    
    device_manager = AudioDeviceManager()
    devices = device_manager.list_all_devices()
    
    if not devices:
        print(f"{Fore.RED}No audio input devices found.{Style.RESET_ALL}")
        return
    
    for device in devices:
        print(f"{Fore.YELLOW}Device {device['index']}:{Style.RESET_ALL} {device['name']}")
        print(f"  Channels: {device['max_input_channels']}")
        print(f"  Sample Rate: {device['default_samplerate']} Hz")
        print(f"  Host API: {device['hostapi']}")
        print()

def apply_device_setting(device_index):
    """Apply audio device setting from command line"""
    from src.utils.audio_device_manager import AudioDeviceManager
    
    try:
        # Validate device exists
        device_manager = AudioDeviceManager()
        devices = device_manager.list_all_devices()
        valid_indices = [device['index'] for device in devices]
        
        if device_index not in valid_indices:
            print(f"⚠️  Invalid device index: {device_index}. Available devices: {valid_indices}")
            print("Use --help-devices to see all available devices.")
            return
        
        # Apply setting
        bulk_update(audio_device_index=device_index)
        
        # Get device name for confirmation
        device_name = next((device['name'] for device in devices if device['index'] == device_index), f"Device {device_index}")
        print(f"🎤 Audio device set to: {device_name} (index {device_index})")
        
    except Exception as e:
        print(f"⚠️  Error setting audio device: {e}")

if __name__ == "__main__":
    sys.exit(main()) 