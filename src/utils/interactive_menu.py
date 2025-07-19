"""
Interactive Menu System for sprachwerk
Provides comprehensive settings management and mode selection
"""

import logging
import msvcrt
from pathlib import Path
from colorama import Fore, Style

from src.core.config import CONFIG, save_user_setting, save_user_settings_to_file, DiarizationBackend, OUTPUT_DIR
from src.utils.cli_helpers import apply_language_setting, apply_speaker_setting
from src.utils.audio_device_manager import AudioDeviceManager
from src.utils.speaker_manager import SpeakerManager
from src.utils.key_input import get_single_keypress

logger = logging.getLogger(__name__)


def prompt_choice(prompt, choices, allow_enter_for_default=False, default_choice=None):
    """Helper for consistent choice input."""
    try:
        return get_single_keypress(prompt, choices, allow_enter_for_default=allow_enter_for_default, default_choice=default_choice)
    except (EOFError, KeyboardInterrupt):
        return None

def prompt_input(prompt):
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return None

class InteractiveMenu:
    """Comprehensive interactive menu for sprachwerk"""
    def __init__(self, app, args):
        self.app = app
        self.args = args
        self.speaker_manager = SpeakerManager()

    def run(self):
        self._show_welcome()
        while True:
            self._show_quick_start_menu()
            choice = prompt_choice(
                f"\n{Fore.CYAN}Select option (1-4, 0 to exit):{Style.RESET_ALL} ",
                ['1', '2', '3', '4', '0']
            )
            if choice is None or choice == '0':
                print(f"{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                break
            if choice == '1':
                if self._quick_confirm_and_start():
                    self._start_transcription()
                break
            elif choice == '2':
                duration = self._get_recording_duration()
                if duration is not None and self._quick_confirm_and_start():
                    self.app.run_batch_mode(duration=duration)
                    break
            elif choice == '3':
                file_path = self._get_audio_file_path()
                if file_path and self._quick_confirm_and_start():
                    self.app.run_batch_mode(input_file=file_path)
                    break
            elif choice == '4':
                self._comprehensive_settings_menu()

    def _show_welcome(self):
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🎙️  SPRACHWERK - INTERACTIVE CONFIGURATION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Configure your transcription settings before starting{Style.RESET_ALL}")

    def _show_quick_start_menu(self):
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🚀 START MENU{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        self._display_current_settings()
        print(f"\n{Fore.YELLOW}Available Options:{Style.RESET_ALL}")
        print(f"  1. {Fore.GREEN}⚡ Start Real-time{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Live transcription with speaker identification{Style.RESET_ALL}")
        print(f"  2. {Fore.BLUE}🎯 Start Batch{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Record and process audio in one go{Style.RESET_ALL}")
        print(f"  3. {Fore.MAGENTA}📁 Process File{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Process pre-recorded audio files{Style.RESET_ALL}")
        print(f"  4. {Fore.CYAN}⚙️ Settings{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Configuration options{Style.RESET_ALL}")
        print(f"  0. {Fore.YELLOW}🚪 Exit{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Quit application{Style.RESET_ALL}")

    def _display_current_settings(self):
        print(f"\n{Fore.CYAN}📋 Current Settings:{Style.RESET_ALL}")
        constraints = CONFIG.get('whisper_language_constraints')
        if CONFIG.get('whisper_language'):
            lang_display = CONFIG['whisper_language'].upper()
            lang_mode = ""
        elif constraints:
            lang_display = ' + '.join(code.upper() for code in constraints)
            lang_mode = " (constrained multilingual)"
        else:
            lang_display = "Auto-detect"
            lang_mode = " (any language)" if CONFIG.get('whisper_multilingual_segments', False) else ""
        if CONFIG['min_speakers'] == 1 and CONFIG['max_speakers'] == 10:
            speaker_display = "Auto-detect"
        elif CONFIG['min_speakers'] != CONFIG['max_speakers']:
            speaker_display = f"{CONFIG['min_speakers']}-{CONFIG['max_speakers']} speakers"
        else:
            speaker_display = f"{CONFIG['min_speakers']} speaker"
        device_info = "Auto-detect"
        if CONFIG.get('audio_device_index') is not None:
            try:
                import sounddevice as sd
                device = sd.query_devices(CONFIG['audio_device_index'])
                device_info = f"Device {CONFIG['audio_device_index']}: {device['name']}"
            except:
                device_info = f"Device {CONFIG['audio_device_index']} (Invalid)"
        if hasattr(self.app, 'audio_capture') and self.app.audio_capture:
            processing_mode = self.app.audio_capture.get_processing_mode()
        else:
            processing_mode = "Pre-processing" if CONFIG.get('enable_audio_preprocessing', True) else "Raw Audio (No Processing)"
        audio_source_mode = CONFIG.get('system_audio_recording_mode', 'mic')
        audio_source_display = {
            'system': 'System Audio Only',
            'mic': 'Microphone Only',
            'both': 'System Audio + Microphone'
        }.get(audio_source_mode, 'Microphone Only')
        print(f"  🌐 Language: {Fore.GREEN}{lang_display}{lang_mode}{Style.RESET_ALL}")
        print(f"  👥 Speakers: {Fore.GREEN}{speaker_display}{Style.RESET_ALL}")
        print(f"  🔊 Audio Source: {Fore.GREEN}{audio_source_display}{Style.RESET_ALL}")

    def _speaker_management_menu(self):
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🎭 SPEAKER MANAGEMENT{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        try:
            self.speaker_manager.run_speaker_management_menu()
        except Exception as e:
            print(f"{Fore.RED}Error in speaker management: {e}{Style.RESET_ALL}")
            logger.error(f"Speaker management error: {e}")
        print(f"{Fore.GREEN}Returning to main menu...{Style.RESET_ALL}")

    def _get_recording_duration(self):
        if self.args.duration:
            return self.args.duration
        print(f"\n{Fore.YELLOW}Recording Duration Options:{Style.RESET_ALL}")
        print(f"  1. 30 seconds (default)")
        print(f"  2. 1 minute")
        print(f"  3. 2 minutes")
        print(f"  4. 5 minutes")
        print(f"  5. Custom duration")
        choice = prompt_choice(
            f"\n{Fore.CYAN}Select duration (1-5, or Enter for default):{Style.RESET_ALL} ",
            ['1', '2', '3', '4', '5'],
            allow_enter_for_default=True,
            default_choice='1'
        )
        if choice is None:
            return None
        if choice == '1':
            print(f"{Fore.GREEN}✓ Using 30 seconds{Style.RESET_ALL}")
            return 30
        if choice == '2':
            print(f"{Fore.GREEN}✓ Using 1 minute{Style.RESET_ALL}")
            return 60
        if choice == '3':
            print(f"{Fore.GREEN}✓ Using 2 minutes{Style.RESET_ALL}")
            return 120
        if choice == '4':
            print(f"{Fore.GREEN}✓ Using 5 minutes{Style.RESET_ALL}")
            return 300
        if choice == '5':
            duration_input = prompt_input(f"{Fore.YELLOW}Enter duration in seconds:{Style.RESET_ALL} ")
            try:
                duration = int(duration_input)
                if duration > 0:
                    print(f"{Fore.GREEN}✓ Using {duration} seconds{Style.RESET_ALL}")
                    return duration
                else:
                    print(f"{Fore.RED}Duration must be positive{Style.RESET_ALL}")
            except (TypeError, ValueError):
                print(f"{Fore.RED}Invalid duration. Please enter a number.{Style.RESET_ALL}")
        return None

    def _get_audio_file_path(self):
        if self.args.file:
            return self.args.file
        print(f"\n{Fore.YELLOW}💡 Supported formats: WAV, MP3, M4A, FLAC, OGG, AAC{Style.RESET_ALL}")
        while True:
            file_path = prompt_input(f"{Fore.CYAN}Enter audio file path (or 'back' to return):{Style.RESET_ALL} ")
            if file_path is None or file_path.lower() == 'back':
                return None
            if file_path and Path(file_path).exists():
                supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
                file_ext = Path(file_path).suffix.lower()
                if file_ext in supported_formats:
                    print(f"{Fore.GREEN}✓ File selected: {Path(file_path).name}{Style.RESET_ALL}")
                    return file_path
                else:
                    print(f"{Fore.RED}Unsupported format. Supported: {', '.join(supported_formats)}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}File not found: {file_path}{Style.RESET_ALL}")

    def _language_selection_menu(self):
        from src.utils.cli_helpers import get_language_preference
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🌐 LANGUAGE SELECTION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        get_language_preference()

    def _speaker_count_menu(self):
        from src.utils.cli_helpers import get_speaker_preference
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}👥 SPEAKER CONFIGURATION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        get_speaker_preference()

    def _audio_device_menu(self):
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🎤 AUDIO DEVICE SELECTION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        device_manager = AudioDeviceManager()
        input_devices = device_manager.list_all_devices()
        if not input_devices:
            print(f"{Fore.RED}No input devices available!{Style.RESET_ALL}")
            return
        print(f"\n{Fore.YELLOW}Available input devices:{Style.RESET_ALL}")
        for device in input_devices:
            print(f"  {device['index']}: {Fore.WHITE}{device['name']}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Options:{Style.RESET_ALL}")
        print(f"  1. {Fore.GREEN}Auto-detect device{Style.RESET_ALL}")
        print(f"  2. {Fore.BLUE}Select specific device{Style.RESET_ALL}")
        print(f"  3. {Fore.CYAN}Test device{Style.RESET_ALL}")
        print(f"  0. {Fore.YELLOW}🔙 Back to audio source menu{Style.RESET_ALL}")
        choice = prompt_choice(
            f"\n{Fore.CYAN}Select option (1-3, 0 to go back):{Style.RESET_ALL} ",
            ['1', '2', '3', '0']
        )
        if choice == '1':
            CONFIG['audio_device_index'] = None
            save_user_setting('audio_device_index', None)
            save_user_settings_to_file()
            print(f"{Fore.GREEN}✓ Audio device set to auto-detect{Style.RESET_ALL}")
        elif choice == '2':
            self._select_specific_device(input_devices)
        elif choice == '3':
            self._test_audio_device(input_devices)

    def _select_specific_device(self, input_devices):
        print(f"\n{Fore.YELLOW}Select device by number:{Style.RESET_ALL}")
        device_input = prompt_input(f"{Fore.CYAN}Enter device number:{Style.RESET_ALL} ")
        if device_input and device_input.isdigit():
            device_idx = int(device_input)
            if any(device['index'] == device_idx for device in input_devices):
                CONFIG['audio_device_index'] = device_idx
                save_user_setting('audio_device_index', device_idx)
                save_user_settings_to_file()
                device_name = next(device['name'] for device in input_devices if device['index'] == device_idx)
                print(f"{Fore.GREEN}✓ Selected device {device_idx}: {device_name}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Invalid device number. Please select from the list above.{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Please enter a valid device number.{Style.RESET_ALL}")

    def _test_audio_device(self, input_devices):
        import sounddevice as sd
        import numpy as np
        print(f"\n{Fore.YELLOW}Audio Device Testing{Style.RESET_ALL}")
        print(f"Select a device to test (or press Enter to test current device):")
        device_input = prompt_input(f"{Fore.CYAN}Enter device number (or Enter for current):{Style.RESET_ALL} ")
        if device_input == '':
            device_idx = CONFIG.get('audio_device_index')
            device_name = next((device['name'] for device in input_devices if device['index'] == device_idx), "Unknown") if device_idx is not None else "Auto-detect"
        elif device_input and device_input.isdigit():
            device_idx = int(device_input)
            device_name = next((device['name'] for device in input_devices if device['index'] == device_idx), "Unknown")
        else:
            print(f"{Fore.RED}Please enter a valid device number or press Enter.{Style.RESET_ALL}")
            return
        print(f"{Fore.YELLOW}Testing device {device_idx}: {device_name}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Recording for 3 seconds... Speak into your microphone!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Press Ctrl+C to cancel recording{Style.RESET_ALL}")
        try:
            duration = 3
            sample_rate = 44100
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, device=device_idx)
            sd.wait()
            rms = np.sqrt(np.mean(recording**2))
            db_level = 20 * np.log10(rms) if rms > 0 else -np.inf
            print(f"{Fore.GREEN}✓ Test completed!{Style.RESET_ALL}")
            print(f"  Audio level: {db_level:.1f} dB")
            if db_level > -30:
                print(f"  {Fore.GREEN}Good audio level detected!{Style.RESET_ALL}")
            elif db_level > -50:
                print(f"  {Fore.YELLOW}Low audio level - consider speaking louder or adjusting microphone{Style.RESET_ALL}")
            else:
                print(f"  {Fore.RED}Very low/no audio detected - check microphone connection{Style.RESET_ALL}")
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Recording cancelled by user{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Test failed: {e}{Style.RESET_ALL}")

    def _start_transcription(self):
        self.app.run_real_time_mode()

    def _audio_quality_menu(self):
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🎵 AUDIO QUALITY SETTINGS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Audio Processing Options:{Style.RESET_ALL}")
        print(f"  1. {Fore.RED}Raw Audio (No Processing){Style.RESET_ALL}")
        print(f"  2. {Fore.GREEN}Pre-processing (Recommended){Style.RESET_ALL}")
        print(f"  0. {Fore.YELLOW}🔙 Back to advanced settings{Style.RESET_ALL}")
        choice = prompt_choice(
            f"\n{Fore.CYAN}Select quality mode (1-2, 0 to go back):{Style.RESET_ALL} ",
            ['1', '2', '0']
        )
        if choice == '1':
            self.app.audio_capture.use_raw_audio_mode(show_info=True)
            save_user_setting('enable_audio_preprocessing', False)
            save_user_settings_to_file()
            print(f"{Fore.GREEN}✓ Raw audio mode enabled and saved{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}💡 Audio will be saved exactly as captured from microphone{Style.RESET_ALL}")
        elif choice == '2':
            self.app.audio_capture.use_minimal_processing(show_info=True)
            save_user_setting('enable_audio_preprocessing', True)
            save_user_settings_to_file()
            print(f"{Fore.GREEN}✓ Pre-processing mode enabled and saved{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}💡 Essential cleanup applied for natural sound{Style.RESET_ALL}")

    def _audio_source_menu(self):
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🔊 AUDIO SOURCE SELECTION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Audio Source Options:{Style.RESET_ALL}")
        print(f"  1. {Fore.GREEN}Microphone Only{Style.RESET_ALL}")
        print(f"  2. {Fore.BLUE}System Audio Only{Style.RESET_ALL}")
        print(f"  3. {Fore.MAGENTA}Both (System + Microphone){Style.RESET_ALL}")
        print(f"  4. {Fore.YELLOW}🎤 Configure Input Device{Style.RESET_ALL}")
        print(f"  5. {Fore.CYAN}🎛️ Configure Output Device{Style.RESET_ALL}")
        print(f"  6. {Fore.YELLOW}⚙️ Advanced System Audio Settings{Style.RESET_ALL}")
        print(f"  0. {Fore.YELLOW}🔙 Back to main menu{Style.RESET_ALL}")
        choice = prompt_choice(
            f"\n{Fore.CYAN}Select audio source (1-6, 0 to go back):{Style.RESET_ALL} ",
            ['1', '2', '3', '4', '5', '6', '0']
        )
        if choice == '1':
            CONFIG['system_audio_recording_mode'] = 'mic'
            save_user_setting('system_audio_recording_mode', 'mic')
            save_user_settings_to_file()
            print(f"{Fore.GREEN}✓ Audio source set to: Microphone Only{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}💡 Will record from your selected microphone/input device{Style.RESET_ALL}")
        elif choice == '2':
            CONFIG['system_audio_recording_mode'] = 'system'
            save_user_setting('system_audio_recording_mode', 'system')
            save_user_settings_to_file()
            print(f"{Fore.GREEN}✓ Audio source set to: System Audio Only{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}💡 Will record your computer's audio output{Style.RESET_ALL}")
        elif choice == '3':
            CONFIG['system_audio_recording_mode'] = 'both'
            save_user_setting('system_audio_recording_mode', 'both')
            save_user_settings_to_file()
            print(f"{Fore.GREEN}✓ Audio source set to: System Audio + Microphone{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}💡 Will record both system audio and microphone{Style.RESET_ALL}")
        elif choice == '4':
            self._audio_device_menu()
        elif choice == '5':
            self._configure_system_audio_devices()
        elif choice == '6':
            self._system_audio_advanced_menu()

    def _system_audio_advanced_menu(self):
        """Advanced system audio configuration menu"""
        while True:
            print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}⚙️  ADVANCED SYSTEM AUDIO SETTINGS{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
            
            print(f"\n{Fore.YELLOW}Advanced Options:{Style.RESET_ALL}")
            print(f"  1. {Fore.MAGENTA}🔊 Configure Audio Gains{Style.RESET_ALL}")
            print(f"     {Fore.WHITE}   • Adjust system and microphone volume levels{Style.RESET_ALL}")
            print(f"  2. {Fore.CYAN}📊 Configure Audio Normalization{Style.RESET_ALL}")
            print(f"     {Fore.WHITE}   • Set audio level balancing options{Style.RESET_ALL}")
            print(f"  3. {Fore.YELLOW}🎵 Audio Quality Settings{Style.RESET_ALL}")
            print(f"     {Fore.WHITE}   • Configure audio processing options{Style.RESET_ALL}")
            print(f"  0. {Fore.YELLOW}🔙 Back to audio source menu{Style.RESET_ALL}")
            
            try:
                choice = get_single_keypress(
                    f"\n{Fore.CYAN}Select option (1-3, 0 to go back):{Style.RESET_ALL} ",
                    ['1', '2', '3', '0']
                )
                
                if choice is None or choice == '0':  # User cancelled or back
                    break
                elif choice == '1':
                    self._configure_audio_gains()
                    # Continue loop to redisplay menu after configuring gains
                elif choice == '2':
                    self._configure_normalization()
                    # Continue loop to redisplay menu after configuring normalization
                elif choice == '3':
                    self._audio_quality_menu()
                    # Continue loop to redisplay menu after configuring quality
                    
            except (EOFError, KeyboardInterrupt):
                break
    
    def _list_system_audio_devices(self):
        """List all available system audio devices"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📋 SYSTEM AUDIO DEVICES{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        try:
            import sounddevice as sd
            
            print(f"\n{Fore.YELLOW}Available System Audio Devices:{Style.RESET_ALL}")
            
            devices = sd.query_devices()
            output_devices = []
            
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:  # Output device
                    output_devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_output_channels'],
                        'sample_rate': device['default_samplerate']
                    })
            
            if output_devices:
                for device in output_devices:
                    print(f"  {device['index']:2d}: {Fore.WHITE}{device['name']}{Style.RESET_ALL}")
                    print(f"      {Fore.CYAN}Channels: {device['channels']}, Sample Rate: {device['sample_rate']:.0f} Hz{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}No output devices found!{Style.RESET_ALL}")
            
            print(f"\n{Fore.GREEN}Press any key to continue...{Style.RESET_ALL}")
            try:
                msvcrt.getch()
            except KeyboardInterrupt:
                pass
            
        except Exception as e:
            print(f"{Fore.RED}Error listing devices: {e}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Press any key to continue...{Style.RESET_ALL}")
            try:
                msvcrt.getch()
            except KeyboardInterrupt:
                pass
    
    def _configure_system_audio_devices(self):
        """Configure system audio device selection"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🎛️  CONFIGURE SYSTEM AUDIO DEVICES{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        try:
            import sounddevice as sd
            
            devices = sd.query_devices()
            output_devices = []
            
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:  # Output device
                    output_devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_output_channels']
                    })
            
            if not output_devices:
                print(f"{Fore.RED}No output devices available!{Style.RESET_ALL}")
                print(f"{Fore.GREEN}Press any key to continue...{Style.RESET_ALL}")
                try:
                    msvcrt.getch()
                except KeyboardInterrupt:
                    pass
                return
            
            print(f"\n{Fore.YELLOW}Available Output Devices:{Style.RESET_ALL}")
            for device in output_devices:
                print(f"  {device['index']:2d}: {Fore.WHITE}{device['name']}{Style.RESET_ALL}")
            
            print(f"\n{Fore.YELLOW}Options:{Style.RESET_ALL}")
            print(f"  1. {Fore.GREEN}Auto-detect system audio device{Style.RESET_ALL}")
            print(f"  2. {Fore.BLUE}Select specific device{Style.RESET_ALL}")
            print(f"  0. {Fore.YELLOW}🔙 Back{Style.RESET_ALL}")
            
            while True:
                try:
                    choice = get_single_keypress(
                        f"\n{Fore.CYAN}Select option (1-2, 0 to go back):{Style.RESET_ALL} ",
                        ['1', '2', '0']
                    )
                    
                    if choice is None or choice == '0':  # User cancelled or back
                        break
                    elif choice == '1':
                        CONFIG['system_audio_device_index'] = None
                        save_user_setting('system_audio_device_index', None)
                        save_user_settings_to_file()
                        print(f"{Fore.GREEN}✓ System audio device set to auto-detect{Style.RESET_ALL}")
                        break
                    elif choice == '2':
                        device_input = input(f"{Fore.CYAN}Enter device number:{Style.RESET_ALL} ").strip()
                        
                        if device_input.isdigit():
                            device_idx = int(device_input)
                            if any(device['index'] == device_idx for device in output_devices):
                                CONFIG['system_audio_device_index'] = device_idx
                                save_user_setting('system_audio_device_index', device_idx)
                                save_user_settings_to_file()
                                device_name = next(device['name'] for device in output_devices if device['index'] == device_idx)
                                print(f"{Fore.GREEN}✓ System audio device set to: {device_name}{Style.RESET_ALL}")
                                break
                            else:
                                print(f"{Fore.RED}Invalid device number. Please select from the list above.{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.RED}Please enter a valid device number.{Style.RESET_ALL}")
                            
                except (EOFError, KeyboardInterrupt):
                    break
                    
        except Exception as e:
            print(f"{Fore.RED}Error configuring devices: {e}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Press any key to continue...{Style.RESET_ALL}")
            try:
                msvcrt.getch()
            except KeyboardInterrupt:
                pass
    
    def _configure_audio_gains(self):
        """Configure audio gain settings"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🔊 CONFIGURE AUDIO GAINS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        current_system_gain = CONFIG.get('system_audio_gain', 1.0)
        current_mic_gain = CONFIG.get('microphone_gain', 1.0)
        
        print(f"\n{Fore.CYAN}Current Settings:{Style.RESET_ALL}")
        print(f"  • System Audio Gain: {Fore.BLUE}{current_system_gain:.1f}x{Style.RESET_ALL}")
        print(f"  • Microphone Gain: {Fore.BLUE}{current_mic_gain:.1f}x{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}Audio Gain Options:{Style.RESET_ALL}")
        print(f"  1. {Fore.GREEN}Configure System Audio Gain{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Adjust volume level of system audio{Style.RESET_ALL}")
        print(f"  2. {Fore.GREEN}Configure Microphone Gain{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Adjust volume level of microphone{Style.RESET_ALL}")
        print(f"  3. {Fore.BLUE}Reset to Defaults{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Set both gains to 1.0x (no change){Style.RESET_ALL}")
        print(f"  0. {Fore.YELLOW}🔙 Back{Style.RESET_ALL}")
        
        while True:
            try:
                choice = get_single_keypress(
                    f"\n{Fore.CYAN}Select option (1-3, 0 to go back):{Style.RESET_ALL} ",
                    ['1', '2', '3', '0']
                )
                
                if choice is None or choice == '0':  # User cancelled or back
                    break
                elif choice == '1':
                    # Configure system audio gain
                    try:
                        gain_input = input(f"{Fore.YELLOW}Enter system audio gain (0.1-5.0, current: {current_system_gain:.1f}):{Style.RESET_ALL} ").strip()
                        gain = float(gain_input)
                        if 0.1 <= gain <= 5.0:
                            CONFIG['system_audio_gain'] = gain
                            save_user_setting('system_audio_gain', gain)
                            save_user_settings_to_file()
                            print(f"{Fore.GREEN}✓ System audio gain set to: {gain:.1f}x{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.RED}Gain must be between 0.1 and 5.0{Style.RESET_ALL}")
                    except ValueError:
                        print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")
                        
                elif choice == '2':
                    # Configure microphone gain
                    try:
                        gain_input = input(f"{Fore.YELLOW}Enter microphone gain (0.1-5.0, current: {current_mic_gain:.1f}):{Style.RESET_ALL} ").strip()
                        gain = float(gain_input)
                        if 0.1 <= gain <= 5.0:
                            CONFIG['microphone_gain'] = gain
                            save_user_setting('microphone_gain', gain)
                            save_user_settings_to_file()
                            print(f"{Fore.GREEN}✓ Microphone gain set to: {gain:.1f}x{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.RED}Gain must be between 0.1 and 5.0{Style.RESET_ALL}")
                    except ValueError:
                        print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")
                        
                elif choice == '3':
                    # Reset to defaults
                    CONFIG['system_audio_gain'] = 1.0
                    CONFIG['microphone_gain'] = 1.0
                    save_user_setting('system_audio_gain', 1.0)
                    save_user_setting('microphone_gain', 1.0)
                    save_user_settings_to_file()
                    print(f"{Fore.GREEN}✓ Audio gains reset to defaults (1.0x){Style.RESET_ALL}")
                    
            except (EOFError, KeyboardInterrupt):
                break
    
    def _configure_normalization(self):
        """Configure audio normalization settings"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 CONFIGURE AUDIO NORMALIZATION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        current_enabled = CONFIG.get('enable_audio_normalization', True)
        current_target = CONFIG.get('normalization_target_db', -20.0)
        
        print(f"\n{Fore.CYAN}Current Settings:{Style.RESET_ALL}")
        print(f"  • Normalization: {Fore.GREEN if current_enabled else Fore.RED}{'Enabled' if current_enabled else 'Disabled'}{Style.RESET_ALL}")
        print(f"  • Target Level: {Fore.BLUE}{current_target:.1f} dB{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}Normalization Options:{Style.RESET_ALL}")
        print(f"  1. {Fore.GREEN}Enable/Disable Normalization{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Turn audio level normalization on/off{Style.RESET_ALL}")
        print(f"  2. {Fore.GREEN}Configure Target Level{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Set desired audio level in dB{Style.RESET_ALL}")
        print(f"  3. {Fore.BLUE}Reset to Defaults{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Enable normalization at -20.0 dB{Style.RESET_ALL}")
        print(f"  0. {Fore.YELLOW}🔙 Back{Style.RESET_ALL}")
        
        while True:
            try:
                choice = get_single_keypress(
                    f"\n{Fore.CYAN}Select option (1-3, 0 to go back):{Style.RESET_ALL} ",
                    ['1', '2', '3', '0']
                )
                
                if choice is None or choice == '0':  # User cancelled or back
                    break
                elif choice == '1':
                    # Toggle normalization
                    new_enabled = not current_enabled
                    CONFIG['enable_audio_normalization'] = new_enabled
                    save_user_setting('enable_audio_normalization', new_enabled)
                    save_user_settings_to_file()
                    print(f"{Fore.GREEN}✓ Audio normalization {'enabled' if new_enabled else 'disabled'}{Style.RESET_ALL}")
                    current_enabled = new_enabled
                    
                elif choice == '2':
                    # Configure target level
                    try:
                        target_input = input(f"{Fore.YELLOW}Enter target level in dB (-60.0 to 0.0, current: {current_target:.1f}):{Style.RESET_ALL} ").strip()
                        target = float(target_input)
                        if -60.0 <= target <= 0.0:
                            CONFIG['normalization_target_db'] = target
                            save_user_setting('normalization_target_db', target)
                            save_user_settings_to_file()
                            print(f"{Fore.GREEN}✓ Normalization target set to: {target:.1f} dB{Style.RESET_ALL}")
                            current_target = target
                        else:
                            print(f"{Fore.RED}Target level must be between -60.0 and 0.0 dB{Style.RESET_ALL}")
                    except ValueError:
                        print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")
                        
                elif choice == '3':
                    # Reset to defaults
                    CONFIG['enable_audio_normalization'] = True
                    CONFIG['normalization_target_db'] = -20.0
                    save_user_setting('enable_audio_normalization', True)
                    save_user_setting('normalization_target_db', -20.0)
                    save_user_settings_to_file()
                    print(f"{Fore.GREEN}✓ Normalization settings reset to defaults{Style.RESET_ALL}")
                    current_enabled = True
                    current_target = -20.0
                    
            except (EOFError, KeyboardInterrupt):
                break
    
    def _show_start_confirmation(self):
        """Show detailed start confirmation with current settings"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🚀 READY TO START TRANSCRIPTION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        # Display current settings
        self._display_current_settings()
        
        print(f"\n{Fore.GREEN}Output Location:{Style.RESET_ALL}")
        print(f"  📁 {OUTPUT_DIR}")
        
        print(f"\n{Fore.YELLOW}Ready to start?{Style.RESET_ALL}")
        print(f"  Y. {Fore.GREEN}Yes, start transcription{Style.RESET_ALL}")
        print(f"  N. {Fore.RED}No, go back to menu{Style.RESET_ALL}")
        
        while True:
            try:
                choice = get_single_keypress(
                    f"\n{Fore.CYAN}Start transcription? (Y/N):{Style.RESET_ALL} ",
                    ['y', 'n']
                )
                
                if choice is None or choice == 'n':  # User cancelled or no
                    return False
                elif choice == 'y':
                    return True
                    
            except (EOFError, KeyboardInterrupt):
                return False
    
    def _quick_confirm_and_start(self):
        """Start immediately without confirmation"""
        return True
    
    def _comprehensive_settings_menu(self):
        """Settings menu with all configuration options"""
        while True:
            print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}⚙️  SETTINGS MENU{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
            
            # Display current settings
            self._display_current_settings()
            
            print(f"\n{Fore.YELLOW}Configuration Options:{Style.RESET_ALL}")
            print(f"  1. {Fore.BLUE}🌐 Language Selection{Style.RESET_ALL}")
            print(f"     {Fore.WHITE}   • Set transcription language(s){Style.RESET_ALL}")
            print(f"  2. {Fore.MAGENTA}👥 Number of Speakers{Style.RESET_ALL}")
            print(f"     {Fore.WHITE}   • Configure speaker count{Style.RESET_ALL}")
            print(f"  3. {Fore.BLUE}🔊 Audio Source Selection{Style.RESET_ALL}")
            print(f"     {Fore.WHITE}   • Choose system audio, microphone, or both{Style.RESET_ALL}")
            print(f"  4. {Fore.YELLOW}🎭 Speaker Management{Style.RESET_ALL}")
            print(f"     {Fore.WHITE}   • Rename speakers, merge duplicates, manage database{Style.RESET_ALL}")
            print(f"  0. {Fore.YELLOW}🔙 Back to Main Menu{Style.RESET_ALL}")
            
            try:
                choice = get_single_keypress(
                    f"\n{Fore.CYAN}Select option (1-4, 0 to go back):{Style.RESET_ALL} ",
                    ['1', '2', '3', '4', '0']
                )
                
                if choice is None or choice == '0':  # User cancelled or back
                    break
                elif choice == '1':
                    self._language_selection_menu()
                elif choice == '2':
                    self._speaker_count_menu()
                elif choice == '3':
                    self._audio_source_menu()
                elif choice == '4':
                    self._speaker_management_menu()
                    
            except (EOFError, KeyboardInterrupt):
                break
    
 
