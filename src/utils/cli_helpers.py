"""
CLI helper functions for language and speaker configuration
"""

from colorama import Fore, Style
from src.core.config import CONFIG, save_user_setting, save_user_settings_to_file
from src.utils.key_input import get_single_keypress


def _bulk_save(**kwargs):
    """Helper to bulk update CONFIG and save settings"""
    CONFIG.update(kwargs)
    for key, value in kwargs.items():
        save_user_setting(key, value)
    save_user_settings_to_file()


def apply_language_setting(language_input):
    """Apply language setting to CONFIG"""
    if language_input.lower() in ['auto', 'unknown', 'detect']:
        _bulk_save(
            whisper_language=None,
            whisper_multilingual_segments=True,
            whisper_language_constraints=None
        )
        print(f"🌐 Language set to: Auto-detect (any language)")
    elif ' ' in language_input:
        # Multiple languages specified (constrained multilingual)
        lang_codes = [code.strip().lower() for code in language_input.split() if len(code.strip()) == 2]
        if len(lang_codes) >= 2:
            _bulk_save(
                whisper_language=None,
                whisper_multilingual_segments=True,
                whisper_language_constraints=lang_codes
            )
            lang_display = ' + '.join(code.upper() for code in lang_codes)
            print(f"🌐 Language set to: {lang_display} (constrained multilingual)")
        else:
            print(f"⚠️  Invalid language format: {language_input}. Using auto-detect.")
            _bulk_save(
                whisper_language=None,
                whisper_multilingual_segments=True,
                whisper_language_constraints=None
            )
    else:
        # Single language
        _bulk_save(
            whisper_language=language_input.lower(),
            whisper_multilingual_segments=False,
            whisper_language_constraints=None
        )
        print(f"🌐 Language set to: {language_input.upper()}")


def apply_speaker_setting(speaker_input):
    """Apply speaker setting from command line argument"""
    try:
        if speaker_input.lower() in ['auto', 'auto-detect', 'autodetect']:
            # Auto-detect mode (1-10 speakers)
            _bulk_save(min_speakers=1, max_speakers=10)
            print(f"👥 Speakers set to: Auto-detect")
        elif '-' in speaker_input:
            # Range format like "1-3"
            min_spk, max_spk = map(int, speaker_input.split('-'))
            _bulk_save(min_speakers=min_spk, max_speakers=max_spk)
            print(f"👥 Speakers set to: {min_spk}-{max_spk} speakers")
        else:
            # Exact number
            num_speakers = int(speaker_input)
            _bulk_save(min_speakers=num_speakers, max_speakers=num_speakers)
            print(f"👥 Speakers set to: exactly {num_speakers} speaker{'s' if num_speakers != 1 else ''}")
    except ValueError:
        print(f"⚠️  Invalid speaker format: {speaker_input}. Using defaults.")


def get_language_preference():
    """Interactive language selection"""
    
    print(f"\n{Fore.YELLOW}Select the expected language(s) in your audio:{Style.RESET_ALL}")
    print(f"  1. {Fore.GREEN}Single Language{Style.RESET_ALL}")
    print(f"     {Fore.WHITE}   • Choose one specific language{Style.RESET_ALL}")
    print(f"  2. {Fore.BLUE}Multilingual (Constrained){Style.RESET_ALL}")
    print(f"     {Fore.WHITE}   • Specify possible languages for better accuracy{Style.RESET_ALL}")
    print(f"  3. {Fore.MAGENTA}Multilingual (Auto-detect){Style.RESET_ALL}")
    print(f"     {Fore.WHITE}   • Any language detection (slowest){Style.RESET_ALL}")
    
    while True:
        try:
            choice = get_single_keypress(
                f"\n{Fore.CYAN}Select language mode (1-3):{Style.RESET_ALL} ",
                ['1', '2', '3']
            )
            
            if choice is None:  # User cancelled
                print(f"\n{Fore.YELLOW}Using auto-detect as default.{Style.RESET_ALL}")
                CONFIG['whisper_language'] = None
                CONFIG['whisper_multilingual_segments'] = True
                CONFIG['whisper_language_constraints'] = None
                return
            elif choice == '1':
                return _select_single_language()
            elif choice == '2':
                return _select_constrained_multilingual()
            elif choice == '3':
                _bulk_save(
                    whisper_language=None,
                    whisper_multilingual_segments=True,
                    whisper_language_constraints=None
                )
                print(f"{Fore.GREEN}✓ Language set to: Auto-detect any language (multilingual){Style.RESET_ALL}")
                return
                
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Fore.YELLOW}Using auto-detect as default.{Style.RESET_ALL}")
            _bulk_save(
                whisper_language=None,
                whisper_multilingual_segments=True,
                whisper_language_constraints=None
            )
            return


def _select_single_language():
    """Select a single language"""
    
    print(f"\n{Fore.CYAN}🎯 SINGLE LANGUAGE SELECTION{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Choose the language for your audio:{Style.RESET_ALL}")
    
    # Common languages with their codes - keeping only English, German, French, Italian
    common_languages = [
        ('en', 'English'), ('de', 'German'), ('fr', 'French'), ('it', 'Italian')
    ]
    
    # Display options
    for i, (code, name) in enumerate(common_languages, 1):
        print(f"  {i:2d}. {Fore.BLUE}{name} ({code}){Style.RESET_ALL}")
    
    print(f"  {len(common_languages)+1:2d}. {Fore.MAGENTA}Other (enter language code){Style.RESET_ALL}")
    
    while True:
        try:
            choice = get_single_keypress(
                f"\n{Fore.CYAN}Select language (1-{len(common_languages)+1}):{Style.RESET_ALL} ",
                [str(i) for i in range(1, len(common_languages)+2)]
            )
            
            if choice is None:  # User cancelled
                print(f"\n{Fore.YELLOW}Using English as default.{Style.RESET_ALL}")
                CONFIG['whisper_language'] = 'en'
                CONFIG['whisper_multilingual_segments'] = False
                CONFIG['whisper_language_constraints'] = None
                save_user_setting('whisper_language', 'en')
                save_user_setting('whisper_multilingual_segments', False)
                save_user_setting('whisper_language_constraints', None)
                save_user_settings_to_file()
                return
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(common_languages):
                lang_code, lang_name = common_languages[choice_num - 1]
                CONFIG['whisper_language'] = lang_code
                CONFIG['whisper_multilingual_segments'] = False
                CONFIG['whisper_language_constraints'] = None
                save_user_setting('whisper_language', lang_code)
                save_user_setting('whisper_multilingual_segments', False)
                save_user_setting('whisper_language_constraints', None)
                save_user_settings_to_file()
                print(f"{Fore.GREEN}✓ Language set to: {lang_name} ({lang_code}){Style.RESET_ALL}")
                return
            elif choice_num == len(common_languages) + 1:
                # Custom language code
                lang_code = input(f"{Fore.YELLOW}Enter language code (e.g., 'sv' for Swedish):{Style.RESET_ALL} ").strip().lower()
                if len(lang_code) == 2 and lang_code.isalpha():
                    _bulk_save(
                        whisper_language=lang_code,
                        whisper_multilingual_segments=False,
                        whisper_language_constraints=None
                    )
                    print(f"{Fore.GREEN}✓ Language set to: {lang_code.upper()}{Style.RESET_ALL}")
                    return
                else:
                    print(f"{Fore.RED}Invalid language code. Please use 2-letter codes (e.g., 'sv', 'da').{Style.RESET_ALL}")
                
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Fore.YELLOW}Using English as default.{Style.RESET_ALL}")
            _bulk_save(
                whisper_language='en',
                whisper_multilingual_segments=False,
                whisper_language_constraints=None
            )
            return


def _select_constrained_multilingual():
    """Select constrained multilingual mode"""
    
    print(f"\n{Fore.CYAN}🌍 CONSTRAINED MULTILINGUAL SELECTION{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Specify the possible languages in your audio:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Examples:{Style.RESET_ALL}")
    print(f"  • {Fore.CYAN}en de{Style.RESET_ALL} - English and German")
    print(f"  • {Fore.CYAN}en fr it{Style.RESET_ALL} - English, French, and Italian")
    print(f"  • {Fore.CYAN}de fr{Style.RESET_ALL} - German and French")
    
    while True:
        try:
            lang_input = input(f"\n{Fore.CYAN}Enter language codes (space-separated):{Style.RESET_ALL} ").strip().lower()
            
            if not lang_input:
                print(f"{Fore.RED}Please enter at least one language code.{Style.RESET_ALL}")
                continue
            
            # Parse and validate language codes
            lang_codes = lang_input.split()
            valid_codes = [code for code in lang_codes if len(code) == 2 and code.isalpha()]
            
            if len(valid_codes) != len(lang_codes):
                print(f"{Fore.RED}Invalid language codes found. Please use 2-letter codes.{Style.RESET_ALL}")
                continue
            
            if len(valid_codes) < 2:
                print(f"{Fore.RED}Please enter at least 2 language codes for multilingual mode.{Style.RESET_ALL}")
                continue
            
            # Remove duplicates while preserving order
            unique_codes = list(dict.fromkeys(valid_codes))
            
            # Set configuration
            _bulk_save(
                whisper_language=None,
                whisper_multilingual_segments=True,
                whisper_language_constraints=unique_codes
            )
            
            lang_display = ' + '.join(code.upper() for code in unique_codes)
            print(f"{Fore.GREEN}✓ Language set to: {lang_display} (constrained multilingual){Style.RESET_ALL}")
            return
            
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Fore.YELLOW}Using auto-detect as default.{Style.RESET_ALL}")
            _bulk_save(
                whisper_language=None,
                whisper_multilingual_segments=True,
                whisper_language_constraints=None
            )
            return


def get_speaker_preference():
    """Interactive speaker count selection"""
    
    print(f"\n{Fore.YELLOW}How many speakers do you expect in your audio?{Style.RESET_ALL}")
    print(f"  1. {Fore.GREEN}Single Speaker (1 speaker){Style.RESET_ALL}")
    print(f"     {Fore.WHITE}   • Very permissive thresholds, aggressive consolidation{Style.RESET_ALL}")
    print(f"     {Fore.WHITE}   • Perfect for monologues, presentations, solo recordings{Style.RESET_ALL}")
    print(f"  2. {Fore.GREEN}Small Groups (2-3 speakers){Style.RESET_ALL}")
    print(f"     {Fore.WHITE}   • Moderately permissive thresholds, smart consolidation{Style.RESET_ALL}")
    print(f"     {Fore.WHITE}   • Merges low-activity speakers into main speakers{Style.RESET_ALL}")
    print(f"  3. {Fore.GREEN}Larger Groups (4+ speakers){Style.RESET_ALL}")
    print(f"     {Fore.WHITE}   • Standard thresholds with minimal consolidation{Style.RESET_ALL}")
    print(f"  4. {Fore.BLUE}Auto-detect{Style.RESET_ALL}")
    print(f"     {Fore.WHITE}   • Automatically detects number of speakers{Style.RESET_ALL}")
    print(f"     {Fore.WHITE}   • Adapts thresholds based on detected speaker count{Style.RESET_ALL}")
    
    speaker_options = [
        (1, 1, "Single Speaker"),
        (2, 3, "Small Groups (2-3 speakers)"),
        (4, 10, "Larger Groups (4+ speakers)"),
        (1, 10, "Auto-detect")
    ]
    
    while True:
        try:
            choice = get_single_keypress(
                f"\n{Fore.CYAN}Select speaker mode (1-4):{Style.RESET_ALL} ",
                ['1', '2', '3', '4']
            )
            
            if choice is None:  # User cancelled
                print(f"\n{Fore.YELLOW}Using Single Speaker mode as default.{Style.RESET_ALL}")
                CONFIG['min_speakers'] = 1
                CONFIG['max_speakers'] = 1
                return
            
            choice_idx = int(choice) - 1
            min_spk, max_spk, description = speaker_options[choice_idx]
            CONFIG['min_speakers'] = min_spk
            CONFIG['max_speakers'] = max_spk
            save_user_setting('min_speakers', min_spk)
            save_user_setting('max_speakers', max_spk)
            save_user_settings_to_file()
            print(f"{Fore.GREEN}✓ Speaker mode set to: {description}{Style.RESET_ALL}")
            return
                
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Fore.YELLOW}Using Single Speaker mode as default.{Style.RESET_ALL}")
            CONFIG['min_speakers'] = 1
            CONFIG['max_speakers'] = 1
            return


def get_adaptive_speaker_settings():
    """Interactive configuration of adaptive speaker threshold settings"""
    from src.core.config import CONFIG, save_user_setting, save_user_settings_to_file
    
    print(f"\n{Fore.CYAN}🎯 ADAPTIVE SPEAKER THRESHOLD CONFIGURATION{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}These settings help improve speaker detection accuracy based on expected speaker count.{Style.RESET_ALL}")
    
    current_enabled = CONFIG.get('enable_adaptive_speaker_thresholds', True)
    current_base = CONFIG.get('base_speaker_similarity_threshold', 0.65)
    current_boost = CONFIG.get('single_speaker_similarity_boost', 0.25)
    
    print(f"\n{Fore.CYAN}Current settings:{Style.RESET_ALL}")
    print(f"  • Adaptive thresholds: {Fore.GREEN if current_enabled else Fore.RED}{'Enabled' if current_enabled else 'Disabled'}{Style.RESET_ALL}")
    print(f"  • Base similarity threshold: {Fore.BLUE}{current_base:.2f}{Style.RESET_ALL}")
    print(f"  • Single speaker boost: {Fore.BLUE}{current_boost:.2f}{Style.RESET_ALL}")
    
    print(f"\n{Fore.YELLOW}What would you like to configure?{Style.RESET_ALL}")
    print(f"  1. {Fore.GREEN}Enable/Disable adaptive thresholds{Style.RESET_ALL}")
    print(f"  2. {Fore.GREEN}Adjust base similarity threshold{Style.RESET_ALL}")
    print(f"  3. {Fore.GREEN}Adjust single speaker boost{Style.RESET_ALL}")
    print(f"  4. {Fore.GREEN}Reset to defaults{Style.RESET_ALL}")
    print(f"  0. {Fore.YELLOW}Keep current settings{Style.RESET_ALL}")
    
    while True:
        try:
            choice = get_single_keypress(
                f"\n{Fore.CYAN}Select option (1-4, 0 to keep current):{Style.RESET_ALL} ",
                ['1', '2', '3', '4', '0']
            )
            
            if choice is None or choice == '0':  # User cancelled or keep current
                print(f"\n{Fore.YELLOW}Keeping current settings{Style.RESET_ALL}")
                return
            elif choice == '1':
                new_enabled = not current_enabled
                CONFIG['enable_adaptive_speaker_thresholds'] = new_enabled
                save_user_setting('enable_adaptive_speaker_thresholds', new_enabled)
                print(f"{Fore.GREEN}✓ Adaptive thresholds {'enabled' if new_enabled else 'disabled'}{Style.RESET_ALL}")
                save_user_settings_to_file()
                return
                
            elif choice == '2':
                try:
                    new_threshold = float(input(f"{Fore.YELLOW}Enter base similarity threshold (0.1-0.9, current: {current_base:.2f}):{Style.RESET_ALL} ").strip())
                    if 0.1 <= new_threshold <= 0.9:
                        CONFIG['base_speaker_similarity_threshold'] = new_threshold
                        save_user_setting('base_speaker_similarity_threshold', new_threshold)
                        print(f"{Fore.GREEN}✓ Base similarity threshold set to: {new_threshold:.2f}{Style.RESET_ALL}")
                        save_user_settings_to_file()
                        return
                    else:
                        print(f"{Fore.RED}Value must be between 0.1 and 0.9{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")
                    
            elif choice == '3':
                try:
                    new_boost = float(input(f"{Fore.YELLOW}Enter single speaker boost (0.0-0.5, current: {current_boost:.2f}):{Style.RESET_ALL} ").strip())
                    if 0.0 <= new_boost <= 0.5:
                        CONFIG['single_speaker_similarity_boost'] = new_boost
                        save_user_setting('single_speaker_similarity_boost', new_boost)
                        print(f"{Fore.GREEN}✓ Single speaker boost set to: {new_boost:.2f}{Style.RESET_ALL}")
                        save_user_settings_to_file()
                        return
                    else:
                        print(f"{Fore.RED}Value must be between 0.0 and 0.5{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")
                    
            elif choice == '4':
                CONFIG['enable_adaptive_speaker_thresholds'] = True
                CONFIG['base_speaker_similarity_threshold'] = 0.65
                CONFIG['single_speaker_similarity_boost'] = 0.25
                CONFIG['single_speaker_clustering_boost'] = 0.15
                save_user_setting('enable_adaptive_speaker_thresholds', True)
                save_user_setting('base_speaker_similarity_threshold', 0.65)
                save_user_setting('single_speaker_similarity_boost', 0.25)
                save_user_setting('single_speaker_clustering_boost', 0.15)
                print(f"{Fore.GREEN}✓ Reset to default adaptive threshold settings{Style.RESET_ALL}")
                save_user_settings_to_file()
                return
                
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Fore.YELLOW}Keeping current settings{Style.RESET_ALL}")
            return 