"""
CLI helper functions for language and speaker configuration
"""

from colorama import Fore, Style
from src.core.config import CONFIG, save_user_setting, save_user_settings_to_file
from src.utils.key_input import get_single_keypress

def _bulk_save(**kwargs):
    CONFIG.update(kwargs)
    for key, value in kwargs.items():
        save_user_setting(key, value)
    save_user_settings_to_file()

def apply_language_setting(language_input):
    lang = language_input.lower()
    if lang in {'auto', 'unknown', 'detect'}:
        _bulk_save(
            whisper_language=None,
            whisper_multilingual_segments=True,
            whisper_language_constraints=None
        )
        print("🌐 Language set to: Auto-detect (any language)")
        return
    lang_codes = [code.strip().lower() for code in lang.split() if len(code.strip()) == 2]
    if len(lang_codes) >= 2:
        _bulk_save(
            whisper_language=None,
            whisper_multilingual_segments=True,
            whisper_language_constraints=lang_codes
        )
        print(f"🌐 Language set to: {' + '.join(code.upper() for code in lang_codes)} (constrained multilingual)")
        return
    if len(lang) == 2 and lang.isalpha():
        _bulk_save(
            whisper_language=lang,
            whisper_multilingual_segments=False,
            whisper_language_constraints=None
        )
        print(f"🌐 Language set to: {lang.upper()}")
        return
    print(f"⚠️  Invalid language format: {language_input}. Using auto-detect.")
    _bulk_save(
        whisper_language=None,
        whisper_multilingual_segments=True,
        whisper_language_constraints=None
    )

def apply_speaker_setting(speaker_input):
    try:
        val = speaker_input.lower()
        if val in {'auto', 'auto-detect', 'autodetect'}:
            _bulk_save(min_speakers=1, max_speakers=10)
            print("👥 Speakers set to: Auto-detect")
            return
        if '-' in val:
            min_spk, max_spk = map(int, val.split('-'))
            _bulk_save(min_speakers=min_spk, max_speakers=max_spk)
            print(f"👥 Speakers set to: {min_spk}-{max_spk} speakers")
            return
        num_speakers = int(val)
        _bulk_save(min_speakers=num_speakers, max_speakers=num_speakers)
        print(f"👥 Speakers set to: exactly {num_speakers} speaker{'s' if num_speakers != 1 else ''}")
    except ValueError:
        print(f"⚠️  Invalid speaker format: {speaker_input}. Using defaults.")

def get_language_preference():
    print(f"\n{Fore.YELLOW}Select the expected language(s) in your audio:{Style.RESET_ALL}")
    print(f"  1. {Fore.GREEN}Single Language{Style.RESET_ALL}")
    print(f"     {Fore.WHITE}   • Choose one specific language{Style.RESET_ALL}")
    print(f"  2. {Fore.BLUE}Multilingual (Constrained){Style.RESET_ALL}")
    print(f"     {Fore.WHITE}   • Specify possible languages for better accuracy{Style.RESET_ALL}")
    print(f"  3. {Fore.MAGENTA}Multilingual (Auto-detect){Style.RESET_ALL}")
    print(f"     {Fore.WHITE}   • Any language detection (slowest){Style.RESET_ALL}")
    try:
        choice = get_single_keypress(
            f"\n{Fore.CYAN}Select language mode (1-3):{Style.RESET_ALL} ",
            ['1', '2', '3']
        )
        if choice == '1':
            _select_single_language()
        elif choice == '2':
            _select_constrained_multilingual()
        else:
            _bulk_save(
                whisper_language=None,
                whisper_multilingual_segments=True,
                whisper_language_constraints=None
            )
            print(f"{Fore.GREEN}✓ Language set to: Auto-detect any language (multilingual){Style.RESET_ALL}")
    except (EOFError, KeyboardInterrupt):
        print(f"\n{Fore.YELLOW}Using auto-detect as default.{Style.RESET_ALL}")
        _bulk_save(
            whisper_language=None,
            whisper_multilingual_segments=True,
            whisper_language_constraints=None
        )

def _select_single_language():
    print(f"\n{Fore.CYAN}🎯 SINGLE LANGUAGE SELECTION{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Choose the language for your audio:{Style.RESET_ALL}")
    common_languages = [
        ('en', 'English'), ('de', 'German'), ('fr', 'French'), ('it', 'Italian')
    ]
    for i, (code, name) in enumerate(common_languages, 1):
        print(f"  {i:2d}. {Fore.BLUE}{name} ({code}){Style.RESET_ALL}")
    print(f"  {len(common_languages)+1:2d}. {Fore.MAGENTA}Other (enter language code){Style.RESET_ALL}")
    try:
        choice = get_single_keypress(
            f"\n{Fore.CYAN}Select language (1-{len(common_languages)+1}):{Style.RESET_ALL} ",
            [str(i) for i in range(1, len(common_languages)+2)]
        )
        if choice is None:
            lang_code = 'en'
        else:
            choice_num = int(choice)
            if 1 <= choice_num <= len(common_languages):
                lang_code = common_languages[choice_num - 1][0]
            else:
                lang_code = input(f"{Fore.YELLOW}Enter language code (e.g., 'sv' for Swedish):{Style.RESET_ALL} ").strip().lower()
                if not (len(lang_code) == 2 and lang_code.isalpha()):
                    print(f"{Fore.RED}Invalid language code. Please use 2-letter codes (e.g., 'sv', 'da').{Style.RESET_ALL}")
                    return _select_single_language()
        _bulk_save(
            whisper_language=lang_code,
            whisper_multilingual_segments=False,
            whisper_language_constraints=None
        )
        print(f"{Fore.GREEN}✓ Language set to: {lang_code.upper()}{Style.RESET_ALL}")
    except (EOFError, KeyboardInterrupt):
        print(f"\n{Fore.YELLOW}Using English as default.{Style.RESET_ALL}")
        _bulk_save(
            whisper_language='en',
            whisper_multilingual_segments=False,
            whisper_language_constraints=None
        )

def _select_constrained_multilingual():
    print(f"\n{Fore.CYAN}🌍 CONSTRAINED MULTILINGUAL SELECTION{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Specify the possible languages in your audio:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Examples:{Style.RESET_ALL}")
    print(f"  • {Fore.CYAN}en de{Style.RESET_ALL} - English and German")
    print(f"  • {Fore.CYAN}en fr it{Style.RESET_ALL} - English, French, and Italian")
    print(f"  • {Fore.CYAN}de fr{Style.RESET_ALL} - German and French")
    try:
        lang_input = input(f"\n{Fore.CYAN}Enter language codes (space-separated):{Style.RESET_ALL} ").strip().lower()
        lang_codes = [code for code in lang_input.split() if len(code) == 2 and code.isalpha()]
        if len(lang_codes) < 2:
            print(f"{Fore.RED}Please enter at least 2 valid 2-letter language codes for multilingual mode.{Style.RESET_ALL}")
            return _select_constrained_multilingual()
        unique_codes = list(dict.fromkeys(lang_codes))
        _bulk_save(
            whisper_language=None,
            whisper_multilingual_segments=True,
            whisper_language_constraints=unique_codes
        )
        print(f"{Fore.GREEN}✓ Language set to: {' + '.join(code.upper() for code in unique_codes)} (constrained multilingual){Style.RESET_ALL}")
    except (EOFError, KeyboardInterrupt):
        print(f"\n{Fore.YELLOW}Using auto-detect as default.{Style.RESET_ALL}")
        _bulk_save(
            whisper_language=None,
            whisper_multilingual_segments=True,
            whisper_language_constraints=None
        )

def get_speaker_preference():
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
    try:
        choice = get_single_keypress(
            f"\n{Fore.CYAN}Select speaker mode (1-4):{Style.RESET_ALL} ",
            ['1', '2', '3', '4']
        )
        if choice is None:
            min_spk, max_spk = 1, 1
            description = "Single Speaker"
        else:
            idx = int(choice) - 1
            min_spk, max_spk, description = speaker_options[idx]
        _bulk_save(min_speakers=min_spk, max_speakers=max_spk)
        print(f"{Fore.GREEN}✓ Speaker mode set to: {description}{Style.RESET_ALL}")
    except (EOFError, KeyboardInterrupt):
        print(f"\n{Fore.YELLOW}Using Single Speaker mode as default.{Style.RESET_ALL}")
        _bulk_save(min_speakers=1, max_speakers=1) 