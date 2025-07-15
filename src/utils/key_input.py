"""
Consolidated single-keypress input utility for Windows
Provides a unified interface for getting single character input without requiring Enter
"""

import msvcrt
from colorama import Fore, Style

def get_single_keypress(prompt, valid_choices, allow_enter_for_default=False, default_choice=None):
    """Get a single keypress from user without requiring Enter (Windows-compatible)
    
    Args:
        prompt: The prompt to display to the user
        valid_choices: List of valid single-character choices
        allow_enter_for_default: Whether to allow Enter key for default choice
        default_choice: Default choice to use when Enter is pressed
        
    Returns:
        The selected choice character, or None if cancelled
    """
    print(prompt, end='', flush=True)
    
    while True:
        try:
            key = msvcrt.getch()
            try:
                key = key.decode('utf-8').lower()
            except UnicodeDecodeError:
                # Ignore special keys (arrows, function keys, etc.)
                print(f"\n{Fore.RED}Invalid key. Please select from: {', '.join(valid_choices)}{Style.RESET_ALL}")
                print(prompt, end='', flush=True)
                continue

            if ord(key) == 3:  # Ctrl+C
                raise KeyboardInterrupt

            if allow_enter_for_default and key == '\r':
                if default_choice:
                    print(f"\n{Fore.GREEN}✓ Using default: {default_choice}{Style.RESET_ALL}")
                    return default_choice
                else:
                    print(f"\n{Fore.RED}No default choice set.{Style.RESET_ALL}")
                    print(prompt, end='', flush=True)
                    continue

            if key in valid_choices:
                print(f"\n{Fore.GREEN}✓ Selected: {key}{Style.RESET_ALL}")
                return key
            else:
                print(f"\n{Fore.RED}Invalid choice. Please select from: {', '.join(valid_choices)}{Style.RESET_ALL}")
                print(prompt, end='', flush=True)
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Fore.YELLOW}Cancelled{Style.RESET_ALL}")
            return None 