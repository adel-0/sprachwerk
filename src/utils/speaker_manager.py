"""
Speaker Management System for sprachwerk
Provides user-friendly interface for managing speaker profiles, renaming, and merging
"""

import logging
from pathlib import Path
from datetime import datetime
from colorama import Fore, Style
from typing import List, Dict, Optional, Tuple

from src.processing.speaker_identification import SpeakerIdentifier
from src.core.config import OUTPUT_DIR
from src.utils.key_input import get_single_keypress

logger = logging.getLogger(__name__)

class SpeakerManager:
    """Interactive speaker management system"""
    
    def __init__(self):
        self.speaker_identifier = SpeakerIdentifier()
    

        
    def run_speaker_management_menu(self):
        """Main speaker management menu"""
        while True:
            try:
                self._show_speaker_management_menu()
                choice = get_single_keypress(
                    f"\n{Fore.CYAN}Select option (1-7):{Style.RESET_ALL} ",
                    ['1', '2', '3', '4', '5', '6', '7']
                )
                
                if choice is None or choice == '7':  # User cancelled or exit
                    print(f"{Fore.YELLOW}Returning to main menu...{Style.RESET_ALL}")
                    break
                elif choice == '1':
                    self._list_speakers()
                elif choice == '2':
                    self._rename_speaker()
                elif choice == '3':
                    self._merge_speakers()
                elif choice == '4':
                    self._view_speaker_details()
                elif choice == '5':
                    self._manage_speaker_database()
                elif choice == '6':
                    self._export_speaker_report()
                    
            except (EOFError, KeyboardInterrupt):
                print(f"\n{Fore.YELLOW}Returning to main menu...{Style.RESET_ALL}")
                break

    def _show_speaker_management_menu(self):
        """Display the speaker management menu"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🎭 SPEAKER MANAGEMENT{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        # Quick stats
        try:
            all_speakers = self.speaker_identifier.list_speakers()
            total_speakers = len(all_speakers)
            named_speakers = sum(1 for s in all_speakers if s['name'] != 'Unnamed')
            
            print(f"\n{Fore.CYAN}📊 Quick Stats:{Style.RESET_ALL}")
            print(f"  • Total speakers: {Fore.GREEN}{total_speakers}{Style.RESET_ALL}")
            print(f"  • Named speakers: {Fore.GREEN}{named_speakers}{Style.RESET_ALL}")
            print(f"  • Unnamed speakers: {Fore.YELLOW}{total_speakers - named_speakers}{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{Fore.RED}Error loading speaker stats: {e}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}Available Options:{Style.RESET_ALL}")
        print(f"  1. {Fore.BLUE}📋 List All Speakers{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • View all speakers with details{Style.RESET_ALL}")
        print(f"  2. {Fore.GREEN}✏️  Rename Speaker{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Give meaningful names to speakers{Style.RESET_ALL}")
        print(f"  3. {Fore.MAGENTA}🔗 Merge Speakers{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Combine duplicate or similar speakers{Style.RESET_ALL}")
        print(f"  4. {Fore.CYAN}🔍 View Speaker Details{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Detailed information about specific speakers{Style.RESET_ALL}")
        print(f"  5. {Fore.YELLOW}🗃️  Manage Database{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Database maintenance and cleanup{Style.RESET_ALL}")
        print(f"  6. {Fore.GREEN}📊 Export Speaker Report{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Generate detailed speaker analysis{Style.RESET_ALL}")
        print(f"  7. {Fore.RED}🔙 Back to Main Menu{Style.RESET_ALL}")

    def _list_speakers(self):
        """List all speakers with their information"""
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📋 ALL SPEAKERS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        
        try:
            speakers = self.speaker_identifier.list_speakers()
            
            if not speakers:
                print(f"\n{Fore.YELLOW}No speakers found in database.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Speakers are added automatically during transcription.{Style.RESET_ALL}")
                self._wait_for_user()
                return
            
            # Sort speakers by total speaking time (descending)
            speakers_sorted = sorted(speakers, key=lambda x: x['total_speaking_time'], reverse=True)
            
            print(f"\n{Fore.YELLOW}Found {len(speakers_sorted)} speakers:{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'ID':<15} {'Name':<25} {'Speaking Time':<15} {'Sessions':<10} {'Last Seen':<12}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'-'*80}{Style.RESET_ALL}")
            
            for speaker in speakers_sorted:
                # Format speaking time
                speaking_time = speaker['total_speaking_time']
                if speaking_time >= 3600:
                    time_str = f"{speaking_time/3600:.1f}h"
                elif speaking_time >= 60:
                    time_str = f"{speaking_time/60:.1f}m"
                else:
                    time_str = f"{speaking_time:.0f}s"
                
                # Format last seen
                try:
                    last_seen = datetime.fromisoformat(speaker['last_seen']).strftime('%Y-%m-%d')
                except:
                    last_seen = "Unknown"
                
                # Color code based on name status
                name_color = Fore.GREEN if speaker['name'] != 'Unnamed' else Fore.YELLOW
                
                print(f"{speaker['speaker_id']:<15} {name_color}{speaker['name']:<25}{Style.RESET_ALL} "
                      f"{time_str:<15} {speaker['session_count']:<10} {last_seen:<12}")
            
            print(f"\n{Fore.CYAN}Legend:{Style.RESET_ALL}")
            print(f"  {Fore.GREEN}Green names{Style.RESET_ALL} - Named speakers")
            print(f"  {Fore.YELLOW}Yellow names{Style.RESET_ALL} - Unnamed speakers")
            
        except Exception as e:
            print(f"{Fore.RED}Error loading speakers: {e}{Style.RESET_ALL}")
            logger.error(f"Error in _list_speakers: {e}")
        
        self._wait_for_user()

    def _rename_speaker(self):
        """Rename a speaker"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✏️  RENAME SPEAKER{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        try:
            speakers = self.speaker_identifier.list_speakers()
            
            if not speakers:
                print(f"\n{Fore.YELLOW}No speakers found in database.{Style.RESET_ALL}")
                self._wait_for_user()
                return
            
            # Sort by speaking time for better UX
            speakers_sorted = sorted(speakers, key=lambda x: x['total_speaking_time'], reverse=True)
            
            print(f"\n{Fore.YELLOW}Select speaker to rename:{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'#':<3} {'Current Name':<25} {'Speaking Time':<15} {'ID':<15}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
            
            for i, speaker in enumerate(speakers_sorted, 1):
                # Format speaking time
                speaking_time = speaker['total_speaking_time']
                if speaking_time >= 3600:
                    time_str = f"{speaking_time/3600:.1f}h"
                elif speaking_time >= 60:
                    time_str = f"{speaking_time/60:.1f}m"
                else:
                    time_str = f"{speaking_time:.0f}s"
                
                name_color = Fore.GREEN if speaker['name'] != 'Unnamed' else Fore.YELLOW
                print(f"{i:<3} {name_color}{speaker['name']:<25}{Style.RESET_ALL} {time_str:<15} {speaker['speaker_id']:<15}")
            
            # Get speaker selection
            while True:
                try:
                    choice = input(f"\n{Fore.CYAN}Enter speaker number (1-{len(speakers_sorted)}) or 'back':{Style.RESET_ALL} ").strip()
                    
                    if choice.lower() == 'back':
                        return
                    
                    speaker_idx = int(choice) - 1
                    if 0 <= speaker_idx < len(speakers_sorted):
                        selected_speaker = speakers_sorted[speaker_idx]
                        break
                    else:
                        print(f"{Fore.RED}Invalid choice. Please enter 1-{len(speakers_sorted)}.{Style.RESET_ALL}")
                        
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
                except (EOFError, KeyboardInterrupt):
                    return
            
            # Get new name
            current_name = selected_speaker['name']
            print(f"\n{Fore.YELLOW}Current name: {Fore.CYAN}{current_name}{Style.RESET_ALL}")
            
            while True:
                try:
                    new_name = input(f"{Fore.CYAN}Enter new name (or 'cancel' to abort):{Style.RESET_ALL} ").strip()
                    
                    if new_name.lower() == 'cancel':
                        return
                    
                    if not new_name:
                        print(f"{Fore.RED}Name cannot be empty.{Style.RESET_ALL}")
                        continue
                    
                    if len(new_name) > 50:
                        print(f"{Fore.RED}Name too long. Maximum 50 characters.{Style.RESET_ALL}")
                        continue
                    
                    # Check if name already exists
                    existing_names = [s['name'] for s in speakers if s['speaker_id'] != selected_speaker['speaker_id']]
                    if new_name in existing_names:
                        print(f"{Fore.RED}Name '{new_name}' already exists. Please choose a different name.{Style.RESET_ALL}")
                        continue
                    
                    break
                    
                except (EOFError, KeyboardInterrupt):
                    return
            
            # Confirm rename
            print(f"\n{Fore.YELLOW}Rename '{current_name}' to '{new_name}'?{Style.RESET_ALL}")
            
            confirm = get_single_keypress(
                f"{Fore.CYAN}Confirm rename? (Y/N):{Style.RESET_ALL} ",
                ['y', 'n']
            )
            
            if confirm == 'y':
                # Perform rename
                success = self.speaker_identifier.assign_speaker_name(selected_speaker['speaker_id'], new_name)
                
                if success:
                    print(f"{Fore.GREEN}✓ Speaker renamed successfully!{Style.RESET_ALL}")
                    print(f"  {Fore.CYAN}'{current_name}' → '{new_name}'{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Failed to rename speaker.{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Rename cancelled.{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error in rename operation: {e}{Style.RESET_ALL}")
            logger.error(f"Error in _rename_speaker: {e}")
        
        self._wait_for_user()

    def _merge_speakers(self):
        """Merge two or more speakers"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🔗 MERGE SPEAKERS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        try:
            speakers = self.speaker_identifier.list_speakers()
            
            if len(speakers) < 2:
                print(f"\n{Fore.YELLOW}Need at least 2 speakers to perform merge.{Style.RESET_ALL}")
                self._wait_for_user()
                return
            
            # Sort by speaking time for better UX
            speakers_sorted = sorted(speakers, key=lambda x: x['total_speaking_time'], reverse=True)
            
            print(f"\n{Fore.YELLOW}⚠️  Speaker merging combines two speakers into one.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}⚠️  This action cannot be undone.{Style.RESET_ALL}")
            
            print(f"\n{Fore.YELLOW}Available speakers:{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'#':<3} {'Name':<25} {'Speaking Time':<15} {'ID':<15}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
            
            for i, speaker in enumerate(speakers_sorted, 1):
                # Format speaking time
                speaking_time = speaker['total_speaking_time']
                if speaking_time >= 3600:
                    time_str = f"{speaking_time/3600:.1f}h"
                elif speaking_time >= 60:
                    time_str = f"{speaking_time/60:.1f}m"
                else:
                    time_str = f"{speaking_time:.0f}s"
                
                name_color = Fore.GREEN if speaker['name'] != 'Unnamed' else Fore.YELLOW
                print(f"{i:<3} {name_color}{speaker['name']:<25}{Style.RESET_ALL} {time_str:<15} {speaker['speaker_id']:<15}")
            
            # Get first speaker (keep this one)
            while True:
                try:
                    choice = input(f"\n{Fore.CYAN}Select speaker to KEEP (1-{len(speakers_sorted)}) or 'back':{Style.RESET_ALL} ").strip()
                    
                    if choice.lower() == 'back':
                        return
                    
                    keep_idx = int(choice) - 1
                    if 0 <= keep_idx < len(speakers_sorted):
                        keep_speaker = speakers_sorted[keep_idx]
                        break
                    else:
                        print(f"{Fore.RED}Invalid choice. Please enter 1-{len(speakers_sorted)}.{Style.RESET_ALL}")
                        
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
                except (EOFError, KeyboardInterrupt):
                    return
            
            # Get second speaker (merge into first)
            remaining_speakers = [s for s in speakers_sorted if s['speaker_id'] != keep_speaker['speaker_id']]
            
            print(f"\n{Fore.GREEN}Keeping: {keep_speaker['name']}{Style.RESET_ALL}")
            print(f"\n{Fore.YELLOW}Select speaker to MERGE INTO the kept speaker:{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'#':<3} {'Name':<25} {'Speaking Time':<15} {'ID':<15}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
            
            for i, speaker in enumerate(remaining_speakers, 1):
                # Format speaking time
                speaking_time = speaker['total_speaking_time']
                if speaking_time >= 3600:
                    time_str = f"{speaking_time/3600:.1f}h"
                elif speaking_time >= 60:
                    time_str = f"{speaking_time/60:.1f}m"
                else:
                    time_str = f"{speaking_time:.0f}s"
                
                name_color = Fore.GREEN if speaker['name'] != 'Unnamed' else Fore.YELLOW
                print(f"{i:<3} {name_color}{speaker['name']:<25}{Style.RESET_ALL} {time_str:<15} {speaker['speaker_id']:<15}")
            
            while True:
                try:
                    choice = input(f"\n{Fore.CYAN}Select speaker to merge (1-{len(remaining_speakers)}) or 'back':{Style.RESET_ALL} ").strip()
                    
                    if choice.lower() == 'back':
                        return
                    
                    merge_idx = int(choice) - 1
                    if 0 <= merge_idx < len(remaining_speakers):
                        merge_speaker = remaining_speakers[merge_idx]
                        break
                    else:
                        print(f"{Fore.RED}Invalid choice. Please enter 1-{len(remaining_speakers)}.{Style.RESET_ALL}")
                        
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
                except (EOFError, KeyboardInterrupt):
                    return
            
            # Name preference
            if keep_speaker['name'] != 'Unnamed' and merge_speaker['name'] != 'Unnamed':
                print(f"\n{Fore.YELLOW}Both speakers have names. Which name should be kept?{Style.RESET_ALL}")
                print(f"  1. Keep '{keep_speaker['name']}' (from speaker with more speaking time)")
                print(f"  2. Keep '{merge_speaker['name']}' (from speaker being merged)")
                
                name_choice = get_single_keypress(
                    f"{Fore.CYAN}Select name preference (1-2):{Style.RESET_ALL} ",
                    ['1', '2']
                )
                
                if name_choice is None:
                    return
                
                keep_name_from = keep_speaker['speaker_id'] if name_choice == '1' else merge_speaker['speaker_id']
            else:
                keep_name_from = None
            
            # Confirm merge
            print(f"\n{Fore.YELLOW}⚠️  Are you sure you want to merge these speakers?{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}⚠️  This action cannot be undone.{Style.RESET_ALL}")
            
            confirm = get_single_keypress(
                f"{Fore.CYAN}Confirm merge? (Y/N):{Style.RESET_ALL} ",
                ['y', 'n']
            )
            
            if confirm == 'y':
                # Perform merge
                success = self.speaker_identifier.merge_speakers(
                    keep_speaker['speaker_id'], 
                    merge_speaker['speaker_id'],
                    keep_name_from=keep_name_from
                )
                
                if success:
                    print(f"{Fore.GREEN}✓ Speakers merged successfully!{Style.RESET_ALL}")
                    print(f"  {Fore.CYAN}'{merge_speaker['name']}' merged into '{keep_speaker['name']}'{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Failed to merge speakers.{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Merge cancelled.{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error in merge operation: {e}{Style.RESET_ALL}")
            logger.error(f"Error in _merge_speakers: {e}")
        
        self._wait_for_user()

    def _view_speaker_details(self):
        """View detailed information about a specific speaker"""
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🔍 SPEAKER DETAILS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        
        try:
            speakers = self.speaker_identifier.list_speakers()
            
            if not speakers:
                print(f"\n{Fore.YELLOW}No speakers found in database.{Style.RESET_ALL}")
                self._wait_for_user()
                return
            
            # Sort by speaking time for better UX
            speakers_sorted = sorted(speakers, key=lambda x: x['total_speaking_time'], reverse=True)
            
            print(f"\n{Fore.YELLOW}Select speaker to view details:{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'#':<3} {'Name':<25} {'Speaking Time':<15} {'ID':<15}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
            
            for i, speaker in enumerate(speakers_sorted, 1):
                # Format speaking time
                speaking_time = speaker['total_speaking_time']
                if speaking_time >= 3600:
                    time_str = f"{speaking_time/3600:.1f}h"
                elif speaking_time >= 60:
                    time_str = f"{speaking_time/60:.1f}m"
                else:
                    time_str = f"{speaking_time:.0f}s"
                
                name_color = Fore.GREEN if speaker['name'] != 'Unnamed' else Fore.YELLOW
                print(f"{i:<3} {name_color}{speaker['name']:<25}{Style.RESET_ALL} {time_str:<15} {speaker['speaker_id']:<15}")
            
            while True:
                try:
                    choice = input(f"\n{Fore.CYAN}Select speaker number (1-{len(speakers_sorted)}) or press Enter to go back:{Style.RESET_ALL} ").strip()
                    
                    if choice == '':
                        return
                    
                    if choice.isdigit() and 1 <= int(choice) <= len(speakers_sorted):
                        selected_speaker = speakers_sorted[int(choice) - 1]
                        break
                    else:
                        print(f"{Fore.RED}Invalid choice. Please select 1-{len(speakers_sorted)} or press Enter to go back.{Style.RESET_ALL}")
                        
                except (EOFError, KeyboardInterrupt):
                    return
            
            # Display detailed information
            speaker = selected_speaker
            print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}📊 DETAILED SPEAKER INFORMATION{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            
            print(f"\n{Fore.YELLOW}Speaker ID:{Style.RESET_ALL} {speaker['speaker_id']}")
            print(f"{Fore.YELLOW}Name:{Style.RESET_ALL} {speaker['name']}")
            
            # Format times
            speaking_time = speaker['total_speaking_time']
            if speaking_time >= 3600:
                time_str = f"{speaking_time/3600:.1f} hours"
            elif speaking_time >= 60:
                time_str = f"{speaking_time/60:.1f} minutes"
            else:
                time_str = f"{speaking_time:.1f} seconds"
            
            print(f"{Fore.YELLOW}Total Speaking Time:{Style.RESET_ALL} {time_str}")
            print(f"{Fore.YELLOW}Number of Sessions:{Style.RESET_ALL} {speaker['session_count']}")
            
            # Format dates
            try:
                first_seen = datetime.fromisoformat(speaker['first_seen']).strftime('%Y-%m-%d %H:%M:%S')
                last_seen = datetime.fromisoformat(speaker['last_seen']).strftime('%Y-%m-%d %H:%M:%S')
                print(f"{Fore.YELLOW}First Seen:{Style.RESET_ALL} {first_seen}")
                print(f"{Fore.YELLOW}Last Seen:{Style.RESET_ALL} {last_seen}")
            except:
                print(f"{Fore.YELLOW}First Seen:{Style.RESET_ALL} Unknown")
                print(f"{Fore.YELLOW}Last Seen:{Style.RESET_ALL} Unknown")
            
            # Additional statistics if available
            if 'voice_characteristics' in speaker:
                print(f"\n{Fore.CYAN}Voice Characteristics:{Style.RESET_ALL}")
                characteristics = speaker['voice_characteristics']
                for key, value in characteristics.items():
                    print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"{Fore.RED}Error viewing speaker details: {e}{Style.RESET_ALL}")
            logger.error(f"Error in _view_speaker_details: {e}")
        
        self._wait_for_user()

    def _manage_speaker_database(self):
        """Database management menu"""
        while True:
            try:
                self._show_database_management_menu()
                choice = get_single_keypress(
                    f"\n{Fore.CYAN}Select option (1-5):{Style.RESET_ALL} ",
                    ['1', '2', '3', '4', '5']
                )
                
                if choice is None or choice == '5':  # User cancelled or back
                    break
                elif choice == '1':
                    self._cleanup_old_speakers()
                elif choice == '2':
                    self._show_database_statistics()
                elif choice == '3':
                    self._reset_database_warning()
                elif choice == '4':
                    # Backup database
                    print(f"\n{Fore.YELLOW}Database backup functionality not yet implemented.{Style.RESET_ALL}")
                    self._wait_for_user()
                    
            except (EOFError, KeyboardInterrupt):
                break

    def _show_database_management_menu(self):
        """Display database management menu"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🗃️  DATABASE MANAGEMENT{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}Database Maintenance Options:{Style.RESET_ALL}")
        print(f"  1. {Fore.BLUE}🧹 Cleanup Old Speakers{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Remove speakers with minimal activity{Style.RESET_ALL}")
        print(f"  2. {Fore.GREEN}📊 Database Statistics{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • View database size and usage statistics{Style.RESET_ALL}")
        print(f"  3. {Fore.RED}⚠️  Reset Database{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • WARNING: Permanently delete all speaker data{Style.RESET_ALL}")
        print(f"  4. {Fore.CYAN}💾 Backup Database{Style.RESET_ALL}")
        print(f"     {Fore.WHITE}   • Create a backup of speaker database{Style.RESET_ALL}")
        print(f"  5. {Fore.YELLOW}🔙 Back to Speaker Management{Style.RESET_ALL}")

    def _cleanup_old_speakers(self):
        """Clean up old or inactive speakers"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🧹 CLEANUP OLD SPEAKERS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}Cleanup Options:{Style.RESET_ALL}")
        print(f"  1. {Fore.BLUE}Remove speakers with < 10 seconds speaking time{Style.RESET_ALL}")
        print(f"  2. {Fore.BLUE}Remove speakers with < 30 seconds speaking time{Style.RESET_ALL}")
        print(f"  3. {Fore.BLUE}Remove speakers not seen in last 30 days{Style.RESET_ALL}")
        print(f"  4. {Fore.BLUE}Custom cleanup criteria{Style.RESET_ALL}")
        print(f"  5. {Fore.YELLOW}🔙 Back{Style.RESET_ALL}")
        
        choice = get_single_keypress(
            f"\n{Fore.CYAN}Select cleanup option (1-5):{Style.RESET_ALL} ",
            ['1', '2', '3', '4', '5']
        )
        
        if choice is None or choice == '5':
            return
        
        try:
            speakers = self.speaker_identifier.list_speakers()
            candidates = []
            
            if choice == '1':
                candidates = [s for s in speakers if s['total_speaking_time'] < 10]
                criteria = "< 10 seconds speaking time"
            elif choice == '2':
                candidates = [s for s in speakers if s['total_speaking_time'] < 30]
                criteria = "< 30 seconds speaking time"
            elif choice == '3':
                from datetime import datetime, timedelta
                cutoff_date = datetime.now() - timedelta(days=30)
                candidates = [s for s in speakers if datetime.fromisoformat(s['last_seen']) < cutoff_date]
                criteria = "not seen in last 30 days"
            elif choice == '4':
                # Custom criteria
                try:
                    min_time = float(input(f"{Fore.YELLOW}Minimum speaking time (seconds):{Style.RESET_ALL} "))
                    candidates = [s for s in speakers if s['total_speaking_time'] < min_time]
                    criteria = f"< {min_time} seconds speaking time"
                except (ValueError, EOFError, KeyboardInterrupt):
                    return
            
            if not candidates:
                print(f"\n{Fore.GREEN}No speakers match the cleanup criteria.{Style.RESET_ALL}")
                self._wait_for_user()
                return
            
            print(f"\n{Fore.YELLOW}Found {len(candidates)} speakers matching criteria: {criteria}{Style.RESET_ALL}")
            
            # Show candidates
            for speaker in candidates[:10]:  # Show first 10
                time_str = f"{speaker['total_speaking_time']:.1f}s"
                print(f"  • {speaker['name']} ({time_str})")
            
            if len(candidates) > 10:
                print(f"  ... and {len(candidates) - 10} more")
            
            confirm = get_single_keypress(
                f"\n{Fore.CYAN}Delete these {len(candidates)} speakers? (Y/N):{Style.RESET_ALL} ",
                ['y', 'n']
            )
            
            if confirm == 'y':
                deleted_count = 0
                for speaker in candidates:
                    success = self.speaker_identifier.delete_speaker(speaker['speaker_id'])
                    if success:
                        deleted_count += 1
                
                print(f"\n{Fore.GREEN}✓ Deleted {deleted_count} speakers.{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.YELLOW}Cleanup cancelled.{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error during cleanup: {e}{Style.RESET_ALL}")
            logger.error(f"Error in _cleanup_old_speakers: {e}")
        
        self._wait_for_user()

    def _show_database_statistics(self):
        """Show database statistics"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 DATABASE STATISTICS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        try:
            speakers = self.speaker_identifier.list_speakers()
            
            if not speakers:
                print(f"\n{Fore.YELLOW}Database is empty.{Style.RESET_ALL}")
                self._wait_for_user()
                return
            
            # Basic statistics
            total_speakers = len(speakers)
            named_speakers = sum(1 for s in speakers if s['name'] != 'Unnamed')
            total_speaking_time = sum(s['total_speaking_time'] for s in speakers)
            total_sessions = sum(s['session_count'] for s in speakers)
            
            print(f"\n{Fore.YELLOW}Basic Statistics:{Style.RESET_ALL}")
            print(f"  • Total speakers: {Fore.GREEN}{total_speakers}{Style.RESET_ALL}")
            print(f"  • Named speakers: {Fore.GREEN}{named_speakers}{Style.RESET_ALL}")
            print(f"  • Unnamed speakers: {Fore.YELLOW}{total_speakers - named_speakers}{Style.RESET_ALL}")
            print(f"  • Total sessions: {Fore.GREEN}{total_sessions}{Style.RESET_ALL}")
            
            # Format total speaking time
            if total_speaking_time >= 3600:
                time_str = f"{total_speaking_time/3600:.1f} hours"
            elif total_speaking_time >= 60:
                time_str = f"{total_speaking_time/60:.1f} minutes"
            else:
                time_str = f"{total_speaking_time:.1f} seconds"
            
            print(f"  • Total speaking time: {Fore.GREEN}{time_str}{Style.RESET_ALL}")
            
            # Top speakers
            top_speakers = sorted(speakers, key=lambda x: x['total_speaking_time'], reverse=True)[:5]
            print(f"\n{Fore.YELLOW}Top 5 Speakers by Speaking Time:{Style.RESET_ALL}")
            for i, speaker in enumerate(top_speakers, 1):
                speaking_time = speaker['total_speaking_time']
                if speaking_time >= 3600:
                    time_str = f"{speaking_time/3600:.1f}h"
                elif speaking_time >= 60:
                    time_str = f"{speaking_time/60:.1f}m"
                else:
                    time_str = f"{speaking_time:.0f}s"
                
                name_color = Fore.GREEN if speaker['name'] != 'Unnamed' else Fore.YELLOW
                print(f"  {i}. {name_color}{speaker['name']}{Style.RESET_ALL} - {time_str}")
            
            # Activity distribution
            short_speakers = sum(1 for s in speakers if s['total_speaking_time'] < 30)
            medium_speakers = sum(1 for s in speakers if 30 <= s['total_speaking_time'] < 300)
            long_speakers = sum(1 for s in speakers if s['total_speaking_time'] >= 300)
            
            print(f"\n{Fore.YELLOW}Activity Distribution:{Style.RESET_ALL}")
            print(f"  • Short activity (< 30s): {Fore.YELLOW}{short_speakers}{Style.RESET_ALL}")
            print(f"  • Medium activity (30s-5m): {Fore.BLUE}{medium_speakers}{Style.RESET_ALL}")
            print(f"  • High activity (> 5m): {Fore.GREEN}{long_speakers}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error loading statistics: {e}{Style.RESET_ALL}")
            logger.error(f"Error in _show_database_statistics: {e}")
        
        self._wait_for_user()

    def _reset_database_warning(self):
        """Show warning and confirm database reset"""
        print(f"\n{Fore.RED}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.RED}⚠️  RESET DATABASE WARNING{Style.RESET_ALL}")
        print(f"{Fore.RED}{'='*70}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}⚠️  This will permanently delete ALL speaker data!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}⚠️  This includes:{Style.RESET_ALL}")
        print(f"  • All speaker profiles and names")
        print(f"  • All voice embeddings and characteristics")
        print(f"  • All session history and statistics")
        print(f"  • All speaking time records")
        
        print(f"\n{Fore.RED}⚠️  THIS ACTION CANNOT BE UNDONE!{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}Are you absolutely sure you want to reset the database?{Style.RESET_ALL}")
        
        confirm1 = get_single_keypress(
            f"{Fore.CYAN}Type 'Y' to continue or 'N' to cancel:{Style.RESET_ALL} ",
            ['y', 'n']
        )
        
        if confirm1 != 'y':
            print(f"{Fore.GREEN}Database reset cancelled.{Style.RESET_ALL}")
            self._wait_for_user()
            return
        
        print(f"\n{Fore.RED}FINAL WARNING: This will delete everything!{Style.RESET_ALL}")
        
        confirm2 = get_single_keypress(
            f"{Fore.CYAN}Type 'Y' to PERMANENTLY DELETE all data:{Style.RESET_ALL} ",
            ['y', 'n']
        )
        
        if confirm2 == 'y':
            try:
                success = self.speaker_identifier.reset_database()
                if success:
                    print(f"\n{Fore.GREEN}✓ Database reset successfully.{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}All speaker data has been permanently deleted.{Style.RESET_ALL}")
                else:
                    print(f"\n{Fore.RED}Failed to reset database.{Style.RESET_ALL}")
            except Exception as e:
                print(f"\n{Fore.RED}Error resetting database: {e}{Style.RESET_ALL}")
                logger.error(f"Error in _reset_database_warning: {e}")
        else:
            print(f"\n{Fore.GREEN}Database reset cancelled.{Style.RESET_ALL}")
        
        self._wait_for_user()

    def _export_speaker_report(self):
        """Export a detailed speaker report"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 EXPORT SPEAKER REPORT{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        try:
            speakers = self.speaker_identifier.list_speakers()
            
            if not speakers:
                print(f"\n{Fore.YELLOW}No speakers found to export.{Style.RESET_ALL}")
                self._wait_for_user()
                return
            
            # Generate report filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"speaker_report_{timestamp}.txt"
            report_path = OUTPUT_DIR / report_filename
            
            # Generate report content
            report_lines = []
            report_lines.append(f"sprachwerk Speaker Report")
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"{'='*70}")
            report_lines.append("")
            
            # Summary statistics
            total_speakers = len(speakers)
            named_speakers = sum(1 for s in speakers if s['name'] != 'Unnamed')
            total_speaking_time = sum(s['total_speaking_time'] for s in speakers)
            
            report_lines.append(f"SUMMARY STATISTICS")
            report_lines.append(f"{'='*70}")
            report_lines.append(f"Total speakers: {total_speakers}")
            report_lines.append(f"Named speakers: {named_speakers}")
            report_lines.append(f"Unnamed speakers: {total_speakers - named_speakers}")
            report_lines.append(f"Total speaking time: {total_speaking_time:.1f} seconds")
            report_lines.append("")
            
            # Detailed speaker list
            speakers_sorted = sorted(speakers, key=lambda x: x['total_speaking_time'], reverse=True)
            
            report_lines.append(f"DETAILED SPEAKER LIST")
            report_lines.append(f"{'='*70}")
            report_lines.append(f"{'Name':<25} {'ID':<15} {'Time':<10} {'Sessions':<10} {'Last Seen':<12}")
            report_lines.append(f"{'-'*70}")
            
            for speaker in speakers_sorted:
                time_str = f"{speaker['total_speaking_time']:.1f}s"
                try:
                    last_seen = datetime.fromisoformat(speaker['last_seen']).strftime('%Y-%m-%d')
                except:
                    last_seen = "Unknown"
                
                report_lines.append(f"{speaker['name']:<25} {speaker['speaker_id']:<15} {time_str:<10} {speaker['session_count']:<10} {last_seen:<12}")
            
            # Write report to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            print(f"\n{Fore.GREEN}✓ Speaker report exported successfully!{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Report saved to: {report_path}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Report contains {len(speakers)} speakers{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error exporting report: {e}{Style.RESET_ALL}")
            logger.error(f"Error in _export_speaker_report: {e}")
        
        self._wait_for_user()

    def _wait_for_user(self, message="Press any key to continue..."):
        """Wait for user input before continuing"""
        print(f"\n{Fore.GREEN}{message}{Style.RESET_ALL}")
        msvcrt.getch()

    def _clear_screen(self):
        """Clear the screen (Windows compatible)"""
        import os
        os.system('cls') 
