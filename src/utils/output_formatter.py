"""
Output formatting module for creating transcript text format
Supports only TXT output format with integrated summary
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

from src.core.config import CONFIG, OUTPUT_DIR

logger = logging.getLogger(__name__)

class OutputFormatter:
    def __init__(self):
        self.output_format = CONFIG['output_format']
        self.output_dir = OUTPUT_DIR
        self.timestamp_format = CONFIG['timestamp_format']
        
    def format_transcript(self, aligned_result):
        """
        Format the aligned transcript as text with timestamps and summary
        
        Args:
            aligned_result: Dictionary with speaker turns from alignment
            
        Returns:
            Formatted transcript string
        """
        return self._format_as_text(aligned_result)
    
    def save_transcript(self, aligned_result, filename=None):
        """
        Save the aligned transcript to file as TXT format
        
        Args:
            aligned_result: Dictionary with speaker turns from alignment
            filename: Custom filename (without extension)
            
        Returns:
            Path to saved file
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcript_{timestamp}"
        
        file_path = self.output_dir / f"{filename}.txt"
        
        # Format the transcript
        formatted_content = self.format_transcript(aligned_result)
        
        # Save to file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            
            logger.info(f"Transcript saved to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save transcript: {e}")
            raise
    
    def _format_as_text(self, aligned_result):
        """Format transcript as plain text with timestamps and summary"""
        speaker_turns = aligned_result.get('speaker_turns', [])
        
        if not speaker_turns:
            return "No transcript available.\n"
        
        lines = []
        
        # Add header with metadata
        lines.extend(self._generate_header(aligned_result))
        
        # Add speaker turns
        for turn in speaker_turns:
            start_time = self._format_timestamp(turn['start'])
            end_time = self._format_timestamp(turn['end'])
            speaker = turn['speaker']
            text = turn['text']
            
            lines.append(f"[{start_time} - {end_time}] {speaker}: {text}")
            lines.append("")
        
        # Add summary at the end
        lines.extend(self._generate_speaker_summary(speaker_turns))
        
        return "\n".join(lines)
    
    def _format_timestamp(self, seconds):
        """Format seconds as HH:MM:SS.mmm"""
        # Handle negative or invalid timestamps
        if seconds < 0 or not isinstance(seconds, (int, float)):
            seconds = 0
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def _generate_header(self, aligned_result):
        """Generate transcript header with metadata"""
        lines = []
        lines.append("=" * 60)
        lines.append("TRANSCRIPT")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Primary Language: {aligned_result.get('language', 'Unknown')}")
        
        # Show multilingual information if available
        detected_languages = aligned_result.get('detected_languages', {})
        language_durations = aligned_result.get('language_durations', {})
        if detected_languages and len(detected_languages) > 1:
            lines.append(f"Multilingual Content: {', '.join(detected_languages.keys())}")
            for lang, count in detected_languages.items():
                duration = language_durations.get(lang, 0)
                lines.append(f"  - {lang.upper()}: {count} chunks, {duration:.1f}s duration")
        
        lines.append(f"Total Speakers: {aligned_result.get('total_speakers', 0)}")
        lines.append(f"Total Words: {aligned_result.get('total_words', 0)}")
        
        stats = aligned_result.get('alignment_stats', {})
        if stats.get('speaker_assignment_rate'):
            rate = stats['speaker_assignment_rate'] * 100
            lines.append(f"Speaker Assignment Rate: {rate:.1f}%")
        
        lines.append("=" * 60)
        lines.append("")
        return lines
    
    def _generate_speaker_summary(self, speaker_turns):
        """Generate speaker breakdown summary"""
        if not speaker_turns:
            return []
        
        lines = []
        lines.append("=" * 60)
        lines.append("TRANSCRIPT SUMMARY")
        lines.append("=" * 60)
        
        # Calculate statistics
        total_duration = max([turn['end'] for turn in speaker_turns]) if speaker_turns else 0
        speaker_stats = {}
        
        for turn in speaker_turns:
            speaker = turn['speaker']
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {'total_time': 0, 'turn_count': 0, 'word_count': 0}
            
            speaker_stats[speaker]['total_time'] += turn.get('duration', 0)
            speaker_stats[speaker]['turn_count'] += 1
            speaker_stats[speaker]['word_count'] += turn.get('word_count', len(turn['text'].split()))
        
        lines.append(f"Total Duration: {self._format_timestamp(total_duration)}")
        lines.append(f"Total Speakers: {len(speaker_stats)}")
        lines.append(f"Total Turns: {len(speaker_turns)}")
        lines.append(f"Total Words: {sum(stats['word_count'] for stats in speaker_stats.values())}")
        lines.append("")
        
        # Speaker breakdown
        lines.append("SPEAKER BREAKDOWN:")
        lines.append("-" * 30)
        
        for speaker, stats in speaker_stats.items():
            time_str = self._format_timestamp(stats['total_time'])
            percentage = (stats['total_time'] / total_duration * 100) if total_duration > 0 else 0
            
            lines.append(f"{speaker}:")
            lines.append(f"  Speaking Time: {time_str} ({percentage:.1f}%)")
            lines.append(f"  Turns: {stats['turn_count']}")
            lines.append(f"  Words: {stats['word_count']}")
            lines.append("")
        
        return lines
    
    def save_real_time_update(self, speaker_turn, session_file=None):
        """
        Append a new speaker turn to an ongoing real-time transcript
        
        Args:
            speaker_turn: Dictionary with speaker turn data
            session_file: Path to existing session file
            
        Returns:
            Path to updated file
        """
        if session_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_file = self.output_dir / f"realtime_session_{timestamp}.txt"
        
        try:
            # Format the turn
            start_time = self._format_timestamp(speaker_turn['start'])
            end_time = self._format_timestamp(speaker_turn['end'])
            speaker = speaker_turn['speaker']
            text = speaker_turn['text']
            
            formatted_line = f"[{start_time} - {end_time}] {speaker}: {text}\n"
            
            # Append to file
            with open(session_file, 'a', encoding='utf-8') as f:
                f.write(formatted_line)
            
            logger.debug(f"Real-time update appended to {session_file}")
            return session_file
            
        except Exception as e:
            logger.error(f"Failed to save real-time update: {e}")
            raise
    
    def export_transcript(self, aligned_result, base_filename=None):
        """Export transcript in TXT format with integrated summary"""
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"transcript_{timestamp}"
        
        try:
            file_path = self.save_transcript(aligned_result, base_filename)
            logger.info(f"Exported TXT format to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to export TXT format: {e}")
            raise 