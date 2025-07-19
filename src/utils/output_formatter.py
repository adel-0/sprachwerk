"""
Output formatting module for creating transcript text format
Supports only TXT output format with integrated summary
"""

import logging
from datetime import datetime
from pathlib import Path
from src.core.config import CONFIG, OUTPUT_DIR

logger = logging.getLogger(__name__)

class OutputFormatter:
    def __init__(self):
        self.output_format = CONFIG['output_format']
        self.output_dir = OUTPUT_DIR

    def format_transcript(self, aligned_result):
        """Format the aligned transcript as text with timestamps and summary"""
        return self._format_as_text(aligned_result)

    def save_transcript(self, aligned_result, filename=None):
        """Save the aligned transcript to file as TXT format"""
        if not filename:
            filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        file_path = self.output_dir / f"{filename}.txt"
        formatted_content = self.format_transcript(aligned_result)
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
        speaker_turns = aligned_result.get('speaker_turns') or []
        if not speaker_turns:
            return "No transcript available.\n"
        lines = []
        lines.extend(self._generate_header(aligned_result))
        for turn in speaker_turns:
            lines.append(f"[{self._format_timestamp(turn['start'])} - {self._format_timestamp(turn['end'])}] {turn['speaker']}: {turn['text']}")
            lines.append("")
        lines.extend(self._generate_speaker_summary(speaker_turns))
        return "\n".join(lines)

    def _format_timestamp(self, seconds):
        """Format seconds as HH:MM:SS.mmm"""
        try:
            seconds = float(seconds)
            if seconds < 0:
                seconds = 0
        except (TypeError, ValueError):
            seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def _generate_header(self, aligned_result):
        """Generate transcript header with metadata"""
        lines = [
            "=" * 60,
            "TRANSCRIPT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Primary Language: {aligned_result.get('language', 'Unknown')}"
        ]
        detected_languages = aligned_result.get('detected_languages') or {}
        language_durations = aligned_result.get('language_durations') or {}
        if len(detected_languages) > 1:
            lines.append(f"Multilingual Content: {', '.join(detected_languages.keys())}")
            for lang, count in detected_languages.items():
                duration = language_durations.get(lang, 0)
                lines.append(f"  - {lang.upper()}: {count} chunks, {duration:.1f}s duration")
        lines.append(f"Total Speakers: {aligned_result.get('total_speakers', 0)}")
        lines.append(f"Total Words: {aligned_result.get('total_words', 0)}")
        stats = aligned_result.get('alignment_stats') or {}
        if stats.get('speaker_assignment_rate'):
            lines.append(f"Speaker Assignment Rate: {stats['speaker_assignment_rate'] * 100:.1f}%")
        lines.append("=" * 60)
        lines.append("")
        return lines

    def _generate_speaker_summary(self, speaker_turns):
        """Generate speaker breakdown summary"""
        if not speaker_turns:
            return []
        lines = [
            "=" * 60,
            "TRANSCRIPT SUMMARY",
            "=" * 60
        ]
        total_duration = max((turn['end'] for turn in speaker_turns), default=0)
        speaker_stats = {}
        for turn in speaker_turns:
            speaker = turn['speaker']
            stats = speaker_stats.setdefault(speaker, {'total_time': 0, 'turn_count': 0, 'word_count': 0})
            stats['total_time'] += turn.get('duration', 0)
            stats['turn_count'] += 1
            stats['word_count'] += turn.get('word_count', len(turn['text'].split()))
        lines.append(f"Total Duration: {self._format_timestamp(total_duration)}")
        lines.append(f"Total Speakers: {len(speaker_stats)}")
        lines.append(f"Total Turns: {len(speaker_turns)}")
        lines.append(f"Total Words: {sum(stats['word_count'] for stats in speaker_stats.values())}")
        lines.append("")
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
        """Append a new speaker turn to an ongoing real-time transcript"""
        if not session_file:
            session_file = self.output_dir / f"realtime_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            formatted_line = f"[{self._format_timestamp(speaker_turn['start'])} - {self._format_timestamp(speaker_turn['end'])}] {speaker_turn['speaker']}: {speaker_turn['text']}\n"
            with open(session_file, 'a', encoding='utf-8') as f:
                f.write(formatted_line)
            logger.debug(f"Real-time update appended to {session_file}")
            return session_file
        except Exception as e:
            logger.error(f"Failed to save real-time update: {e}")
            raise

    def export_transcript(self, aligned_result, base_filename=None):
        """Export transcript in TXT format with integrated summary"""
        if not base_filename:
            base_filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            file_path = self.save_transcript(aligned_result, base_filename)
            logger.info(f"Exported TXT format to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to export TXT format: {e}")
            raise 