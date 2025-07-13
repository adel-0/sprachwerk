"""
Alignment module for combining Whisper transcription with speaker diarization
Creates speaker-labeled transcripts by matching words with speaker segments
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from src.processing.speaker_identification import SpeakerIdentifier

logger = logging.getLogger(__name__)

class TranscriptionAligner:
    def __init__(self):
        self.overlap_threshold = 0.5  # Minimum overlap ratio to assign speaker
        self.gap_tolerance = 0.5      # Maximum gap between words to group them
        self.speaker_identifier = SpeakerIdentifier()  # Speaker identification system
        
    def align_transcription_with_speakers(self, transcription_result, diarization_result, audio_data: Optional[np.ndarray] = None):
        """
        Align transcription words with speaker segments
        """
        words = transcription_result.get('words', [])
        speaker_segments = diarization_result.get('speaker_segments', [])
        
        if not words:
            # Don't warn for empty transcription - silence periods are normal
            logger.debug("No words found in transcription result")
            if 'error' in transcription_result:
                logger.warning(f"Transcription error: {transcription_result['error']}")
            else:
                logger.debug(f"Transcription result keys: {list(transcription_result.keys())}")
                logger.debug(f"Transcription text length: {len(transcription_result.get('text', ''))}")
            return self._create_empty_result(transcription_result, diarization_result)
        
        if not speaker_segments:
            logger.warning("No speaker segments found in diarization result")
            logger.warning(f"Diarization result keys: {list(diarization_result.keys())}")
            logger.warning(f"Number of speaker segments: {len(diarization_result.get('speaker_segments', []))}")
            if 'error' in diarization_result:
                logger.warning(f"Diarization error: {diarization_result['error']}")
            return self._assign_unknown_speaker(transcription_result)
        
        # Assign speakers to words
        aligned_words = self._assign_speakers_to_words(words, speaker_segments)
        
        # Group words by speaker turns
        speaker_turns = self._group_words_into_turns(aligned_words)
        
        # Apply speaker identification to get consistent speaker names across sessions
        # Pass audio data for acoustic voice signatures
        speaker_turns = self.speaker_identifier.process_speaker_turns(speaker_turns, audio_data)
        
        # Create final aligned result
        result = {
            'speaker_turns': speaker_turns,
            'total_words': len(aligned_words),
            'total_speakers': len(set(word.get('speaker', 'Unknown') for word in aligned_words)),
            'language': transcription_result.get('language', 'unknown'),
            'language_probability': transcription_result.get('language_probability', 0.0),
            'detected_languages': transcription_result.get('detected_languages', {}),
            'language_durations': transcription_result.get('language_durations', {}),
            'multilingual': transcription_result.get('multilingual', False),
            'alignment_stats': self._calculate_alignment_stats(aligned_words, speaker_segments)
        }
        
        logger.info(f"Alignment completed: {result['total_words']} words, {result['total_speakers']} speakers")
        return result
    
    def _assign_speakers_to_words(self, words, speaker_segments):
        """Assign speakers to individual words based on temporal overlap"""
        aligned_words = []
        
        for word in words:
            word_start = word['start']
            word_end = word['end']
            word_duration = word_end - word_start
            
            best_speaker = None
            best_overlap = 0.0
            
            # Find the speaker segment with maximum overlap
            for segment in speaker_segments:
                segment_start = segment['start']
                segment_end = segment['end']
                
                # Calculate overlap
                overlap_start = max(word_start, segment_start)
                overlap_end = min(word_end, segment_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if word_duration > 0:
                    overlap_ratio = overlap_duration / word_duration
                else:
                    overlap_ratio = 0.0
                
                # Check if this is the best overlap so far
                if overlap_ratio > best_overlap and overlap_ratio >= self.overlap_threshold:
                    best_overlap = overlap_ratio
                    best_speaker = segment.get('speaker_normalized', segment['speaker'])
            
            # Create aligned word entry
            aligned_word = word.copy()
            aligned_word['speaker'] = best_speaker if best_speaker else 'Unknown'
            aligned_word['speaker_confidence'] = best_overlap
            
            aligned_words.append(aligned_word)
        
        return aligned_words
    
    def _group_words_into_turns(self, aligned_words):
        """Group consecutive words by the same speaker into turns"""
        if not aligned_words:
            return []
        
        turns = []
        current_turn = {
            'speaker': aligned_words[0]['speaker'],
            'words': [],
            'start': aligned_words[0]['start'],
            'end': aligned_words[0]['end'],
            'text': ''
        }
        
        for word in aligned_words:
            # Check if we should start a new turn
            if (word['speaker'] != current_turn['speaker'] or 
                word['start'] - current_turn['end'] > self.gap_tolerance):
                
                # Finish current turn
                if current_turn['words']:
                    current_turn['text'] = ' '.join([w['word'] for w in current_turn['words']])
                    current_turn['duration'] = current_turn['end'] - current_turn['start']
                    current_turn['word_count'] = len(current_turn['words'])
                    turns.append(current_turn)
                
                # Start new turn
                current_turn = {
                    'speaker': word['speaker'],
                    'words': [],
                    'start': word['start'],
                    'end': word['end'],
                    'text': ''
                }
            
            # Add word to current turn
            current_turn['words'].append(word)
            current_turn['end'] = word['end']
        
        # Add the final turn
        if current_turn['words']:
            current_turn['text'] = ' '.join([w['word'] for w in current_turn['words']])
            current_turn['duration'] = current_turn['end'] - current_turn['start']
            current_turn['word_count'] = len(current_turn['words'])
            turns.append(current_turn)
        
        logger.debug(f"Grouped {len(aligned_words)} words into {len(turns)} speaker turns")
        return turns
    
    def _create_result_template(self, transcription_result, speaker_segments_count=0):
        """Create a standardized result template"""
        return {
            'speaker_turns': [],
            'total_words': 0,
            'total_speakers': 0,
            'language': transcription_result.get('language', 'unknown'),
            'language_probability': transcription_result.get('language_probability', 0.0),
            'detected_languages': transcription_result.get('detected_languages', {}),
            'language_durations': transcription_result.get('language_durations', {}),
            'multilingual': transcription_result.get('multilingual', False),
            'alignment_stats': {
                'total_words': 0,
                'words_with_known_speaker': 0,
                'words_with_unknown_speaker': 0,
                'speaker_assignment_rate': 0.0,
                'total_speaker_segments': speaker_segments_count
            }
        }
    
    def _calculate_alignment_stats(self, aligned_words, speaker_segments):
        """Calculate statistics about the alignment quality"""
        total_words = len(aligned_words)
        unknown_words = sum(1 for word in aligned_words if word['speaker'] == 'Unknown')
        known_words = total_words - unknown_words
        
        return {
            'total_words': total_words,
            'words_with_known_speaker': known_words,
            'words_with_unknown_speaker': unknown_words,
            'speaker_assignment_rate': known_words / total_words if total_words > 0 else 0.0,
            'total_speaker_segments': len(speaker_segments)
        }
    
    def _create_empty_result(self, transcription_result, diarization_result):
        """Create empty result when alignment fails"""
        return self._create_result_template(
            transcription_result, 
            len(diarization_result.get('speaker_segments', []))
        )
    
    def _assign_unknown_speaker(self, transcription_result):
        """Assign all words to unknown speaker when diarization fails"""
        words = transcription_result.get('words', [])
        
        if not words:
            return self._create_result_template(transcription_result)
        
        # Create speaker turns with unknown speaker
        speaker_turns = [{
            'speaker': 'Unknown',
            'words': words,
            'start': words[0]['start'],
            'end': words[-1]['end'],
            'text': ' '.join([w['word'] for w in words]),
            'duration': words[-1]['end'] - words[0]['start'],
            'word_count': len(words)
        }]
        
        result = self._create_result_template(transcription_result)
        result.update({
            'speaker_turns': speaker_turns,
            'total_words': len(words),
            'total_speakers': 1,
            'alignment_stats': self._calculate_alignment_stats(
                [{'speaker': 'Unknown'} for _ in words], []
            )
        })
        
        return result
    
    def merge_chunk_results(self, chunk_alignments):
        """Merge alignment results from multiple chunks"""
        if not chunk_alignments:
            return self._create_empty_result({}, {})
        
        all_turns = []
        total_words = 0
        all_speakers = set()
        
        # Collect all speaker turns from chunks
        for alignment in chunk_alignments:
            chunk_turns = alignment.get('speaker_turns', [])
            all_turns.extend(chunk_turns)
            total_words += alignment.get('total_words', 0)
            
            for turn in chunk_turns:
                all_speakers.add(turn['speaker'])
        
        # Sort turns by start time
        all_turns.sort(key=lambda x: x['start'])
        
        # Get language info from first non-empty chunk
        language = 'unknown'
        language_probability = 0.0
        for alignment in chunk_alignments:
            if alignment.get('language'):
                language = alignment['language']
                language_probability = alignment.get('language_probability', 0.0)
                break
        
        return {
            'speaker_turns': all_turns,
            'total_words': total_words,
            'total_speakers': len(all_speakers),
            'language': language,
            'language_probability': language_probability,
            'alignment_stats': {
                'total_words': total_words,
                'total_chunks_processed': len(chunk_alignments)
            }
        } 