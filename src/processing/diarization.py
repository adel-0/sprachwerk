"""
Speaker diarization module using pyannote-audio
Identifies and segments different speakers in audio data
"""

import logging
import numpy as np
import torch

from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import threading
import queue
import time
from pathlib import Path
import tempfile
import scipy.io.wavfile as wavfile

from src.core.config import CONFIG, PYANNOTE_MODEL_PATH, TEMP_DIR, HF_AUTH_TOKEN
from src.utils.warning_suppressor import configure_torch_tf32

# Configure TF32 settings
configure_torch_tf32()

logger = logging.getLogger(__name__)

class SpeakerDiarizer:
    def __init__(self):
        self.model_path = PYANNOTE_MODEL_PATH
        self.device = CONFIG['diarization_device']
        self.sample_rate = CONFIG['sample_rate']
        self.min_speakers = CONFIG['min_speakers']
        self.max_speakers = CONFIG['max_speakers']
        
        # Enhanced diarization parameters for better accuracy
        self.min_segment_duration = CONFIG.get('diarization_min_segment_duration', 0.5)
        self.max_segment_duration = CONFIG.get('diarization_max_segment_duration', 30.0)
        self.base_clustering_threshold = CONFIG.get('diarization_clustering_threshold', 0.7)
        self.onset_threshold = CONFIG.get('diarization_onset_threshold', 0.5)
        self.offset_threshold = CONFIG.get('diarization_offset_threshold', 0.5)
        self.min_duration_on = CONFIG.get('diarization_min_duration_on', 0.1)
        self.min_duration_off = CONFIG.get('diarization_min_duration_off', 0.1)
        
        # Adaptive clustering threshold (will be updated based on speaker expectations)
        self.clustering_threshold = self.base_clustering_threshold
        
        self.pipeline = None
        self.is_loaded = False
        
        # Processing queues for real-time mode
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
        
        # Initialize adaptive thresholds
        self._update_adaptive_thresholds()
        
    def _update_adaptive_thresholds(self):
        """Update adaptive thresholds based on current speaker configuration"""
        # Import here to avoid circular imports
        from src.processing.speaker_identification import SpeakerIdentifier
        
        # Create a temporary speaker identifier to get adaptive thresholds
        temp_identifier = SpeakerIdentifier()
        self.clustering_threshold = temp_identifier.get_current_clustering_threshold()
        
        logger.debug(f"Updated diarization clustering threshold to: {self.clustering_threshold:.3f}")
    
    def update_speaker_expectations(self, min_speakers: int, max_speakers: int):
        """Update speaker expectations and recalculate adaptive thresholds"""
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        # Update adaptive thresholds
        self._update_adaptive_thresholds()
        
        logger.info(f"Updated speaker expectations: {min_speakers}-{max_speakers} speakers, "
                   f"clustering threshold: {self.clustering_threshold:.3f}")
    
    def load_model(self):
        """Load the speaker diarization pipeline"""
        if self.is_loaded:
            logger.info("Diarization pipeline already loaded")
            return
        
        logger.info(f"Loading speaker diarization pipeline: {self.model_path}")
        
        try:
            # Check CUDA availability
            if self.device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA not available for diarization, falling back to CPU")
                self.device = 'cpu'
            
            # Load the pipeline
            self.pipeline = Pipeline.from_pretrained(
                self.model_path,
                use_auth_token=HF_AUTH_TOKEN
            )
            
            # Set device
            if self.device == 'cuda':
                self.pipeline = self.pipeline.to(torch.device('cuda'))
            
            self.is_loaded = True
            logger.info(f"Diarization pipeline loaded successfully on {self.device}")
            
            # Log GPU memory usage if CUDA
            if self.device == 'cuda':
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"GPU Memory after diarization load - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            logger.warning("You may need to accept pyannote terms: https://huggingface.co/pyannote/speaker-diarization")
            raise
    
    def _process_diarization_segments(self, diarization, chunk_timestamp):
        """Process diarization results into standardized format"""
        speaker_segments = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Adjust timestamps with chunk offset
            start_time = turn.start + chunk_timestamp
            end_time = turn.end + chunk_timestamp
            duration = turn.end - turn.start
            
            # Apply segment filtering
            if self._is_valid_segment(duration):
                speaker_segments.append({
                    'speaker': speaker,
                    'start': start_time,
                    'end': end_time,
                    'duration': duration
                })
        
        # Apply speaker consolidation based on adaptive thresholds
        consolidated_segments = self._consolidate_speakers(speaker_segments)
        
        return consolidated_segments
    
    def _consolidate_speakers(self, speaker_segments):
        """Consolidate speakers that are likely the same person based on adaptive thresholds"""
        if not speaker_segments or len(speaker_segments) <= 1:
            return speaker_segments
        
        # Skip consolidation if we expect many speakers or have wide range
        if self.max_speakers > 3 and (self.max_speakers - self.min_speakers) > 2:
            logger.debug("Skipping speaker consolidation for wide speaker range")
            return speaker_segments
        
        # Group segments by speaker
        from collections import defaultdict
        speaker_groups = defaultdict(list)
        for segment in speaker_segments:
            speaker_groups[segment['speaker']].append(segment)
        
        # If we already have the expected number or fewer speakers, minimal consolidation
        unique_speakers = list(speaker_groups.keys())
        if len(unique_speakers) <= self.max_speakers:
            logger.debug(f"Speaker count ({len(unique_speakers)}) within expected range, minimal consolidation")
            return self._minimal_speaker_consolidation(speaker_segments, speaker_groups)
        
        # Aggressive consolidation for single speaker scenarios
        if self.min_speakers == 1 and self.max_speakers == 1:
            logger.info(f"Single speaker expected, consolidating {len(unique_speakers)} speakers")
            return self._aggressive_speaker_consolidation(speaker_segments, speaker_groups)
        
        # Moderate consolidation for small speaker counts
        if self.max_speakers <= 2:
            logger.info(f"Small speaker count expected ({self.max_speakers}), consolidating {len(unique_speakers)} speakers")
            return self._moderate_speaker_consolidation(speaker_segments, speaker_groups)
        
        return speaker_segments
    
    def _minimal_speaker_consolidation(self, speaker_segments, speaker_groups):
        """Minimal consolidation - only merge very similar adjacent speakers"""
        # Only merge if speakers have very close timestamps (likely same person with slight detection gaps)
        consolidated_segments = []
        segments_by_time = sorted(speaker_segments, key=lambda x: x['start'])
        
        i = 0
        while i < len(segments_by_time):
            current_segment = segments_by_time[i].copy()
            
            # Look for immediately following segments from different speakers within 2 seconds
            j = i + 1
            while (j < len(segments_by_time) and 
                   segments_by_time[j]['start'] - current_segment['end'] < 2.0 and
                   segments_by_time[j]['speaker'] != current_segment['speaker']):
                
                # Merge if the gap is very small (likely same speaker)
                if segments_by_time[j]['start'] - current_segment['end'] < 0.5:
                    logger.debug(f"Merging adjacent speakers: {current_segment['speaker']} -> {segments_by_time[j]['speaker']}")
                    current_segment['end'] = segments_by_time[j]['end']
                    current_segment['duration'] = current_segment['end'] - current_segment['start']
                    j += 1
                else:
                    break
            
            consolidated_segments.append(current_segment)
            i = j if j > i + 1 else i + 1
        
        return consolidated_segments
    
    def _moderate_speaker_consolidation(self, speaker_segments, speaker_groups):
        """Moderate consolidation for 2-3 expected speakers"""
        # Calculate speaker statistics for similarity-based merging
        speaker_stats = {}
        for speaker, segments in speaker_groups.items():
            total_duration = sum(seg['duration'] for seg in segments)
            avg_duration = total_duration / len(segments)
            speaker_stats[speaker] = {
                'total_duration': total_duration,
                'avg_duration': avg_duration,
                'segment_count': len(segments),
                'segments': segments
            }
        
        # Sort speakers by total speaking time (most active first)
        sorted_speakers = sorted(speaker_stats.items(), key=lambda x: x[1]['total_duration'], reverse=True)
        
        # Keep top speakers, merge others into most similar
        keep_count = min(self.max_speakers, len(sorted_speakers))
        main_speakers = [speaker for speaker, _ in sorted_speakers[:keep_count]]
        merge_candidates = [speaker for speaker, _ in sorted_speakers[keep_count:]]
        
        # Merge short-duration speakers into main speakers
        speaker_mapping = {speaker: speaker for speaker in main_speakers}
        
        for candidate in merge_candidates:
            # Find the most similar main speaker based on temporal proximity
            best_target = main_speakers[0]  # Default to most active speaker
            
            candidate_segments = speaker_stats[candidate]['segments']
            min_avg_distance = float('inf')
            
            for main_speaker in main_speakers:
                main_segments = speaker_stats[main_speaker]['segments']
                distances = []
                
                for c_seg in candidate_segments:
                    min_dist = min(abs(c_seg['start'] - m_seg['start']) for m_seg in main_segments)
                    distances.append(min_dist)
                
                avg_distance = sum(distances) / len(distances) if distances else float('inf')
                if avg_distance < min_avg_distance:
                    min_avg_distance = avg_distance
                    best_target = main_speaker
            
            speaker_mapping[candidate] = best_target
            logger.debug(f"Merging speaker {candidate} into {best_target} (avg distance: {min_avg_distance:.2f}s)")
        
        # Apply mapping to segments
        consolidated_segments = []
        for segment in speaker_segments:
            new_segment = segment.copy()
            new_segment['speaker'] = speaker_mapping.get(segment['speaker'], segment['speaker'])
            consolidated_segments.append(new_segment)
        
        return consolidated_segments
    
    def _aggressive_speaker_consolidation(self, speaker_segments, speaker_groups):
        """Aggressive consolidation for single speaker scenarios"""
        # For single speaker expected, merge all speakers into the most active one
        if not speaker_groups:
            return speaker_segments
        
        # Find the speaker with the most total speaking time
        main_speaker = max(speaker_groups.keys(), 
                          key=lambda s: sum(seg['duration'] for seg in speaker_groups[s]))
        
        logger.info(f"Consolidating all speakers into main speaker: {main_speaker}")
        
        # Reassign all segments to the main speaker
        consolidated_segments = []
        for segment in speaker_segments:
            new_segment = segment.copy()
            new_segment['speaker'] = main_speaker
            consolidated_segments.append(new_segment)
        
        return consolidated_segments
    
    def _is_valid_segment(self, duration):
        """Check if segment meets duration requirements"""
        if duration < self.min_segment_duration:
            logger.debug(f"Skipping short segment: {duration:.2f}s < {self.min_segment_duration}s")
            return False
        
        if duration > self.max_segment_duration:
            logger.debug(f"Truncating long segment: {duration:.2f}s > {self.max_segment_duration}s")
            # Could implement segment splitting here if needed
        
        return True
    
    def diarize_chunk(self, audio_chunk, chunk_timestamp=0.0):
        """Perform speaker diarization on a single audio chunk"""
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load_model() first.")
        
        try:
            # Save audio chunk to temporary file (pyannote requires file input)
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav', 
                delete=False, 
                dir=TEMP_DIR
            )
            
            # Ensure audio is in the right format
            if audio_chunk.dtype != np.int16:
                audio_int16 = (audio_chunk * 32767).astype(np.int16)
            else:
                audio_int16 = audio_chunk
            
            # Write audio to temp file
            wavfile.write(temp_file.name, self.sample_rate, audio_int16)
            temp_file.close()
            
            # Perform diarization
            diarization = self.pipeline(
                temp_file.name,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )
            
            # Process diarization results
            speaker_segments = self._process_diarization_segments(diarization, chunk_timestamp)
            
            # Clean up temp file
            Path(temp_file.name).unlink(missing_ok=True)
            
            result = {
                'speaker_segments': speaker_segments,
                'chunk_timestamp': chunk_timestamp,
                'num_speakers': len(set(seg['speaker'] for seg in speaker_segments))
            }
            
            logger.debug(f"Diarized chunk at {chunk_timestamp:.2f}s: {result['num_speakers']} speakers, {len(speaker_segments)} segments")
            return result
            
        except Exception as e:
            logger.error(f"Diarization failed for chunk at {chunk_timestamp:.2f}s: {e}")
            # Clean up temp file on error
            try:
                Path(temp_file.name).unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to clean up temp file on error")
            
            return {
                'speaker_segments': [],
                'chunk_timestamp': chunk_timestamp,
                'num_speakers': 0,
                'error': str(e)
            }
    
    def diarize_file(self, audio_file_path):
        """Perform speaker diarization on a complete audio file"""
        logger.info(f"Diarizing audio file: {audio_file_path}")
        
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Perform diarization on the entire file
            diarization = self.pipeline(
                str(audio_file_path),
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )
            
            # Process diarization results
            speaker_segments = self._process_diarization_segments(diarization, 0.0)
            
            result = {
                'speaker_segments': speaker_segments,
                'num_speakers': len(set(seg['speaker'] for seg in speaker_segments)),
                'total_duration': max([seg['end'] for seg in speaker_segments]) if speaker_segments else 0,
                'speakers': list(set(seg['speaker'] for seg in speaker_segments))
            }
            
            logger.info(f"File diarization completed. Found {len(set(seg['speaker'] for seg in speaker_segments))} speakers in {len(speaker_segments)} segments")
            return result
            
        except Exception as e:
            logger.error(f"File diarization failed: {e}")
            raise
    
    def start_real_time_processing(self):
        """Start real-time diarization processing thread"""
        if self.is_processing:
            logger.warning("Real-time diarization processing already started")
            return
        
        if not self.is_loaded:
            self.load_model()
        
        logger.info("Starting real-time diarization processing")
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
    
    def stop_real_time_processing(self):
        """Stop real-time diarization processing"""
        if not self.is_processing:
            return
        
        logger.info("Stopping real-time diarization processing")
        self.is_processing = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
    
    def _processing_loop(self):
        """Main processing loop for real-time diarization"""
        while self.is_processing:
            try:
                # Get audio chunk from queue (with timeout)
                chunk_data = self.input_queue.get(timeout=1)
                if chunk_data is None:
                    continue
                
                audio_chunk, chunk_timestamp = chunk_data
                
                # Perform diarization
                start_time = time.time()
                result = self.diarize_chunk(audio_chunk, chunk_timestamp)
                processing_time = time.time() - start_time
                
                # Add processing time to result
                result['processing_time'] = processing_time
                
                # Put result in output queue
                try:
                    self.output_queue.put(result, block=False)
                except queue.Full:
                    logger.warning("Diarization output queue full, dropping result")
                
                # Log performance
                chunk_duration = len(audio_chunk) / self.sample_rate
                real_time_factor = processing_time / chunk_duration
                logger.debug(f"Diarization RTF: {real_time_factor:.2f}x")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in diarization processing loop: {e}")
    
    def add_audio_chunk(self, audio_chunk, chunk_timestamp):
        """Add audio chunk to processing queue"""
        try:
            self.input_queue.put((audio_chunk, chunk_timestamp), block=False)
        except queue.Full:
            logger.warning("Diarization input queue full, dropping chunk")
    
    def get_diarization_result(self, timeout=1):
        """Get next diarization result"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def diarize_chunks(self, audio_chunks):
        """Diarize multiple audio chunks (for batch processing)"""
        results = []
        
        logger.info(f"Diarizing {len(audio_chunks)} audio chunks")
        
        for i, (chunk, timestamp) in enumerate(audio_chunks):
            logger.info(f"Diarizing chunk {i+1}/{len(audio_chunks)}")
            result = self.diarize_chunk(chunk, timestamp)
            results.append(result)
        
        return results
    
    def merge_speaker_segments(self, chunk_results):
        """Merge speaker segments from multiple chunks"""
        all_segments = []
        
        for result in chunk_results:
            if 'speaker_segments' in result:
                all_segments.extend(result['speaker_segments'])
        
        # Sort by start time
        all_segments.sort(key=lambda x: x['start'])
        
        # Merge consecutive segments from the same speaker
        merged_segments = []
        current_segment = None
        
        for segment in all_segments:
            if current_segment is None:
                current_segment = segment.copy()
            elif (current_segment['speaker'] == segment['speaker'] and 
                  abs(segment['start'] - current_segment['end']) < 1.0):  # 1 second gap tolerance
                # Merge segments
                current_segment['end'] = segment['end']
                current_segment['duration'] = current_segment['end'] - current_segment['start']
            else:
                # Start new segment
                merged_segments.append(current_segment)
                current_segment = segment.copy()
        
        if current_segment is not None:
            merged_segments.append(current_segment)
        
        # Assign normalized speaker names
        unique_speakers = list(set(seg['speaker'] for seg in merged_segments))
        timestamp = time.time_ns() // 1000000  # Use milliseconds for uniqueness
        speaker_mapping = {speaker: f"TEMP_SPEAKER_{i:02d}_{timestamp}" for i, speaker in enumerate(unique_speakers)}
        
        for segment in merged_segments:
            segment['speaker_normalized'] = speaker_mapping[segment['speaker']]
        
        logger.info(f"Merged {len(all_segments)} segments into {len(merged_segments)} segments with {len(unique_speakers)} speakers")
        
        return {
            'speaker_segments': merged_segments,
            'num_speakers': len(unique_speakers),
            'speaker_mapping': speaker_mapping
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_real_time_processing()
        
        if self.pipeline is not None:
            # pyannote doesn't need explicit cleanup
            self.pipeline = None
            self.is_loaded = False
        
        # Clear GPU cache if using CUDA
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache after diarization cleanup") 