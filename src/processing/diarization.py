"""
Speaker diarization module using SpeechBrain ECAPA-TDNN embeddings and clustering
Identifies and segments different speakers in audio data
"""

import logging
import numpy as np
import torch
import threading
import queue
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.core.config import CONFIG, DiarizationBackend
from src.processing.embedding_extractor import SpeakerEmbeddingExtractor
from src.processing.speaker_clustering import SpeakerClustering

logger = logging.getLogger(__name__)

class SpeakerDiarizer:
    """
    SpeechBrain-based speaker diarization system
    Uses ECAPA-TDNN embeddings and clustering for speaker identification
    """
    
    def __init__(self):
        # Get configuration
        self.config = CONFIG if CONFIG else {}
        self.sample_rate = self.config.get('sample_rate', 48000)
        self.min_speakers = self.config.get('min_speakers', 1)
        self.max_speakers = self.config.get('max_speakers', 2)
        
        # Diarization backend selection
        backend_str = self.config.get('diarization_backend', 'speechbrain')
        if isinstance(backend_str, str):
            self.backend = DiarizationBackend.SPEECHBRAIN if backend_str == 'speechbrain' else DiarizationBackend.PYANNOTE
        else:
            self.backend = backend_str
        
        # SpeechBrain parameters
        self.window_length = self.config.get('window_length', 1.5)  # seconds
        self.hop_length = self.config.get('hop_length', 0.75)  # seconds
        self.cluster_threshold = self.config.get('cluster_threshold', 0.3)
        self.min_cluster_size = self.config.get('min_cluster_size', 2)
        
        # Clustering parameters
        self.clustering_algorithm = self.config.get('clustering_algorithm', 'agglomerative')
        
        # Components
        self.embedding_extractor = SpeakerEmbeddingExtractor()
        self.clustering = SpeakerClustering(algorithm=self.clustering_algorithm)
        
        # State
        self.is_loaded = False
        
        # Processing queues for real-time mode
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
        
        # Segment filtering parameters
        self.min_segment_duration = self.config.get('diarization_min_segment_duration', 0.5)
        self.max_segment_duration = self.config.get('diarization_max_segment_duration', 30.0)
        
        logger.info(f"Initialized SpeakerDiarizer with backend: {self.backend.value}")
    
    def load_model(self):
        """Load the required models"""
        if self.is_loaded:
            logger.info("Diarization models already loaded")
            return
        
        logger.info("Loading SpeechBrain diarization models...")
        
        try:
            # Load embedding extractor
            self.embedding_extractor.load_model()
            
            # Update clustering parameters
            self.clustering.update_parameters(
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
                cluster_threshold=self.cluster_threshold,
                algorithm=self.clustering_algorithm
            )
            
            self.is_loaded = True
            logger.info("SpeechBrain diarization models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load diarization models: {e}")
            raise
    
    def update_speaker_expectations(self, min_speakers: int, max_speakers: int):
        """Update speaker expectations"""
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        # Update clustering parameters
        if hasattr(self, 'clustering'):
            self.clustering.update_parameters(
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
        
        logger.info(f"Updated speaker expectations: {min_speakers}-{max_speakers} speakers")
    
    def diarize_chunk(self, audio_chunk, chunk_timestamp=0.0):
        """Perform speaker diarization on a single audio chunk"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_model() first.")
        
        try:
            # Ensure audio is in the right format
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Extract speaker segments using sliding window
            speaker_segments = self._extract_speaker_segments(audio_chunk, chunk_timestamp)
            
            if not speaker_segments:
                logger.debug(f"No speaker segments found for chunk at {chunk_timestamp:.2f}s")
                return {
                    'speaker_segments': [],
                    'chunk_timestamp': chunk_timestamp,
                    'num_speakers': 0
                }
            
            # Process segments to ensure they're valid
            processed_segments = self._process_segments(speaker_segments)
            
            result = {
                'speaker_segments': processed_segments,
                'chunk_timestamp': chunk_timestamp,
                'num_speakers': len(set(seg['speaker'] for seg in processed_segments))
            }
            
            logger.debug(f"Diarized chunk at {chunk_timestamp:.2f}s: {result['num_speakers']} speakers, {len(processed_segments)} segments")
            return result
            
        except Exception as e:
            logger.error(f"Diarization failed for chunk at {chunk_timestamp:.2f}s: {e}")
            
            return {
                'speaker_segments': [],
                'chunk_timestamp': chunk_timestamp,
                'num_speakers': 0,
                'error': str(e)
            }
    
    def _extract_speaker_segments(self, audio_chunk, chunk_timestamp):
        """Extract speaker segments using sliding window and clustering"""
        
        # Calculate window parameters
        window_samples = int(self.window_length * self.sample_rate)
        hop_samples = int(self.hop_length * self.sample_rate)
        
        # Extract overlapping windows
        windows = []
        window_timestamps = []
        
        for start_sample in range(0, len(audio_chunk) - window_samples + 1, hop_samples):
            end_sample = start_sample + window_samples
            window = audio_chunk[start_sample:end_sample]
            
            # Only process windows with sufficient audio
            if len(window) >= window_samples:
                windows.append(window)
                window_timestamps.append(chunk_timestamp + start_sample / self.sample_rate)
        
        if not windows:
            logger.debug("No valid windows extracted from audio chunk")
            return []
        
        # Extract embeddings for each window
        embeddings = self.embedding_extractor.extract_embeddings(windows)
        
        if len(embeddings) == 0:
            logger.warning("No embeddings extracted from windows")
            return []
        
        # Cluster embeddings to identify speakers
        clustering_result = self.clustering.cluster_embeddings(embeddings, window_timestamps)
        
        if clustering_result['n_clusters'] == 0:
            logger.warning("No clusters found in embeddings")
            return []
        
        # Convert clustering results to speaker segments
        speaker_segments = self._clustering_to_segments(
            clustering_result['labels'],
            window_timestamps,
            self.window_length
        )
        
        return speaker_segments
    
    def _clustering_to_segments(self, labels, timestamps, window_length):
        """Convert clustering labels to speaker segments"""
        if not labels or not timestamps:
            return []
        
        segments = []
        
        # Group consecutive windows with same speaker
        current_speaker = labels[0]
        current_start = timestamps[0]
        current_end = timestamps[0] + window_length
        
        for i in range(1, len(labels)):
            if labels[i] == current_speaker:
                # Continue current segment
                current_end = timestamps[i] + window_length
            else:
                # End current segment and start new one
                segments.append({
                    'speaker': f'SPEAKER_{current_speaker:03d}',
                    'start': current_start,
                    'end': current_end,
                    'duration': current_end - current_start
                })
                
                current_speaker = labels[i]
                current_start = timestamps[i]
                current_end = timestamps[i] + window_length
        
        # Add the last segment
        segments.append({
            'speaker': f'SPEAKER_{current_speaker:03d}',
            'start': current_start,
            'end': current_end,
            'duration': current_end - current_start
        })
        
        return segments
    
    def _process_segments(self, speaker_segments):
        """Process segments to ensure they meet minimum requirements"""
        processed_segments = []
        
        for segment in speaker_segments:
            # Apply segment filtering
            if self._is_valid_segment(segment['duration']):
                processed_segments.append(segment)
        
        # Merge adjacent segments from the same speaker
        if processed_segments:
            processed_segments = self._merge_adjacent_segments(processed_segments)
        
        return processed_segments
    
    def _is_valid_segment(self, duration):
        """Check if a segment meets minimum duration requirements"""
        return self.min_segment_duration <= duration <= self.max_segment_duration
    
    def _merge_adjacent_segments(self, segments):
        """Merge adjacent segments from the same speaker"""
        if len(segments) <= 1:
            return segments
        
        merged = []
        current_segment = segments[0].copy()
        
        for next_segment in segments[1:]:
            # Check if segments are from same speaker and adjacent (within 0.5 seconds)
            if (current_segment['speaker'] == next_segment['speaker'] and 
                abs(next_segment['start'] - current_segment['end']) <= 0.5):
                # Merge segments
                current_segment['end'] = next_segment['end']
                current_segment['duration'] = current_segment['end'] - current_segment['start']
            else:
                # Add current segment and start new one
                merged.append(current_segment)
                current_segment = next_segment.copy()
        
        # Add the last segment
        merged.append(current_segment)
        
        return merged
    
    def diarize_file(self, audio_file_path):
        """Perform speaker diarization on a complete audio file"""
        logger.info(f"Diarizing audio file: {audio_file_path}")
        
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Load audio file
            import librosa
            audio_data, sr = librosa.load(audio_file_path, sr=self.sample_rate)
            
            # Process in chunks to handle large files
            chunk_duration = 30.0  # seconds
            chunk_samples = int(chunk_duration * self.sample_rate)
            overlap_samples = int(0.5 * self.sample_rate)  # 0.5 second overlap
            
            all_segments = []
            
            for start_sample in range(0, len(audio_data), chunk_samples - overlap_samples):
                end_sample = min(start_sample + chunk_samples, len(audio_data))
                chunk = audio_data[start_sample:end_sample]
                chunk_timestamp = start_sample / self.sample_rate
                
                # Diarize chunk
                chunk_result = self.diarize_chunk(chunk, chunk_timestamp)
                all_segments.extend(chunk_result['speaker_segments'])
            
            # Merge overlapping segments
            merged_segments = self._merge_overlapping_segments(all_segments)
            
            result = {
                'speaker_segments': merged_segments,
                'num_speakers': len(set(seg['speaker'] for seg in merged_segments)),
                'total_duration': max([seg['end'] for seg in merged_segments]) if merged_segments else 0,
                'speakers': list(set(seg['speaker'] for seg in merged_segments))
            }
            
            logger.info(f"File diarization completed. Found {result['num_speakers']} speakers in {len(merged_segments)} segments")
            return result
            
        except Exception as e:
            logger.error(f"File diarization failed: {e}")
            raise
    
    def _merge_overlapping_segments(self, segments):
        """Merge overlapping segments from different chunks"""
        if not segments:
            return []
        
        # Sort segments by start time
        segments.sort(key=lambda x: x['start'])
        
        merged = []
        current_segment = segments[0].copy()
        
        for next_segment in segments[1:]:
            # Check if segments overlap and are from the same speaker
            if (current_segment['speaker'] == next_segment['speaker'] and 
                next_segment['start'] <= current_segment['end'] + 1.0):  # 1 second tolerance
                # Merge segments
                current_segment['end'] = max(current_segment['end'], next_segment['end'])
                current_segment['duration'] = current_segment['end'] - current_segment['start']
            else:
                # Add current segment and start new one
                merged.append(current_segment)
                current_segment = next_segment.copy()
        
        # Add the last segment
        merged.append(current_segment)
        
        return merged
    
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
        merged_segments = self._merge_overlapping_segments(all_segments)
        
        # Assign normalized speaker names
        unique_speakers = list(set(seg['speaker'] for seg in merged_segments))
        timestamp = time.time_ns() // 1000000
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
        
        # Cleanup embedding extractor
        if hasattr(self, 'embedding_extractor'):
            self.embedding_extractor.cleanup()
        
        self.is_loaded = False
        logger.info("Diarization cleanup completed") 