"""
Speaker identification and naming system
Creates voice signatures for speakers and enables cross-session identification

ADAPTIVE THRESHOLD SYSTEM:
This module now includes an adaptive threshold system that automatically adjusts
speaker detection sensitivity based on the expected number of speakers:

1. Single Speaker (min=1, max=1):
   - Very permissive thresholds (similarity ~0.50, clustering ~0.55)
   - Aggressive consolidation - merges all detected speakers into one
   - Ideal for monologues, presentations, single-person recordings

2. Small Groups (2-3 speakers):
   - Moderately permissive thresholds 
   - Smart consolidation based on speaking time and temporal proximity
   - Merges low-activity speakers into main speakers

3. Larger Groups (4+ speakers):
   - Uses base thresholds with minimal consolidation
   - Only merges very similar adjacent segments

CONFIGURATION OPTIONS:
- enable_adaptive_speaker_thresholds: Enable/disable the adaptive system
- base_speaker_similarity_threshold: Base threshold for speaker matching (0.65)
- single_speaker_similarity_boost: Additional permissiveness for single speaker (0.25)
- single_speaker_clustering_boost: Clustering threshold reduction for single speaker (0.15)

USAGE:
The system automatically adapts when speaker expectations are set via:
- Command line: --speakers 1 or --speakers 1-2
- Interactive menu: speaker preference selection
- Programmatic: SpeakerIdentifier.set_expected_speakers(min, max)
"""

import logging
import json
import numpy as np
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pickle
from dataclasses import dataclass, asdict
from collections import defaultdict
import librosa

from src.core.config import CONFIG, OUTPUT_DIR
from src.core.config import get_outputs_dir

logger = logging.getLogger(__name__)

@dataclass
class SpeakerProfile:
    speaker_id: str
    name: Optional[str] = None
    voice_signature: Optional[List] = None
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    total_speaking_time: float = 0.0
    session_count: int = 0
    confidence_score: float = 0.0
    notes: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class SpeakerIdentifier:
    def __init__(self, speakers_db_path=None):
        outputs_dir = Path(get_outputs_dir())
        self.speakers_db_path = speakers_db_path or (outputs_dir / "speakers_database.json")
        self.voice_signatures_path = outputs_dir / "voice_signatures.pkl"
        self.speakers_db: Dict[str, SpeakerProfile] = {}
        self.voice_signatures: Dict[str, np.ndarray] = {}
        self.current_session_mapping: Dict[str, str] = {}
        self.sample_rate = CONFIG.get('sample_rate', 48000)
        self.frame_length = int(0.025 * self.sample_rate)
        self.hop_length = int(0.010 * self.sample_rate)
        self.base_similarity_threshold = CONFIG.get('base_speaker_similarity_threshold', 0.75)
        self.base_clustering_threshold = CONFIG.get('diarization_clustering_threshold', 0.7)
        self.min_speech_duration = 2.0
        self.max_signature_duration = 30.0
        self.enable_adaptive_thresholds = CONFIG.get('enable_adaptive_speaker_thresholds', True)
        self.single_speaker_similarity_boost = CONFIG.get('single_speaker_similarity_boost', 0.20)
        self.single_speaker_clustering_boost = CONFIG.get('single_speaker_clustering_boost', 0.15)
        self.similarity_threshold = self.base_similarity_threshold
        self.clustering_threshold = self.base_clustering_threshold
        self.load_speakers_database()
        self.update_adaptive_thresholds()

    def _handle_database_operation(self, operation_func, operation_name):
        try:
            return operation_func()
        except Exception as e:
            logger.error(f"Error {operation_name}: {e}")
            return False

    def load_speakers_database(self):
        def load_operation():
            if self.speakers_db_path.exists():
                with open(self.speakers_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.speakers_db = {speaker_id: SpeakerProfile.from_dict(profile_data) for speaker_id, profile_data in data.items()}
                logger.info(f"Loaded {len(self.speakers_db)} speaker profiles")
            if self.voice_signatures_path.exists():
                with open(self.voice_signatures_path, 'rb') as f:
                    self.voice_signatures = pickle.load(f)
                logger.info(f"Loaded {len(self.voice_signatures)} voice signatures")
            return True
        if not self._handle_database_operation(load_operation, "loading speakers database"):
            self.speakers_db = {}
            self.voice_signatures = {}

    def save_speakers_database(self):
        def save_operation():
            for speaker_id, signature in self.voice_signatures.items():
                if speaker_id in self.speakers_db:
                    self.speakers_db[speaker_id].voice_signature = signature.tolist()
            data = {speaker_id: profile.to_dict() for speaker_id, profile in self.speakers_db.items()}
            with open(self.speakers_db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            with open(self.voice_signatures_path, 'wb') as f:
                pickle.dump(self.voice_signatures, f)
            logger.info(f"Saved {len(self.speakers_db)} speaker profiles and {len(self.voice_signatures)} voice signatures")
            return True
        return self._handle_database_operation(save_operation, "saving speakers database")

    def create_voice_signature(self, speaker_segments: List[Dict], audio_data: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if audio_data is None:
            logger.warning("No audio data provided for voice signature creation")
            return None
        valid_segments = [seg for seg in speaker_segments if seg.get('duration', 0) >= 0.5]
        if not valid_segments:
            logger.debug("No valid segments after filtering")
            return None
        total_duration = sum(seg.get('duration', 0) for seg in valid_segments)
        if total_duration < self.min_speech_duration:
            logger.debug(f"Insufficient speech duration ({total_duration:.1f}s)")
            return None
        speaker_audio_segments = []
        cumulative_duration = 0
        for segment in valid_segments:
            if cumulative_duration >= self.max_signature_duration:
                break
            start_sample = int(segment.get('start', 0) * self.sample_rate)
            end_sample = int(segment.get('end', 0) * self.sample_rate)
            if 0 <= start_sample < end_sample <= len(audio_data):
                segment_audio = audio_data[start_sample:end_sample]
                if len(segment_audio) > 0:
                    speaker_audio_segments.append(segment_audio)
                    cumulative_duration += len(segment_audio) / self.sample_rate
        if not speaker_audio_segments:
            logger.debug("No audio segments extracted")
            return None
        speaker_audio = np.concatenate(speaker_audio_segments)
        max_samples = int(self.max_signature_duration * self.sample_rate)
        if len(speaker_audio) > max_samples:
            speaker_audio = speaker_audio[:max_samples]
        features = self._extract_acoustic_features(speaker_audio)
        if not features or len(features) < 10:
            logger.debug("Insufficient acoustic features extracted")
            return None
        signature = np.array(features, dtype=np.float32)
        signature = self._normalize_features(signature)
        logger.debug(f"Created audio-based voice signature with {len(features)} features")
        return signature

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return features
        features = np.clip(features, -10, 10)
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        feat_min, feat_max = np.min(features), np.max(features)
        return (features - feat_min) / (feat_max - feat_min) if feat_max - feat_min > 1e-8 else np.full_like(features, 0.5)

    def _extract_acoustic_features(self, audio: np.ndarray) -> Optional[List[float]]:
        if len(audio) < self.frame_length:
            return None
        features = []
        features += self._extract_pitch_features(audio)
        features += self._extract_mfcc_features(audio)
        features += self._extract_spectral_features(audio)
        features += self._extract_energy_features(audio)
        return features

    def _extract_pitch_features(self, audio: np.ndarray) -> List[float]:
        try:
            f0 = librosa.yin(audio, fmin=80, fmax=400, sr=self.sample_rate, frame_length=self.frame_length, hop_length=self.hop_length)
            f0_clean = f0[f0 > 0]
            return [np.mean(f0_clean), np.std(f0_clean), np.median(f0_clean), np.max(f0_clean) - np.min(f0_clean)] if len(f0_clean) else [0.0] * 4
        except Exception as e:
            logger.warning(f"Error extracting pitch features: {e}")
            return [0.0] * 4

    def _extract_mfcc_features(self, audio: np.ndarray) -> List[float]:
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13, n_fft=2048, hop_length=self.hop_length, n_mels=26)
            return [stat for coeff in mfccs for stat in (np.mean(coeff), np.std(coeff))]
        except Exception as e:
            logger.warning(f"Error extracting MFCC features: {e}")
            return [0.0] * 26

    def _extract_spectral_features(self, audio: np.ndarray) -> List[float]:
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            return [np.mean(spectral_centroids), np.std(spectral_centroids), np.mean(spectral_rolloff), np.std(spectral_rolloff), np.mean(zcr), np.std(zcr)]
        except Exception as e:
            logger.warning(f"Error extracting spectral features: {e}")
            return [0.0] * 6

    def _extract_energy_features(self, audio: np.ndarray) -> List[float]:
        try:
            rms_energy = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            return [np.mean(rms_energy), np.std(rms_energy), np.max(rms_energy)]
        except Exception as e:
            logger.warning(f"Error extracting energy features: {e}")
            return [0.0] * 3

    def calculate_speaker_similarity(self, signature1: np.ndarray, signature2: np.ndarray) -> float:
        if signature1 is None or signature2 is None:
            return 0.0
        min_len = min(len(signature1), len(signature2))
        if min_len == 0:
            return 0.0
        sig1 = np.nan_to_num(signature1[:min_len], nan=0.0, posinf=1.0, neginf=-1.0)
        sig2 = np.nan_to_num(signature2[:min_len], nan=0.0, posinf=1.0, neginf=-1.0)
        norm1, norm2 = np.linalg.norm(sig1), np.linalg.norm(sig2)
        if norm1 > 1e-8 and norm2 > 1e-8:
            cosine_sim = np.dot(sig1, sig2) / (norm1 * norm2)
            cosine_sim = min(1.0, max(0.0, cosine_sim * 1.05 if cosine_sim > 0.85 else cosine_sim))
            logger.debug(f"Speaker similarity: {cosine_sim:.3f}")
            return cosine_sim
        return 0.0

    def identify_speaker(self, original_speaker: str, speaker_segments: List[Dict], audio_data: Optional[np.ndarray] = None) -> str:
        if original_speaker in self.current_session_mapping:
            return self.current_session_mapping[original_speaker]
        voice_signature = self.create_voice_signature(speaker_segments, audio_data)
        identified_id = None
        if voice_signature is not None:
            best_match_id, best_similarity = None, 0.0
            for speaker_id, existing_signature in self.voice_signatures.items():
                similarity = self.calculate_speaker_similarity(voice_signature, existing_signature)
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity, best_match_id = similarity, speaker_id
            if best_match_id:
                identified_id = best_match_id
                logger.info(f"Matched {original_speaker} to existing speaker {identified_id} (similarity: {best_similarity:.3f})")
                self._update_voice_signature(identified_id, voice_signature, best_similarity)
            else:
                identified_id = self._generate_speaker_id()
                self.voice_signatures[identified_id] = voice_signature
                self.speakers_db[identified_id] = SpeakerProfile(
                    speaker_id=identified_id,
                    voice_signature=voice_signature.tolist(),
                    first_seen=datetime.now().isoformat(),
                    session_count=1,
                    confidence_score=1.0
                )
                logger.info(f"Created new speaker profile: {identified_id}")
        else:
            identified_id = self._generate_speaker_id()
            self.speakers_db[identified_id] = SpeakerProfile(
                speaker_id=identified_id,
                voice_signature=None,
                first_seen=datetime.now().isoformat(),
                session_count=1,
                confidence_score=0.5
            )
            logger.debug(f"Created speaker profile without voice signature: {identified_id}")
        self.current_session_mapping[original_speaker] = identified_id
        self._update_speaker_profile(identified_id, speaker_segments)
        return identified_id

    def _generate_speaker_id(self) -> str:
        return f"SPEAKER_{len(self.speakers_db)+1:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _update_speaker_profile(self, speaker_id: str, speaker_segments: List[Dict]):
        if speaker_id not in self.speakers_db:
            return
        profile = self.speakers_db[speaker_id]
        total_duration = sum(seg.get('duration', 0) for seg in speaker_segments)
        profile.total_speaking_time += total_duration
        profile.last_seen = datetime.now().isoformat()
        profile.session_count += 1
        profile.confidence_score = min(1.0, profile.confidence_score + 0.1)

    def assign_speaker_name(self, speaker_id: str, name: str, notes: Optional[str] = None) -> bool:
        if speaker_id not in self.speakers_db:
            logger.error(f"Speaker {speaker_id} not found in database")
            return False
        profile = self.speakers_db[speaker_id]
        profile.name = name
        if notes:
            profile.notes = notes
        self.save_speakers_database()
        logger.info(f"Assigned name '{name}' to speaker {speaker_id}")
        return True

    def get_speaker_display_name(self, speaker_id: str) -> str:
        return self.speakers_db[speaker_id].name if speaker_id in self.speakers_db and self.speakers_db[speaker_id].name else speaker_id

    def process_speaker_turns(self, speaker_turns: List[Dict], audio_data: Optional[np.ndarray] = None) -> List[Dict]:
        if not speaker_turns:
            return speaker_turns
        speaker_segments = defaultdict(list)
        for turn in speaker_turns:
            speaker_segments[turn.get('speaker', 'Unknown')].append(turn)
        speaker_mapping = {orig_speaker: self.identify_speaker(orig_speaker, segments, audio_data) for orig_speaker, segments in speaker_segments.items()}
        speaker_mapping = self._merge_excess_speakers(speaker_mapping, speaker_segments)
        processed_turns = []
        for turn in speaker_turns:
            processed_turn = turn.copy()
            original_speaker = turn.get('speaker', 'Unknown')
            identified_id = speaker_mapping.get(original_speaker, original_speaker)
            processed_turn['speaker'] = self.get_speaker_display_name(identified_id)
            processed_turn['speaker_id'] = identified_id
            processed_turn['original_speaker'] = original_speaker
            processed_turns.append(processed_turn)
        self.save_speakers_database()
        return processed_turns

    def _merge_excess_speakers(self, speaker_mapping: Dict[str, str], speaker_segments: Dict[str, List[Dict]]) -> Dict[str, str]:
        max_speakers = CONFIG.get('max_speakers', 2)
        unique_speakers = list(set(speaker_mapping.values()))
        if len(unique_speakers) <= max_speakers:
            return speaker_mapping
        speaker_durations = {}
        for orig_speaker, identified_id in speaker_mapping.items():
            speaker_durations[identified_id] = speaker_durations.get(identified_id, 0) + sum(seg.get('duration', 0) for seg in speaker_segments.get(orig_speaker, []))
        sorted_speakers = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
        keep_speakers = [speaker_id for speaker_id, _ in sorted_speakers[:max_speakers]]
        merge_speakers = [speaker_id for speaker_id, _ in sorted_speakers[max_speakers:]]
        if not merge_speakers:
            return speaker_mapping
        logger.info(f"Merging {len(merge_speakers)} excess speakers into {len(keep_speakers)} main speakers")
        target_speaker = keep_speakers[0]
        updated_mapping = {orig_speaker: (target_speaker if identified_id in merge_speakers else identified_id) for orig_speaker, identified_id in speaker_mapping.items()}
        for merged_id in merge_speakers:
            logger.debug(f"Merging speaker {merged_id} into {target_speaker}")
        return updated_mapping

    def _update_voice_signature(self, speaker_id: str, new_signature: np.ndarray, confidence: float):
        if speaker_id not in self.voice_signatures:
            self.voice_signatures[speaker_id] = new_signature
            return
        existing_signature = self.voice_signatures[speaker_id]
        new_weight = 0.2 + (confidence * 0.3)
        existing_weight = 1.0 - new_weight
        self.voice_signatures[speaker_id] = existing_weight * existing_signature + new_weight * new_signature
        logger.debug(f"Updated voice signature for {speaker_id} (weight: {new_weight:.2f})")

    def get_session_statistics(self) -> Dict:
        return {
            'speakers_in_session': len(self.current_session_mapping),
            'total_known_speakers': len(self.speakers_db),
            'session_mappings': dict(self.current_session_mapping),
            'speakers_with_signatures': len(self.voice_signatures),
            'database_path': str(self.speakers_db_path)
        }

    def cleanup_old_speakers(self, days_old: int = 30):
        cutoff_date = datetime.now() - timedelta(days=days_old)
        to_remove = [speaker_id for speaker_id, profile in self.speakers_db.items()
                     if profile.last_seen and datetime.fromisoformat(profile.last_seen) < cutoff_date and profile.session_count < 3]
        for speaker_id in to_remove:
            self.speakers_db.pop(speaker_id, None)
            self.voice_signatures.pop(speaker_id, None)
            logger.info(f"Removed old speaker {speaker_id}")
        if to_remove:
            self.save_speakers_database()
            logger.info(f"Cleaned up {len(to_remove)} old speakers")

    def list_speakers(self) -> List[Dict]:
        speakers_info = [{
            'speaker_id': speaker_id,
            'name': profile.name or 'Unnamed',
            'total_speaking_time': profile.total_speaking_time,
            'session_count': profile.session_count,
            'first_seen': profile.first_seen,
            'last_seen': profile.last_seen,
            'confidence_score': profile.confidence_score,
            'notes': profile.notes
        } for speaker_id, profile in self.speakers_db.items()]
        speakers_info.sort(key=lambda x: x['total_speaking_time'], reverse=True)
        return speakers_info

    def merge_speakers(self, speaker_id1: str, speaker_id2: str, keep_name_from: Optional[str] = None) -> bool:
        if speaker_id1 not in self.speakers_db or speaker_id2 not in self.speakers_db:
            logger.error("One or both speakers not found in database")
            return False
        profile1, profile2 = self.speakers_db[speaker_id1], self.speakers_db[speaker_id2]
        if profile1.total_speaking_time >= profile2.total_speaking_time:
            keep_profile, remove_profile = profile1, profile2
            keep_id, remove_id = speaker_id1, speaker_id2
        else:
            keep_profile, remove_profile = profile2, profile1
            keep_id, remove_id = speaker_id2, speaker_id1
        keep_profile.total_speaking_time += remove_profile.total_speaking_time
        keep_profile.session_count += remove_profile.session_count
        if remove_profile.first_seen and (not keep_profile.first_seen or remove_profile.first_seen < keep_profile.first_seen):
            keep_profile.first_seen = remove_profile.first_seen
        if remove_profile.last_seen and (not keep_profile.last_seen or remove_profile.last_seen > keep_profile.last_seen):
            keep_profile.last_seen = remove_profile.last_seen
        if keep_name_from == remove_id and remove_profile.name:
            keep_profile.name = remove_profile.name
        if remove_profile.notes:
            keep_profile.notes = f"{keep_profile.notes}; {remove_profile.notes}" if keep_profile.notes else remove_profile.notes
        self.speakers_db.pop(remove_id, None)
        self.voice_signatures.pop(remove_id, None)
        for orig_speaker, mapped_id in self.current_session_mapping.items():
            if mapped_id == remove_id:
                self.current_session_mapping[orig_speaker] = keep_id
        self.save_speakers_database()
        logger.info(f"Merged speaker {remove_id} into {keep_id}")
        return True

    def export_speakers_report(self, output_path: Optional[Path] = None) -> Path:
        if output_path is None:
            output_path = OUTPUT_DIR / f"speakers_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        speakers_info = self.list_speakers()
        lines = [
            "=" * 80,
            "SPEAKER IDENTIFICATION REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Speakers: {len(speakers_info)}",
            ""
        ]
        for i, speaker in enumerate(speakers_info, 1):
            lines.append(f"{i}. {speaker['name']} ({speaker['speaker_id']})")
            lines.append(f"   Total Speaking Time: {speaker['total_speaking_time']:.1f} seconds")
            lines.append(f"   Sessions: {speaker['session_count']}")
            lines.append(f"   First Seen: {speaker['first_seen'] or 'Unknown'}")
            lines.append(f"   Last Seen: {speaker['last_seen'] or 'Unknown'}")
            lines.append(f"   Confidence: {speaker['confidence_score']:.2f}")
            if speaker['notes']:
                lines.append(f"   Notes: {speaker['notes']}")
            lines.append("")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        logger.info(f"Speakers report exported to {output_path}")
        return output_path

    def reset_session_mapping(self):
        self.current_session_mapping.clear()
        logger.debug("Reset session speaker mapping")

    def update_adaptive_thresholds(self):
        if not self.enable_adaptive_thresholds:
            self.similarity_threshold = self.base_similarity_threshold
            self.clustering_threshold = self.base_clustering_threshold
            return
        from src.core.config import CONFIG
        min_speakers = CONFIG.get('min_speakers', 1)
        max_speakers = CONFIG.get('max_speakers', 2)
        self.similarity_threshold, self.clustering_threshold = self._calculate_adaptive_thresholds(min_speakers, max_speakers)
        logger.info(f"Updated adaptive thresholds - Similarity: {self.similarity_threshold:.3f}, Clustering: {self.clustering_threshold:.3f} (for {min_speakers}-{max_speakers} speakers)")

    def _calculate_adaptive_thresholds(self, min_speakers: int, max_speakers: int) -> Tuple[float, float]:
        min_similarity_threshold = 0.50
        if min_speakers == 1 and max_speakers == 1:
            similarity_threshold = max(min_similarity_threshold, self.base_similarity_threshold - self.single_speaker_similarity_boost)
            clustering_threshold = max(0.45, self.base_clustering_threshold - self.single_speaker_clustering_boost)
            logger.debug("Applied single-speaker optimizations")
        elif min_speakers == max_speakers and max_speakers <= 2:
            adaptation_factor = 0.5 / max_speakers
            similarity_threshold = max(min_similarity_threshold, self.base_similarity_threshold - adaptation_factor)
            clustering_threshold = max(0.55, self.base_clustering_threshold - adaptation_factor * 0.6)
            logger.debug(f"Applied fixed {max_speakers}-speaker optimizations")
        elif max_speakers - min_speakers <= 1:
            avg_speakers = (min_speakers + max_speakers) / 2
            adaptation_factor = max(0.1, 0.4 / avg_speakers)
            similarity_threshold = max(min_similarity_threshold, self.base_similarity_threshold - adaptation_factor * 0.5)
            clustering_threshold = max(0.60, self.base_clustering_threshold - adaptation_factor * 0.3)
            logger.debug(f"Applied narrow-range optimizations for {min_speakers}-{max_speakers} speakers")
        else:
            similarity_threshold = max(min_similarity_threshold, self.base_similarity_threshold)
            clustering_threshold = self.base_clustering_threshold
            logger.debug("Using base thresholds for wide speaker range")
        return similarity_threshold, clustering_threshold

    def set_expected_speakers(self, min_speakers: int, max_speakers: int):
        if min_speakers < 1 or max_speakers < min_speakers:
            raise ValueError("Invalid speaker range")
        self.similarity_threshold, self.clustering_threshold = self._calculate_adaptive_thresholds(min_speakers, max_speakers)
        logger.info(f"Manually set speaker expectations: {min_speakers}-{max_speakers} speakers")
        logger.info(f"Updated thresholds - Similarity: {self.similarity_threshold:.3f}, Clustering: {self.clustering_threshold:.3f}")

    def get_current_clustering_threshold(self) -> float:
        return self.clustering_threshold

    def delete_speaker(self, speaker_id: str) -> bool:
        try:
            if speaker_id not in self.speakers_db:
                logger.warning(f"Speaker {speaker_id} not found in database")
                return False
            self.speakers_db.pop(speaker_id, None)
            self.voice_signatures.pop(speaker_id, None)
            self.save_speakers_database()
            logger.info(f"Successfully deleted speaker {speaker_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting speaker {speaker_id}: {e}")
            return False

    def reset_database(self) -> bool:
        try:
            self.speakers_db.clear()
            self.voice_signatures.clear()
            self.reset_session_mapping()
            self.save_speakers_database()
            logger.info("Speaker database has been reset")
            return True
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            return False 