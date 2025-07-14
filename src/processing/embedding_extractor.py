"""
Speaker embedding extractor using SpeechBrain ECAPA-TDNN model
Provides speaker embeddings for diarization clustering
"""

import logging
import numpy as np
import torch
from pathlib import Path
from speechbrain.inference import EncoderClassifier
from src.core.config import CONFIG

logger = logging.getLogger(__name__)

class SpeakerEmbeddingExtractor:
    """
    Wrapper for SpeechBrain ECAPA-TDNN model to extract speaker embeddings
    """
    
    def __init__(self, model_path=None):
        self.model_path = model_path or Path("models/speechbrain_ecapa")
        self.device = CONFIG.get('diarization_device', 'cuda')
        self.sample_rate = CONFIG.get('sample_rate', 48000)
        self.model = None
        self.is_loaded = False
        self.min_segment_length = int(0.5 * self.sample_rate)  # 0.5 seconds minimum
        self.max_segment_length = int(10.0 * self.sample_rate)  # 10 seconds maximum
        logger.info(f"Initialized SpeakerEmbeddingExtractor with device: {self.device}")
    
    def load_model(self):
        """Load the ECAPA-TDNN model"""
        if self.is_loaded:
            logger.info("ECAPA model already loaded")
            return
        logger.info(f"Loading ECAPA-TDNN model from {self.model_path}")
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available for embedding extraction, falling back to CPU")
            self.device = 'cpu'
        try:
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(self.model_path),
                run_opts={"device": self.device}
            )
            self.is_loaded = True
            logger.info(f"ECAPA model loaded successfully on {self.device}")
            if self.device == 'cuda':
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"GPU Memory after ECAPA load - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        except Exception as e:
            logger.error(f"Failed to load ECAPA model: {e}")
            raise

    def extract_embeddings(self, audio_segments):
        """
        Extract speaker embeddings from audio segments
        Args:
            audio_segments: List of audio segments (numpy arrays)
        Returns:
            numpy array of embeddings (n_segments, embedding_dim)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if not audio_segments:
            logger.warning("No audio segments provided for embedding extraction")
            return np.array([])
        embeddings = []
        for i, segment in enumerate(audio_segments):
            try:
                segment = np.asarray(segment, dtype=np.float32)
                if len(segment) < self.min_segment_length:
                    logger.debug(f"Segment {i} too short ({len(segment)} samples), skipping")
                    continue
                if len(segment) > self.max_segment_length:
                    logger.debug(f"Segment {i} too long ({len(segment)} samples), truncating")
                    segment = segment[:self.max_segment_length]
                audio_tensor = torch.from_numpy(segment).unsqueeze(0)
                if self.device == 'cuda':
                    audio_tensor = audio_tensor.cuda()
                with torch.no_grad():
                    embedding = self.model.encode_batch(audio_tensor)
                    embedding = embedding.squeeze(0).cpu().numpy()
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to extract embedding for segment {i}: {e}")
        if not embeddings:
            logger.warning("No valid embeddings extracted from audio segments")
            return np.array([])
        embeddings_array = np.vstack(embeddings)
        logger.debug(f"Extracted {len(embeddings)} embeddings with shape {embeddings_array.shape}")
        return embeddings_array

    def extract_embedding_from_audio(self, audio_data):
        """
        Extract single embedding from audio data
        Args:
            audio_data: numpy array of audio samples
        Returns:
            numpy array of embedding
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        try:
            audio_data = np.asarray(audio_data, dtype=np.float32)
            if len(audio_data) < self.min_segment_length:
                logger.warning(f"Audio segment too short ({len(audio_data)} samples)")
                return None
            if len(audio_data) > self.max_segment_length:
                audio_data = audio_data[:self.max_segment_length]
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            if self.device == 'cuda':
                audio_tensor = audio_tensor.cuda()
            with torch.no_grad():
                embedding = self.model.encode_batch(audio_tensor)
                embedding = embedding.squeeze(0).cpu().numpy()
            return embedding
        except Exception as e:
            logger.error(f"Failed to extract embedding from audio: {e}")
            return None

    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        Returns:
            float: Cosine similarity score
        """
        try:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if not norm1 or not norm2:
                return 0.0
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0

    def cleanup(self):
        """Clean up model resources"""
        self.model = None
        self.is_loaded = False
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache after embedding extractor cleanup") 