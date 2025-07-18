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
from src.core.config import get_models_dir

logger = logging.getLogger(__name__)

class SpeakerEmbeddingExtractor:
    """
    Wrapper for SpeechBrain ECAPA-TDNN model to extract speaker embeddings
    """
    
    def __init__(self, model_path=None):
        self.model_path = model_path or Path(get_models_dir()) / "speechbrain_ecapa"
        self.sample_rate = CONFIG.get('sample_rate', 48000)
        self.min_segment_length = int(0.5 * self.sample_rate)
        self.max_segment_length = int(10.0 * self.sample_rate)
        
        # Set device once during initialization
        self.device = CONFIG.get('diarization_device', 'cuda')
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            logger.warning("CUDA not available, using CPU")
        
        self.model = None
        self.is_loaded = False
        logger.info(f"Initialized SpeakerEmbeddingExtractor with device: {self.device}")
    
    def load_model(self):
        """Load the ECAPA-TDNN model"""
        if self.is_loaded:
            return
            
        logger.info(f"Loading ECAPA-TDNN model from {self.model_path}")
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
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

    def _prepare_audio_tensor(self, audio_data):
        """Prepare audio data for model input"""
        audio_data = np.asarray(audio_data, dtype=np.float32)
        
        if len(audio_data) < self.min_segment_length:
            return None
            
        if len(audio_data) > self.max_segment_length:
            audio_data = audio_data[:self.max_segment_length]
            
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        if self.device == 'cuda':
            audio_tensor = audio_tensor.cuda()
        return audio_tensor

    def _extract_single_embedding(self, audio_tensor):
        """Extract embedding from prepared audio tensor"""
        with torch.no_grad():
            embedding = self.model.encode_batch(audio_tensor)
            return embedding.squeeze(0).cpu().numpy()

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
            return np.array([])
            
        embeddings = []
        for segment in audio_segments:
            audio_tensor = self._prepare_audio_tensor(segment)
            if audio_tensor is not None:
                embedding = self._extract_single_embedding(audio_tensor)
                embeddings.append(embedding)
        
        return np.vstack(embeddings) if embeddings else np.array([])

    def extract_embedding_from_audio(self, audio_data):
        """
        Extract single embedding from audio data
        Args:
            audio_data: numpy array of audio samples
        Returns:
            numpy array of embedding or None if invalid
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        audio_tensor = self._prepare_audio_tensor(audio_data)
        if audio_tensor is None:
            return None
            
        return self._extract_single_embedding(audio_tensor)

    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        Returns:
            float: Cosine similarity score
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def cleanup(self):
        """Clean up model resources"""
        self.model = None
        self.is_loaded = False
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache after embedding extractor cleanup") 