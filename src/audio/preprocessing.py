"""
Audio preprocessing module for enhancing audio quality in real-world meeting conditions
Specifically optimized for distant microphones, low volume scenarios, and noisy environments
"""

import numpy as np
import scipy.signal
import scipy.ndimage
from scipy.signal import butter, filtfilt
import logging

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """
    Advanced audio preprocessing for real-world meeting conditions
    
    Features:
    - Adaptive gain control for distant microphones
    - Noise reduction and spectral subtraction
    - Audio normalization and dynamic range compression
    - Frequency domain filtering
    - Voice activity detection based enhancement
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate // 2
        
        # Default preprocessing parameters
        self.params = {
            'noise_gate_threshold': 0.01,
            'max_gain_db': 20.0,
            'target_rms': 0.2,
            'compression_ratio': 4.0,
            'compression_threshold': 0.5,
            'noise_reduction_factor': 0.5,
            'spectral_floor': 0.1,
            'energy_threshold': 0.02,
            'highpass_freq': 80,
            'lowpass_freq': 8000,
            'adaptation_window_sec': 2.0,
            'noise_update_threshold': 0.1,
            'gain_smoothing_factor': 0.9,
            'min_adaptation_samples': 1600
        }
        
        # Voice activity detection parameters
        self.vad_frame_length = int(0.025 * sample_rate)
        self.vad_hop_length = int(0.010 * sample_rate)
        
        # Adaptive noise profiling
        self.noise_profile = None
        self.noise_profile_length = int(0.5 * sample_rate)
        self.noise_profiles_history = []
        self.max_noise_profiles = 5
        
        # Adaptive gain control
        self.adaptation_window_samples = int(self.params['adaptation_window_sec'] * sample_rate)
        self.recent_gain_values = []
        self.max_gain_history = 10
        
        # Statistics tracking for adaptation
        self.signal_statistics = {
            'recent_rms_values': [],
            'recent_snr_estimates': [],
            'voice_activity_ratio': 0.0
        }
        
        logger.info(f"AudioPreprocessor initialized for {sample_rate}Hz with adaptive features")
    
    def _validate_audio_input(self, audio: np.ndarray) -> bool:
        """Validate audio input parameters"""
        return (len(audio) > 0 and 
                isinstance(audio, np.ndarray) and 
                audio.dtype in [np.float32, np.float64])
    
    def preprocess_audio(self, audio: np.ndarray, **processing_options) -> np.ndarray:
        """
        Apply comprehensive audio preprocessing pipeline with simplified configuration
        
        Args:
            audio: Input audio signal
            **processing_options: Override default processing options
            
        Returns:
            Preprocessed audio signal
        """
        if not self._validate_audio_input(audio):
            return audio
            
        processed_audio = audio.copy().astype(np.float32)
        
        # Pipeline configuration
        pipeline_config = {
            'enable_filtering': True,
            'enable_gain_boost': True,
            'enable_normalization': True
        }
        pipeline_config.update(processing_options)
        
        logger.debug(f"Starting audio preprocessing: {len(processed_audio)} samples")
        
        # Apply processing pipeline
        if pipeline_config['enable_filtering']:
            processed_audio = self._apply_frequency_filtering(processed_audio)
            logger.debug("Applied filtering")
            
        processed_audio = self._apply_noise_gate(processed_audio)
        logger.debug("Applied noise gate")
        
        if pipeline_config['enable_gain_boost']:
            processed_audio = self._apply_adaptive_gain(processed_audio)
            logger.debug("Applied gain boost")
            
        if pipeline_config['enable_normalization']:
            processed_audio = self._apply_normalization(processed_audio)
            logger.debug("Applied normalization")
        
        # Final clipping protection
        processed_audio = np.clip(processed_audio, -1.0, 1.0)
        
        logger.debug(f"Audio preprocessing completed: {len(processed_audio)} samples")
        return processed_audio
    
    def _apply_frequency_filtering(self, audio: np.ndarray) -> np.ndarray:
        """Apply bandpass filtering to remove unwanted frequencies"""
        try:
            # Design bandpass filter
            low = self.params['highpass_freq'] / self.nyquist
            high = min(self.params['lowpass_freq'], self.nyquist - 100) / self.nyquist
            
            if low >= high:
                logger.warning("Invalid filter frequencies, skipping filtering")
                return audio
            
            # Use butterworth filter for smooth response
            b, a = butter(4, [low, high], btype='band')
            
            # Apply zero-phase filtering to avoid phase distortion
            filtered_audio = filtfilt(b, a, audio)
            
            return filtered_audio.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Frequency filtering failed: {e}")
            return audio
    
    def _apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise gate to remove very quiet background noise"""
        window_size = int(0.01 * self.sample_rate)  # 10ms windows
        
        if len(audio) < window_size:
            return audio
        
        # Calculate windowed RMS
        audio_squared = audio ** 2
        kernel = np.ones(window_size) / window_size
        rms = np.sqrt(np.convolve(audio_squared, kernel, mode='same'))
        
        # Create gate mask and apply smooth transitions
        gate_mask = rms > self.params['noise_gate_threshold']
        gate_mask = scipy.ndimage.uniform_filter1d(gate_mask.astype(float), size=window_size//2)
        
        return audio * gate_mask
    
    def _apply_adaptive_gain(self, audio: np.ndarray) -> np.ndarray:
        """Apply adaptive gain boost for distant microphones with sliding window analysis"""
        if len(audio) < self.adaptation_window_samples:
            return self._apply_simple_gain(audio)
        
        # Adaptive gain using sliding windows
        window_size = self.adaptation_window_samples
        hop_size = window_size // 4  # 75% overlap for smooth transitions
        output_audio = np.zeros_like(audio)
        
        # Process audio in overlapping windows
        for i in range(0, len(audio) - window_size + 1, hop_size):
            window_start = i
            window_end = i + window_size
            window_audio = audio[window_start:window_end]
            
            # Calculate adaptive gain for this window
            window_gain = self._calculate_adaptive_gain(window_audio)
            
            # Apply gain with smooth transitions
            gain_window = np.ones(window_size) * window_gain
            
            # Smooth gain transitions using a window
            transition_samples = hop_size // 2
            if i > 0:  # Fade in
                fade_in = np.linspace(0, 1, transition_samples)
                gain_window[:transition_samples] *= fade_in
            
            if window_end < len(audio):  # Fade out
                fade_out = np.linspace(1, 0, transition_samples)
                gain_window[-transition_samples:] *= fade_out
            
            # Apply gain to window
            processed_window = window_audio * gain_window
            processed_window = self._soft_limit(processed_window)
            
            # Add to output with overlap-add
            output_audio[window_start:window_end] += processed_window
        
        # Handle remaining audio at the end
        if len(audio) % hop_size != 0:
            remaining_start = len(audio) - (len(audio) % hop_size)
            remaining_audio = audio[remaining_start:]
            remaining_gain = self._calculate_adaptive_gain(remaining_audio)
            output_audio[remaining_start:] = remaining_audio * remaining_gain
        
        # Normalize overlap regions
        overlap_factor = window_size / hop_size
        output_audio = output_audio / max(overlap_factor, 1.0)
        
        return self._soft_limit(output_audio)
    
    def _apply_simple_gain(self, audio: np.ndarray) -> np.ndarray:
        """Fallback simple gain for short audio segments"""
        current_rms = np.sqrt(np.mean(audio ** 2))
        
        if current_rms < 1e-6:
            return audio
        
        target_rms = self.params['target_rms'] * 0.5
        gain_linear = target_rms / current_rms
        gain_db = 20 * np.log10(gain_linear)
        gain_db = min(gain_db, self.params['max_gain_db'])
        gain_linear = 10 ** (gain_db / 20)
        
        return self._soft_limit(audio * gain_linear)
    
    def _calculate_adaptive_gain(self, window_audio: np.ndarray) -> float:
        """Calculate adaptive gain for a window of audio"""
        current_rms = np.sqrt(np.mean(window_audio ** 2))
        
        if current_rms < 1e-6:
            return 1.0
        
        # Estimate SNR for this window
        snr_estimate = self._estimate_snr(window_audio)
        
        # Adaptive target based on SNR
        if snr_estimate > 10:  # Good SNR
            target_rms = self.params['target_rms'] * 0.7
        elif snr_estimate > 5:  # Moderate SNR
            target_rms = self.params['target_rms'] * 0.5
        else:  # Poor SNR - be more conservative
            target_rms = self.params['target_rms'] * 0.3
        
        # Calculate required gain
        gain_linear = target_rms / current_rms
        gain_db = 20 * np.log10(gain_linear)
        
        # Limit gain based on SNR
        max_gain = self.params['max_gain_db']
        if snr_estimate < 5:  # Reduce max gain for noisy signals
            max_gain *= 0.7
        
        gain_db = min(gain_db, max_gain)
        gain_linear = 10 ** (gain_db / 20)
        
        # Smooth gain changes
        if self.recent_gain_values:
            smoothing = self.params['gain_smoothing_factor']
            previous_gain = self.recent_gain_values[-1]
            gain_linear = smoothing * previous_gain + (1 - smoothing) * gain_linear
        
        # Update gain history
        self.recent_gain_values.append(gain_linear)
        if len(self.recent_gain_values) > self.max_gain_history:
            self.recent_gain_values.pop(0)
        
        return gain_linear
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio for adaptive processing"""
        if len(audio) < 1024:
            return 5.0  # Default moderate SNR
        
        # Use spectral analysis to estimate SNR
        f, psd = scipy.signal.welch(audio, fs=self.sample_rate, nperseg=min(1024, len(audio)//4))
        
        # Estimate speech band energy (300-3400 Hz typical for speech)
        speech_mask = (f >= 300) & (f <= 3400)
        speech_energy = np.mean(psd[speech_mask]) if np.any(speech_mask) else np.mean(psd)
        
        # Estimate noise energy from low and high frequencies
        noise_mask = (f < 300) | (f > 4000)
        noise_energy = np.mean(psd[noise_mask]) if np.any(noise_mask) else np.mean(psd) * 0.1
        
        # Calculate SNR in dB - with better zero handling
        if noise_energy > 1e-10:  # Prevent divide by zero
            snr_db = 10 * np.log10(speech_energy / noise_energy)
        else:
            snr_db = 20  # High SNR if no detectable noise
        
        return max(snr_db, 0)  # Ensure non-negative SNR
    
    def _spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction using estimated noise profile"""
        if self.noise_profile is None:
            return audio
        
        # Frame the audio
        frame_length = 1024
        hop_length = frame_length // 2
        
        if len(audio) < frame_length:
            return audio
        
        # Process in overlapping frames
        output = np.zeros_like(audio)
        window = np.hanning(frame_length)
        
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length] * window
            
            # FFT
            fft_frame = np.fft.rfft(frame)
            magnitude = np.abs(fft_frame)
            phase = np.angle(fft_frame)
            
            # Spectral subtraction
            power = magnitude ** 2
            
            # Interpolate noise profile to match frame length
            noise_power = np.interp(
                np.linspace(0, len(self.noise_profile)-1, len(magnitude)),
                np.arange(len(self.noise_profile)),
                self.noise_profile
            )
            
            # Subtract noise power
            clean_power = power - self.params['noise_reduction_factor'] * noise_power
            
            # Apply spectral floor
            clean_power = np.maximum(clean_power, self.params['spectral_floor'] * power)
            
            # Reconstruct magnitude
            clean_magnitude = np.sqrt(clean_power)
            
            # Reconstruct complex spectrum
            clean_fft = clean_magnitude * np.exp(1j * phase)
            
            # IFFT and overlap-add
            clean_frame = np.fft.irfft(clean_fft, n=frame_length)
            clean_frame *= window
            
            output[i:i + frame_length] += clean_frame
        
        return output
    
    def _apply_normalization(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio normalization to target RMS level"""
        if len(audio) == 0:
            return audio
        
        # Calculate current RMS
        current_rms = np.sqrt(np.mean(audio ** 2))
        
        if current_rms < 1e-8:
            return audio
        
        # Normalize to target RMS
        normalization_factor = self.params['target_rms'] / current_rms
        
        # Limit normalization to prevent excessive amplification
        max_normalization = 10.0  # 20dB max
        normalization_factor = min(normalization_factor, max_normalization)
        
        normalized_audio = audio * normalization_factor
        
        # Apply soft limiting
        return self._soft_limit(normalized_audio)
    
    def _soft_limit(self, audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """Apply soft limiting to prevent clipping"""
        # Soft limiting using tanh
        limited_audio = np.where(
            np.abs(audio) > threshold,
            np.sign(audio) * threshold * np.tanh(np.abs(audio) / threshold),
            audio
        )
        return limited_audio
    
    def detect_voice_activity(self, audio: np.ndarray) -> np.ndarray:
        """Enhanced voice activity detection for meeting scenarios"""
        if len(audio) < self.vad_frame_length:
            return np.ones(len(audio), dtype=bool)
        
        # Frame the audio
        num_frames = (len(audio) - self.vad_frame_length) // self.vad_hop_length + 1
        vad_result = np.zeros(num_frames, dtype=bool)
        
        # Calculate multiple features for robust VAD
        energy_features = []
        zcr_features = []
        spectral_features = []
        
        for i in range(num_frames):
            start = i * self.vad_hop_length
            end = start + self.vad_frame_length
            frame = audio[start:end]
            
            # Energy-based features
            energy = np.mean(frame ** 2)
            energy_features.append(energy)
            
            # Zero crossing rate
            zcr = np.mean(np.diff(np.sign(frame)) != 0)
            zcr_features.append(zcr)
            
            # Spectral centroid (frequency distribution)
            if len(frame) >= 256:
                fft_frame = np.fft.rfft(frame)
                magnitude = np.abs(fft_frame)
                freqs = np.fft.rfftfreq(len(frame), 1/self.sample_rate)
                
                spectral_centroid = (np.sum(freqs * magnitude) / np.sum(magnitude) 
                                   if np.sum(magnitude) > 0 else 0)
                spectral_features.append(spectral_centroid)
            else:
                spectral_features.append(0)
        
        # Convert to numpy arrays
        energy_features = np.array(energy_features)
        zcr_features = np.array(zcr_features)
        spectral_features = np.array(spectral_features)
        
        # Adaptive thresholding
        energy_threshold = self._calculate_adaptive_threshold(energy_features, base_threshold=self.params['energy_threshold'])
        zcr_threshold = 0.15  # More conservative ZCR threshold for meetings
        spectral_threshold = 800  # Typical speech spectral centroid
        
        # Multi-feature VAD decision
        for i in range(num_frames):
            # Energy condition
            energy_active = energy_features[i] > energy_threshold
            
            # ZCR condition (speech typically has moderate ZCR)
            zcr_active = 0.05 < zcr_features[i] < zcr_threshold
            
            # Spectral condition (speech frequency range)
            spectral_active = 200 < spectral_features[i] < 4000
            
            # Combined decision with relaxed criteria for meeting scenarios
            vad_result[i] = energy_active and (zcr_active or spectral_active)
        
        # Apply temporal smoothing to reduce false negatives
        vad_result = self._apply_vad_smoothing(vad_result)
        
        # Interpolate VAD result to original audio length
        vad_interpolated = np.interp(
            np.arange(len(audio)),
            np.linspace(0, len(audio)-1, len(vad_result)),
            vad_result.astype(float)
        ) > 0.5
        
        # Update statistics
        self.signal_statistics['voice_activity_ratio'] = np.mean(vad_interpolated)
        
        return vad_interpolated
    
    def _calculate_adaptive_threshold(self, values: np.ndarray, base_threshold: float) -> float:
        """Calculate adaptive threshold based on signal statistics"""
        if len(values) == 0:
            return base_threshold
        
        # Use percentile-based adaptive threshold
        median_value = np.median(values)
        std_value = np.std(values)
        
        # Adaptive threshold: base + factor * (median + std)
        adaptive_threshold = max(base_threshold, 0.5 * (median_value + 0.5 * std_value))
        
        return adaptive_threshold
    
    def _apply_vad_smoothing(self, vad_result: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to VAD results using efficient morphological operations"""
        if vad_result.size == 0:
            return vad_result

        # Define structuring element sizes (in frames)
        min_gap_frames = max(1, int(0.1 * self.sample_rate / self.vad_hop_length))   # 100 ms
        min_voice_frames = max(1, int(0.05 * self.sample_rate / self.vad_hop_length))  # 50 ms

        # Fill short gaps (closing) then remove short bursts (opening)
        smoothed = scipy.ndimage.binary_closing(
            vad_result, structure=np.ones(min_gap_frames, dtype=bool)
        )
        smoothed = scipy.ndimage.binary_opening(
            smoothed, structure=np.ones(min_voice_frames, dtype=bool)
        )

        return smoothed
    
    def enhance_for_whisper(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply minimal audio enhancements - only the essentials
        
        This method applies only the most basic preprocessing to avoid
        audio quality degradation while still helping Whisper.
        """
        logger.debug("Applying minimal audio enhancements")
        
        if not self._validate_audio_input(audio):
            return audio
            
        processed_audio = audio.copy().astype(np.float32)
        
        # 1. Remove DC offset (essential)
        processed_audio = processed_audio - np.mean(processed_audio)
        
        # 2. Very basic high-pass filter to remove rumble/AC hum (60Hz and below)
        if len(processed_audio) > 100:  # Only if we have enough samples
            try:
                nyquist = self.sample_rate / 2
                low_freq = 50.0  # Remove everything below 50Hz
                if low_freq < nyquist:
                    b, a = butter(2, low_freq / nyquist, btype='high')
                    processed_audio = filtfilt(b, a, processed_audio)
            except Exception as e:
                logger.debug(f"Minimal filtering failed: {e}")
        
        # 3. Very conservative normalization - only if audio is extremely quiet
        max_amplitude = np.max(np.abs(processed_audio))
        if max_amplitude > 0 and max_amplitude < 0.01:  # Only if very quiet (< 1%)
            # Gentle boost to bring it to at least 5% of full scale
            target_level = 0.05
            gain = target_level / max_amplitude
            gain = min(gain, 3.0)  # Limit to 3x gain maximum
            processed_audio = processed_audio * gain
            logger.debug(f"Applied minimal gain boost: {20*np.log10(gain):.1f}dB")
        
        # 4. Soft clipping protection
        processed_audio = np.clip(processed_audio, -0.98, 0.98)
        
        logger.debug("Minimal audio enhancement completed")
        return processed_audio
    
    def _estimate_noise_profile(self, noise_audio: np.ndarray):
        """Estimate noise profile from noise sample"""
        if len(noise_audio) < 1024:
            return
        
        # Calculate power spectral density of noise
        f, psd = scipy.signal.welch(noise_audio, fs=self.sample_rate, nperseg=1024)
        
        # Store noise profile
        self.noise_profile = psd
        
        # Update noise profiles history
        self.noise_profiles_history.append(psd.copy())
        if len(self.noise_profiles_history) > self.max_noise_profiles:
            self.noise_profiles_history.pop(0)
    
    def update_noise_profile(self, noise_audio: np.ndarray):
        """Update noise profile with new noise sample"""
        if len(noise_audio) > 0:
            self._estimate_noise_profile(noise_audio)
            logger.debug("Updated noise profile")
    
    def reset_noise_profile(self):
        """Reset the noise profile"""
        self.noise_profile = None
        logger.debug("Reset noise profile")
    
    def set_parameters(self, **kwargs):
        """Update preprocessing parameters"""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
                logger.debug(f"Updated parameter {key} = {value}")
            else:
                logger.warning(f"Unknown parameter: {key}")


# Convenience function for quick audio enhancement
def enhance_audio_for_whisper(audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    """
    Quick audio enhancement function optimized for Whisper and distant microphones
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate of the audio
        
    Returns:
        Enhanced audio signal
    """
    preprocessor = AudioPreprocessor(sample_rate)
    return preprocessor.enhance_for_whisper(audio) 