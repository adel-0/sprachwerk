"""
Processing modes base classes and common functionality
Extracts shared patterns between batch and real-time processing
"""

import logging
import time
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from colorama import Fore, Style
import numpy as np

from src.core.config import CONFIG

logger = logging.getLogger(__name__)

class ProcessingMode(ABC):
    """Base class for processing modes with common functionality"""
    
    def __init__(self, transcriber, diarizer, aligner, formatter):
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.aligner = aligner
        self.formatter = formatter
        self.is_running = False
        self.stop_requested = False

    def _get_audio_source_mode(self):
        """Get the current audio source mode dynamically"""
        return CONFIG.get('system_audio_recording_mode', 'mic')
    
    def _get_use_system_audio(self):
        """Get whether to use system audio dynamically"""
        audio_source_mode = self._get_audio_source_mode()
        return audio_source_mode in ['system', 'both']

    def _get_audio_capture(self, audio_capture, system_audio_capture):
        use_system_audio = self._get_use_system_audio()
        selected_capture = system_audio_capture if use_system_audio else audio_capture
        logger.info(f"Audio capture selection: use_system_audio={use_system_audio}, selected_type={type(selected_capture).__name__}")
        return selected_capture

    def _get_source_description(self):
        audio_source_mode = self._get_audio_source_mode()
        return {
            'system': 'system audio',
            'both': 'system audio and microphone',
            'mic': 'microphone'
        }.get(audio_source_mode, 'audio')

    def _setup_audio_capture(self, audio_capture, system_audio_capture):
        use_system_audio = self._get_use_system_audio()
        audio_source_mode = self._get_audio_source_mode()
        logger.info(f"Setting up audio capture: use_system_audio={use_system_audio}, audio_source_mode={audio_source_mode}")
        if not use_system_audio:
            logger.info("Using regular AudioCapture (microphone only)")
            if not audio_capture.select_device():
                logger.error("Failed to setup audio device")
                return False
            logger.info("Microphone audio setup completed successfully")
            return True
        try:
            logger.info("Using SystemAudioCapture (system audio or both)")
            system_device_index = CONFIG.get('system_audio_device_index')
            mic_device_index = CONFIG.get('system_audio_mic_device_index')
            if not system_audio_capture.set_recording_mode(
                audio_source_mode, system_device_index, mic_device_index
            ):
                logger.error("Failed to setup system audio recording")
                return False
            system_audio_capture.set_gains(
                CONFIG.get('system_audio_gain', 0.7),
                CONFIG.get('system_audio_mic_gain', 1.0)
            )
            system_audio_capture.set_normalization(
                CONFIG.get('system_audio_auto_normalize', True),
                CONFIG.get('system_audio_target_level', 0.1)
            )
            logger.info(f"System audio setup completed successfully (mode: {audio_source_mode})")
            return True
        except ImportError:
            logger.error("System audio recording requires pyaudiowpatch. Install with: pip install pyaudiowpatch")
            print(f"{Fore.RED}✗ System audio recording requires pyaudiowpatch{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}  Install with: pip install pyaudiowpatch{Style.RESET_ALL}")
            return False
        except Exception as e:
            logger.error(f"Audio setup failed: {e}")
            return False

    def _save_audio_recording(self, audio_data, filename_base, audio_capture):
        if audio_data is None or (hasattr(audio_data, 'size') and audio_data.size == 0):
            print(f"{Fore.YELLOW}⚠ No audio data to save{Style.RESET_ALL}")
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            use_system_audio = self._get_use_system_audio()
            if use_system_audio:
                audio_filename = f"{filename_base}_{timestamp}.wav"
                saved_audio_path = audio_capture.save_mixed_audio(audio_data, audio_filename)
                print(f"{Fore.GREEN}✓ System audio recording saved to: {saved_audio_path}{Style.RESET_ALL}")
                if CONFIG.get('system_audio_save_separate', False):
                    separate_files = audio_capture.save_separate_audio_files(f"{filename_base}_{timestamp}")
                    if separate_files.get('system'):
                        print(f"  📄 System audio: {separate_files['system']}")
                    if separate_files.get('microphone'):
                        print(f"  📄 Microphone: {separate_files['microphone']}")
                return saved_audio_path
            raw_recording = getattr(audio_capture, 'realtime_recording_raw', None)
            if raw_recording:
                processed_path, raw_path = audio_capture.save_dual_audio(
                    audio_data, np.array(raw_recording), filename_base
                )
                print(f"{Fore.GREEN}✓ Audio recordings saved:{Style.RESET_ALL}")
                print(f"  📄 Processed: {processed_path}")
                print(f"  📄 Raw: {raw_path}")
                print(f"{Fore.YELLOW}💡 Compare both files - use raw version if processed sounds distorted{Style.RESET_ALL}")
                return processed_path
            audio_filename = f"{filename_base}_{timestamp}.wav"
            saved_audio_path = audio_capture.save_audio(audio_data, audio_filename)
            print(f"{Fore.GREEN}✓ Audio recording saved to: {saved_audio_path}{Style.RESET_ALL}")
            return saved_audio_path
        except Exception as e:
            logger.error(f"Failed to save audio recording: {e}")
            print(f"{Fore.RED}✗ Failed to save audio recording: {e}{Style.RESET_ALL}")
            return None

    def _process_audio_data(self, audio_data, audio_file_path=None):
        try:
            if audio_file_path:
                print(f"{Fore.YELLOW}Transcribing audio...{Style.RESET_ALL}")
                transcription_result = self.transcriber.transcribe_file(audio_file_path)
            else:
                transcription_result = self.transcriber.transcribe_chunk(audio_data)
            if not transcription_result or not transcription_result.get('words'):
                print(f"{Fore.RED}✗ Transcription failed or produced no results{Style.RESET_ALL}")
                return None
            print(f"{Fore.YELLOW}Performing speaker diarization...{Style.RESET_ALL}")
            diarization_result = self.diarizer.diarize_file(audio_file_path) if audio_file_path else self.diarizer.diarize_chunk(audio_data)
            if not diarization_result:
                print(f"{Fore.RED}✗ Diarization failed{Style.RESET_ALL}")
                return None
            print(f"{Fore.YELLOW}Aligning transcription with speakers...{Style.RESET_ALL}")
            aligned_result = self.aligner.align_transcription_with_speakers(
                transcription_result, diarization_result, audio_data
            )
            return {
                'transcription': transcription_result,
                'diarization': diarization_result,
                'aligned': aligned_result
            }
        except Exception as e:
            print(f"{Fore.RED}✗ Audio processing failed: {e}{Style.RESET_ALL}")
            logger.error(f"Audio processing error: {e}")
            return None

    def _display_results(self, results):
        aligned_result = results['aligned']
        transcription_result = results['transcription']
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}PROCESSING RESULTS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Language:{Style.RESET_ALL} {transcription_result.get('language', 'Unknown')}")
        print(f"{Fore.YELLOW}Language Confidence:{Style.RESET_ALL} {transcription_result.get('language_probability', 0.0):.2%}")
        print(f"{Fore.YELLOW}Total Speakers:{Style.RESET_ALL} {aligned_result.get('total_speakers', 0)}")
        print(f"{Fore.YELLOW}Total Words:{Style.RESET_ALL} {aligned_result.get('total_words', 0)}")
        detected_languages = transcription_result.get('detected_languages', {})
        if detected_languages and len(detected_languages) > 1:
            print(f"{Fore.YELLOW}Multilingual Content:{Style.RESET_ALL}")
            for lang, count in detected_languages.items():
                duration = transcription_result.get('language_durations', {}).get(lang, 0)
                print(f"  - {lang.upper()}: {count} segments, {duration:.1f}s")
        stats = aligned_result.get('alignment_stats', {})
        if stats.get('speaker_assignment_rate'):
            rate = stats['speaker_assignment_rate'] * 100
            print(f"{Fore.YELLOW}Speaker Assignment Rate:{Style.RESET_ALL} {rate:.1f}%")
        print(f"\n{Fore.CYAN}TRANSCRIPT:{Style.RESET_ALL}")
        print("-" * 60)
        for turn in aligned_result.get('speaker_turns', []):
            start_time = self.formatter._format_timestamp(turn['start'])
            end_time = self.formatter._format_timestamp(turn['end'])
            speaker = turn['speaker']
            text = turn['text']
            print(f"{Fore.GREEN}[{start_time} - {end_time}] {speaker}:{Style.RESET_ALL} {text}")
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

    def _save_results(self, results, filename_base):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_filename = f"{filename_base}_{timestamp}"
            self.formatter.export_transcript(results['aligned'], session_filename)
            print(f"\n{Fore.GREEN}Results saved to:{Style.RESET_ALL}")
            print(f"  📄 txt")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            print(f"{Fore.RED}✗ Failed to save results: {e}{Style.RESET_ALL}")

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

class BatchProcessingMode(ProcessingMode):
    def run(self, audio_capture, system_audio_capture, duration=None, input_file=None):
        print(f"{Fore.CYAN}Starting Batch Mode{Style.RESET_ALL}")
        if input_file:
            print(f"{Fore.YELLOW}Processing audio file: {input_file}{Style.RESET_ALL}")
            return self._process_file(input_file)
        duration = duration or 30
        print(f"{Fore.YELLOW}Recording {duration} seconds of audio...{Style.RESET_ALL}")
        return self._record_and_process(audio_capture, system_audio_capture, duration)

    def _process_file(self, audio_file):
        try:
            import librosa
            audio_data, sample_rate = librosa.load(audio_file, sr=48000)
            results = self._process_audio_data(audio_data, audio_file)
            if not results:
                return False
            self._display_results(results)
            audio_filename = Path(audio_file).stem
            self._save_results(results, f"batch_session_{audio_filename}")
            return True
        except Exception as e:
            print(f"{Fore.RED}✗ File processing failed: {e}{Style.RESET_ALL}")
            logger.error(f"File processing error: {e}")
            return False

    def _record_and_process(self, audio_capture, system_audio_capture, duration):
        if not self._setup_audio_capture(audio_capture, system_audio_capture):
            print(f"{Fore.RED}✗ Audio setup failed. Unable to start batch recording.{Style.RESET_ALL}")
            return False
        try:
            source_desc = self._get_source_description()
            print(f"{Fore.GREEN}Recording started! Capturing {source_desc} for {duration} seconds...{Style.RESET_ALL}")
            capture = self._get_audio_capture(audio_capture, system_audio_capture)
            audio_data = capture.record_batch(duration)
            if audio_data is None or (hasattr(audio_data, 'size') and audio_data.size == 0):
                print(f"{Fore.RED}✗ No audio data recorded{Style.RESET_ALL}")
                return False
            temp_audio_file = self._save_audio_recording(audio_data, "batch_recording", capture)
            if not temp_audio_file:
                return False
            print(f"{Fore.YELLOW}Processing recorded audio...{Style.RESET_ALL}")
            return self._process_file(temp_audio_file)
        except Exception as e:
            print(f"{Fore.RED}✗ Batch recording failed: {e}{Style.RESET_ALL}")
            logger.error(f"Batch recording error: {e}")
            return False

class RealTimeProcessingMode(ProcessingMode):
    def __init__(self, transcriber, diarizer, aligner, formatter):
        super().__init__(transcriber, diarizer, aligner, formatter)
        self.session_file = None
        self.input_thread = None
        self.last_chunk_words = []  # Buffer for sliding window correction
        self.overlap_window_sec = 1.0  # Overlap window in seconds

    def run(self, audio_capture, system_audio_capture):
        print(f"{Fore.CYAN}Starting Real-time Mode{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Press Ctrl+C to stop recording and save transcript{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Or type 'q' or 'quit' and press Enter to stop gracefully{Style.RESET_ALL}")
        audio_source_mode = self._get_audio_source_mode()
        use_system_audio = self._get_use_system_audio()
        logger.info(f"Real-time mode: audio_source_mode={audio_source_mode}, use_system_audio={use_system_audio}")
        if not self._setup_audio_capture(audio_capture, system_audio_capture):
            print(f"{Fore.RED}✗ Audio setup failed. Unable to start real-time transcription.{Style.RESET_ALL}")
            return False
        self._initialize_session()
        capture = self._get_audio_capture(audio_capture, system_audio_capture)
        logger.info(f"Selected capture object: {type(capture).__name__}")
        try:
            self._start_processing_components()
            source_desc = self._get_source_description()
            print(f"{Fore.GREEN}Recording started! Capturing {source_desc}...{Style.RESET_ALL}")
            capture.start_real_time_recording()
            self.is_running = True
            self.input_thread = threading.Thread(target=self._monitor_input, daemon=True)
            self.input_thread.start()
            self._real_time_processing_loop(capture)
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Stopping recording (KeyboardInterrupt)...{Style.RESET_ALL}")
            self.stop_requested = True
            self.is_running = False
        except Exception as e:
            print(f"\n{Fore.RED}Error in real-time mode: {e}{Style.RESET_ALL}")
            logger.error(f"Real-time mode error: {e}")
            self.stop_requested = True
            self.is_running = False
        finally:
            # Always stop recording and join input thread
            print(f"{Fore.YELLOW}Finalizing and saving output...{Style.RESET_ALL}")
            complete_recording = capture.stop_real_time_recording()
            self._stop_processing_components()
            if self.input_thread and self.input_thread.is_alive():
                try:
                    self.input_thread.join(timeout=2.0)
                except Exception:
                    pass
            saved_path = self._save_audio_recording(complete_recording, "realtime_recording", capture)
            if saved_path:
                print(f"{Fore.GREEN}✓ Real-time session completed. Check outputs folder for transcript and audio.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}✗ No audio was saved. Please check your input devices and try again.{Style.RESET_ALL}")
            return True

    def _initialize_session(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.formatter.output_dir / f"realtime_session_{timestamp}.txt"
        with open(self.session_file, 'w', encoding='utf-8') as f:
            f.write(f"REAL-TIME TRANSCRIPTION SESSION\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")

    def _start_processing_components(self):
        self.transcriber.start_real_time_processing()
        self.diarizer.start_real_time_processing()
        self.stop_requested = False

    def _stop_processing_components(self):
        self.is_running = False
        if hasattr(self.transcriber, 'stop_real_time_processing'):
            self.transcriber.stop_real_time_processing()
        if hasattr(self.diarizer, 'stop_real_time_processing'):
            self.diarizer.stop_real_time_processing()

    def _monitor_input(self):
        while self.is_running and not self.stop_requested:
            try:
                user_input = input().strip().lower()
                if user_input in ['q', 'quit', 'stop', 'exit']:
                    print(f"\n{Fore.YELLOW}Graceful shutdown requested...{Style.RESET_ALL}")
                    self.stop_requested = True
                    self.is_running = False
                    break
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                logger.debug(f"Input monitoring thread error: {e}")

    def _real_time_processing_loop(self, audio_capture):
        pending_transcriptions = {}
        pending_diarizations = {}
        pending_audio_chunks = {}
        transcription_time_received = {}
        stale_threshold = 5
        while self.is_running and not self.stop_requested:
            try:
                chunk_data = audio_capture.get_audio_chunk(timeout=0.1)
                if chunk_data:
                    audio_chunk, chunk_timestamp = chunk_data
                    pending_audio_chunks[chunk_timestamp] = audio_chunk
                    transcription_time_received[chunk_timestamp] = time.time()
                    self.transcriber.add_audio_chunk(audio_chunk, chunk_timestamp)
                    self.diarizer.add_audio_chunk(audio_chunk, chunk_timestamp)
                transcription_result = self.transcriber.get_transcription_result(timeout=0.1)
                if transcription_result:
                    ts = transcription_result['chunk_timestamp']
                    pending_transcriptions[ts] = transcription_result
                diarization_result = self.diarizer.get_diarization_result(timeout=0.1)
                if diarization_result:
                    ts = diarization_result['chunk_timestamp']
                    pending_diarizations[ts] = diarization_result
                self._process_completed_chunks(pending_transcriptions, pending_diarizations, pending_audio_chunks)
                current_time = time.time()
                stale_timestamps = [ts for ts, recv_time in transcription_time_received.items() 
                                  if current_time - recv_time > stale_threshold and ts in pending_transcriptions and ts not in pending_diarizations]
                for ts in stale_timestamps:
                    self._output_transcription_only(pending_transcriptions.pop(ts))
                    transcription_time_received.pop(ts, None)
                    pending_audio_chunks.pop(ts, None)
                cutoff_time = current_time - 30
                pending_diarizations = {ts: result for ts, result in pending_diarizations.items() if ts > cutoff_time}
                pending_audio_chunks = {ts: chunk for ts, chunk in pending_audio_chunks.items() if ts > cutoff_time}
                transcription_time_received = {ts: t for ts, t in transcription_time_received.items() if ts > cutoff_time}
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)

    def _process_completed_chunks(self, transcriptions, diarizations, audio_chunks):
        completed_timestamps = set(transcriptions.keys()) & set(diarizations.keys())
        for timestamp in sorted(completed_timestamps):
            transcription_result = transcriptions.pop(timestamp)
            diarization_result = diarizations.pop(timestamp)
            audio_chunk = audio_chunks.pop(timestamp, None)

            # Sliding window correction logic
            words = transcription_result.get('words', [])
            if self.last_chunk_words and words:
                # Find overlap region (by time)
                overlap_start = words[0]['start']
                overlap_end = overlap_start + self.overlap_window_sec
                # Get words in overlap from last chunk
                prev_overlap = [w for w in self.last_chunk_words if w['end'] > overlap_start and w['start'] < overlap_end]
                curr_overlap = [w for w in words if w['end'] > overlap_start and w['start'] < overlap_end]
                # If both have overlap, compare and correct
                if prev_overlap and curr_overlap:
                    # Simple correction: if word text differs, prefer the new chunk's word
                    corrected_words = []
                    prev_idx = 0
                    for w in curr_overlap:
                        if prev_idx < len(prev_overlap):
                            prev_word = prev_overlap[prev_idx]
                            if w['word'] != prev_word['word']:
                                # Replace previous word with new one
                                prev_overlap[prev_idx] = w
                        prev_idx += 1
                    # Rebuild last_chunk_words with corrected overlap
                    # Remove old overlap, append corrected overlap, then append new non-overlap words
                    non_overlap = [w for w in self.last_chunk_words if w['end'] <= overlap_start]
                    self.last_chunk_words = non_overlap + prev_overlap
                    # Add new words after overlap
                    post_overlap = [w for w in words if w['start'] >= overlap_end]
                    self.last_chunk_words += post_overlap
                else:
                    # No overlap, just append
                    self.last_chunk_words += words
            else:
                self.last_chunk_words += words

            # Use corrected words for alignment
            corrected_transcription = dict(transcription_result)
            corrected_transcription['words'] = self.last_chunk_words.copy()
            aligned_result = self.aligner.align_transcription_with_speakers(
                corrected_transcription, diarization_result, audio_chunk
            )
            self._output_real_time_result(aligned_result, timestamp)

    def _output_real_time_result(self, aligned_result, timestamp):
        for turn in aligned_result.get('speaker_turns', []):
            start_time = self.formatter._format_timestamp(turn['start'])
            speaker = turn['speaker']
            text = turn['text']
            print(f"{Fore.GREEN}[{start_time}] {speaker}:{Style.RESET_ALL} {text}")
            if self.session_file:
                self.formatter.save_real_time_update(turn, self.session_file)

    def _output_transcription_only(self, transcription_result):
        words = transcription_result.get('words', [])
        if not words:
            return
        start_time = self.formatter._format_timestamp(words[0]['start']) if words else "0:00"
        text = transcription_result.get('text', '').strip()
        if not text:
            return
        print(f"{Fore.GREEN}[{start_time}] Speaker:{Style.RESET_ALL} {text}")
        if self.session_file:
            self.formatter.save_real_time_update({
                'speaker': 'Speaker', 
                'text': text, 
                'start': words[0]['start'], 
                'end': words[-1]['end']
            }, self.session_file) 