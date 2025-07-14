"""
Main application class for Offline Whisper + Speaker Diarization
Coordinates all modules and handles mode selection
"""

import logging
import sys
import time
import signal
import threading
from pathlib import Path
from datetime import datetime
from colorama import init, Fore, Style
import numpy as np

# Initialize colorama for colored output
init()

from src.core.config import CONFIG, LOGGING_CONFIG, set_mode
from src.audio.capture import AudioCapture
from src.audio.system_audio_capture import SystemAudioCapture
from src.processing.transcription import WhisperTranscriber
from src.processing.diarization import SpeakerDiarizer
from src.processing.alignment import TranscriptionAligner
from src.utils.output_formatter import OutputFormatter

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(CONFIG['output_directory']) / 'transcription.log')
    ]
)

logger = logging.getLogger(__name__)

class TranscriptionApp:
    def __init__(self, enable_signal_handlers=True):
        self.audio_capture = AudioCapture()
        self.system_audio_capture = SystemAudioCapture()
        self.transcriber = WhisperTranscriber()
        self.diarizer = SpeakerDiarizer()
        self.aligner = TranscriptionAligner()
        self.formatter = OutputFormatter()
        self.session_file = None
        
        # State management
        self.is_running = False
        self.stop_requested = False
        self._cleanup_called = False
        
        # Audio source configuration
        self.audio_source_mode = CONFIG.get('system_audio_recording_mode', 'mic')
        self.use_system_audio = self.audio_source_mode in ['system', 'both']
        
        if enable_signal_handlers:
            self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.debug("Signal handlers set up successfully")
        except (ValueError, Exception) as e:
            logger.debug(f"Could not set up signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_processing()
        sys.exit(0)
    
    def models_loaded(self):
        """Check if models are already loaded"""
        return (hasattr(self.transcriber, 'is_loaded') and self.transcriber.is_loaded and
                hasattr(self.diarizer, 'is_loaded') and self.diarizer.is_loaded)
    
    def initialize_models(self):
        """Load all required models"""
        print(f"{Fore.CYAN}Initializing models...{Style.RESET_ALL}")
        
        try:
            print(f"{Fore.YELLOW}Loading Whisper model ({CONFIG['whisper_model']})...{Style.RESET_ALL}")
            self.transcriber.load_model()
            
            print(f"{Fore.YELLOW}Loading speaker diarization model...{Style.RESET_ALL}")
            self.diarizer.load_model()
            
            print(f"{Fore.GREEN}✓ All models loaded successfully!{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}✗ Failed to load models: {e}{Style.RESET_ALL}")
            logger.error(f"Model loading failed: {e}")
            return False
    
    def setup_audio(self):
        """Setup audio capture with device selection"""
        try:
            if not self.use_system_audio:
                if not self.audio_capture.select_device():
                    logger.error("Failed to setup audio device")
                    return False
                logger.info("Microphone audio setup completed successfully")
                return True
            
            # System audio setup
            try:
                system_device_index = CONFIG.get('system_audio_device_index')
                mic_device_index = CONFIG.get('system_audio_mic_device_index')
                
                if not self.system_audio_capture.set_recording_mode(
                    self.audio_source_mode, system_device_index, mic_device_index
                ):
                    logger.error("Failed to setup system audio recording")
                    return False
                
                self.system_audio_capture.set_gains(
                    CONFIG.get('system_audio_gain', 0.7),
                    CONFIG.get('system_audio_mic_gain', 1.0)
                )
                
                self.system_audio_capture.set_normalization(
                    CONFIG.get('system_audio_auto_normalize', True),
                    CONFIG.get('system_audio_target_level', 0.1)
                )
                
                logger.info(f"System audio setup completed successfully (mode: {self.audio_source_mode})")
                return True
                
            except ImportError:
                logger.error("System audio recording requires pyaudiowpatch. Install with: pip install pyaudiowpatch")
                print(f"{Fore.RED}✗ System audio recording requires pyaudiowpatch{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}  Install with: pip install pyaudiowpatch{Style.RESET_ALL}")
                return False
                
        except Exception as e:
            logger.error(f"Audio setup failed: {e}")
            return False
    
    def _get_audio_capture(self):
        """Get the appropriate audio capture instance"""
        return self.system_audio_capture if self.use_system_audio else self.audio_capture
    
    def _get_source_description(self):
        """Get description of audio source"""
        descriptions = {
            'system': 'system audio',
            'both': 'system audio and microphone',
            'mic': 'microphone'
        }
        return descriptions.get(self.audio_source_mode, 'audio')
    
    def run_real_time_mode(self):
        """Run real-time transcription mode"""
        print(f"{Fore.CYAN}Starting Real-time Mode{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Press Ctrl+C to stop recording and save transcript{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Or type 'q' or 'quit' and press Enter to stop gracefully{Style.RESET_ALL}")
        
        if not self.setup_audio():
            print(f"{Fore.RED}✗ Audio setup failed. Unable to start real-time transcription.{Style.RESET_ALL}")
            logger.error("Audio setup failed – aborting real-time mode start")
            return

        # Initialize session file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.formatter.output_dir / f"realtime_session_{timestamp}.txt"
        
        with open(self.session_file, 'w', encoding='utf-8') as f:
            f.write(f"REAL-TIME TRANSCRIPTION SESSION\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
        
        self.stop_requested = False
        
        # Start input monitoring thread
        input_thread = threading.Thread(target=self._monitor_input, daemon=True)
        input_thread.start()
        
        try:
            self.transcriber.start_real_time_processing()
            self.diarizer.start_real_time_processing()
            
            source_desc = self._get_source_description()
            print(f"{Fore.GREEN}Recording started! Capturing {source_desc}...{Style.RESET_ALL}")
            
            audio_capture = self._get_audio_capture()
            audio_capture.start_real_time_recording()
            self.is_running = True
            
            self._real_time_processing_loop(audio_capture)
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Stopping recording...{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{Fore.RED}Error in real-time mode: {e}{Style.RESET_ALL}")
            logger.error(f"Real-time mode error: {e}")
        finally:
            audio_capture = self._get_audio_capture()
            complete_recording = audio_capture.stop_real_time_recording()
            self.stop_processing()
            
            self._save_real_time_recording(complete_recording)
            print(f"{Fore.GREEN}✓ Real-time session completed. Check outputs folder for transcript.{Style.RESET_ALL}")
    
    def _monitor_input(self):
        """Monitor for user input to gracefully stop real-time mode"""
        try:
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
    
    def _output_transcription_only(self, transcription_result):
        """Fallback output when diarization is not available"""
        words = transcription_result.get('words', [])
        if not words:
            return
        start_time = self.formatter._format_timestamp(words[0]['start']) if words else "0:00"
        text = transcription_result.get('text', '').strip()
        if not text:
            return
        print(f"{Fore.GREEN}[{start_time}] Speaker:{Style.RESET_ALL} {text}")
        if self.session_file:
            self.formatter.save_real_time_update({'speaker': 'Speaker', 'text': text, 'start': words[0]['start'], 'end': words[-1]['end']}, self.session_file)

    def _real_time_processing_loop(self, audio_capture):
        """Main processing loop for real-time mode"""
        pending_transcriptions = {}
        pending_diarizations = {}
        pending_audio_chunks = {}
        transcription_time_received = {}
        stale_threshold = 5

        while self.is_running and not self.stop_requested:
            try:
                # Get audio chunks and send to processors
                chunk_data = audio_capture.get_audio_chunk(timeout=0.1)
                if chunk_data:
                    audio_chunk, chunk_timestamp = chunk_data
                    pending_audio_chunks[chunk_timestamp] = audio_chunk
                    transcription_time_received[chunk_timestamp] = time.time()
                    self.transcriber.add_audio_chunk(audio_chunk, chunk_timestamp)
                    self.diarizer.add_audio_chunk(audio_chunk, chunk_timestamp)

                # Process results
                transcription_result = self.transcriber.get_transcription_result(timeout=0.1)
                if transcription_result:
                    ts = transcription_result['chunk_timestamp']
                    pending_transcriptions[ts] = transcription_result

                diarization_result = self.diarizer.get_diarization_result(timeout=0.1)
                if diarization_result:
                    ts = diarization_result['chunk_timestamp']
                    pending_diarizations[ts] = diarization_result

                # Align completed chunks
                self._process_completed_chunks(pending_transcriptions, pending_diarizations, pending_audio_chunks)

                # Fallback: output stale transcriptions
                current_time = time.time()
                stale_timestamps = [ts for ts, recv_time in transcription_time_received.items() 
                                  if current_time - recv_time > stale_threshold and ts in pending_transcriptions and ts not in pending_diarizations]
                for ts in stale_timestamps:
                    self._output_transcription_only(pending_transcriptions.pop(ts))
                    transcription_time_received.pop(ts, None)
                    pending_audio_chunks.pop(ts, None)

                # Cleanup old data
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
        """Process chunks that have both transcription and diarization results"""
        completed_timestamps = set(transcriptions.keys()) & set(diarizations.keys())
        
        for timestamp in sorted(completed_timestamps):
            transcription_result = transcriptions.pop(timestamp)
            diarization_result = diarizations.pop(timestamp)
            audio_chunk = audio_chunks.pop(timestamp, None)
            
            aligned_result = self.aligner.align_transcription_with_speakers(
                transcription_result, diarization_result, audio_chunk
            )
            
            self._output_real_time_result(aligned_result, timestamp)
    
    def _output_real_time_result(self, aligned_result, timestamp):
        """Output real-time results to console and file"""
        speaker_turns = aligned_result.get('speaker_turns', [])
        
        for turn in speaker_turns:
            start_time = self.formatter._format_timestamp(turn['start'])
            speaker = turn['speaker']
            text = turn['text']
            
            print(f"{Fore.GREEN}[{start_time}] {speaker}:{Style.RESET_ALL} {text}")
            
            if self.session_file:
                self.formatter.save_real_time_update(turn, self.session_file)
    
    def _save_real_time_recording(self, complete_recording):
        """Save the complete real-time recording"""
        if not complete_recording or len(complete_recording) == 0:
            print(f"{Fore.YELLOW}⚠ No audio data to save{Style.RESET_ALL}")
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_capture = self._get_audio_capture()
            
            if self.use_system_audio:
                audio_filename = f"realtime_recording_{timestamp}.wav"
                saved_audio_path = audio_capture.save_mixed_audio(complete_recording, audio_filename)
                print(f"{Fore.GREEN}✓ System audio recording saved to: {saved_audio_path}{Style.RESET_ALL}")
                
                if CONFIG.get('system_audio_save_separate', False):
                    separate_files = audio_capture.save_separate_audio_files(f"realtime_recording_{timestamp}")
                    if separate_files.get('system'):
                        print(f"  📄 System audio: {separate_files['system']}")
                    if separate_files.get('microphone'):
                        print(f"  📄 Microphone: {separate_files['microphone']}")
            else:
                # Save microphone recording with dual audio support
                raw_recording = getattr(audio_capture, 'realtime_recording_raw', None)
                if raw_recording and len(raw_recording) > 0:
                    processed_path, raw_path = audio_capture.save_dual_audio(
                        complete_recording, np.array(raw_recording), "realtime_recording"
                    )
                    print(f"{Fore.GREEN}✓ Audio recordings saved:{Style.RESET_ALL}")
                    print(f"  📄 Processed: {processed_path}")
                    print(f"  📄 Raw: {raw_path}")
                    print(f"{Fore.YELLOW}💡 Compare both files - use raw version if processed sounds distorted{Style.RESET_ALL}")
                else:
                    audio_filename = f"realtime_recording_{timestamp}.wav"
                    saved_audio_path = audio_capture.save_audio(complete_recording, audio_filename)
                    print(f"{Fore.GREEN}✓ Audio recording saved to: {saved_audio_path}{Style.RESET_ALL}")
                    
        except Exception as e:
            logger.error(f"Failed to save real-time recording: {e}")
            print(f"{Fore.RED}✗ Failed to save audio recording: {e}{Style.RESET_ALL}")
    
    def run_batch_mode(self, duration=None, input_file=None):
        """Run batch processing mode"""
        print(f"{Fore.CYAN}Starting Batch Mode{Style.RESET_ALL}")
        
        if input_file:
            print(f"{Fore.YELLOW}Processing audio file: {input_file}{Style.RESET_ALL}")
            self._process_audio_file(input_file)
        else:
            duration = duration or 30
            print(f"{Fore.YELLOW}Recording {duration} seconds of audio...{Style.RESET_ALL}")
            self._record_and_process_batch(duration)
    
    def _record_and_process_batch(self, duration):
        """Record audio and process it in batch mode"""
        if not self.setup_audio():
            print(f"{Fore.RED}✗ Audio setup failed. Unable to start batch recording.{Style.RESET_ALL}")
            logger.error("Audio setup failed – aborting batch mode start")
            return
            
        try:
            source_desc = self._get_source_description()
            print(f"{Fore.GREEN}Recording started! Capturing {source_desc} for {duration} seconds...{Style.RESET_ALL}")
            
            audio_capture = self._get_audio_capture()
            audio_data = audio_capture.record_batch(duration)
            
            if not audio_data or len(audio_data) == 0:
                print(f"{Fore.RED}✗ No audio data recorded{Style.RESET_ALL}")
                return
            
            # Save recorded audio to temp file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_audio_filename = f"batch_recording_{timestamp}.wav"
            
            if self.use_system_audio:
                temp_audio_file = audio_capture.save_mixed_audio(audio_data, temp_audio_filename)
                if CONFIG.get('system_audio_save_separate', False):
                    audio_capture.save_separate_audio_files(f"batch_recording_{timestamp}")
            else:
                temp_audio_file = audio_capture.save_audio(audio_data, temp_audio_filename)
            
            print(f"{Fore.YELLOW}Processing recorded audio...{Style.RESET_ALL}")
            self._process_audio_file(temp_audio_file)
            
        except Exception as e:
            print(f"{Fore.RED}✗ Batch recording failed: {e}{Style.RESET_ALL}")
            logger.error(f"Batch recording error: {e}")
    
    def _process_audio_file(self, audio_file):
        """Process a complete audio file"""
        try:
            import librosa
            audio_data, sample_rate = librosa.load(audio_file, sr=48000)
            
            print(f"{Fore.YELLOW}Transcribing audio...{Style.RESET_ALL}")
            transcription_result = self.transcriber.transcribe_file(audio_file)
            
            if not transcription_result or not transcription_result.get('words'):
                print(f"{Fore.RED}✗ Transcription failed or produced no results{Style.RESET_ALL}")
                return
            
            print(f"{Fore.YELLOW}Performing speaker diarization...{Style.RESET_ALL}")
            diarization_result = self.diarizer.diarize_file(audio_file)
            
            if not diarization_result:
                print(f"{Fore.RED}✗ Diarization failed{Style.RESET_ALL}")
                return
            
            print(f"{Fore.YELLOW}Aligning transcription with speakers...{Style.RESET_ALL}")
            aligned_result = self.aligner.align_transcription_with_speakers(
                transcription_result, diarization_result, audio_data
            )
            
            self._display_batch_results(aligned_result, transcription_result, diarization_result)
            
            audio_filename = Path(audio_file).stem
            self._save_batch_results(aligned_result, audio_filename)
            
        except Exception as e:
            print(f"{Fore.RED}✗ Audio processing failed: {e}{Style.RESET_ALL}")
            logger.error(f"Audio processing error: {e}")
    
    def _display_batch_results(self, aligned_result, transcription_result, diarization_result):
        """Display batch processing results"""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}BATCH PROCESSING RESULTS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        # Show metadata
        print(f"{Fore.YELLOW}Language:{Style.RESET_ALL} {transcription_result.get('language', 'Unknown')}")
        print(f"{Fore.YELLOW}Language Confidence:{Style.RESET_ALL} {transcription_result.get('language_probability', 0.0):.2%}")
        print(f"{Fore.YELLOW}Total Speakers:{Style.RESET_ALL} {aligned_result.get('total_speakers', 0)}")
        print(f"{Fore.YELLOW}Total Words:{Style.RESET_ALL} {aligned_result.get('total_words', 0)}")
        
        # Show multilingual info if available
        detected_languages = transcription_result.get('detected_languages', {})
        if detected_languages and len(detected_languages) > 1:
            print(f"{Fore.YELLOW}Multilingual Content:{Style.RESET_ALL}")
            for lang, count in detected_languages.items():
                duration = transcription_result.get('language_durations', {}).get(lang, 0)
                print(f"  - {lang.upper()}: {count} segments, {duration:.1f}s")
        
        # Show alignment stats
        stats = aligned_result.get('alignment_stats', {})
        if stats.get('speaker_assignment_rate'):
            rate = stats['speaker_assignment_rate'] * 100
            print(f"{Fore.YELLOW}Speaker Assignment Rate:{Style.RESET_ALL} {rate:.1f}%")
        
        print(f"\n{Fore.CYAN}TRANSCRIPT:{Style.RESET_ALL}")
        print("-" * 60)
        
        # Display speaker turns
        speaker_turns = aligned_result.get('speaker_turns', [])
        for turn in speaker_turns:
            start_time = self.formatter._format_timestamp(turn['start'])
            end_time = self.formatter._format_timestamp(turn['end'])
            speaker = turn['speaker']
            text = turn['text']
            
            print(f"{Fore.GREEN}[{start_time} - {end_time}] {speaker}:{Style.RESET_ALL} {text}")
        
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    def _save_batch_results(self, aligned_result, filename_base):
        """Save batch processing results to files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_filename = f"batch_session_{filename_base}_{timestamp}"
            
            self.formatter.export_transcript(aligned_result, session_filename)
            
            print(f"\n{Fore.GREEN}Results saved to:{Style.RESET_ALL}")
            print(f"  📄 txt")
            
        except Exception as e:
            logger.error(f"Failed to save batch results: {e}")
            print(f"{Fore.RED}✗ Failed to save results: {e}{Style.RESET_ALL}")
    
    def stop_processing(self):
        """Stop all processing and cleanup"""
        if self._cleanup_called:
            return
        
        self._cleanup_called = True
        logger.info("Stopping all processing...")
        self.is_running = False
        
        # Stop processing threads
        if hasattr(self.transcriber, 'stop_real_time_processing'):
            self.transcriber.stop_real_time_processing()
        
        if hasattr(self.diarizer, 'stop_real_time_processing'):
            self.diarizer.stop_real_time_processing()
        
        # Cleanup resources
        for component in [self.transcriber, self.diarizer, self.audio_capture, self.system_audio_capture]:
            if component:
                component.cleanup()
        
        print(f"{Fore.GREEN}✓ Processing stopped and resources cleaned up{Style.RESET_ALL}") 