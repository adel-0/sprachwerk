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
        self._cleanup_called = False  # Prevent multiple cleanup calls
        
        # Setup signal handlers for graceful shutdown
        if enable_signal_handlers:
            self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.debug("Signal handlers set up successfully")
        except ValueError as e:
            # This happens when not in the main thread
            logger.debug(f"Could not set up signal handlers (not in main thread): {e}")
        except Exception as e:
            logger.warning(f"Failed to set up signal handlers: {e}")
    
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
            # Load Whisper model
            print(f"{Fore.YELLOW}Loading Whisper model ({CONFIG['whisper_model']})...{Style.RESET_ALL}")
            self.transcriber.load_model()
            
            # Load Diarization model
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
            # Get audio source mode from CONFIG
            audio_source_mode = CONFIG.get('system_audio_recording_mode', 'mic')
            
            if audio_source_mode == 'mic':
                # Traditional microphone recording
                if hasattr(self.audio_capture, 'sync_with_config'):
                    self.audio_capture.sync_with_config()

                if not self.audio_capture.select_device():
                    logger.error("Failed to setup audio device")
                    return False

                logger.info("Microphone audio setup completed successfully")

            elif audio_source_mode in ['system', 'both']:
                # System audio recording or both
                try:
                    # Configure system audio capture
                    system_device_index = CONFIG.get('system_audio_device_index')
                    mic_device_index = CONFIG.get('system_audio_mic_device_index')
                    
                    # Set up system audio capture
                    if not self.system_audio_capture.set_recording_mode(
                        audio_source_mode, system_device_index, mic_device_index
                    ):
                        logger.error("Failed to setup system audio recording")
                        return False
                    
                    # Configure gains and normalization
                    self.system_audio_capture.set_gains(
                        CONFIG.get('system_audio_gain', 0.7),
                        CONFIG.get('system_audio_mic_gain', 1.0)
                    )
                    
                    self.system_audio_capture.set_normalization(
                        CONFIG.get('system_audio_auto_normalize', True),
                        CONFIG.get('system_audio_target_level', 0.1)
                    )
                    
                    logger.info(f"System audio setup completed successfully (mode: {audio_source_mode})")
                    
                except ImportError:
                    logger.error("System audio recording requires pyaudiowpatch. Install with: pip install pyaudiowpatch")
                    print(f"{Fore.RED}✗ System audio recording requires pyaudiowpatch{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}  Install with: pip install pyaudiowpatch{Style.RESET_ALL}")
                    return False
                except Exception as e:
                    logger.error(f"System audio setup failed: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Audio setup failed: {e}")
            return False
    
    def run_real_time_mode(self):
        """Run real-time transcription mode"""
        print(f"{Fore.CYAN}Starting Real-time Mode{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Press Ctrl+C to stop recording and save transcript{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Or type 'q' or 'quit' and press Enter to stop gracefully{Style.RESET_ALL}")
        
        # Ensure audio devices are configured before we start anything
        if not self.setup_audio():
            print(f"{Fore.RED}✗ Audio setup failed. Unable to start real-time transcription.{Style.RESET_ALL}")
            logger.error("Audio setup failed – aborting real-time mode start")
            return

        # Initialize session file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.formatter.output_dir / f"realtime_session_{timestamp}.txt"
        
        # Create session header
        with open(self.session_file, 'w', encoding='utf-8') as f:
            f.write(f"REAL-TIME TRANSCRIPTION SESSION\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
        
        # Flag for graceful shutdown
        self.stop_requested = False
        
        # Start input monitoring thread for graceful shutdown
        input_thread = threading.Thread(target=self._monitor_input, daemon=True)
        input_thread.start()
        
        try:
            # Start processing threads
            self.transcriber.start_real_time_processing()
            self.diarizer.start_real_time_processing()
            
            # Check audio source mode and start appropriate recording
            audio_source_mode = CONFIG.get('system_audio_recording_mode', 'mic')
            
            if audio_source_mode in ['system', 'both']:
                # Use system audio capture for real-time recording
                source_desc = {
                    'system': 'system audio',
                    'both': 'system audio and microphone'
                }.get(audio_source_mode, 'audio')
                
                print(f"{Fore.GREEN}Recording started! Capturing {source_desc}...{Style.RESET_ALL}")
                
                # Start system audio recording
                self.system_audio_capture.start_real_time_recording()
                self.is_running = True
                
                # Use system audio capture for processing loop
                self._real_time_processing_loop_system_audio()
                
            else:
                # Use traditional microphone recording
                print(f"{Fore.GREEN}Recording started! Speak into your microphone...{Style.RESET_ALL}")
                
                # Start microphone recording
                self.audio_capture.start_real_time_recording()
                self.is_running = True
                
                # Use traditional processing loop
                self._real_time_processing_loop()
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Stopping recording...{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{Fore.RED}Error in real-time mode: {e}{Style.RESET_ALL}")
            logger.error(f"Real-time mode error: {e}")
        finally:
            # Stop recording and save the complete audio based on audio source mode
            audio_source_mode = CONFIG.get('system_audio_recording_mode', 'mic')
            
            if audio_source_mode in ['system', 'both']:
                # Stop system audio recording
                complete_recording = self.system_audio_capture.stop_real_time_recording()
            else:
                # Stop microphone recording
                complete_recording = self.audio_capture.stop_real_time_recording()
            
            self.stop_processing()
            
            # Save the complete real-time recording
            if complete_recording is not None and len(complete_recording) > 0:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    if audio_source_mode in ['system', 'both']:
                        # Save system audio recording
                        audio_filename = f"realtime_recording_{timestamp}.wav"
                        saved_audio_path = self.system_audio_capture.save_mixed_audio(complete_recording, audio_filename)
                        print(f"{Fore.GREEN}✓ System audio recording saved to: {saved_audio_path}{Style.RESET_ALL}")
                        
                        # Save separate files if configured
                        if CONFIG.get('system_audio_save_separate', False):
                            separate_files = self.system_audio_capture.save_separate_audio_files(f"realtime_recording_{timestamp}")
                            if separate_files.get('system'):
                                print(f"  📄 System audio: {separate_files['system']}")
                            if separate_files.get('microphone'):
                                print(f"  📄 Microphone: {separate_files['microphone']}")
                    else:
                        # Save microphone recording with dual audio support
                        raw_recording = getattr(self.audio_capture, 'realtime_recording_raw', None)
                        if raw_recording and len(raw_recording) > 0:
                            processed_path, raw_path = self.audio_capture.save_dual_audio(
                                complete_recording, 
                                np.array(raw_recording), 
                                "realtime_recording"
                            )
                            print(f"{Fore.GREEN}✓ Audio recordings saved:{Style.RESET_ALL}")
                            print(f"  📄 Processed: {processed_path}")
                            print(f"  📄 Raw: {raw_path}")
                            print(f"{Fore.YELLOW}💡 Compare both files - use raw version if processed sounds distorted{Style.RESET_ALL}")
                        else:
                            # Fallback to single file
                            audio_filename = f"realtime_recording_{timestamp}.wav"
                            saved_audio_path = self.audio_capture.save_audio(complete_recording, audio_filename)
                            print(f"{Fore.GREEN}✓ Audio recording saved to: {saved_audio_path}{Style.RESET_ALL}")
                            
                except Exception as e:
                    logger.error(f"Failed to save real-time recording: {e}")
                    print(f"{Fore.RED}✗ Failed to save audio recording: {e}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}⚠ No audio data to save{Style.RESET_ALL}")
            
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
                    # Handle cases where input is interrupted
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
        # Save to file if session file exists
        if self.session_file:
            self.formatter.save_real_time_update({'speaker': 'Speaker', 'text': text, 'start': words[0]['start'], 'end': words[-1]['end']}, self.session_file)

    def _real_time_processing_loop(self):
        """Main processing loop for real-time mode with fallback transcription output"""
        pending_transcriptions = {}
        pending_diarizations = {}
        pending_audio_chunks = {}
        transcription_time_received = {}
        stale_threshold = 5  # seconds to wait before falling back without diarization

        while self.is_running and not self.stop_requested:
            try:
                # Get audio chunks and send to processors
                chunk_data = self.audio_capture.get_audio_chunk(timeout=0.1)
                if chunk_data:
                    audio_chunk, chunk_timestamp = chunk_data
                    pending_audio_chunks[chunk_timestamp] = audio_chunk
                    transcription_time_received[chunk_timestamp] = time.time()
                    self.transcriber.add_audio_chunk(audio_chunk, chunk_timestamp)
                    self.diarizer.add_audio_chunk(audio_chunk, chunk_timestamp)

                # Process transcription results
                transcription_result = self.transcriber.get_transcription_result(timeout=0.1)
                if transcription_result:
                    ts = transcription_result['chunk_timestamp']
                    pending_transcriptions[ts] = transcription_result

                # Process diarization results
                diarization_result = self.diarizer.get_diarization_result(timeout=0.1)
                if diarization_result:
                    ts = diarization_result['chunk_timestamp']
                    pending_diarizations[ts] = diarization_result

                # Align completed chunks
                self._process_completed_chunks(pending_transcriptions, pending_diarizations, pending_audio_chunks)

                # Fallback: output any transcription older than stale_threshold without diarization
                current_time = time.time()
                stale_timestamps = [ts for ts, recv_time in transcription_time_received.items() if current_time - recv_time > stale_threshold and ts in pending_transcriptions and ts not in pending_diarizations]
                for ts in stale_timestamps:
                    self._output_transcription_only(pending_transcriptions.pop(ts))
                    transcription_time_received.pop(ts, None)
                    pending_audio_chunks.pop(ts, None)

                # Cleanup old pending diarizations as well
                cutoff_time = current_time - 30
                pending_diarizations = {ts: result for ts, result in pending_diarizations.items() if ts > cutoff_time}
                pending_audio_chunks = {ts: chunk for ts, chunk in pending_audio_chunks.items() if ts > cutoff_time}
                transcription_time_received = {ts: t for ts, t in transcription_time_received.items() if ts > cutoff_time}

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def _real_time_processing_loop_system_audio(self):
        """Main processing loop for real-time mode with system audio capture"""
        pending_transcriptions = {}
        pending_diarizations = {}
        pending_audio_chunks = {}
        transcription_time_received = {}
        stale_threshold = 5  # seconds to wait before falling back without diarization

        while self.is_running and not self.stop_requested:
            try:
                # Get audio chunks from system audio capture and send to processors
                chunk_data = self.system_audio_capture.get_audio_chunk(timeout=0.1)
                if chunk_data:
                    audio_chunk, chunk_timestamp = chunk_data
                    pending_audio_chunks[chunk_timestamp] = audio_chunk
                    transcription_time_received[chunk_timestamp] = time.time()
                    self.transcriber.add_audio_chunk(audio_chunk, chunk_timestamp)
                    self.diarizer.add_audio_chunk(audio_chunk, chunk_timestamp)

                # Process transcription results
                transcription_result = self.transcriber.get_transcription_result(timeout=0.1)
                if transcription_result:
                    ts = transcription_result['chunk_timestamp']
                    pending_transcriptions[ts] = transcription_result

                # Process diarization results
                diarization_result = self.diarizer.get_diarization_result(timeout=0.1)
                if diarization_result:
                    ts = diarization_result['chunk_timestamp']
                    pending_diarizations[ts] = diarization_result

                # Align completed chunks
                self._process_completed_chunks(pending_transcriptions, pending_diarizations, pending_audio_chunks)

                # Fallback: output any transcription older than stale_threshold without diarization
                current_time = time.time()
                stale_timestamps = [ts for ts, recv_time in transcription_time_received.items() if current_time - recv_time > stale_threshold and ts in pending_transcriptions and ts not in pending_diarizations]
                for ts in stale_timestamps:
                    self._output_transcription_only(pending_transcriptions.pop(ts))
                    transcription_time_received.pop(ts, None)
                    pending_audio_chunks.pop(ts, None)

                # Cleanup old pending diarizations as well
                cutoff_time = current_time - 30
                pending_diarizations = {ts: result for ts, result in pending_diarizations.items() if ts > cutoff_time}
                pending_audio_chunks = {ts: chunk for ts, chunk in pending_audio_chunks.items() if ts > cutoff_time}
                transcription_time_received = {ts: t for ts, t in transcription_time_received.items() if ts > cutoff_time}

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in system audio processing loop: {e}")
                time.sleep(0.1)
    
    def _process_completed_chunks(self, transcriptions, diarizations, audio_chunks):
        """Process chunks that have both transcription and diarization results"""
        # Find chunks that have both results
        completed_timestamps = set(transcriptions.keys()) & set(diarizations.keys())
        
        for timestamp in sorted(completed_timestamps):
            transcription_result = transcriptions.pop(timestamp)
            diarization_result = diarizations.pop(timestamp)
            audio_chunk = audio_chunks.pop(timestamp, None)  # Get audio chunk if available
            
            # Align transcription with speakers using audio data for voice signatures
            aligned_result = self.aligner.align_transcription_with_speakers(
                transcription_result, diarization_result, audio_chunk
            )
            
            # Output the results
            self._output_real_time_result(aligned_result, timestamp)
    
    def _output_real_time_result(self, aligned_result, timestamp):
        """Output real-time results to console and file"""
        speaker_turns = aligned_result.get('speaker_turns', [])
        
        for turn in speaker_turns:
            # Console output
            start_time = self.formatter._format_timestamp(turn['start'])
            speaker = turn['speaker']
            text = turn['text']
            
            print(f"{Fore.GREEN}[{start_time}] {speaker}:{Style.RESET_ALL} {text}")
            
            # File output
            if self.session_file:
                self.formatter.save_real_time_update(turn, self.session_file)
    
    def run_batch_mode(self, duration=None, input_file=None):
        """Run batch processing mode"""
        print(f"{Fore.CYAN}Starting Batch Mode{Style.RESET_ALL}")
        
        if input_file:
            # Process existing audio file
            print(f"{Fore.YELLOW}Processing audio file: {input_file}{Style.RESET_ALL}")
            self._process_audio_file(input_file)
        else:
            # Record and process
            duration = duration or 30  # Default 30 seconds
            print(f"{Fore.YELLOW}Recording {duration} seconds of audio...{Style.RESET_ALL}")
            self._record_and_process_batch(duration)
    
    def _record_and_process_batch(self, duration):
        """Record audio and process it in batch mode"""
        # Make sure audio devices are ready before we start recording
        if not self.setup_audio():
            print(f"{Fore.RED}✗ Audio setup failed. Unable to start batch recording.{Style.RESET_ALL}")
            logger.error("Audio setup failed – aborting batch mode start")
            return
        try:
            # Get audio source mode from CONFIG
            audio_source_mode = CONFIG.get('system_audio_recording_mode', 'mic')
            
            if audio_source_mode == 'mic':
                # Traditional microphone recording
                print(f"{Fore.GREEN}Recording started! Speak for {duration} seconds...{Style.RESET_ALL}")
                audio_data = self.audio_capture.record_batch(duration)
                
                if audio_data is None or len(audio_data) == 0:
                    print(f"{Fore.RED}✗ No audio data recorded{Style.RESET_ALL}")
                    return
                
                # Save recorded audio to temp file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_audio_filename = f"batch_recording_{timestamp}.wav"
                temp_audio_file = self.audio_capture.save_audio(audio_data, temp_audio_filename)
                
            elif audio_source_mode in ['system', 'both']:
                # System audio recording or both
                source_desc = {
                    'system': 'system audio',
                    'both': 'system audio and microphone'
                }.get(audio_source_mode, 'audio')
                
                print(f"{Fore.GREEN}Recording started! Capturing {source_desc} for {duration} seconds...{Style.RESET_ALL}")
                audio_data = self.system_audio_capture.record_batch(duration)
                
                if audio_data is None or len(audio_data) == 0:
                    print(f"{Fore.RED}✗ No audio data recorded{Style.RESET_ALL}")
                    return
                
                # Save recorded audio to temp file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_audio_filename = f"batch_recording_{timestamp}.wav"
                temp_audio_file = self.system_audio_capture.save_mixed_audio(audio_data, temp_audio_filename)
                
                # Save separate files if configured
                if CONFIG.get('system_audio_save_separate', False):
                    self.system_audio_capture.save_separate_audio_files(f"batch_recording_{timestamp}")
                
            else:
                print(f"{Fore.RED}✗ Invalid audio source mode: {audio_source_mode}{Style.RESET_ALL}")
                return
            
            print(f"{Fore.YELLOW}Processing recorded audio...{Style.RESET_ALL}")
            
            # Process the recorded audio
            self._process_audio_file(temp_audio_file)
            
        except Exception as e:
            print(f"{Fore.RED}✗ Batch recording failed: {e}{Style.RESET_ALL}")
            logger.error(f"Batch recording error: {e}")
    
    def _process_audio_file(self, audio_file):
        """Process a complete audio file"""
        try:
            # Load audio data for voice signatures
            import librosa
            audio_data, sample_rate = librosa.load(audio_file, sr=16000)
            
            # Transcribe the entire file
            print(f"{Fore.YELLOW}Transcribing audio...{Style.RESET_ALL}")
            transcription_result = self.transcriber.transcribe_file(audio_file)
            
            if not transcription_result or not transcription_result.get('words'):
                print(f"{Fore.RED}✗ Transcription failed or produced no results{Style.RESET_ALL}")
                return
            
            # Perform speaker diarization
            print(f"{Fore.YELLOW}Performing speaker diarization...{Style.RESET_ALL}")
            diarization_result = self.diarizer.diarize_file(audio_file)
            
            if not diarization_result:
                print(f"{Fore.RED}✗ Diarization failed{Style.RESET_ALL}")
                return
            
            # Align transcription with speakers using audio data for voice signatures
            print(f"{Fore.YELLOW}Aligning transcription with speakers...{Style.RESET_ALL}")
            aligned_result = self.aligner.align_transcription_with_speakers(
                transcription_result, diarization_result, audio_data
            )
            
            # Display and save results
            self._display_batch_results(aligned_result, transcription_result, diarization_result)
            
            # Save results
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
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename with session info
            session_filename = f"batch_session_{filename_base}_{timestamp}"
            
            # Save as TXT format with integrated summary
            saved_file = self.formatter.export_transcript(aligned_result, session_filename)
            
            print(f"\n{Fore.GREEN}Results saved to:{Style.RESET_ALL}")
            print(f"  📄 txt")
            
        except Exception as e:
            logger.error(f"Failed to save batch results: {e}")
            print(f"{Fore.RED}✗ Failed to save results: {e}{Style.RESET_ALL}")
    
    def stop_processing(self):
        """Stop all processing and cleanup"""
        if self._cleanup_called:
            return  # Avoid multiple cleanup calls
        
        self._cleanup_called = True
        logger.info("Stopping all processing...")
        self.is_running = False
        
        # Stop processing threads (but not audio recording - that's handled separately in real-time mode)
        if hasattr(self.transcriber, 'stop_real_time_processing'):
            self.transcriber.stop_real_time_processing()
        
        if hasattr(self.diarizer, 'stop_real_time_processing'):
            self.diarizer.stop_real_time_processing()
        
        # Cleanup resources
        if self.transcriber:
            self.transcriber.cleanup()
        if self.diarizer:
            self.diarizer.cleanup()
        if self.audio_capture:
            self.audio_capture.cleanup()
        if self.system_audio_capture:
            self.system_audio_capture.cleanup()
        
        print(f"{Fore.GREEN}✓ Processing stopped and resources cleaned up{Style.RESET_ALL}") 