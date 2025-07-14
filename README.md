# sprachwerk

A real-time and batch voice transcription engine with speaker diarization, built using OpenAI's Whisper and SpeechBrain ECAPA-TDNN. Perfect for integrating speech-to-text capabilities with speaker identification into applications, services, and workflows.

## ✨ Core Features

- **Multiple Processing Modes:**
  - Real-time transcription with live speaker identification
  - Batch processing for recorded audio files
  - File processing for existing audio files (WAV, MP3, M4A, FLAC)

- **Advanced Audio Processing:**
  - Enhanced audio preprocessing for distant microphones
  - Noise reduction and audio enhancement
  - Automatic gain control and voice activity detection
  - Support for various audio formats and devices

- **Speaker Diarization:**
  - Modern SpeechBrain ECAPA-TDNN embeddings for speaker identification
  - Automatic speaker detection and identification
  - Consistent speaker naming across sessions
  - Support for 1-10 speakers
  - Speaker profile management
  - Configurable clustering algorithms (AgglomerativeClustering, DBSCAN)

- **Smart Language Detection:**
  - Automatic language detection
  - Multilingual content support
  - Language-specific optimization
  - Support for 100+ languages

- **Engine Architecture:**
  - Modular design for easy integration
  - Processing pipeline with clear interfaces
  - Real-time and batch processing modes
  - Comprehensive error handling and diagnostics

- **Output Formats:**
  - Real-time transcription results with timestamps
  - Speaker-labeled transcripts in TXT format
  - Complete audio recordings with metadata
  - Detailed processing logs and statistics

## 🔧 Diarization Technology

sprachwerk uses modern SpeechBrain ECAPA-TDNN embeddings for speaker diarization:

### SpeechBrain ECAPA-TDNN (Primary)
- **Modern embedding-based approach** for superior accuracy
- **Fast inference** with GPU acceleration
- **Works completely offline** - no external API calls required
- **No authentication required** - no HuggingFace token needed
- **Configurable clustering algorithms** (AgglomerativeClustering, DBSCAN)
- **Apache 2.0 license** for commercial use
- **Real-time processing** optimized for live transcription

### Configuration Options
- Clustering algorithms: AgglomerativeClustering (default), DBSCAN
- Adjustable window length and clustering thresholds
- Adaptive speaker count detection
- Custom similarity thresholds

### Model Management
- Models are automatically downloaded and cached locally
- Run `python tools/download_speechbrain_models.py` to pre-download models
- Models stored in `models/speechbrain_ecapa/` directory
- No external dependencies for core functionality

## 🚀 Quick Start

### As an Engine

1. **Installation:**
   ```bash
   git clone <repository-url>
   cd sprachwerk
   pip install -r requirements.txt
   ```

2. **Command Line Usage:**
   ```bash
   # Interactive mode (default)
   python main.py
   
   # Batch mode - record 30 seconds then process
   python main.py --mode batch --duration 30
   
   # Real-time mode
   python main.py --mode realtime
   
   # Process existing audio file
   python main.py --mode batch --file audio.wav
   
   # With language and speaker constraints
   python main.py -l "en de" -s "2-3" --mode batch
   
   # Use specific audio device
   python main.py -i 1 --mode realtime
   
   # Show available audio devices
   python main.py --help-devices
   ```

### Interactive Application

The engine includes an interactive application for direct use:

```bash
python main.py
```

The interactive interface provides:
- Configuration menu for audio devices and processing settings
- Real-time visual feedback during processing
- Session management and output organization

## 🏗️ Engine Architecture

The sprachwerk engine is organized into modular components:

- **`src/audio/`** - Audio capture and preprocessing pipeline
- **`src/processing/`** - Transcription, diarization, and alignment engines
- **`src/utils/`** - Helper utilities and configuration management
- **`src/core/`** - Core engine coordination and configuration

## 📋 System Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- 4GB+ GPU memory for optimal performance
- Microphone access for real-time transcription (optional)

## 🎯 Integration Use Cases

- **Business Applications:** Meeting transcription services
- **Media Processing:** Podcast and interview analysis
- **Research Platforms:** Audio data processing pipelines
- **Accessibility Tools:** Real-time captioning systems
- **Content Management:** Audio indexing and search
- **Call Centers:** Conversation analysis and quality assurance

## 🔍 Processing Pipeline

The engine implements a sophisticated processing pipeline:

1. **Audio Capture/Loading** - Multi-format audio input handling
2. **Preprocessing** - Noise reduction and audio enhancement
3. **Transcription** - Whisper-based speech-to-text with word timestamps
4. **Speaker Embedding** - SpeechBrain ECAPA-TDNN feature extraction
5. **Clustering** - Advanced speaker identification and grouping
6. **Alignment** - Precise word-to-speaker assignment
7. **Output Generation** - Formatted results with speaker labels

## 📁 Output Management

All transcription results are organized in the `outputs/` directory:
- Session-based transcript organization
- Speaker-labeled text files with timestamps
- Complete audio recordings with processing metadata
- Detailed logs and performance statistics
- JSON format results for programmatic access

## 🔧 Optional Dependencies

Enhanced features available with additional packages:

- **System Audio Recording** (Windows): `PyAudioWPatch>=0.2.12`
- **Enhanced Terminal Output**: `colorama>=0.4.0`
- **Environment Configuration**: `python-dotenv>=1.0.0`

---

*sprachwerk - Powering the next generation of voice-enabled applications*
