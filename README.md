# sprachwerk

A real-time and batch speech transcription application with speaker diarization, built using OpenAI's Whisper and pyannote.audio. Perfect for meetings, interviews, lectures, and any multi-speaker audio content.

## ✨ Features

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
  - Automatic speaker detection and identification
  - Consistent speaker naming across sessions
  - Support for 1-10 speakers
  - Speaker profile management

- **Smart Language Detection:**
  - Automatic language detection
  - Multilingual content support
  - Language-specific optimization
  - Support for 100+ languages

- **User-Friendly Interface:**
  - Interactive configuration menu
  - Settings confirmation before starting
  - Real-time visual feedback
  - Comprehensive error handling and diagnostics

- **Output Formats:**
  - Real-time console output with timestamps
  - Saved transcripts in TXT format with speaker summary
  - Complete audio recordings
  - Detailed processing logs

## 🚀 Quick Start

1. **Installation:**
   ```bash
   git clone <repository-url>
   cd sprachwerk
   pip install -r requirements.txt
   ```

   For minimal installation (without optional features):
   ```bash
   pip install -r requirements-minimal.txt
   ```

2. **Run the application:**
   ```bash
   python main.py
   ```

3. **Configure your settings:**
   - Select your recording mode (real-time, batch, or file processing)
   - Choose your audio device
   - Set language preferences
   - Configure speaker count
   - Review and confirm settings before starting

## 🏗️ Project Structure

The codebase is organized into three main modules:

- **`src/audio/`** - Audio capture and preprocessing
- **`src/processing/`** - Transcription, diarization, and alignment
- **`src/utils/`** - Helper utilities and configuration management

## 📋 System Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- Microphone access for real-time transcription
- HuggingFace account (for speaker diarization models)

## 🔧 Configuration

The application provides an interactive menu system that guides you through:

- **Recording Mode Selection:** Choose between real-time, batch, or file processing
- **Audio Device Configuration:** Automatic detection or manual selection
- **Language Settings:** Auto-detect or specify target languages
- **Speaker Configuration:** Set expected number of speakers
- **Settings Confirmation:** Review all settings before starting transcription

## 🎯 Use Cases

- **Business Meetings:** Real-time transcription with speaker identification
- **Interviews:** Batch processing with high accuracy
- **Lectures:** Long-form content processing
- **Podcasts:** Multi-speaker content analysis
- **Research:** Audio data processing and analysis

## 🔍 Troubleshooting

The application includes comprehensive error handling and diagnostics:

- **Audio Issues:** Automatic device testing and fallback options
- **Quality Monitoring:** Real-time audio quality assessment
- **Processing Errors:** Detailed error messages and recovery suggestions
- **Performance Tracking:** Real-time factor monitoring for optimization

## 📁 Output

All transcripts and recordings are saved to the `outputs/` directory:
- Real-time session transcripts with speaker identification
- Batch processing results with detailed summaries
- Complete audio recordings (processed and raw versions)
- Processing logs and statistics

## 🔧 Optional Features

Some features require additional dependencies:

- **System Audio Recording** (Windows): Install `PyAudioWPatch>=0.2.12`
- **Colored Terminal Output**: Install `colorama>=0.4.0`
- **Environment File Support**: Install `python-dotenv>=1.0.0`

---
