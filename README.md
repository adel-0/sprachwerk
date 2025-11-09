# sprachwerk

Offline speech transcription with speaker diarization using Whisper and SpeechBrain ECAPA-TDNN.

## Overview

sprachwerk is a Python application that transcribes audio with speaker identification. It works completely offline using local models - no API keys or internet required after initial setup.

## Features

- **Real-time transcription** with live speaker identification
- **Batch processing** for recorded audio files
- **Speaker diarization** using SpeechBrain ECAPA-TDNN embeddings
- **Multiple audio formats** (WAV, MP3, M4A, FLAC)
- **Multilingual support** with automatic language detection
- **GPU acceleration** for faster processing

## Installation

This project uses [uv](https://github.com/astral-sh/uv), a fast Python package manager.

```bash
# Install uv if you haven't already
# On Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# On Unix/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <repository-url>
cd sprachwerk

# Install dependencies (creates virtual environment automatically)
uv sync
```

## Usage

### Interactive Mode (Default)
```bash
uv run main.py
```
Launches an interactive menu for configuration and mode selection.

### Command Line Examples

```bash
# Record 30 seconds then process
uv run main.py --mode batch --duration 30

# Real-time transcription
uv run main.py --mode realtime

# Process existing audio file
uv run main.py --mode batch --file audio.wav

# Specify language and speaker count
uv run main.py -l "en" -s "2" --mode batch

# Use specific audio device
uv run main.py -i 1 --mode realtime

# Show available audio devices
uv run main.py --help-devices
```

### Command Line Options

- `--mode`: `batch`, `realtime`, or `interactive`
- `--duration`: Recording duration in seconds (batch mode)
- `--file`: Audio file to process (batch mode)
- `--language`: Language code(s) or "auto" for detection
- `--speakers`: Number of speakers or "auto" for detection
- `--device-index`: Audio input device index

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 4GB+ GPU memory for optimal performance
- Microphone access (for real-time mode)

## Architecture

```
src/
├── audio/          # Audio capture and preprocessing
├── processing/     # Transcription, diarization, alignment
├── utils/          # Helper utilities and configuration
└── core/           # Core application logic
```

## Processing Pipeline

1. **Audio Input** - Capture from microphone or load from file
2. **Preprocessing** - Noise reduction and audio enhancement
3. **Transcription** - Whisper-based speech-to-text
4. **Speaker Embedding** - ECAPA-TDNN feature extraction
5. **Clustering** - Speaker identification and grouping
6. **Alignment** - Word-to-speaker assignment
7. **Output** - Speaker-labeled transcript with timestamps

## Output

Results are saved in the `outputs/` directory:
- Speaker-labeled transcripts in TXT format
- Complete audio recordings
- Processing logs and statistics

## Dependencies

### Core
- `faster-whisper` - Whisper transcription
- `speechbrain` - Speaker diarization
- `torch` - PyTorch backend
- `sounddevice` - Audio capture
- `librosa` - Audio processing
- `scikit-learn` - Clustering algorithms

### Optional
- `PyAudioWPatch` - System audio recording (Windows)
- `colorama` - Enhanced terminal output

## Configuration

Speaker diarization settings can be configured in `src/core/config.py`:
- Clustering algorithm (AgglomerativeClustering, DBSCAN)
- Window length and similarity thresholds
- Speaker count detection

## Troubleshooting

### Audio Device Issues
```bash
uv run main.py --help-devices
```
