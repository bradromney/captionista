# Captionista

A Python toolkit for audio diarization and subtitle generation from video/audio files.

## Overview

Captionista is a two-stage process for creating subtitles with speaker identification:

1. **Speaker Diarization** (`diarize.py`) - Identifies and separates different speakers in audio
2. **Subtitle Generation** (`merge_rechunk.py`) - Combines transcription with speaker info to create formatted subtitles

## Features

- Speaker diarization using AI models
- Automatic transcription integration
- Multiple output formats (SRT, VTT, etc.)
- Configurable chunking and timing
- Support for various audio/video input formats

## Usage

### Step 1: Audio Extraction and Diarization
```bash
# Extract audio segment for testing (optional)
ffmpeg -y -i input.wav -t 60 test.wav

# Activate virtual environment
source ~/dia/bin/activate

# Run speaker diarization
python diarize.py test.wav
```

### Step 2: Generate Subtitles
```bash
# Merge transcription with speaker data
python merge_rechunk.py original.json test.spk.json output.srt
```

## Requirements

- Python 3.x
- ffmpeg
- Virtual environment with required packages (see `~/dia/`)

## Future Plans

- Combine both scripts into a single end-to-end application
- Direct video file input support
- Improved speaker identification accuracy
- Additional output format options

## File Structure

- `diarize.py` - Speaker diarization functionality
- `merge_rechunk.py` - Subtitle generation and formatting
- `.gitignore` - Excludes audio/video files and generated outputs

## Contributing

This project is in active development. Current focus is on merging the two-step process into a unified captionista application.
