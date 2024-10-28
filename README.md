# Video Silence Remover

An AI-powered video editor that automatically removes silences, filler words, and false starts from videos while maintaining context and natural flow.

## Features

- Automatic transcription using Deepgram API with filler word detection
- Intelligent content editing using Claude AI
- Removes:
  - Silences
  - Filler words
  - False starts
  - Repetitions
- Maintains natural speech patterns and context
- Supports MP3/MP4 input formats

## Setup

1. Clone the repository:
   git clone https://github.com/zensajnani/video_remove_silences.git
   cd video_remove_silences

2. Create and activate virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Create a .env file with your API keys:
   ANTHROPIC_API_KEY=your_anthropic_key
   DEEPGRAM_API_KEY=your-deepgram-api-key

## Usage

1. Place your video/audio file in the project directory
2. Run: python app/editor.py
3. The edited video will be saved as ai_output.mp4

## Requirements

- Python 3.9+
- FFmpeg installed on your system
- Deepgram API key
- Anthropic API key

## Note

Make sure to keep your API keys confidential and never commit them to the repository.