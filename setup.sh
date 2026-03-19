#!/bin/bash
set -e

echo "Installing yt-transcribe dependencies..."
pip3 install --user youtube-transcript-api yt-dlp openai

echo ""
echo "Done! Usage:"
echo "  python3 yt_transcribe.py 'https://www.youtube.com/watch?v=VIDEO_ID'"
echo ""
echo "For Whisper fallback (videos without captions), set your OpenAI API key:"
echo "  export OPENAI_API_KEY='your-key-here'"
