#!/usr/bin/env python3
"""Transcribe YouTube videos using captions or OpenAI Whisper API."""

import argparse
import json
import os
import re
import signal
import sys
import tempfile


CAPTION_TIMEOUT = 30  # seconds


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Request timed out")

# Gated imports with helpful error messages
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import (
        JSONFormatter,
        SRTFormatter,
        TextFormatter,
        WebVTTFormatter,
    )
except ImportError:
    print(
        "Missing: youtube-transcript-api\n"
        "Install: pip3 install youtube-transcript-api",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import yt_dlp
except ImportError:
    yt_dlp = None  # Only needed for Whisper fallback

try:
    import openai
except ImportError:
    openai = None  # Only needed for Whisper fallback


YOUTUBE_PATTERN = re.compile(
    r"(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/embed/"
    r"|youtube\.com/v/|youtube\.com/shorts/|youtube\.com/live/)"
    r"([a-zA-Z0-9_-]{11})"
)
BARE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{11}$")

FORMATTERS = {
    "text": TextFormatter(),
    "srt": SRTFormatter(),
    "vtt": WebVTTFormatter(),
    "json": JSONFormatter(),
}

WHISPER_FORMAT_MAP = {
    "text": "text",
    "srt": "srt",
    "vtt": "vtt",
    "json": "verbose_json",
}


def log(msg, verbose_only=False, verbose=False):
    """Print a status message to stderr."""
    if verbose_only and not verbose:
        return
    print(msg, file=sys.stderr)


def extract_video_id(url):
    """Extract the 11-character video ID from a YouTube URL or bare ID."""
    match = YOUTUBE_PATTERN.search(url)
    if match:
        return match.group(1)
    if BARE_ID_PATTERN.match(url.strip()):
        return url.strip()
    print(f"Error: Could not extract video ID from: {url}", file=sys.stderr)
    sys.exit(1)


def fetch_captions(video_id, language, fmt, verbose=False):
    """Try to fetch existing YouTube captions. Returns formatted string or None."""
    ytt = YouTubeTranscriptApi()

    # Try fetching with preferred language, falling back to English
    languages = [language] if language == "en" else [language, "en"]

    # Set a timeout to prevent hanging on unresponsive requests.
    # signal.alarm only works in the main thread (not inside web servers),
    # so we gracefully skip it when running in a background thread.
    use_alarm = True
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(CAPTION_TIMEOUT)
    except (ValueError, OSError):
        use_alarm = False

    try:
        transcript = ytt.fetch(video_id, languages=languages)
    except TimeoutError:
        log("Caption fetch timed out.", verbose_only=False, verbose=verbose)
        return None
    except Exception as e:
        error_name = type(e).__name__

        if error_name in ("VideoUnavailable", "InvalidVideoId", "VideoUnplayable"):
            return None

        if error_name in ("TranscriptsDisabled", "NoTranscriptFound",
                          "NoTranscriptAvailable"):
            log(f"No captions available ({error_name}).", verbose_only=False, verbose=verbose)
            # Try to find any auto-generated transcript
            try:
                transcript_list = ytt.list(video_id)
                log("Available transcripts: " + str(transcript_list),
                    verbose_only=True, verbose=verbose)
                transcript = transcript_list.find_generated_transcript(languages).fetch()
            except (Exception, TimeoutError):
                return None
        elif error_name in ("RequestBlocked", "IpBlocked"):
            log("Warning: Request was blocked by YouTube. Try again later.",
                verbose_only=False, verbose=verbose)
            return None
        elif error_name == "AgeRestricted":
            log("Video is age-restricted. Falling back to audio download.",
                verbose_only=False, verbose=verbose)
            return None
        else:
            log(f"Caption fetch failed ({error_name}): {e}",
                verbose_only=False, verbose=verbose)
            return None
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    formatter = FORMATTERS.get(fmt, FORMATTERS["text"])
    return formatter.format_transcript(transcript)


def check_fallback_deps():
    """Check that yt-dlp and openai are installed for the Whisper fallback."""
    missing = []
    if yt_dlp is None:
        missing.append("yt-dlp")
    if openai is None:
        missing.append("openai")
    if missing:
        print(
            f"Missing packages for Whisper fallback: {', '.join(missing)}\n"
            f"Install: pip3 install {' '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is required for Whisper transcription.\n"
            "Set it with: export OPENAI_API_KEY='your-key-here'\n"
            "Get a key at: https://platform.openai.com/api-keys",
            file=sys.stderr,
        )
        sys.exit(1)


def download_audio(video_id, output_dir, verbose=False):
    """Download audio using yt-dlp. Returns path to the downloaded file."""
    # Try audio-only first, fall back to smallest combined format.
    # YouTube's SABR streaming may block audio-only formats, so the
    # fallback to 'worst[ext=mp4]' ensures we always get something
    # the Whisper API can handle.
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/worst[ext=mp4]/worst",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": not verbose,
        "no_warnings": not verbose,
        # No postprocessors — avoids ffmpeg requirement
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except yt_dlp.utils.DownloadError as e:
        print(f"Error downloading audio: {e}", file=sys.stderr)
        sys.exit(1)


def transcribe_with_whisper(audio_path, fmt, verbose=False):
    """Transcribe audio file using OpenAI Whisper API."""
    client = openai.OpenAI()
    response_format = WHISPER_FORMAT_MAP.get(fmt, "text")

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    log(f"Audio file: {file_size_mb:.1f} MB", verbose_only=True, verbose=verbose)

    if file_size_mb > 25:
        print(
            f"Error: Audio file is {file_size_mb:.1f} MB (max 25 MB for Whisper API).\n"
            "For long videos, install ffmpeg to enable automatic splitting:\n"
            "  brew install ffmpeg",
            file=sys.stderr,
        )
        sys.exit(1)

    log("Transcribing with Whisper API...", verbose_only=False, verbose=verbose)

    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format=response_format,
        )

    if response_format == "verbose_json":
        if hasattr(result, "model_dump"):
            return json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
        return json.dumps(result, indent=2, ensure_ascii=False)

    return result if isinstance(result, str) else result.text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe a YouTube video.",
        epilog="Examples:\n"
               "  %(prog)s 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'\n"
               "  %(prog)s -o transcript.srt -f srt 'https://youtu.be/VIDEO_ID'\n"
               "  %(prog)s --no-fallback -l es 'URL'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url", metavar="URL", help="YouTube video URL or video ID")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="Save transcript to file (default: stdout only)")
    parser.add_argument("-f", "--format", choices=["text", "srt", "vtt", "json"],
                        default="text", help="Output format (default: text)")
    parser.add_argument("-l", "--language", default="en",
                        help="Preferred caption language code (default: en)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Only use YouTube captions, skip Whisper fallback")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed progress")
    return parser.parse_args()


def main():
    args = parse_args()

    video_id = extract_video_id(args.url)
    log(f"Video ID: {video_id}", verbose_only=True, verbose=args.verbose)

    # Try YouTube captions first
    log("Fetching YouTube captions...", verbose_only=False, verbose=args.verbose)
    transcript = fetch_captions(video_id, args.language, args.format, verbose=args.verbose)

    if transcript:
        log("Captions found.", verbose_only=False, verbose=args.verbose)
    elif args.no_fallback:
        print("No captions available and --no-fallback was specified.", file=sys.stderr)
        sys.exit(1)
    else:
        # Fall back to Whisper
        log("Downloading audio for Whisper transcription...",
            verbose_only=False, verbose=args.verbose)
        check_fallback_deps()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_audio(video_id, tmpdir, verbose=args.verbose)
            transcript = transcribe_with_whisper(audio_path, args.format, verbose=args.verbose)

    # Output
    print(transcript)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(transcript)
            log(f"Saved to {args.output}", verbose_only=False, verbose=args.verbose)
        except (OSError, IOError) as e:
            print(f"Error writing to {args.output}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
