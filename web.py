#!/usr/bin/env python3
"""Web interface for Audio Transcriber."""

import json
import os
import re
import sys
import tempfile
import urllib.request
import warnings
import webbrowser
import threading

# Suppress the urllib3/OpenSSL warning
warnings.filterwarnings("ignore", message=".*OpenSSL.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from flask import Flask, request, jsonify, render_template_string

# Import transcription functions from our CLI tool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yt_transcribe import extract_video_id, fetch_captions
from youtube_transcript_api import YouTubeTranscriptApi

# Optional imports for AI features
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    import openai
except ImportError:
    openai = None


# ── Configuration ────────────────────────────────────────────────────────────

CONFIG_FILE = os.path.expanduser("~/.yt-transcribe.json")


def load_config():
    """Load config from local file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_config_file(config):
    """Save config to local file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_api_key():
    """Get OpenAI API key from config file or environment."""
    config = load_config()
    return config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")


# ── Spotify Support ──────────────────────────────────────────────────────────

SPOTIFY_PATTERN = re.compile(
    r'open\.spotify\.com/episode/([a-zA-Z0-9]+)'
)


def is_spotify_url(url):
    """Check if a URL is a Spotify episode link."""
    return bool(SPOTIFY_PATTERN.search(url))


def extract_spotify_id(url):
    """Extract Spotify episode ID from a URL."""
    m = SPOTIFY_PATTERN.search(url)
    return m.group(1) if m else None


def get_spotify_metadata(episode_id):
    """Fetch episode title and thumbnail from Spotify's free oEmbed API."""
    oembed_url = f"https://open.spotify.com/oembed?url=https://open.spotify.com/episode/{episode_id}"
    try:
        req = urllib.request.Request(oembed_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return {
                "title": data.get("title", ""),
                "thumbnail_url": data.get("thumbnail_url", ""),
            }
    except Exception:
        return {"title": "", "thumbnail_url": ""}


def find_youtube_match(search_query):
    """Search YouTube for a video matching the query. Returns (video_id, title) or (None, None)."""
    if not yt_dlp:
        return None, None
    try:
        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch3:{search_query}", download=False)
            if info and "entries" in info and info["entries"]:
                best = info["entries"][0]
                return best.get("id"), best.get("title", "")
    except Exception:
        pass
    return None, None


# ── Speaker Labels ───────────────────────────────────────────────────────────

def add_speaker_labels(text):
    """Detect speaker changes from >> markers and add speaker labels."""
    if ">>" not in text:
        return text, []

    parts = re.split(r'(?:^|\n)\s*>>\s*', text)

    if len(parts) < 2:
        return text, []

    speaker_map = {}
    current_speaker = 0

    all_text = text[:3000]
    name_hints = re.findall(
        r'(?:welcome back[, ]+|thanks[, ]+|thank you[, ]+|'
        r"(?:I'm |my name is |this is |here with |welcome |joining us[, ]+))"
        r'([A-Z][a-z]+(?: [A-Z][a-z]+)?)',
        all_text
    )
    seen = set()
    unique_names = []
    for name in name_hints:
        if name.lower() not in seen and len(name) > 2:
            seen.add(name.lower())
            unique_names.append(name)

    labeled_parts = []
    if parts[0].strip():
        speaker_label = unique_names[0] if unique_names else "Speaker 1"
        labeled_parts.append(f"**{speaker_label}:** {parts[0].strip()}")
        current_speaker = 0

    for i, part in enumerate(parts[1:], 1):
        part = part.strip()
        if not part:
            continue
        current_speaker = i % 2
        if current_speaker < len(unique_names):
            label = unique_names[current_speaker]
        else:
            label = f"Speaker {current_speaker + 1}"
        labeled_parts.append(f"**{label}:** {part}")

    all_names = unique_names if unique_names else [f"Speaker {i+1}" for i in range(2)]
    return "\n\n".join(labeled_parts), all_names


# ── Flask App ────────────────────────────────────────────────────────────────

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Transcriber</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f;
            color: #e1e1e1;
            min-height: 100vh;
        }

        .container {
            position: relative;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
        }

        header {
            text-align: center;
            margin-bottom: 40px;
        }

        header h1 {
            font-size: 32px;
            font-weight: 700;
            color: #fff;
            margin-bottom: 8px;
        }

        header h1 .play { color: #ff4444; }

        header p {
            color: #888;
            font-size: 16px;
        }

        .settings-btn {
            display: inline-block;
            background: #222;
            border: 1px solid #555;
            color: #ccc;
            font-size: 14px;
            cursor: pointer;
            padding: 8px 20px;
            transition: all 0.2s;
            border-radius: 8px;
            font-weight: normal;
            margin-top: 16px;
        }

        .settings-btn:hover { background: #333; color: #fff; border-color: #888; }
        .settings-btn.has-key { border-color: #3a6a3a; color: #6c6; }

        .input-group {
            display: flex;
            gap: 12px;
            margin-bottom: 12px;
        }

        input[type="text"] {
            flex: 1;
            padding: 14px 18px;
            font-size: 16px;
            background: #1a1a1a;
            border: 2px solid #333;
            border-radius: 12px;
            color: #fff;
            outline: none;
            transition: border-color 0.2s;
        }

        input[type="text"]:focus { border-color: #ff4444; }
        input[type="text"]::placeholder { color: #555; }

        button {
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            border: none;
            border-radius: 12px;
            transition: all 0.2s;
        }

        .btn-primary {
            padding: 14px 28px;
            background: #ff4444;
            color: #fff;
        }

        .btn-primary:hover { background: #e03333; }
        .btn-primary:disabled { background: #555; cursor: not-allowed; }

        .options {
            display: flex;
            gap: 12px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }

        .options label {
            display: flex;
            align-items: center;
            gap: 6px;
            color: #888;
            font-size: 14px;
            cursor: pointer;
        }

        .options select {
            padding: 6px 10px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            color: #ccc;
            font-size: 14px;
            outline: none;
        }

        .status {
            text-align: center;
            padding: 40px;
            display: none;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #333;
            border-top-color: #ff4444;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 16px;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        .status-text { color: #888; font-size: 15px; }

        .result {
            display: none;
            margin-top: 20px;
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            flex-wrap: wrap;
            gap: 8px;
        }

        .result-header h2 {
            font-size: 18px;
            color: #fff;
        }

        .result-actions {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .btn-sm {
            padding: 8px 16px;
            font-size: 13px;
            background: #222;
            color: #ccc;
            border: 1px solid #444;
        }

        .btn-sm:hover { background: #333; color: #fff; }

        .btn-sm.copied {
            background: #1a3a1a;
            border-color: #4a4;
            color: #4a4;
        }

        .btn-cleanup {
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 600;
            background: linear-gradient(135deg, #1a1a3a, #2a1a3a);
            color: #b8a9f0;
            border: 1px solid #3a3a5a;
        }

        .btn-cleanup:hover {
            background: linear-gradient(135deg, #2a2a4a, #3a2a4a);
            color: #d4c8f8;
        }

        .btn-cleanup:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .btn-original {
            padding: 8px 16px;
            font-size: 13px;
            background: #1a2a2a;
            color: #8ed8d8;
            border: 1px solid #2a4a4a;
        }

        .btn-original:hover { background: #2a3a3a; }

        .speaker-rename-bar {
            display: none;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
            padding: 12px 16px;
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 10px;
            margin-bottom: 10px;
        }

        .speaker-rename-bar.visible { display: flex; }

        .speaker-rename-bar .rename-label {
            font-size: 12px;
            color: #888;
            margin-right: 4px;
        }

        .speaker-chip {
            display: flex;
            align-items: center;
            gap: 6px;
            background: #222;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 4px 10px;
        }

        .speaker-chip .color-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            flex-shrink: 0;
        }

        .speaker-chip input {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 4px;
            color: #ddd;
            font-size: 13px;
            font-weight: 600;
            padding: 4px 6px;
            width: 140px;
            font-family: inherit;
        }

        .speaker-chip input:hover {
            border-color: #444;
        }

        .speaker-chip input:focus {
            outline: none;
            border-color: #5a5a8a;
            background: #1a1a2a;
        }

        .download-menu {
            position: relative;
            display: inline-block;
        }

        .download-dropdown {
            display: none;
            position: absolute;
            right: 0;
            top: 100%;
            margin-top: 4px;
            background: #222;
            border: 1px solid #444;
            border-radius: 8px;
            min-width: 200px;
            z-index: 100;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        }

        .download-dropdown.open { display: block; }

        .download-option {
            display: block;
            width: 100%;
            padding: 10px 16px;
            background: none;
            border: none;
            border-bottom: 1px solid #333;
            color: #ccc;
            font-size: 13px;
            text-align: left;
            cursor: pointer;
            font-family: inherit;
        }

        .download-option:last-child { border-bottom: none; }

        .download-option:hover { background: #333; color: #fff; }

        .download-option .dl-desc {
            display: block;
            font-size: 11px;
            color: #777;
            margin-top: 2px;
        }

        .summary-box {
            display: none;
            background: linear-gradient(135deg, #1a1a2a, #1a2a1a);
            border: 1px solid #2a3a3a;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 12px;
        }

        .summary-box.visible { display: block; }

        .summary-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .summary-header h3 {
            font-size: 15px;
            color: #8ed8d8;
            margin: 0;
        }

        .summary-toggle {
            background: none;
            border: none;
            color: #666;
            font-size: 12px;
            cursor: pointer;
            padding: 4px 8px;
        }

        .summary-toggle:hover { color: #999; }

        .summary-content ul {
            margin: 0;
            padding: 0 0 0 20px;
            list-style: none;
        }

        .summary-content li {
            color: #ccc;
            font-size: 14px;
            line-height: 1.7;
            margin-bottom: 8px;
            position: relative;
            padding-left: 4px;
        }

        .summary-content li::before {
            content: "\\2022";
            color: #8ed8d8;
            font-weight: bold;
            position: absolute;
            left: -16px;
        }

        .ai-badge {
            display: inline-block;
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            background: #2a1a3a;
            color: #b8a9f0;
            border-radius: 6px;
            margin-left: 8px;
            vertical-align: middle;
        }

        .transcript-box {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 24px;
            max-height: 60vh;
            overflow-y: auto;
            line-height: 1.8;
            font-size: 15px;
            color: #ddd;
            word-wrap: break-word;
        }

        .transcript-box .speaker {
            font-weight: 700;
            margin-top: 16px;
            display: block;
        }

        .transcript-box .speaker:first-child { margin-top: 0; }

        .speaker-1 { color: #5eb5f7; }
        .speaker-2 { color: #f7a85e; }
        .speaker-3 { color: #8ef79e; }
        .speaker-4 { color: #f78ef0; }

        .toggle-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .toggle {
            position: relative;
            width: 40px;
            height: 22px;
            appearance: none;
            -webkit-appearance: none;
            background: #333;
            border-radius: 11px;
            cursor: pointer;
            transition: background 0.2s;
        }

        .toggle:checked { background: #ff4444; }

        .toggle::before {
            content: '';
            position: absolute;
            top: 3px;
            left: 3px;
            width: 16px;
            height: 16px;
            background: #fff;
            border-radius: 50%;
            transition: transform 0.2s;
        }

        .toggle:checked::before { transform: translateX(18px); }

        .transcript-box::-webkit-scrollbar { width: 8px; }
        .transcript-box::-webkit-scrollbar-track { background: #1a1a1a; }
        .transcript-box::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }

        .error-box {
            background: #2a1515;
            border: 1px solid #4a2020;
            border-radius: 12px;
            padding: 20px;
            color: #ff6b6b;
            text-align: center;
            display: none;
            margin-top: 20px;
        }

        .video-info {
            display: none;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            background: #1a1a1a;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        .video-info img { width: 80px; border-radius: 6px; }
        .video-info .title { font-size: 14px; color: #ccc; }

        .spotify-note {
            font-size: 12px;
            color: #1db954;
            padding: 6px 16px;
            background: #1a2a1a;
            border-radius: 6px;
            margin-top: 6px;
        }

        /* ── Paste Workaround (shown when YouTube blocks) ── */

        .paste-workaround {
            display: none;
            background: #161622;
            border: 1px solid #2a2a4a;
            border-radius: 12px;
            padding: 20px;
            margin-top: 16px;
        }

        .paste-workaround h3 {
            font-size: 15px;
            color: #b8a9f0;
            margin-bottom: 12px;
        }

        .paste-area {
            width: 100%;
            min-height: 180px;
            padding: 16px;
            font-size: 14px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            border: 2px solid #333;
            border-radius: 12px;
            color: #ddd;
            outline: none;
            resize: vertical;
            line-height: 1.6;
            margin-bottom: 12px;
            transition: border-color 0.2s;
        }

        .paste-area:focus { border-color: #7c5cbf; }
        .paste-area::placeholder { color: #444; }

        .paste-steps {
            font-size: 13px;
            color: #999;
            margin-bottom: 14px;
            line-height: 1.6;
        }

        .paste-steps ol {
            margin: 8px 0 0 20px;
            padding: 0;
        }

        .paste-steps li { margin: 4px 0; }
        .paste-steps strong { color: #ccc; }

        footer {
            text-align: center;
            margin-top: 40px;
            color: #444;
            font-size: 13px;
        }

        /* ── Settings Modal ───────────────────── */

        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 100;
            justify-content: center;
            align-items: center;
            backdrop-filter: blur(4px);
        }

        .modal-overlay.active { display: flex; }

        .modal {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 16px;
            padding: 32px;
            max-width: 480px;
            width: 90%;
            position: relative;
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .modal-header h2 {
            font-size: 20px;
            color: #fff;
        }

        .modal-close {
            background: none;
            border: none;
            color: #666;
            font-size: 20px;
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 8px;
            font-weight: normal;
        }

        .modal-close:hover { color: #fff; background: #333; }

        .modal .subtitle {
            color: #888;
            font-size: 14px;
            margin-bottom: 20px;
        }

        .key-input-group {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }

        .key-input-group input {
            flex: 1;
            padding: 12px 16px;
            font-size: 14px;
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
            background: #0f0f0f;
            border: 2px solid #333;
            border-radius: 10px;
            color: #fff;
            outline: none;
        }

        .key-input-group input:focus { border-color: #ff4444; }
        .key-input-group input::placeholder { color: #444; }

        .show-key-btn {
            padding: 12px 14px;
            background: #222;
            border: 1px solid #444;
            color: #888;
            font-size: 13px;
            border-radius: 10px;
            cursor: pointer;
            min-width: 52px;
            font-weight: normal;
        }

        .show-key-btn:hover { background: #333; color: #ccc; }

        .btn-save-key {
            width: 100%;
            padding: 12px;
            background: #ff4444;
            color: #fff;
            font-size: 15px;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            margin-bottom: 12px;
        }

        .btn-save-key:hover { background: #e03333; }

        .key-status {
            font-size: 13px;
            text-align: center;
            margin-bottom: 16px;
            min-height: 18px;
        }

        .key-status.active { color: #5a5; }
        .key-status.inactive { color: #f77; }
        .key-status.neutral { color: #888; }

        .key-link {
            display: block;
            text-align: center;
            color: #5eb5f7;
            font-size: 13px;
            text-decoration: none;
            margin-bottom: 20px;
        }

        .key-link:hover { text-decoration: underline; }

        .features-box {
            background: #0f0f0f;
            border-radius: 10px;
            padding: 16px;
        }

        .features-box h3 {
            font-size: 13px;
            color: #888;
            margin-bottom: 10px;
            font-weight: 600;
        }

        .features-box ul {
            list-style: none;
            padding: 0;
        }

        .features-box li {
            font-size: 13px;
            color: #aaa;
            padding: 4px 0;
        }

        .features-box li strong { color: #ccc; }

        /* ── AI Options ────────────────────────── */

        .toggle-ai:checked { background: #7c5cbf; }

        .ai-options {
            padding: 14px 18px;
            background: #161622;
            border: 1px solid #2a2a4a;
            border-radius: 10px;
            margin-bottom: 30px;
        }

        .ai-options label {
            display: block;
            font-size: 13px;
            color: #b8a9f0;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .ai-options .optional {
            font-weight: normal;
            color: #666;
            font-size: 12px;
        }

        .speaker-input {
            width: 100%;
            padding: 10px 14px;
            font-size: 15px;
            background: #0f0f1a;
            border: 1px solid #3a3a5a;
            border-radius: 8px;
            color: #fff;
            outline: none;
        }

        .speaker-input:focus { border-color: #b8a9f0; }
        .speaker-input::placeholder { color: #444; }

        .btn-sm.claude-copied {
            background: #1a2a1a;
            border-color: #3a5a3a;
            color: #8ef09e;
        }

        /* ── Timestamps ── */
        .timestamp {
            display: inline-block;
            font-size: 11px;
            color: #5eb5f7;
            background: #1a1a2a;
            padding: 2px 6px;
            border-radius: 4px;
            margin-right: 8px;
            cursor: pointer;
            text-decoration: none;
            font-family: monospace;
            vertical-align: middle;
        }

        .timestamp:hover { background: #2a2a3a; color: #8ed4ff; }

        /* ── Search bar ── */
        .search-bar {
            display: none;
            margin-bottom: 10px;
            position: relative;
        }

        .search-bar.visible { display: flex; gap: 8px; align-items: center; }

        .search-bar input {
            flex: 1;
            padding: 8px 12px;
            font-size: 14px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            color: #ddd;
            outline: none;
            font-family: inherit;
        }

        .search-bar input:focus { border-color: #5a5a8a; }

        .search-info {
            font-size: 12px;
            color: #888;
            white-space: nowrap;
        }

        .search-nav {
            background: #222;
            border: 1px solid #444;
            color: #ccc;
            padding: 6px 10px;
            font-size: 13px;
            cursor: pointer;
            border-radius: 6px;
        }

        .search-nav:hover { background: #333; color: #fff; }

        mark.search-highlight {
            background: #5a4a00;
            color: #ffd;
            border-radius: 2px;
            padding: 0 1px;
        }

        mark.search-highlight.current {
            background: #8a7a00;
            color: #fff;
            outline: 2px solid #cc0;
        }

        /* ── Batch mode ── */
        .batch-toggle {
            font-size: 12px;
            color: #888;
            background: none;
            border: none;
            cursor: pointer;
            padding: 4px 8px;
            margin-top: 8px;
        }

        .batch-toggle:hover { color: #ccc; }

        .batch-area {
            display: none;
            margin-top: 12px;
        }

        .batch-area.visible { display: block; }

        .batch-area textarea {
            width: 100%;
            min-height: 80px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 12px;
            color: #ddd;
            font-size: 14px;
            font-family: inherit;
            resize: vertical;
            box-sizing: border-box;
        }

        .batch-area textarea:focus { outline: none; border-color: #5a5a8a; }

        .batch-area .batch-hint {
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }

        .batch-progress {
            display: none;
            margin-top: 12px;
        }

        .batch-progress.visible { display: block; }

        .batch-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            margin-bottom: 6px;
            font-size: 13px;
            color: #999;
        }

        .batch-item.active { border-color: #3a3a5a; color: #ddd; }
        .batch-item.done { border-color: #2a4a2a; color: #8ef09e; }
        .batch-item.failed { border-color: #4a2a2a; color: #f09e8e; }

        .batch-item .batch-status {
            flex-shrink: 0;
            width: 20px;
            text-align: center;
        }

        .batch-item .batch-url {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .batch-results-list { margin-top: 12px; }

        .batch-result-link {
            display: block;
            padding: 10px 14px;
            background: #1a1a2a;
            border: 1px solid #2a3a3a;
            border-radius: 8px;
            margin-bottom: 6px;
            color: #8ed8d8;
            text-decoration: none;
            font-size: 13px;
            cursor: pointer;
        }

        .batch-result-link:hover { background: #2a2a3a; }

        .upload-divider {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 16px 0 12px;
            color: #555;
            font-size: 13px;
        }
        .upload-divider::before, .upload-divider::after {
            content: '';
            flex: 1;
            border-top: 1px solid #333;
        }

        .upload-zone {
            border: 2px dashed #333;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s, background 0.2s;
            color: #777;
            font-size: 14px;
        }
        .upload-zone:hover {
            border-color: #555;
            background: #1a1a1a;
        }
        .upload-zone.dragover {
            border-color: #5eb5f7;
            background: #1a1a2a;
            color: #5eb5f7;
        }
        .upload-zone .upload-icon {
            font-size: 28px;
            display: block;
            margin-bottom: 8px;
        }
        .upload-zone .upload-hint {
            font-size: 12px;
            color: #555;
            margin-top: 6px;
        }
        .upload-file-name {
            display: none;
            margin-top: 8px;
            padding: 8px 14px;
            background: #1a2a1a;
            border: 1px solid #2a4a2a;
            border-radius: 8px;
            color: #8ef09e;
            font-size: 13px;
        }
        .upload-file-name.visible { display: flex; align-items: center; justify-content: space-between; }
        .upload-file-name .remove-file {
            background: none;
            border: none;
            color: #f09e8e;
            cursor: pointer;
            font-size: 16px;
            padding: 0 4px;
        }

        .batch-tabs {
            display: none;
            gap: 0;
            overflow-x: auto;
            margin-bottom: 12px;
            border-bottom: 2px solid #2a2a2a;
        }

        .batch-tabs.visible { display: flex; }

        .batch-tab {
            padding: 10px 18px;
            font-size: 13px;
            color: #888;
            background: none;
            border: none;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            cursor: pointer;
            white-space: nowrap;
            max-width: 220px;
            overflow: hidden;
            text-overflow: ellipsis;
            font-family: inherit;
        }

        .batch-tab:hover { color: #ccc; background: #1a1a1a; }

        .batch-tab.active {
            color: #5eb5f7;
            border-bottom-color: #5eb5f7;
            background: #1a1a2a;
        }

        /* ── History ── */
        .history-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: #222;
            border: 1px solid #444;
            color: #ccc;
            font-size: 20px;
            cursor: pointer;
            z-index: 200;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }

        .history-btn:hover { background: #333; color: #fff; }

        .history-badge {
            position: absolute;
            top: -4px;
            right: -4px;
            background: #5eb5f7;
            color: #000;
            font-size: 10px;
            font-weight: 700;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .history-panel {
            display: none;
            position: fixed;
            bottom: 80px;
            right: 20px;
            width: 340px;
            max-height: 60vh;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 12px;
            z-index: 200;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        }

        .history-panel.open { display: flex; flex-direction: column; }

        .history-panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 16px;
            border-bottom: 1px solid #2a2a2a;
        }

        .history-panel-header h3 { margin: 0; font-size: 15px; color: #fff; }

        .history-clear {
            background: none;
            border: none;
            color: #f06b6b;
            font-size: 12px;
            cursor: pointer;
        }

        .history-clear:hover { color: #ff8888; }

        .history-list {
            overflow-y: auto;
            flex: 1;
            padding: 8px;
        }

        .history-item {
            padding: 10px 12px;
            background: #222;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            margin-bottom: 6px;
            cursor: pointer;
        }

        .history-item:hover { background: #2a2a2a; border-color: #444; }

        .history-item .hist-title {
            font-size: 13px;
            color: #ddd;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .history-item .hist-meta {
            font-size: 11px;
            color: #666;
            margin-top: 4px;
        }

        .history-empty {
            text-align: center;
            color: #555;
            padding: 24px;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><span class="play">&#127911;</span> Audio Transcriber</h1>
            <p>Paste a YouTube or Spotify link, or upload an audio file</p>
            <button class="settings-btn" id="settingsBtn" onclick="openSettings()">&#9881; Settings</button>
        </header>

        <div class="input-group">
            <input type="text" id="url" placeholder="YouTube or Spotify episode link..." autofocus>
            <button class="btn-primary" id="goBtn" onclick="transcribe()">Transcribe</button>
        </div>
        <button class="batch-toggle" id="batchToggle" onclick="toggleBatchMode()">&#43; Multiple URLs</button>
        <div class="batch-area" id="batchArea">
            <textarea id="batchUrls" placeholder="Paste one URL per line (YouTube or Spotify)..."></textarea>
            <div class="batch-hint">One URL per line. They'll be transcribed one at a time.</div>
            <button class="btn-primary" id="batchGoBtn" onclick="startBatch()" style="width:100%;margin-top:8px;">Transcribe all</button>
            <div class="batch-progress" id="batchProgress"></div>
            <div class="batch-results-list" id="batchResults"></div>
        </div>

        <div class="upload-divider">or upload a file</div>
        <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
            <span class="upload-icon">&#127908;</span>
            Drop an audio file here, or click to browse
            <div class="upload-hint">MP3, M4A, WAV, MP4, WEBM (max 25 MB)</div>
        </div>
        <input type="file" id="fileInput" accept=".mp3,.m4a,.wav,.mp4,.webm,.mpeg,.mpga" style="display:none" onchange="handleFileSelect(this)">
        <div class="upload-file-name" id="uploadFileName">
            <span id="uploadFileLabel"></span>
            <button class="remove-file" onclick="clearUploadedFile()">&times;</button>
        </div>

        <div class="options">
            <label>
                Format:
                <select id="format">
                    <option value="text">Plain text</option>
                    <option value="srt">SRT (with timestamps)</option>
                    <option value="vtt">VTT (subtitles)</option>
                    <option value="json">JSON (structured)</option>
                </select>
            </label>
            <label>
                Language:
                <select id="language">
                    <option value="en">English</option>
                    <option value="es">Spanish</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                    <option value="pt">Portuguese</option>
                    <option value="ja">Japanese</option>
                    <option value="ko">Korean</option>
                    <option value="zh-Hans">Chinese</option>
                    <option value="ar">Arabic</option>
                    <option value="hi">Hindi</option>
                    <option value="nl">Dutch</option>
                    <option value="it">Italian</option>
                </select>
            </label>
            <label class="toggle-group">
                <input type="checkbox" id="speakers" class="toggle" checked>
                Label speakers
            </label>
            <label class="toggle-group">
                <input type="checkbox" id="aiCleanup" class="toggle toggle-ai" onchange="toggleAiOptions()" checked>
                &#10024; AI cleanup
            </label>
        </div>

        <div class="ai-options" id="aiOptions">
            <label for="speakerNames">Speaker names <span class="optional">(optional &mdash; helps identify who's talking)</span></label>
            <input type="text" id="speakerNames" class="speaker-input" placeholder="e.g. Joe Rogan, Elon Musk">
        </div>

        <div class="video-info" id="videoInfo">
            <img id="thumbnail" src="" alt="">
            <span class="title" id="videoTitle"></span>
        </div>
        <div class="spotify-note" id="spotifyNote" style="display:none"></div>

        <div class="status" id="status">
            <div class="spinner"></div>
            <div class="status-text" id="statusText">Fetching captions...</div>
        </div>

        <div class="error-box" id="error"></div>

        <!-- Paste workaround: only appears when YouTube blocks automatic fetching -->
        <div class="paste-workaround" id="pasteWorkaround">
            <h3>&#128221; Workaround: grab it from YouTube yourself</h3>
            <div class="paste-steps">
                YouTube is blocking our tool right now, but the transcript is still on YouTube &mdash; you just need to copy it:
                <ol>
                    <li>Open the video on <strong>YouTube.com</strong> in your browser</li>
                    <li>Click on the <strong>video description</strong> (the text below the title) to expand it</li>
                    <li>Scroll down and click <strong>Show transcript</strong></li>
                    <li>A transcript panel will open &mdash; click inside it, then press <strong>Cmd+A</strong> to select all</li>
                    <li>Press <strong>Cmd+C</strong> to copy, then come back here and press <strong>Cmd+V</strong> to paste below</li>
                </ol>
            </div>
            <textarea class="paste-area" id="pasteText" placeholder="Paste the transcript here..."></textarea>
            <div class="ai-options" style="display:block; margin-bottom: 12px;">
                <label for="pasteSpeakerNames">Speaker names <span class="optional">(optional)</span></label>
                <input type="text" id="pasteSpeakerNames" class="speaker-input" placeholder="e.g. Joe Rogan, Elon Musk">
            </div>
            <button type="button" class="btn-primary" id="pasteGoBtn" style="width:100%;">&#10024; Clean up with AI</button>
        </div>

        <div class="result" id="result">
            <div class="result-header">
                <h2 id="resultTitle">Transcript</h2>
                <div class="result-actions">
                    <button class="btn-original" id="originalBtn" onclick="showOriginal()" style="display:none;">&#8617; Show original</button>
                    <button class="btn-sm" onclick="toggleSearch()">&#128269; Search</button>
                    <button class="btn-sm" id="copyBtn" onclick="copyText()">&#128203; Copy</button>
                    <button class="btn-sm" onclick="copyForClaude()" id="claudeBtn" title="Copy transcript formatted for pasting into Claude">&#129302; Send to Claude</button>
                    <div class="download-menu">
                        <button class="btn-sm" onclick="toggleDownloadMenu()" id="downloadBtn">&#11015; Download</button>
                        <div class="download-dropdown" id="downloadDropdown">
                            <button class="download-option" onclick="downloadAs('txt')">
                                Plain Text (.txt)
                                <span class="dl-desc">Simple text, no formatting</span>
                            </button>
                            <button class="download-option" onclick="downloadAs('md')">
                                Markdown (.md)
                                <span class="dl-desc">Speaker names in bold &mdash; works in Notion, GitHub, etc.</span>
                            </button>
                            <button class="download-option" onclick="downloadAs('html')">
                                HTML (.html)
                                <span class="dl-desc">Color-coded speakers &mdash; paste into Docs, email, etc.</span>
                            </button>
                            <button class="download-option" onclick="downloadAs('csv')">
                                CSV (.csv)
                                <span class="dl-desc">Spreadsheet with Speaker &amp; Text columns</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            <div class="batch-tabs" id="batchTabs"></div>
            <div class="summary-box" id="summaryBox">
                <div class="summary-header">
                    <h3>&#128273; Key Points</h3>
                    <button class="summary-toggle" onclick="toggleSummary()">hide</button>
                </div>
                <div class="summary-content" id="summaryContent"></div>
            </div>
            <div class="search-bar" id="searchBar">
                <input type="text" id="searchInput" placeholder="Search transcript..." oninput="doSearch()">
                <span class="search-info" id="searchInfo"></span>
                <button class="search-nav" onclick="searchNav(-1)">&#9650;</button>
                <button class="search-nav" onclick="searchNav(1)">&#9660;</button>
            </div>
            <div class="speaker-rename-bar" id="speakerRenameBar">
                <span class="rename-label">&#9998; Rename speakers:</span>
            </div>
            <div class="transcript-box" id="transcript"></div>

        </div>

        <footer>
            YouTube &bull; Spotify &bull; Audio files &bull; AI powered by OpenAI
        </footer>
    </div>

    <!-- Settings Modal -->
    <div class="modal-overlay" id="settingsModal" onclick="if(event.target===this)closeSettings()">
        <div class="modal">
            <div class="modal-header">
                <h2>&#9881; Settings</h2>
                <button class="modal-close" onclick="closeSettings()">&#10005;</button>
            </div>
            <p class="subtitle">Add your OpenAI API key to unlock AI features</p>

            <div class="key-input-group">
                <input type="password" id="apiKeyInput" placeholder="sk-proj-...">
                <button class="show-key-btn" id="showKeyBtn" onclick="toggleKeyVisibility()">Show</button>
            </div>

            <button class="btn-save-key" onclick="saveApiKey()">Save Key</button>
            <div class="key-status neutral" id="keyStatus"></div>

            <a class="key-link" href="https://platform.openai.com/api-keys" target="_blank">
                Get an API key from OpenAI &#8594;
            </a>

            <div class="features-box">
                <h3>What this unlocks:</h3>
                <ul>
                    <li>&#10024; <strong>AI Cleanup</strong> &mdash; fix punctuation, add paragraphs, identify speakers</li>
                    <li>&#127908; <strong>Whisper</strong> &mdash; transcribe videos that don't have captions</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- History -->
    <button class="history-btn" id="historyBtn" onclick="toggleHistory()">
        &#128337;
        <span class="history-badge" id="historyBadge" style="display:none;">0</span>
    </button>
    <div class="history-panel" id="historyPanel">
        <div class="history-panel-header">
            <h3>&#128337; History</h3>
            <button class="history-clear" onclick="clearHistory()">Clear all</button>
        </div>
        <div class="history-list" id="historyList">
            <div class="history-empty">No transcripts yet</div>
        </div>
    </div>

    <script>
        const urlInput = document.getElementById('url');
        let hasApiKey = false;
        let originalData = null;
        let cleanedData = null;
        let showingCleaned = false;
        let videoMeta = { title: '', channel: '', description: '' };
        let currentVideoId = '';
        let currentTimestamps = [];
        let searchIndex = -1;
        let searchMatches = [];

        function toggleAiOptions() {
            const on = document.getElementById('aiCleanup').checked;
            document.getElementById('aiOptions').style.display = on ? 'block' : 'none';
        }

        function showPasteWorkaround() {
            document.getElementById('pasteWorkaround').style.display = 'block';
            // Scroll down so the user sees it
            document.getElementById('pasteWorkaround').scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        function showSummary(summaryText) {
            const box = document.getElementById('summaryBox');
            const content = document.getElementById('summaryContent');
            content.innerHTML = '';

            if (!summaryText || !summaryText.trim()) {
                box.classList.remove('visible');
                return;
            }

            // Parse bullet points (lines starting with - )
            const ul = document.createElement('ul');
            const lines = summaryText.split('\\n').filter(function(l) { return l.trim(); });
            lines.forEach(function(line) {
                const li = document.createElement('li');
                li.textContent = line.replace(/^[-*]\\s*/, '').trim();
                ul.appendChild(li);
            });
            content.appendChild(ul);
            box.classList.add('visible');

            // Reset toggle button
            content.style.display = '';
            box.querySelector('.summary-toggle').textContent = 'hide';
        }

        function hideSummary() {
            document.getElementById('summaryBox').classList.remove('visible');
        }

        function toggleSummary() {
            const content = document.getElementById('summaryContent');
            const btn = document.getElementById('summaryBox').querySelector('.summary-toggle');
            if (content.style.display === 'none') {
                content.style.display = '';
                btn.textContent = 'hide';
            } else {
                content.style.display = 'none';
                btn.textContent = 'show';
            }
        }

        // Check API key status on page load
        checkApiKey();

        // Submit on Enter key
        urlInput.addEventListener('keydown', e => {
            if (e.key === 'Enter') transcribe();
        });

        // Close modal on Escape
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') closeSettings();
        });

        async function checkApiKey() {
            try {
                const resp = await fetch('/settings');
                const data = await resp.json();
                hasApiKey = data.has_key;
                updateSettingsIcon();
            } catch (e) {}
        }

        function updateSettingsIcon() {
            const btn = document.getElementById('settingsBtn');
            if (hasApiKey) {
                btn.classList.add('has-key');
                btn.title = 'Settings (API key configured)';
            } else {
                btn.classList.remove('has-key');
                btn.title = 'Settings';
            }
        }

        function openSettings() {
            document.getElementById('settingsModal').classList.add('active');
            fetch('/settings')
                .then(r => r.json())
                .then(data => {
                    const status = document.getElementById('keyStatus');
                    if (data.has_key) {
                        status.className = 'key-status active';
                        status.textContent = 'Key saved: ' + data.key_preview;
                        document.getElementById('apiKeyInput').placeholder = data.key_preview;
                    } else {
                        status.className = 'key-status neutral';
                        status.textContent = 'No API key configured yet';
                    }
                });
        }

        function closeSettings() {
            document.getElementById('settingsModal').classList.remove('active');
        }

        function toggleKeyVisibility() {
            const input = document.getElementById('apiKeyInput');
            const btn = document.getElementById('showKeyBtn');
            if (input.type === 'password') {
                input.type = 'text';
                btn.textContent = 'Hide';
            } else {
                input.type = 'password';
                btn.textContent = 'Show';
            }
        }

        async function saveApiKey() {
            const key = document.getElementById('apiKeyInput').value.trim();
            const status = document.getElementById('keyStatus');

            if (!key) {
                status.className = 'key-status inactive';
                status.textContent = 'Please paste your API key above';
                return;
            }

            try {
                const resp = await fetch('/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ api_key: key })
                });
                const data = await resp.json();

                if (data.ok) {
                    hasApiKey = true;
                    updateSettingsIcon();
                    status.className = 'key-status active';
                    status.textContent = 'Key saved successfully!';
                    document.getElementById('apiKeyInput').value = '';
                    document.getElementById('apiKeyInput').placeholder = key.substring(0, 7) + '...' + key.slice(-4);
                    setTimeout(closeSettings, 1500);
                } else {
                    status.className = 'key-status inactive';
                    status.textContent = data.error || 'Failed to save';
                }
            } catch (e) {
                status.className = 'key-status inactive';
                status.textContent = 'Error saving key. Is the server running?';
            }
        }

        async function transcribe() {
            // If a file is uploaded, transcribe that instead
            if (uploadedFile) {
                return transcribeFile();
            }
            const url = urlInput.value.trim();
            if (!url) { urlInput.focus(); return; }

            const fmt = document.getElementById('format').value;
            const lang = document.getElementById('language').value;
            const labelSpeakers = document.getElementById('speakers').checked;
            const doCleanup = document.getElementById('aiCleanup').checked;
            const speakerNames = document.getElementById('speakerNames').value.trim();

            // Reset state

            originalData = null;
            cleanedData = null;
            showingCleaned = false;
            videoMeta = { title: '', channel: '', description: '' };
            document.getElementById('originalBtn').style.display = 'none';
            document.getElementById('pasteWorkaround').style.display = 'none';
            document.getElementById('speakerRenameBar').classList.remove('visible');
            document.getElementById('batchTabs').classList.remove('visible');
            document.getElementById('batchTabs').innerHTML = '';
            hideSummary();

            // Show loading
            document.getElementById('status').style.display = 'block';
            document.getElementById('statusText').textContent = 'Fetching captions...';
            document.getElementById('result').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            document.getElementById('goBtn').disabled = true;

            showThumbnail(url);

            try {
                // Step 1: Get the transcript
                const resp = await fetch('/transcribe', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url, format: fmt, language: lang, speakers: labelSpeakers })
                });

                const data = await resp.json();

                if (data.error) {
                    document.getElementById('status').style.display = 'none';
                    const errorBox = document.getElementById('error');
                    errorBox.textContent = data.error;
                    errorBox.style.display = 'block';
                    if (data.blocked) {
                        showPasteWorkaround();
                    }
                    document.getElementById('goBtn').disabled = false;
                    return;
                }

                // Store the raw caption version
                originalData = { transcript: data.transcript, speakers: data.speakers || [], source: data.source };
                currentVideoId = data.video_id || '';
                currentTimestamps = data.timestamps || [];

                // Show Spotify match notice if applicable
                if (data.spotify_match) {
                    document.getElementById('spotifyNote').textContent = 'Transcribed via YouTube match: ' + data.spotify_match;
                    document.getElementById('spotifyNote').style.display = 'block';
                } else {
                    document.getElementById('spotifyNote').style.display = 'none';
                }

                // If AI cleanup is OFF, just show raw captions and stop
                if (!doCleanup) {
                    document.getElementById('status').style.display = 'none';
                    renderTranscript(data.transcript, data.speakers || []);
                    const sourceLabel = data.source === 'whisper' ? ' (from Whisper)' : ' (from captions)';
                    document.getElementById('resultTitle').innerHTML = 'Transcript' + sourceLabel;
                    document.getElementById('result').style.display = 'block';
                    saveToHistory(url, data.transcript, data.speakers || [], '', videoMeta.title);
                    document.getElementById('goBtn').disabled = false;
                    return;
                }

                // Step 2: AI cleanup is ON — fetch metadata first
                document.getElementById('statusText').textContent = 'Getting video info...';

                try {
                    const metaResp = await fetch('/metadata', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ url })
                    });
                    videoMeta = await metaResp.json();
                    if (videoMeta.title) {
                        document.getElementById('videoTitle').textContent = videoMeta.title;
                    }
                } catch (e) {}

                // Step 3: Run AI cleanup
                document.getElementById('statusText').textContent = 'Cleaning up with AI...';

                const rawText = data.transcript;
                // Strip the **Speaker:** markdown for cleaner input to GPT
                const plainText = rawText.replace(/\\*\\*([^*]+):\\*\\*/g, '$1:');

                const cleanResp = await fetch('/cleanup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        transcript: plainText,
                        speaker_names: speakerNames,
                        video_title: videoMeta.title,
                        video_channel: videoMeta.channel,
                        video_description: videoMeta.description
                    })
                });

                const cleanData = await cleanResp.json();
                document.getElementById('status').style.display = 'none';

                if (cleanData.error) {
                    // Cleanup failed — show raw captions with error note
                    renderTranscript(data.transcript, data.speakers || []);
                    document.getElementById('resultTitle').innerHTML = 'Transcript (from captions)';
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('error').textContent = 'AI cleanup failed: ' + cleanData.error + ' — showing raw captions instead.';
                    document.getElementById('error').style.display = 'block';
                } else {
                    // Store cleaned version and show it
                    cleanedData = { transcript: cleanData.transcript, speakers: cleanData.speakers || [], summary: cleanData.summary || '' };
                    showingCleaned = true;

                    showSummary(cleanData.summary || '');
                    renderTranscript(cleanData.transcript, cleanData.speakers || []);
                    document.getElementById('resultTitle').innerHTML =
                        'Transcript <span class="ai-badge">&#10024; Cleaned with AI</span>';
                    document.getElementById('result').style.display = 'block';

                    // Show toggle button
                    const origBtn = document.getElementById('originalBtn');
                    origBtn.style.display = 'inline-block';
                    origBtn.innerHTML = '&#8617; Show raw captions';

                    saveToHistory(url, cleanData.transcript, cleanData.speakers || [], cleanData.summary || '', videoMeta.title);
                }

            } catch (err) {
                document.getElementById('status').style.display = 'none';
                document.getElementById('error').textContent = 'Something went wrong. Is the server still running?';
                document.getElementById('error').style.display = 'block';
            }

            document.getElementById('goBtn').disabled = false;
        }

        async function processPasted() {
            const text = document.getElementById('pasteText').value.trim();
            if (!text) {
                document.getElementById('pasteText').focus();
                return;
            }

            const speakerNames = document.getElementById('pasteSpeakerNames').value.trim();

            // Reset state

            originalData = null;
            cleanedData = null;
            showingCleaned = false;
            document.getElementById('originalBtn').style.display = 'none';
            document.getElementById('speakerRenameBar').classList.remove('visible');
            hideSummary();

            // Show loading
            document.getElementById('status').style.display = 'block';
            document.getElementById('statusText').textContent = 'Cleaning up with AI...';
            document.getElementById('result').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            document.getElementById('pasteGoBtn').disabled = true;

            // Store the raw pasted version
            originalData = { transcript: text, speakers: [], source: 'pasted' };

            try {
                const cleanResp = await fetch('/cleanup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        transcript: text,
                        speaker_names: speakerNames,
                        video_title: '',
                        video_channel: '',
                        video_description: ''
                    })
                });

                const cleanData = await cleanResp.json();
                document.getElementById('status').style.display = 'none';

                if (cleanData.error) {
                    // Show raw pasted text as fallback
                    document.getElementById('transcript').style.whiteSpace = 'pre-wrap';
                    document.getElementById('transcript').textContent = text;
                    document.getElementById('resultTitle').innerHTML = 'Transcript (pasted)';
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('error').textContent = 'AI cleanup failed: ' + cleanData.error;
                    document.getElementById('error').style.display = 'block';
                } else {
                    // Hide the workaround and error since we got a result
                    document.getElementById('pasteWorkaround').style.display = 'none';
                    document.getElementById('error').style.display = 'none';

                    cleanedData = { transcript: cleanData.transcript, speakers: cleanData.speakers || [], summary: cleanData.summary || '' };
                    showingCleaned = true;

                    showSummary(cleanData.summary || '');
                    renderTranscript(cleanData.transcript, cleanData.speakers || []);
                    document.getElementById('resultTitle').innerHTML =
                        'Transcript <span class="ai-badge">&#10024; Cleaned with AI</span>';
                    document.getElementById('result').style.display = 'block';

                    const origBtn = document.getElementById('originalBtn');
                    origBtn.style.display = 'inline-block';
                    origBtn.innerHTML = '&#8617; Show raw pasted';
                }
            } catch (err) {
                document.getElementById('status').style.display = 'none';
                document.getElementById('error').textContent = 'Something went wrong. Is the server running?';
                document.getElementById('error').style.display = 'block';
            }

            document.getElementById('pasteGoBtn').disabled = false;
        }

        function showOriginal() {
            // Cancel any active editing first


            const btn = document.getElementById('originalBtn');

            if (showingCleaned) {
                // Switch to raw captions
                if (!originalData) return;
                hideSummary();
                renderTranscript(originalData.transcript, originalData.speakers);
                showingCleaned = false;
                const sourceLabel = originalData.source === 'whisper' ? ' (from Whisper)' : ' (from captions)';
                document.getElementById('resultTitle').innerHTML = 'Transcript' + sourceLabel;
                btn.innerHTML = '&#10024; Show cleaned';
            } else {
                // Switch back to cleaned
                if (!cleanedData) return;
                if (cleanedData.summary) showSummary(cleanedData.summary);
                renderTranscript(cleanedData.transcript, cleanedData.speakers);
                showingCleaned = true;
                document.getElementById('resultTitle').innerHTML =
                    'Transcript <span class="ai-badge">&#10024; Cleaned with AI</span>';
                btn.innerHTML = '&#8617; Show raw captions';
            }
        }

        function showThumbnail(url) {
            const ytMatch = url.match(/(?:v=|youtu\\.be\\/|shorts\\/)([a-zA-Z0-9_-]{11})/);
            if (ytMatch) {
                document.getElementById('thumbnail').src = 'https://img.youtube.com/vi/' + ytMatch[1] + '/mqdefault.jpg';
                document.getElementById('videoInfo').style.display = 'flex';
                return;
            }
            // Spotify episode — fetch thumbnail from oEmbed via our metadata endpoint
            if (url.indexOf('spotify.com/episode') >= 0) {
                fetch('/metadata', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: url })
                })
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.title) {
                        document.getElementById('videoTitle').textContent = data.title;
                        if (data.thumbnail_url) {
                            document.getElementById('thumbnail').src = data.thumbnail_url;
                        }
                        document.getElementById('videoInfo').style.display = 'flex';
                    }
                })
                .catch(function() {});
            }
        }

        function findTimestamp(contentSnippet) {
            // Find the closest timestamp for a text snippet
            if (!currentTimestamps.length || !currentVideoId) return null;
            const words = contentSnippet.toLowerCase().replace(/[^a-z0-9\\s]/g, '').split(/\\s+/).slice(0, 6).join(' ');
            if (!words) return null;
            for (let t = 0; t < currentTimestamps.length; t++) {
                const tsWords = currentTimestamps[t].text.toLowerCase().replace(/[^a-z0-9\\s]/g, '').split(/\\s+/).slice(0, 6).join(' ');
                if (tsWords && words.indexOf(tsWords.slice(0, 20)) >= 0 || tsWords.indexOf(words.slice(0, 20)) >= 0) {
                    return currentTimestamps[t].start;
                }
            }
            return null;
        }

        function makeTimestampLink(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            const label = h > 0 ? h + ':' + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0')
                                : m + ':' + String(s).padStart(2,'0');
            const a = document.createElement('a');
            a.className = 'timestamp';
            a.textContent = label;
            a.href = 'https://www.youtube.com/watch?v=' + currentVideoId + '&t=' + Math.floor(seconds) + 's';
            a.target = '_blank';
            a.title = 'Jump to ' + label + ' on YouTube';
            return a;
        }

        function renderTranscript(text, speakers) {
            const box = document.getElementById('transcript');
            box.innerHTML = '';
            box.style.whiteSpace = '';

            if (speakers && speakers.length > 0) {
                const colors = ['speaker-1', 'speaker-2', 'speaker-3', 'speaker-4'];
                const parts = text.split(/\\*\\*([^*]+):\\*\\*/);

                for (let i = 1; i < parts.length; i += 2) {
                    const name = parts[i];
                    const content = (parts[i + 1] || '').trim();
                    const speakerIdx = speakers.indexOf(name);
                    const colorClass = colors[speakerIdx >= 0 ? speakerIdx % colors.length : 0];

                    const label = document.createElement('span');
                    label.className = 'speaker ' + colorClass;

                    // Try to add a timestamp link
                    const ts = findTimestamp(content);
                    if (ts !== null) {
                        label.appendChild(makeTimestampLink(ts));
                    }
                    label.appendChild(document.createTextNode(name + ':'));
                    box.appendChild(label);

                    const para = document.createElement('p');
                    para.style.marginTop = '4px';
                    para.style.marginBottom = '16px';
                    para.textContent = content;
                    box.appendChild(para);
                }

                if (parts[0].trim()) {
                    const pre = document.createElement('p');
                    pre.style.marginBottom = '16px';
                    pre.textContent = parts[0].trim();
                    box.insertBefore(pre, box.firstChild);
                }
            } else {
                box.style.whiteSpace = 'pre-wrap';
                box.textContent = text;
            }

            // Show/hide speaker rename bar
            showSpeakerRenameBar(speakers);
        }

        // ── Speaker rename ──

        function showSpeakerRenameBar(speakers) {
            const bar = document.getElementById('speakerRenameBar');
            // Clear old chips (keep the label)
            while (bar.children.length > 1) bar.removeChild(bar.lastChild);

            if (!speakers || speakers.length === 0) {
                bar.classList.remove('visible');
                return;
            }

            const colors = ['#5eb5f7', '#f7a85e', '#8ef79e', '#f78ef0'];

            speakers.forEach(function(name, idx) {
                const chip = document.createElement('div');
                chip.className = 'speaker-chip';

                const dot = document.createElement('span');
                dot.className = 'color-dot';
                dot.style.background = colors[idx % colors.length];
                chip.appendChild(dot);

                const input = document.createElement('input');
                input.type = 'text';
                input.value = name;
                input.dataset.original = name;
                input.style.width = Math.max(100, name.length * 9) + 'px';

                // Auto-resize as they type
                input.addEventListener('input', function() {
                    this.style.width = Math.max(100, this.value.length * 9) + 'px';
                });

                // Rename on Enter or blur
                input.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter') { this.blur(); }
                });
                input.addEventListener('blur', function() {
                    const oldName = this.dataset.original;
                    const newName = this.value.trim();
                    if (newName && newName !== oldName) {
                        renameSpeaker(oldName, newName);
                        this.dataset.original = newName;
                    } else {
                        this.value = oldName;  // revert if empty
                    }
                });

                chip.appendChild(input);
                bar.appendChild(chip);
            });

            bar.classList.add('visible');
        }

        function renameSpeaker(oldName, newName) {
            // Update in stored data
            function updateData(data) {
                if (!data) return;
                // Replace in transcript text: **OldName:** → **NewName:**
                data.transcript = data.transcript.split('**' + oldName + ':**').join('**' + newName + ':**');
                // Update speakers array
                const idx = data.speakers.indexOf(oldName);
                if (idx >= 0) data.speakers[idx] = newName;
            }

            if (showingCleaned && cleanedData) {
                updateData(cleanedData);
                renderTranscript(cleanedData.transcript, cleanedData.speakers);
                showSpeakerRenameBar(cleanedData.speakers);
            } else if (originalData) {
                updateData(originalData);
                renderTranscript(originalData.transcript, originalData.speakers);
                showSpeakerRenameBar(originalData.speakers);
            }
        }

        // ── File upload ──

        let uploadedFile = null;

        // Drag and drop
        const uploadZone = document.getElementById('uploadZone');
        uploadZone.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
        uploadZone.addEventListener('dragleave', function() {
            uploadZone.classList.remove('dragover');
        });
        uploadZone.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                setUploadedFile(e.dataTransfer.files[0]);
            }
        });

        function handleFileSelect(input) {
            if (input.files.length > 0) {
                setUploadedFile(input.files[0]);
            }
        }

        function setUploadedFile(file) {
            const maxSize = 25 * 1024 * 1024;
            if (file.size > maxSize) {
                document.getElementById('error').textContent = 'File is too large (max 25 MB for Whisper).';
                document.getElementById('error').style.display = 'block';
                return;
            }
            uploadedFile = file;
            document.getElementById('uploadFileLabel').textContent = file.name + ' (' + (file.size / (1024 * 1024)).toFixed(1) + ' MB)';
            document.getElementById('uploadFileName').classList.add('visible');
            document.getElementById('uploadZone').style.display = 'none';
        }

        function clearUploadedFile() {
            uploadedFile = null;
            document.getElementById('fileInput').value = '';
            document.getElementById('uploadFileName').classList.remove('visible');
            document.getElementById('uploadZone').style.display = '';
        }

        async function transcribeFile() {
            if (!uploadedFile) return;

            const doCleanup = document.getElementById('aiCleanup').checked;
            const speakerNames = document.getElementById('speakerNames').value.trim();

            // Reset state
            originalData = null;
            cleanedData = null;
            showingCleaned = false;
            videoMeta = { title: uploadedFile.name, channel: '', description: '' };
            document.getElementById('originalBtn').style.display = 'none';
            document.getElementById('pasteWorkaround').style.display = 'none';
            document.getElementById('speakerRenameBar').classList.remove('visible');
            document.getElementById('batchTabs').classList.remove('visible');
            document.getElementById('batchTabs').innerHTML = '';
            hideSummary();

            // Show loading
            document.getElementById('status').style.display = 'block';
            document.getElementById('statusText').textContent = 'Uploading and transcribing with Whisper...';
            document.getElementById('result').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            document.getElementById('goBtn').disabled = true;

            document.getElementById('videoTitle').textContent = uploadedFile.name;
            document.getElementById('thumbnail').src = '';
            document.getElementById('videoInfo').style.display = 'none';

            try {
                const formData = new FormData();
                formData.append('file', uploadedFile);

                const resp = await fetch('/upload', { method: 'POST', body: formData });
                const data = await resp.json();

                if (data.error) {
                    document.getElementById('status').style.display = 'none';
                    document.getElementById('error').textContent = data.error;
                    document.getElementById('error').style.display = 'block';
                    document.getElementById('goBtn').disabled = false;
                    return;
                }

                originalData = { transcript: data.transcript, speakers: data.speakers || [], source: 'whisper' };
                currentVideoId = '';
                currentTimestamps = [];

                if (!doCleanup) {
                    document.getElementById('status').style.display = 'none';
                    renderTranscript(data.transcript, data.speakers || []);
                    document.getElementById('resultTitle').innerHTML = 'Transcript (from Whisper)';
                    document.getElementById('result').style.display = 'block';
                    saveToHistory(uploadedFile.name, data.transcript, data.speakers || [], '');
                    document.getElementById('goBtn').disabled = false;
                    return;
                }

                // AI cleanup
                document.getElementById('statusText').textContent = 'Cleaning up with AI...';

                const cleanResp = await fetch('/cleanup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        transcript: data.transcript,
                        speaker_names: speakerNames,
                        video_title: uploadedFile.name,
                        video_channel: '',
                        video_description: ''
                    })
                });

                const cleanData = await cleanResp.json();
                document.getElementById('status').style.display = 'none';

                if (cleanData.error) {
                    renderTranscript(data.transcript, data.speakers || []);
                    document.getElementById('resultTitle').innerHTML = 'Transcript (from Whisper)';
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('error').textContent = 'AI cleanup failed: ' + cleanData.error;
                    document.getElementById('error').style.display = 'block';
                } else {
                    cleanedData = { transcript: cleanData.transcript, speakers: cleanData.speakers || [], summary: cleanData.summary || '' };
                    showingCleaned = true;
                    showSummary(cleanData.summary || '');
                    renderTranscript(cleanData.transcript, cleanData.speakers || []);
                    document.getElementById('resultTitle').innerHTML =
                        'Transcript <span class="ai-badge">&#10024; Cleaned with AI</span>';
                    document.getElementById('result').style.display = 'block';

                    const origBtn = document.getElementById('originalBtn');
                    origBtn.style.display = 'inline-block';
                    origBtn.innerHTML = '&#8617; Show raw transcript';

                    saveToHistory(uploadedFile.name, cleanData.transcript, cleanData.speakers || [], cleanData.summary || '', uploadedFile.name);
                }
            } catch (err) {
                document.getElementById('status').style.display = 'none';
                document.getElementById('error').textContent = 'Upload failed. Is the server still running?';
                document.getElementById('error').style.display = 'block';
            }

            document.getElementById('goBtn').disabled = false;
        }

        // ── Download menu ──

        function toggleDownloadMenu() {
            const dd = document.getElementById('downloadDropdown');
            dd.classList.toggle('open');
        }

        // Close dropdown when clicking elsewhere
        document.addEventListener('click', function(e) {
            const menu = document.getElementById('downloadDropdown');
            const btn = document.getElementById('downloadBtn');
            if (menu && !menu.contains(e.target) && e.target !== btn) {
                menu.classList.remove('open');
            }
        });

        function getCurrentSpeakerData() {
            // Get the current transcript text and speakers
            if (showingCleaned && cleanedData) {
                return { transcript: cleanedData.transcript, speakers: cleanedData.speakers || [] };
            } else if (originalData) {
                return { transcript: originalData.transcript, speakers: originalData.speakers || [] };
            }
            return { transcript: document.getElementById('transcript').textContent, speakers: [] };
        }

        function parseSpeakerBlocks(text, speakers) {
            // Split transcript into [{speaker, text}] blocks
            if (!speakers || speakers.length === 0) {
                return [{ speaker: '', text: text }];
            }
            const parts = text.split(/\\*\\*([^*]+):\\*\\*/);
            const blocks = [];
            // parts[0] is text before first speaker (if any)
            if (parts[0] && parts[0].trim()) {
                blocks.push({ speaker: '', text: parts[0].trim() });
            }
            for (let i = 1; i < parts.length; i += 2) {
                blocks.push({ speaker: parts[i], text: (parts[i + 1] || '').trim() });
            }
            return blocks;
        }

        function downloadAs(format) {
            document.getElementById('downloadDropdown').classList.remove('open');

            const data = getCurrentSpeakerData();
            const blocks = parseSpeakerBlocks(data.transcript, data.speakers);
            const hasSpeakers = data.speakers && data.speakers.length > 0;

            let content, ext, mimeType;

            if (format === 'txt') {
                // Plain text with speaker names
                if (hasSpeakers) {
                    content = blocks.map(function(b) {
                        return b.speaker ? b.speaker + ':\\n' + b.text : b.text;
                    }).join('\\n\\n');
                } else {
                    content = data.transcript;
                }
                ext = 'txt';
                mimeType = 'text/plain';

            } else if (format === 'md') {
                // Markdown with bold speaker names
                if (hasSpeakers) {
                    content = blocks.map(function(b) {
                        return b.speaker ? '**' + b.speaker + ':**\\n\\n' + b.text : b.text;
                    }).join('\\n\\n---\\n\\n');
                } else {
                    content = data.transcript;
                }
                ext = 'md';
                mimeType = 'text/markdown';

            } else if (format === 'html') {
                // HTML with colored speaker names
                const colors = ['#5eb5f7', '#f7a85e', '#8ef79e', '#f78ef0'];
                let body;
                if (hasSpeakers) {
                    body = blocks.map(function(b) {
                        if (!b.speaker) return '<p>' + b.text.replace(/\\n/g, '<br>') + '</p>';
                        const idx = data.speakers.indexOf(b.speaker);
                        const color = colors[idx >= 0 ? idx % colors.length : 0];
                        return '<h3 style="color:' + color + '; margin: 24px 0 8px 0;">' + b.speaker + '</h3>' +
                               '<p style="margin: 0 0 16px 0; line-height: 1.7;">' + b.text.replace(/\\n/g, '<br>') + '</p>';
                    }).join('\\n');
                } else {
                    body = '<p style="line-height: 1.7;">' + data.transcript.replace(/\\n/g, '<br>') + '</p>';
                }
                content = '<!DOCTYPE html>\\n<html><head><meta charset="utf-8"><title>Transcript</title>' +
                          '<style>body{font-family:-apple-system,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;background:#111;color:#ddd;}</style>' +
                          '</head><body>\\n' + body + '\\n</body></html>';
                ext = 'html';
                mimeType = 'text/html';

            } else if (format === 'csv') {
                // CSV with Speaker, Text columns
                const rows = ['"Speaker","Text"'];
                blocks.forEach(function(b) {
                    const speaker = (b.speaker || 'N/A').replace(/"/g, '""');
                    const text = b.text.replace(/"/g, '""').replace(/\\n/g, ' ');
                    rows.push('"' + speaker + '","' + text + '"');
                });
                content = rows.join('\\n');
                ext = 'csv';
                mimeType = 'text/csv';
            }

            const blob = new Blob([content], { type: mimeType });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'transcript.' + ext;
            a.click();
        }

        function copyText() {
            const text = document.getElementById('transcript').textContent;
            navigator.clipboard.writeText(text).then(function() {
                const btn = document.getElementById('copyBtn');
                btn.textContent = '\u2713 Copied!';
                btn.classList.add('copied');
                setTimeout(function() {
                    btn.innerHTML = '&#128203; Copy';
                    btn.classList.remove('copied');
                }, 2000);
            });
        }

        function copyForClaude() {
            const data = getCurrentSpeakerData();
            const blocks = parseSpeakerBlocks(data.transcript, data.speakers);
            const hasSpeakers = data.speakers && data.speakers.length > 0;

            // Build a Claude-friendly prompt with the transcript
            let parts = [];

            // Title
            const title = videoMeta.title || document.getElementById('videoTitle').textContent || '';
            if (title) {
                parts.push('Here is a transcript from the YouTube video "' + title + '":');
            } else {
                parts.push('Here is a transcript from a YouTube video:');
            }
            parts.push('');

            // Summary if available
            const summaryEl = document.getElementById('summaryContent');
            if (summaryEl && summaryEl.textContent.trim()) {
                parts.push('KEY POINTS:');
                const items = summaryEl.querySelectorAll('li');
                items.forEach(function(li) {
                    parts.push('- ' + li.textContent);
                });
                parts.push('');
                parts.push('FULL TRANSCRIPT:');
                parts.push('');
            }

            // Transcript
            if (hasSpeakers) {
                blocks.forEach(function(b) {
                    if (b.speaker) {
                        parts.push(b.speaker + ':');
                        parts.push(b.text);
                        parts.push('');
                    } else {
                        parts.push(b.text);
                        parts.push('');
                    }
                });
            } else {
                parts.push(data.transcript);
            }

            const fullText = parts.join('\\n');

            navigator.clipboard.writeText(fullText).then(function() {
                const btn = document.getElementById('claudeBtn');
                btn.textContent = '\\u2713 Copied for Claude!';
                btn.classList.add('claude-copied');
                setTimeout(function() {
                    btn.innerHTML = '&#129302; Send to Claude';
                    btn.classList.remove('claude-copied');
                }, 2500);
            });
        }

        function downloadText() {
            // Fallback — opens the download menu
            toggleDownloadMenu();
        }

        // ── Search ──

        function toggleSearch() {
            const bar = document.getElementById('searchBar');
            if (bar.classList.contains('visible')) {
                bar.classList.remove('visible');
                clearSearch();
            } else {
                bar.classList.add('visible');
                document.getElementById('searchInput').focus();
            }
        }

        function doSearch() {
            const query = document.getElementById('searchInput').value.trim().toLowerCase();
            const box = document.getElementById('transcript');

            // Clear previous highlights
            box.querySelectorAll('mark.search-highlight').forEach(function(m) {
                const parent = m.parentNode;
                parent.replaceChild(document.createTextNode(m.textContent), m);
                parent.normalize();
            });

            searchMatches = [];
            searchIndex = -1;

            if (!query) {
                document.getElementById('searchInfo').textContent = '';
                return;
            }

            // Walk text nodes and highlight matches
            const walker = document.createTreeWalker(box, NodeFilter.SHOW_TEXT, null, false);
            const textNodes = [];
            while (walker.nextNode()) textNodes.push(walker.currentNode);

            textNodes.forEach(function(node) {
                const text = node.textContent;
                const lower = text.toLowerCase();
                let idx = lower.indexOf(query);
                if (idx < 0) return;

                const frag = document.createDocumentFragment();
                let lastIdx = 0;
                while (idx >= 0) {
                    frag.appendChild(document.createTextNode(text.slice(lastIdx, idx)));
                    const mark = document.createElement('mark');
                    mark.className = 'search-highlight';
                    mark.textContent = text.slice(idx, idx + query.length);
                    frag.appendChild(mark);
                    searchMatches.push(mark);
                    lastIdx = idx + query.length;
                    idx = lower.indexOf(query, lastIdx);
                }
                frag.appendChild(document.createTextNode(text.slice(lastIdx)));
                node.parentNode.replaceChild(frag, node);
            });

            document.getElementById('searchInfo').textContent = searchMatches.length + ' found';
            if (searchMatches.length > 0) searchNav(1);
        }

        function searchNav(dir) {
            if (searchMatches.length === 0) return;
            if (searchIndex >= 0 && searchMatches[searchIndex]) {
                searchMatches[searchIndex].classList.remove('current');
            }
            searchIndex = (searchIndex + dir + searchMatches.length) % searchMatches.length;
            searchMatches[searchIndex].classList.add('current');
            searchMatches[searchIndex].scrollIntoView({ behavior: 'smooth', block: 'center' });
            document.getElementById('searchInfo').textContent = (searchIndex + 1) + ' / ' + searchMatches.length;
        }

        function clearSearch() {
            document.getElementById('searchInput').value = '';
            const box = document.getElementById('transcript');
            box.querySelectorAll('mark.search-highlight').forEach(function(m) {
                const parent = m.parentNode;
                parent.replaceChild(document.createTextNode(m.textContent), m);
                parent.normalize();
            });
            searchMatches = [];
            searchIndex = -1;
            document.getElementById('searchInfo').textContent = '';
        }

        // ── Batch mode ──

        function toggleBatchMode() {
            const area = document.getElementById('batchArea');
            const toggle = document.getElementById('batchToggle');
            if (area.classList.contains('visible')) {
                area.classList.remove('visible');
                toggle.innerHTML = '&#43; Multiple URLs';
            } else {
                area.classList.add('visible');
                toggle.innerHTML = '&#8722; Multiple URLs';
                document.getElementById('batchUrls').focus();
            }
        }

        async function startBatch() {
            const text = document.getElementById('batchUrls').value.trim();
            if (!text) return;

            const urls = text.split('\\n').map(function(l) { return l.trim(); }).filter(function(l) { return l; });
            if (urls.length === 0) return;

            const progress = document.getElementById('batchProgress');
            const results = document.getElementById('batchResults');
            progress.innerHTML = '';
            results.innerHTML = '';
            progress.classList.add('visible');
            document.getElementById('batchGoBtn').disabled = true;

            // Build progress items
            urls.forEach(function(u, i) {
                const item = document.createElement('div');
                item.className = 'batch-item';
                item.id = 'batch-' + i;
                item.innerHTML = '<span class="batch-status">&#9711;</span><span class="batch-url">' + u + '</span>';
                progress.appendChild(item);
            });

            const batchHistory = [];

            for (let i = 0; i < urls.length; i++) {
                const item = document.getElementById('batch-' + i);
                item.classList.add('active');
                item.querySelector('.batch-status').innerHTML = '&#9203;';

                try {
                    // Use the same transcribe flow
                    const doCleanup = document.getElementById('aiCleanup').checked;
                    const speakerNames = document.getElementById('speakerNames').value.trim();
                    const language = document.getElementById('language').value;

                    const resp = await fetch('/transcribe', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            url: urls[i],
                            format: 'text',
                            language: language,
                            speakers: true
                        })
                    });
                    const data = await resp.json();

                    if (data.error) {
                        item.classList.remove('active');
                        item.classList.add('failed');
                        item.querySelector('.batch-status').innerHTML = '&#10007;';
                        continue;
                    }

                    let finalTranscript = data.transcript;
                    let finalSpeakers = data.speakers || [];
                    let summary = '';
                    let vidTitle = '';

                    // Fetch metadata for title
                    try {
                        const metaResp = await fetch('/metadata', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ url: urls[i] })
                        });
                        const meta = await metaResp.json();
                        vidTitle = meta.title || '';
                    } catch(e) {}

                    // AI cleanup if enabled
                    if (doCleanup && hasApiKey) {
                        const cleanResp = await fetch('/cleanup', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                transcript: data.transcript,
                                speaker_names: speakerNames,
                                video_title: vidTitle,
                                video_channel: '',
                                video_description: ''
                            })
                        });
                        const cleanData = await cleanResp.json();
                        if (!cleanData.error) {
                            finalTranscript = cleanData.transcript;
                            finalSpeakers = cleanData.speakers || [];
                            summary = cleanData.summary || '';
                        }
                    }

                    item.classList.remove('active');
                    item.classList.add('done');
                    item.querySelector('.batch-status').innerHTML = '&#10003;';

                    batchHistory.push({
                        url: urls[i],
                        videoId: data.video_id || '',
                        transcript: finalTranscript,
                        speakers: finalSpeakers,
                        summary: summary,
                        timestamps: data.timestamps || [],
                        title: vidTitle
                    });

                } catch (err) {
                    item.classList.remove('active');
                    item.classList.add('failed');
                    item.querySelector('.batch-status').innerHTML = '&#10007;';
                }
            }

            // Build tabs for switching between results
            const tabsBar = document.getElementById('batchTabs');
            tabsBar.innerHTML = '';

            if (batchHistory.length > 0) {
                // Store batch results globally so tabs can access them
                window._batchResults = batchHistory;

                batchHistory.forEach(function(entry, i) {
                    const tab = document.createElement('button');
                    tab.className = 'batch-tab';
                    tab.textContent = entry.title || ('Video ' + (i + 1));
                    tab.onclick = function() { loadBatchResult(i); };
                    tabsBar.appendChild(tab);

                    // Save each to history
                    saveToHistory(entry.url, entry.transcript, entry.speakers, entry.summary, entry.title);
                });

                tabsBar.classList.add('visible');

                // Auto-load the first result
                loadBatchResult(0);
            }

            document.getElementById('batchGoBtn').disabled = false;
        }

        function loadBatchResult(index) {
            const results = window._batchResults;
            if (!results || !results[index]) return;

            const entry = results[index];

            // Highlight the active tab
            const tabs = document.querySelectorAll('.batch-tab');
            tabs.forEach(function(t, i) {
                t.classList.toggle('active', i === index);
            });

            // Load into main view
            currentVideoId = entry.videoId;
            currentTimestamps = entry.timestamps;
            originalData = null;
            cleanedData = { transcript: entry.transcript, speakers: entry.speakers, summary: entry.summary };
            showingCleaned = true;
            document.getElementById('originalBtn').style.display = 'none';

            showSummary(entry.summary);
            renderTranscript(entry.transcript, entry.speakers);

            const titleText = entry.title ? entry.title : 'Transcript';
            document.getElementById('resultTitle').innerHTML =
                titleText + ' <span class="ai-badge">&#10024; Cleaned with AI</span>';
            document.getElementById('result').style.display = 'block';
            document.getElementById('result').scrollIntoView({ behavior: 'smooth' });

            // Clear any active search
            if (document.getElementById('searchBar').classList.contains('visible')) {
                clearSearch();
            }
        }

        // ── History ──

        function getHistory() {
            try {
                return JSON.parse(localStorage.getItem('yt_transcribe_history') || '[]');
            } catch(e) { return []; }
        }

        function saveToHistory(url, transcript, speakers, summary, title) {
            const history = getHistory();
            // Use provided title, or fall back to URL
            if (!title) {
                const match = url.match(/(?:v=|youtu\\.be\\/|shorts\\/)([a-zA-Z0-9_-]{11})/);
                title = match ? 'Video ' + match[1] : url;
            }

            // Don't duplicate
            const existing = history.findIndex(function(h) { return h.url === url; });
            if (existing >= 0) history.splice(existing, 1);

            history.unshift({
                url: url,
                title: title,
                transcript: transcript,
                speakers: speakers || [],
                summary: summary || '',
                date: new Date().toISOString()
            });

            // Keep max 50
            if (history.length > 50) history.length = 50;

            localStorage.setItem('yt_transcribe_history', JSON.stringify(history));
            updateHistoryBadge();
        }

        function updateHistoryBadge() {
            const count = getHistory().length;
            const badge = document.getElementById('historyBadge');
            if (count > 0) {
                badge.textContent = count;
                badge.style.display = 'flex';
            } else {
                badge.style.display = 'none';
            }
        }

        function toggleHistory() {
            const panel = document.getElementById('historyPanel');
            if (panel.classList.contains('open')) {
                panel.classList.remove('open');
            } else {
                renderHistoryList();
                panel.classList.add('open');
            }
        }

        function renderHistoryList() {
            const list = document.getElementById('historyList');
            const history = getHistory();
            list.innerHTML = '';

            if (history.length === 0) {
                list.innerHTML = '<div class="history-empty">No transcripts yet</div>';
                return;
            }

            history.forEach(function(entry, i) {
                const item = document.createElement('div');
                item.className = 'history-item';

                const d = new Date(entry.date);
                const dateStr = d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

                item.innerHTML = '<div class="hist-title">' + entry.title.replace(/</g, '&lt;') + '</div>' +
                                 '<div class="hist-meta">' + dateStr + '</div>';

                item.onclick = function() {
                    // Load from history
                    cleanedData = { transcript: entry.transcript, speakers: entry.speakers || [], summary: entry.summary || '' };
                    showingCleaned = true;
                    showSummary(entry.summary || '');
                    renderTranscript(entry.transcript, entry.speakers || []);
                    document.getElementById('resultTitle').innerHTML =
                        'Transcript <span class="ai-badge">&#10024; Cleaned with AI</span>';
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('result').scrollIntoView({ behavior: 'smooth' });
                    document.getElementById('historyPanel').classList.remove('open');
                };
                list.appendChild(item);
            });
        }

        function clearHistory() {
            localStorage.removeItem('yt_transcribe_history');
            renderHistoryList();
            updateHistoryBadge();
        }

        // Initialize history badge on load
        updateHistoryBadge();

        // ── Wire up paste button ──
        document.getElementById('pasteGoBtn').addEventListener('click', function() {
            processPasted();
        });
    </script>
</body>
</html>
"""


# ── Routes ───────────────────────────────────────────────────────────────────

@app.after_request
def add_no_cache(response):
    """Prevent browser caching so changes show up immediately."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/settings", methods=["GET"])
def get_settings():
    key = get_api_key()
    has_key = bool(key and len(key) > 10)
    preview = ""
    if has_key:
        preview = key[:7] + "..." + key[-4:]
    return jsonify({"has_key": has_key, "key_preview": preview})


@app.route("/settings", methods=["POST"])
def update_settings():
    data = request.get_json()
    key = data.get("api_key", "").strip()
    config = load_config()
    config["openai_api_key"] = key
    save_config_file(config)
    return jsonify({"ok": True})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json()
    url = data.get("url", "").strip()
    fmt = data.get("format", "text")
    language = data.get("language", "en")
    label_speakers = data.get("speakers", False)

    if not url:
        return jsonify({"error": "Please enter a YouTube or Spotify episode URL."})

    # ── Spotify episode handling ──
    if is_spotify_url(url):
        spotify_id = extract_spotify_id(url)
        if not spotify_id:
            return jsonify({"error": "Couldn't read that Spotify link. Try copying it again."})

        meta = get_spotify_metadata(spotify_id)
        if not meta["title"]:
            return jsonify({"error": "Couldn't fetch episode info from Spotify."})

        # Search YouTube for a matching video
        yt_id, yt_title = find_youtube_match(meta["title"])
        if not yt_id:
            return jsonify({
                "error": f"Couldn't find \"{meta['title']}\" on YouTube. "
                         "This episode may not be available there. "
                         "Try pasting the YouTube link directly if you can find it."
            })

        # Redirect to the normal YouTube flow with the found video ID
        url = f"https://www.youtube.com/watch?v={yt_id}"
        spotify_match = yt_title or meta["title"]
        # Fall through to normal YouTube transcription below
    else:
        spotify_match = None

    try:
        video_id = extract_video_id(url)
    except SystemExit:
        return jsonify({"error": "That doesn't look like a valid YouTube or Spotify URL."})

    # Try captions first
    try:
        transcript = fetch_captions(video_id, language, fmt)
    except Exception as e:
        error_name = type(e).__name__
        if error_name in ("RequestBlocked", "IpBlocked"):
            return jsonify({
                "error": "YouTube is temporarily blocking automatic requests from your network. "
                         "But don't worry \u2014 you can still get the transcript manually (see below).",
                "blocked": True
            })
        return jsonify({"error": f"Error fetching captions: {error_name}"})

    if transcript:
        speakers = []
        if label_speakers and fmt == "text":
            transcript, speakers = add_speaker_labels(transcript)

        # Also fetch raw timestamps for the timeline
        timestamps = []
        try:
            ytt = YouTubeTranscriptApi()
            raw = ytt.fetch(video_id, languages=[language] if language == "en" else [language, "en"])
            timestamps = [{"start": s.start, "text": s.text.replace("\n", " ")} for s in raw.snippets]
        except Exception:
            pass  # Timestamps are optional — don't fail the request

        resp = {
            "transcript": transcript, "source": "captions",
            "speakers": speakers, "timestamps": timestamps, "video_id": video_id
        }
        if spotify_match:
            resp["spotify_match"] = spotify_match
        return jsonify(resp)

    # Try Whisper fallback if API key is available
    api_key = get_api_key()
    if api_key and yt_dlp and openai:
        try:
            client = openai.OpenAI(api_key=api_key)
            with tempfile.TemporaryDirectory() as tmpdir:
                ydl_opts = {
                    "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/worst[ext=mp4]/worst",
                    "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
                    "quiet": True,
                    "no_warnings": True,
                }
                yt_url = f"https://www.youtube.com/watch?v={video_id}"
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(yt_url, download=True)
                    audio_path = ydl.prepare_filename(info)

                file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                if file_size_mb > 25:
                    return jsonify({
                        "error": f"Audio file is {file_size_mb:.0f} MB (max 25 MB for Whisper). Try a shorter video."
                    })

                whisper_fmt = {
                    "text": "text", "srt": "srt", "vtt": "vtt", "json": "verbose_json"
                }.get(fmt, "text")

                with open(audio_path, "rb") as f:
                    result = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format=whisper_fmt,
                    )

                if whisper_fmt == "verbose_json":
                    if hasattr(result, "model_dump"):
                        transcript = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
                    else:
                        transcript = json.dumps(result, indent=2, ensure_ascii=False)
                else:
                    transcript = result if isinstance(result, str) else result.text

                return jsonify({"transcript": transcript, "source": "whisper", "speakers": []})

        except Exception as e:
            # Strip ANSI color codes from yt-dlp error messages
            error_msg = re.sub(r'\x1b\[[0-9;]*m', '', str(e))[:200]
            if "403" in error_msg or "Forbidden" in error_msg:
                return jsonify({
                    "error": "YouTube is blocking downloads from your network. "
                             "But you can still get the transcript manually (see below).",
                    "blocked": True
                })
            return jsonify({"error": f"Whisper transcription failed: {error_msg}"})

    # No captions and no Whisper available
    if api_key:
        return jsonify({
            "error": "No captions found and audio download was blocked. "
                     "But you can still get the transcript manually (see below).",
            "blocked": True
        })

    return jsonify({
        "error": "No captions available for this video. "
                 "Add your OpenAI API key in Settings for AI transcription, "
                 "or grab the transcript manually from YouTube (see below).",
        "blocked": True
    })


@app.route("/upload", methods=["POST"])
def upload_audio():
    """Transcribe an uploaded audio file using Whisper."""
    api_key = get_api_key()
    if not api_key:
        return jsonify({"error": "Please add your OpenAI API key in Settings first."})

    if openai is None:
        return jsonify({"error": "OpenAI package not installed. Run: pip3 install openai"})

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."})

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected."})

    # Check file size (25 MB limit for Whisper)
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > 25 * 1024 * 1024:
        return jsonify({"error": f"File is {size / (1024*1024):.0f} MB (max 25 MB for Whisper)."})

    # Save to temp file and send to Whisper
    try:
        client = openai.OpenAI(api_key=api_key)
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1], delete=False) as tmp:
            file.save(tmp)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text",
                )

            transcript = result if isinstance(result, str) else result.text
            return jsonify({"transcript": transcript, "source": "whisper", "speakers": []})
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        error_msg = str(e)[:200]
        return jsonify({"error": f"Whisper transcription failed: {error_msg}"})


@app.route("/metadata", methods=["POST"])
def metadata():
    """Fetch video title, channel name, and description from YouTube."""
    if yt_dlp is None:
        return jsonify({"title": "", "channel": "", "description": ""})

    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"title": "", "channel": "", "description": ""})

    # Handle Spotify URLs — use oEmbed for metadata
    if is_spotify_url(url):
        spotify_id = extract_spotify_id(url)
        if spotify_id:
            meta = get_spotify_metadata(spotify_id)
            return jsonify({
                "title": meta.get("title", ""),
                "channel": "Spotify",
                "description": "",
                "thumbnail_url": meta.get("thumbnail_url", ""),
            })
        return jsonify({"title": "", "channel": "", "description": ""})

    try:
        video_id = extract_video_id(url)
        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        yt_url = f"https://www.youtube.com/watch?v={video_id}"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(yt_url, download=False)
            return jsonify({
                "title": info.get("title", ""),
                "channel": info.get("channel") or info.get("uploader", ""),
                "description": (info.get("description") or "")[:1000],
            })
    except Exception:
        return jsonify({"title": "", "channel": "", "description": ""})


@app.route("/cleanup", methods=["POST"])
def cleanup():
    if openai is None:
        return jsonify({"error": "OpenAI package not installed. Run: pip3 install openai"})

    api_key = get_api_key()
    if not api_key:
        return jsonify({"error": "No API key configured. Click the gear icon to add your OpenAI API key."})

    data = request.get_json()
    transcript = data.get("transcript", "")
    video_title = data.get("video_title", "")
    video_channel = data.get("video_channel", "")
    video_description = data.get("video_description", "")
    speaker_names = data.get("speaker_names", "")

    if not transcript.strip():
        return jsonify({"error": "No transcript to clean up."})

    try:
        client = openai.OpenAI(api_key=api_key)

        # Limit input length to stay within practical token limits
        max_chars = 80000
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars]

        # Build context about the video
        video_context = ""
        if video_title or video_channel:
            video_context = "\nVIDEO CONTEXT:\n"
            if video_title:
                video_context += f"- Video title: {video_title}\n"
            if video_channel:
                video_context += f"- Channel/uploader: {video_channel}\n"
            if video_description:
                video_context += f"- Description excerpt: {video_description[:500]}\n"
            video_context += "Use this context to help identify speakers and understand the topic.\n"

        # Build speaker name instructions
        speaker_instruction = ""
        if speaker_names.strip():
            speaker_instruction = (
                f"\nSPEAKER NAMES PROVIDED BY USER: {speaker_names}\n"
                "Use these exact names as speaker labels. Match them to the correct speakers "
                "based on context (e.g., the channel owner is likely the host).\n"
            )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional transcript editor. Clean up this YouTube video transcript.\n"
                        + video_context
                        + speaker_instruction
                        + "\nSPEAKER IDENTIFICATION (very important):\n"
                        "- Search the ENTIRE transcript for speaker names. Look for:\n"
                        "  * Introductions: \"I'm John\", \"my name is Sarah\", \"this is Mike\"\n"
                        "  * Greetings: \"Hey John\", \"Thanks Sarah\", \"Welcome back Mike\"\n"
                        "  * References: \"as John said\", \"John mentioned\"\n"
                        "  * The channel name or video title often reveals the host's name\n"
                        "- Replace generic labels (Speaker 1, Speaker 2) with actual names you find\n"
                        "- If you cannot find names, use descriptive labels like Host, Guest, Interviewer, etc.\n"
                        "- Format speaker labels exactly like this: **Name:** followed by their text\n\n"
                        "TEXT CLEANUP:\n"
                        "1. Fix all punctuation, capitalization, and grammar\n"
                        "2. Organize into clear paragraphs (separate with blank lines)\n"
                        "3. Remove filler words (um, uh, like, you know) unless they add meaning\n"
                        "4. Fix obvious transcription errors where you can confidently guess the intended word\n"
                        "5. Preserve the exact meaning and content \u2014 do not add or remove information\n\n"
                        "KEY POINTS SUMMARY:\n"
                        "After the cleaned transcript, add a section starting with exactly this marker on its own line:\n"
                        "===KEY_POINTS===\n"
                        "Then list 4-7 key takeaways or main points from the conversation as bullet points.\n"
                        "Each bullet should start with '- ' and be a concise one-sentence insight.\n"
                        "Focus on the most important ideas, revelations, or actionable takeaways.\n\n"
                        "RESPONSE FORMAT:\n"
                        "1. The cleaned transcript (with **Speaker:** labels)\n"
                        "2. Then the ===KEY_POINTS=== marker\n"
                        "3. Then the bullet-point summary\n"
                        "No other commentary."
                    )
                },
                {
                    "role": "user",
                    "content": transcript
                }
            ],
            temperature=0.3,
            max_tokens=16000,
        )

        raw_output = response.choices[0].message.content

        # Split out the key points summary if present
        summary = ""
        cleaned = raw_output
        if "===KEY_POINTS===" in raw_output:
            parts = raw_output.split("===KEY_POINTS===", 1)
            cleaned = parts[0].strip()
            summary = parts[1].strip()

        # Extract speaker names from **Name:** patterns in the cleaned text
        speakers = list(dict.fromkeys(re.findall(r'\*\*([^*]+):\*\*', cleaned)))

        return jsonify({"transcript": cleaned, "speakers": speakers, "summary": summary})

    except Exception as e:
        error_name = type(e).__name__
        error_msg = str(e)[:200]

        if "AuthenticationError" in error_name:
            return jsonify({"error": "Invalid API key. Please check your key in Settings."})
        elif "RateLimitError" in error_name:
            return jsonify({"error": "API rate limit reached. Please try again in a moment."})
        else:
            return jsonify({"error": f"AI cleanup failed: {error_msg}"})


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Audio Transcriber is running!")
    print("  Open your browser to: http://localhost:8080\n")

    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:8080")).start()

    app.run(debug=False, port=8080)
