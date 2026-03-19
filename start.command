#!/bin/bash
cd "$(dirname "$0")"
echo ""
echo "  Starting YouTube Transcriber..."
echo "  (Your browser will open automatically)"
echo ""
echo "  To stop: close this window or press Ctrl+C"
echo ""
python3 web.py
