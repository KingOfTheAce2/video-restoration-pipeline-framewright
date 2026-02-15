#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  FrameWright Installer"
echo "============================================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[X] Python 3 not found."
    echo "    Install Python 3.9+:"
    echo "      macOS:  brew install python"
    echo "      Ubuntu: sudo apt install python3 python3-venv python3-pip"
    echo "      Fedora: sudo dnf install python3 python3-pip"
    exit 1
fi

PYVER=$(python3 --version 2>&1 | awk '{print $2}')
echo "[OK] Python $PYVER found."

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "[OK] Virtual environment created."
else
    echo "[OK] Virtual environment already exists."
fi

# Activate virtual environment
source .venv/bin/activate
echo "[OK] Virtual environment activated."

# Upgrade pip
echo
echo "Upgrading pip..."
python -m pip install --upgrade pip > /dev/null 2>&1

# Install FrameWright
echo
echo "Installing FrameWright..."
pip install -e ".[full,progress]"
echo "[OK] FrameWright installed."

# Check FFmpeg
echo
if ! command -v ffmpeg &> /dev/null; then
    echo "[!!] FFmpeg not found."
    echo
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "    Install via Homebrew:"
        echo "      brew install ffmpeg"
    elif command -v apt &> /dev/null; then
        echo "    Install via apt:"
        echo "      sudo apt install ffmpeg"
    elif command -v dnf &> /dev/null; then
        echo "    Install via dnf:"
        echo "      sudo dnf install ffmpeg"
    elif command -v pacman &> /dev/null; then
        echo "    Install via pacman:"
        echo "      sudo pacman -S ffmpeg"
    else
        echo "    Install FFmpeg from https://ffmpeg.org/download.html"
    fi
    echo
else
    echo "[OK] FFmpeg found."
fi

# Run hardware check
echo
echo "Running hardware check..."
if python -m framewright --help > /dev/null 2>&1; then
    echo "[OK] FrameWright is ready."
else
    echo "[!!] FrameWright installed but help command returned an error."
    echo "    Try: python -m framewright --help"
fi

echo
echo "============================================================"
echo "  Installation complete!"
echo
echo "  To activate the environment in a new terminal:"
echo "    source .venv/bin/activate"
echo
echo "  Quick start:"
echo "    framewright --help"
echo "    framewright-check"
echo "    framewright video.mp4"
echo "============================================================"
