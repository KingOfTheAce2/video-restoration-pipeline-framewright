#!/usr/bin/env python
"""Launch FrameWright UI development server."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from framewright.ui import launch_ui

if __name__ == "__main__":
    print("\nStarting FrameWright UI Development Server...")
    print("=" * 60)

    # Launch on all interfaces so it's accessible
    launch_ui(
        share=False,
        server_port=7860,
        server_name="0.0.0.0",  # Accessible from network
    )
