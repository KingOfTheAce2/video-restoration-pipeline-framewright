#!/usr/bin/env python
"""Launch FrameWright Interactive Wizard."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import os
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # Try to enable VT100 mode for rich colors
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        pass

from framewright._ui_pkg.wizard import run_wizard

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    input_path = Path(input_file) if input_file else None

    result = run_wizard(input_path=input_path)

    if result.cancelled:
        print("\nWizard cancelled.")
        sys.exit(0)

    if result.completed:
        print("\n" + "=" * 60)
        print("WIZARD COMPLETE!")
        print("=" * 60)
        print(f"\nConfiguration saved. Ready to process:")
        print(f"  Input:  {result.input_path}")
        print(f"  Output: {result.output_path}")
        print(f"  Preset: {result.preset}")
        print(f"  Scale:  {result.scale_factor}x")
        print("\nTo start restoration, run:")
        print(f"  python -m framewright restore \"{result.input_path}\"")
        print("\nOr import and use the settings:")
        print("  from framewright.config import Config")
        print("  config = Config(**result.to_config_dict())")
