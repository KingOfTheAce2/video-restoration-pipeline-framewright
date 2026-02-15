"""Model management commands for AI models."""

import sys
from pathlib import Path


def models_list_command(args):
    """List all available AI models.

    Shows installed models, download status, and versions.

    Args:
        args: Parsed command-line arguments
    """
    from ..cli import print_colored, Colors
    from ..utils.model_manager import ModelManager

    print_colored("\n=== Available Models ===\n", Colors.HEADER)

    manager = ModelManager()
    models = manager.list_models()

    if not models:
        print_colored("No models found.", Colors.WARNING)
        print_colored("Download models with: framewright models download <model-name>", Colors.OKCYAN)
        return

    for model in models:
        status = "✓ Installed" if model.is_installed else "✗ Not installed"
        color = Colors.OKGREEN if model.is_installed else Colors.WARNING

        print_colored(f"\n{model.name}:", Colors.OKBLUE)
        print_colored(f"  Status: {status}", color)
        print_colored(f"  Type: {model.model_type}", Colors.OKGREEN)

        if model.is_installed:
            print_colored(f"  Version: {model.version}", Colors.OKGREEN)
            print_colored(f"  Size: {model.size_mb:.1f} MB", Colors.OKGREEN)
            print_colored(f"  Path: {model.path}", Colors.OKCYAN)


def models_download_command(args):
    """Download AI model weights.

    Args:
        args: Parsed command-line arguments with model name
    """
    from ..cli import print_colored, Colors
    from ..utils.model_manager import ModelManager

    if not hasattr(args, 'model_name') or not args.model_name:
        print_colored("Error: Please specify a model name", Colors.FAIL)
        print_colored("Usage: framewright models download <model-name>", Colors.OKCYAN)
        sys.exit(1)

    model_name = args.model_name

    print_colored(f"\n=== Downloading Model: {model_name} ===\n", Colors.HEADER)

    manager = ModelManager()

    # Check if model exists
    if not manager.model_exists(model_name):
        print_colored(f"Error: Unknown model: {model_name}", Colors.FAIL)
        print_colored("List available models with: framewright models list", Colors.OKCYAN)
        sys.exit(1)

    # Check if already downloaded
    if manager.is_model_installed(model_name):
        print_colored(f"Model '{model_name}' is already installed.", Colors.WARNING)

        if not getattr(args, 'force', False):
            print_colored("Use --force to re-download.", Colors.OKCYAN)
            return

        print_colored("Re-downloading due to --force flag...", Colors.OKCYAN)

    # Download with progress
    def progress_callback(progress: float):
        percent = int(progress * 100)
        print(f"\rProgress: [{('=' * (percent // 2)).ljust(50)}] {percent}%", end='', flush=True)

    try:
        success = manager.download_model(model_name, progress_callback=progress_callback)

        print()  # New line after progress bar

        if success:
            print_colored(f"\n✓ Model '{model_name}' downloaded successfully!", Colors.OKGREEN)
        else:
            print_colored(f"\n✗ Failed to download model '{model_name}'", Colors.FAIL)
            sys.exit(1)

    except Exception as e:
        print()
        print_colored(f"\n✗ Error downloading model: {e}", Colors.FAIL)
        sys.exit(1)


def models_verify_command(args):
    """Verify integrity of downloaded models.

    Checks file hashes and validates model files.

    Args:
        args: Parsed command-line arguments (optional: specific model)
    """
    from ..cli import print_colored, Colors
    from ..utils.model_manager import ModelManager

    print_colored("\n=== Verifying Models ===\n", Colors.HEADER)

    manager = ModelManager()

    # Verify specific model or all models
    if hasattr(args, 'model_name') and args.model_name:
        models_to_verify = [args.model_name]
    else:
        models_to_verify = manager.get_installed_models()

    if not models_to_verify:
        print_colored("No models to verify.", Colors.WARNING)
        return

    all_valid = True

    for model_name in models_to_verify:
        print_colored(f"\nVerifying {model_name}...", Colors.OKCYAN)

        is_valid, error_msg = manager.verify_model(model_name)

        if is_valid:
            print_colored(f"  ✓ Valid", Colors.OKGREEN)
        else:
            print_colored(f"  ✗ Invalid: {error_msg}", Colors.FAIL)
            all_valid = False

    if all_valid:
        print_colored("\n✓ All models verified successfully!", Colors.OKGREEN)
    else:
        print_colored("\n✗ Some models failed verification", Colors.FAIL)
        print_colored("Re-download failed models with: framewright models download <model-name> --force", Colors.OKCYAN)
        sys.exit(1)


def models_path_command(args):
    """Show paths where models are stored.

    Args:
        args: Parsed command-line arguments
    """
    from ..cli import print_colored, Colors
    from ..utils.model_manager import ModelManager

    print_colored("\n=== Model Storage Paths ===\n", Colors.HEADER)

    manager = ModelManager()
    paths = manager.get_model_paths()

    for model_type, path in paths.items():
        print_colored(f"{model_type}:", Colors.OKBLUE)
        print_colored(f"  {path}", Colors.OKGREEN)

        if path.exists():
            # Count models in directory
            model_files = list(path.glob("*.pth")) + list(path.glob("*.pt"))
            print_colored(f"  Files: {len(model_files)}", Colors.OKCYAN)
        else:
            print_colored(f"  (directory not created yet)", Colors.WARNING)
