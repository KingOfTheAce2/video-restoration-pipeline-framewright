"""Hardware detection and benchmarking commands."""

import sys


def list_gpus_command(args):
    """List all available GPUs with details.

    Shows GPU vendor, model, VRAM, and capabilities.

    Args:
        args: Parsed command-line arguments
    """
    from ..cli import print_colored, Colors
    from ..utils.gpu import get_all_gpus_multivendor

    print_colored("\n=== GPU Detection ===\n", Colors.HEADER)

    gpus = get_all_gpus_multivendor()

    if not gpus:
        print_colored("No GPUs detected.", Colors.WARNING)
        print_colored(
            "Make sure GPU drivers are installed and up to date.",
            Colors.WARNING
        )
        return

    for i, gpu in enumerate(gpus):
        print_colored(f"\nGPU {i}:", Colors.OKBLUE)
        print_colored(f"  Name: {gpu.name}", Colors.OKGREEN)
        print_colored(f"  Vendor: {gpu.vendor.value}", Colors.OKGREEN)
        print_colored(f"  VRAM: {gpu.vram_mb:.0f} MB", Colors.OKGREEN)
        print_colored(f"  CUDA: {gpu.cuda_available}", Colors.OKGREEN)
        print_colored(f"  Vulkan: {gpu.vulkan_available}", Colors.OKGREEN)

    print_colored(f"\nTotal GPUs: {len(gpus)}", Colors.OKCYAN)


def check_hardware(args):
    """Check hardware capabilities and compatibility.

    Validates:
    - GPU availability and VRAM
    - CUDA/Vulkan support
    - CPU capabilities
    - Available RAM
    - Disk space

    Args:
        args: Parsed command-line arguments (optional output_file)
    """
    from ..cli import print_colored, Colors
    from ..hardware import HardwareChecker

    print_colored("\n=== Hardware Check ===\n", Colors.HEADER)

    checker = HardwareChecker()
    report = checker.check_all()

    # Print report
    print_colored(report.summary(), Colors.OKGREEN if report.is_suitable else Colors.WARNING)

    # Save to file if requested
    if hasattr(args, 'output_file') and args.output_file:
        output_path = Path(args.output_file)
        output_path.write_text(report.to_json())
        print_colored(f"\nReport saved to: {output_path}", Colors.OKCYAN)


def run_benchmark(args):
    """Run performance benchmark on current hardware.

    Tests:
    - Frame extraction speed
    - Enhancement speed (Real-ESRGAN)
    - Interpolation speed (RIFE)
    - Encoding speed

    Args:
        args: Parsed command-line arguments
    """
    from ..cli import print_colored, Colors
    from ..benchmarks import BenchmarkRunner
    from pathlib import Path

    print_colored("\n=== FrameWright Benchmark ===\n", Colors.HEADER)

    # Determine test video
    test_video = None
    if hasattr(args, 'input') and args.input:
        test_video = Path(args.input)
        if not test_video.exists():
            print_colored(f"Error: Test video not found: {test_video}", Colors.FAIL)
            sys.exit(1)

    # Run benchmark
    runner = BenchmarkRunner()

    print_colored("Running benchmark (this may take 5-10 minutes)...\n", Colors.OKCYAN)

    results = runner.run_full_benchmark(
        test_video=test_video,
        scale_factor=getattr(args, 'scale', 4),
        enable_interpolation=getattr(args, 'interpolate', False),
    )

    # Print results
    print_colored("\n=== Benchmark Results ===\n", Colors.HEADER)
    print_colored(results.summary(), Colors.OKGREEN)

    # Save results if requested
    if hasattr(args, 'output') and args.output:
        output_path = Path(args.output)
        output_path.write_text(results.to_json())
        print_colored(f"\nResults saved to: {output_path}", Colors.OKCYAN)
