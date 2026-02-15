"""Frame processing mixin for VideoRestorer.

Contains all frame enhancement logic including:
- Backend detection and selection
- Single frame enhancement (PyTorch and ncnn-vulkan)
- Parallel and sequential batch processing
- VRAM optimization and tile sizing
- Error handling and retry logic
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class FrameProcessingMixin:
    """Mixin providing frame enhancement capabilities to VideoRestorer.

    This mixin encapsulates all frame processing logic, keeping it separate
    from video assembly and core restoration orchestration.

    Methods included:
    - _get_ncnn_vulkan_binary: Locate ncnn-vulkan binary
    - _get_enhancement_backend: Select PyTorch vs ncnn-vulkan
    - _enhance_single_frame: Enhance one frame (dispatches to backend)
    - _enhance_single_frame_pytorch: PyTorch/CUDA enhancement
    - _enhance_single_frame_ncnn: ncnn-vulkan enhancement
    - enhance_frames: Main orchestrator for batch enhancement
    - _enhance_frames_sequential: Sequential processing
    - _enhance_frames_parallel: Parallel multi-threaded processing
    """

    def _get_ncnn_vulkan_binary(self) -> Optional[Path]:
        """Get the path to the ncnn-vulkan binary.

        Searches in order:
        1. System PATH
        2. ~/.framewright/bin/
        3. Project bin/ directory

        Returns:
            Path to binary or None if not found
        """
        from ..processors.ncnn_vulkan import get_ncnn_vulkan_path

        ncnn_path = get_ncnn_vulkan_path()
        if ncnn_path:
            return ncnn_path

        # Fallback: check if it's in PATH directly
        binary = shutil.which("realesrgan-ncnn-vulkan")
        if binary:
            return Path(binary)

        return None

    def _get_enhancement_backend(self) -> str:
        """Determine which enhancement backend to use.

        Order of preference:
        1. FRAMEWRIGHT_BACKEND environment variable (explicit override)
        2. PyTorch if available and ncnn-vulkan is not
        3. ncnn-vulkan (default if available)
        4. PyTorch as fallback

        Returns:
            'pytorch' or 'ncnn-vulkan'
        """
        from ..processors.pytorch_realesrgan import is_pytorch_esrgan_available
        from ..processors.ncnn_vulkan import is_ncnn_vulkan_available

        # Check environment variable for explicit override
        env_backend = os.environ.get("FRAMEWRIGHT_BACKEND", "").lower()
        if env_backend == "pytorch":
            if is_pytorch_esrgan_available():
                logger.info("Using PyTorch backend (FRAMEWRIGHT_BACKEND=pytorch)")
                return "pytorch"
            else:
                logger.warning("FRAMEWRIGHT_BACKEND=pytorch but PyTorch Real-ESRGAN not installed")
        elif env_backend == "ncnn-vulkan" or env_backend == "ncnn":
            if is_ncnn_vulkan_available():
                return "ncnn-vulkan"
            else:
                logger.warning("FRAMEWRIGHT_BACKEND=ncnn-vulkan but ncnn-vulkan not installed")

        # Auto-detect: prefer ncnn-vulkan if available, otherwise PyTorch
        if is_ncnn_vulkan_available():
            return "ncnn-vulkan"
        elif is_pytorch_esrgan_available():
            logger.info("ncnn-vulkan not found, using PyTorch backend")
            return "pytorch"

        # Neither available - will fail later with helpful error
        return "ncnn-vulkan"

    def _enhance_single_frame(
        self,
        input_path: Path,
        output_dir: Path,
        tile_size: int
    ) -> Tuple[Path, bool, Optional[str]]:
        """Enhance a single frame with Real-ESRGAN.

        Supports two backends:
        - PyTorch: Uses CUDA (best for cloud/Docker with NVIDIA GPUs)
        - ncnn-vulkan: Uses Vulkan API (supports AMD/Intel/NVIDIA)

        Backend is selected automatically or via FRAMEWRIGHT_BACKEND env var.

        Args:
            input_path: Path to input frame
            output_dir: Directory for output
            tile_size: Tile size for processing

        Returns:
            Tuple of (output_path, success, error_message)

        Note:
            Includes CPU fallback detection when require_gpu=True to prevent
            runaway CPU usage that can freeze the system.
        """
        output_path = output_dir / input_path.name
        backend = self._get_enhancement_backend()

        if backend == "pytorch":
            return self._enhance_single_frame_pytorch(input_path, output_path, tile_size)
        else:
            return self._enhance_single_frame_ncnn(input_path, output_dir, tile_size)

    def _enhance_single_frame_pytorch(
        self,
        input_path: Path,
        output_path: Path,
        tile_size: int
    ) -> Tuple[Path, bool, Optional[str]]:
        """Enhance a single frame using PyTorch Real-ESRGAN (CUDA).

        Args:
            input_path: Path to input frame
            output_path: Path for output frame
            tile_size: Tile size for processing (0 = auto)

        Returns:
            Tuple of (output_path, success, error_message)
        """
        from ..processors.pytorch_realesrgan import (
            is_pytorch_esrgan_available,
            enhance_frame_pytorch,
            PyTorchESRGANConfig,
            convert_ncnn_model_name,
        )
        from ..validators import validate_frame_integrity

        if not is_pytorch_esrgan_available():
            return output_path, False, (
                "PyTorch Real-ESRGAN not available. Install with:\n"
                "  pip install realesrgan basicsr"
            )

        # Convert ncnn model name to PyTorch model name
        pytorch_model = convert_ncnn_model_name(self.config.model_name)

        config = PyTorchESRGANConfig(
            model_name=pytorch_model,
            scale_factor=self.config.scale_factor,
            tile_size=tile_size,
            gpu_id=self.config.gpu_id if self.config.gpu_id is not None else 0,
        )

        success, error = enhance_frame_pytorch(input_path, output_path, config)

        if success:
            # Validate output
            validation = validate_frame_integrity(output_path)
            if not validation.is_valid:
                return output_path, False, validation.error_message

        return output_path, success, error

    def _enhance_single_frame_ncnn(
        self,
        input_path: Path,
        output_dir: Path,
        tile_size: int
    ) -> Tuple[Path, bool, Optional[str]]:
        """Enhance a single frame with Real-ESRGAN using ncnn-vulkan.

        Automatically finds the ncnn-vulkan binary in common locations,
        supporting AMD, Intel, and NVIDIA GPUs via Vulkan.

        Args:
            input_path: Path to input frame
            output_dir: Directory for output
            tile_size: Tile size for processing

        Returns:
            Tuple of (output_path, success, error_message)

        Note:
            Includes CPU fallback detection when require_gpu=True to prevent
            runaway CPU usage that can freeze the system.
        """
        from ..validators import validate_frame_integrity
        from ..errors import classify_error

        output_path = output_dir / input_path.name

        # Find the ncnn-vulkan binary
        ncnn_binary = self._get_ncnn_vulkan_binary()
        if not ncnn_binary:
            return output_path, False, (
                "realesrgan-ncnn-vulkan not found. "
                "Install it with: python -c \"from framewright.processors.ncnn_vulkan import install_ncnn_vulkan; install_ncnn_vulkan()\""
            )

        # Get model directory (bundled with ncnn-vulkan)
        model_dir = ncnn_binary.parent / "models"

        cmd = [
            str(ncnn_binary),
            '-i', str(input_path),
            '-o', str(output_path),
            '-n', self.config.model_name,
            '-s', str(self.config.scale_factor),
            '-f', 'png'
        ]

        # Add model path if it exists
        if model_dir.exists():
            cmd.extend(['-m', str(model_dir)])

        if tile_size > 0:
            cmd.extend(['-t', str(tile_size)])

        # GPU/CPU mode selection
        if self.config.require_gpu:
            gpu_id = self.config.gpu_id if self.config.gpu_id is not None else 0
            cmd.extend(['-g', str(gpu_id)])
            logger.debug(f"Explicit GPU selection: GPU {gpu_id}")
        else:
            # Force CPU mode with -g -1
            cmd.extend(['-g', '-1'])
            logger.debug("Forcing CPU mode (-g -1)")

        # CPU fallback indicators to detect if ncnn-vulkan falls back to CPU
        cpu_fallback_indicators = [
            "using cpu", "no vulkan device", "vulkan not found",
            "failed to create gpu instance", "cpu mode", "fallback to cpu"
        ]

        try:
            # Use CREATE_NO_WINDOW on Windows to avoid console popups
            creationflags = 0
            if hasattr(subprocess, 'CREATE_NO_WINDOW'):
                creationflags = subprocess.CREATE_NO_WINDOW

            # Set environment variables for Vulkan compatibility
            env = os.environ.copy()
            # Fix for AMD switchable graphics causing vkEnumeratePhysicalDevices to fail
            env['DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1'] = '1'
            # Disable problematic AMD switchable graphics Vulkan layer
            env['VK_LOADER_LAYERS_DISABLE'] = 'VK_LAYER_AMD_switchable_graphics'
            # Alternative: disable all implicit layers that might interfere
            env['VK_LOADER_LAYERS_ENABLE'] = ''
            # Ensure Vulkan finds the correct GPU
            env['VK_ICD_FILENAMES'] = env.get('VK_ICD_FILENAMES', '')

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per frame
                creationflags=creationflags,
                env=env,
            )

            # Check for CPU fallback in output when GPU is required
            if self.config.require_gpu:
                combined_output = f"{result.stdout or ''} {result.stderr or ''}".lower()
                for indicator in cpu_fallback_indicators:
                    if indicator in combined_output:
                        # Delete output if created (don't trust CPU-processed results)
                        if output_path.exists():
                            output_path.unlink()
                        return output_path, False, (
                            f"CPU fallback detected: '{indicator}'. "
                            "Processing would use CPU instead of GPU, which can freeze your system. "
                            "Check GPU drivers, Vulkan installation, or set require_gpu=False."
                        )

            # Validate output
            if not output_path.exists():
                return output_path, False, "Output file not created"

            validation = validate_frame_integrity(output_path)
            if not validation.is_valid:
                return output_path, False, validation.error_message

            return output_path, True, None

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or str(e)

            # Check for CPU fallback in error output
            if self.config.require_gpu:
                error_lower = error_msg.lower()
                for indicator in cpu_fallback_indicators:
                    if indicator in error_lower:
                        return output_path, False, (
                            f"CPU fallback detected in error: '{indicator}'. "
                            "GPU processing failed and would fall back to CPU."
                        )

            error_class = classify_error(e, e.stderr)
            return output_path, False, f"{error_class.__name__}: {error_msg}"
        except subprocess.TimeoutExpired:
            if self.config.require_gpu:
                return output_path, False, (
                    "Enhancement timed out (>5 min). "
                    "This may indicate CPU fallback causing extremely slow processing."
                )
            return output_path, False, "Enhancement timed out"

    def enhance_frames(self) -> int:
        """Enhance all extracted frames using Real-ESRGAN.

        Supports checkpointing, retry logic, and parallel processing.
        Uses ThreadPoolExecutor for concurrent frame enhancement when
        parallel_frames > 1.

        When deduplication is enabled, enhances only unique frames from
        unique_frames_dir instead of all frames from frames_dir.

        Returns:
            Number of frames enhanced

        Raises:
            EnhancementError: If frame enhancement fails
        """
        from ..errors import EnhancementError, ErrorReport
        from ..utils.gpu import get_adaptive_tile_sequence

        # Use unique_frames_dir if deduplication was performed
        if self._dedup_result is not None and self.config.unique_frames_dir.exists():
            source_dir = self.config.unique_frames_dir
            logger.info(
                f"Using deduplicated frames: {self._dedup_result.unique_frames} unique "
                f"(from {self._dedup_result.total_frames} total)"
            )
        else:
            source_dir = self.config.frames_dir

        frames = sorted(source_dir.glob("frame_*.png"))
        total_frames = len(frames)

        if total_frames == 0:
            raise EnhancementError("No frames found to enhance")

        logger.info(f"Enhancing {total_frames} frames using {self.config.model_name}")

        # Check for resume from checkpoint
        if self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint and checkpoint.stage == "enhance":
                frames = self.checkpoint_manager.get_unprocessed_frames(frames)
                logger.info(f"Resuming enhancement: {len(frames)} frames remaining")

        if not frames:
            logger.info("All frames already enhanced")
            return total_frames

        # Prepare output directory
        if not self.config.enhanced_dir.exists():
            self.config.enhanced_dir.mkdir(parents=True)

        # Update checkpoint stage
        if self.checkpoint_manager:
            self.checkpoint_manager.update_stage("enhance")

        # Get initial tile size
        tile_size = self._get_tile_size()
        tile_sequence = get_adaptive_tile_sequence(
            frame_resolution=(self.metadata.get('width', 1920), self.metadata.get('height', 1080)),
            scale_factor=self.config.scale_factor,
        )

        self._update_progress(
            stage="enhance_frames",
            progress=0.0,
            frames_completed=0,
            frames_total=len(frames),
        )
        error_report = ErrorReport(total_operations=len(frames))

        # Determine processing mode
        num_workers = self.config.parallel_frames
        use_parallel = num_workers > 1 and len(frames) > 1

        if use_parallel:
            logger.info(f"Using parallel processing with {num_workers} workers")
            enhanced_count = self._enhance_frames_parallel(
                frames, tile_size, tile_sequence, error_report
            )
        else:
            logger.info("Using sequential processing")
            enhanced_count = self._enhance_frames_sequential(
                frames, tile_size, tile_sequence, error_report
            )

        # Final count from directory
        final_count = len(list(self.config.enhanced_dir.glob("*.png")))

        if final_count == 0:
            raise EnhancementError("No enhanced frames were created")

        self._update_progress(
            stage="enhance_frames",
            progress=1.0,
            frames_completed=final_count,
            frames_total=final_count,
            eta_seconds=0.0,
        )
        logger.info(f"Enhanced {final_count} frames ({error_report.summary()})")

        # Store error report
        self._error_report = error_report

        return final_count

    def _enhance_frames_sequential(
        self,
        frames: List[Path],
        tile_size: int,
        tile_sequence: List[int],
        error_report: 'ErrorReport',
    ) -> int:
        """Enhance frames sequentially (original behavior).

        Args:
            frames: List of frame paths to enhance
            tile_size: Initial tile size
            tile_sequence: Sequence of fallback tile sizes
            error_report: Error report to update

        Returns:
            Number of successfully enhanced frames
        """
        from ..errors import EnhancementError

        tile_index = 0
        enhanced_count = 0
        total_frames = len(frames)

        # Reset timing for this stage
        self._reset_stage_timing("enhance_frames")

        for i, frame_path in enumerate(frames):
            frame_start_time = time.time()
            retry_count = 0
            success = False

            while retry_count <= self.config.max_retries and not success:
                output_path, success, error_msg = self._enhance_single_frame(
                    frame_path,
                    self.config.enhanced_dir,
                    tile_size
                )

                if not success:
                    # Check if VRAM error
                    if error_msg and ("vram" in error_msg.lower() or "memory" in error_msg.lower()):
                        # Try smaller tile size
                        tile_index += 1
                        if tile_index < len(tile_sequence):
                            tile_size = tile_sequence[tile_index]
                            logger.info(f"VRAM error, reducing tile size to {tile_size}")
                            retry_count += 1
                            time.sleep(self.config.retry_delay)
                            continue
                        else:
                            logger.error("Exhausted tile size options")
                            break

                    retry_count += 1
                    if retry_count <= self.config.max_retries:
                        delay = self.config.retry_delay * (2 ** retry_count)
                        logger.warning(f"Frame {frame_path.name} failed, retrying in {delay}s...")
                        time.sleep(delay)

            # Record frame processing time for ETA calculation
            frame_time = time.time() - frame_start_time
            self._record_frame_time(frame_time)

            if success:
                enhanced_count += 1
                error_report.add_success()

                # Update checkpoint
                if self.checkpoint_manager:
                    frame_num = int(frame_path.stem.split("_")[-1])
                    self.checkpoint_manager.update_frame(
                        frame_number=frame_num,
                        input_path=frame_path,
                        output_path=output_path,
                    )
            else:
                error_report.add_error(
                    frame_path.name,
                    EnhancementError(error_msg or "Unknown error"),
                )
                if not self.config.continue_on_error:
                    raise EnhancementError(
                        f"Failed to enhance frame {frame_path.name}: {error_msg}"
                    )
                else:
                    # Copy original frame to output when enhancement fails
                    try:
                        shutil.copy2(frame_path, output_path)
                        logger.warning(
                            f"Frame {frame_path.name} enhancement failed, using original. "
                            f"Error: {error_msg}"
                        )
                        enhanced_count += 1  # Count as processed (with original)
                    except Exception as copy_err:
                        logger.error(f"Could not copy original frame: {copy_err}")

            # Update progress with frame counts for ETA calculation
            frames_completed = i + 1
            progress = frames_completed / total_frames
            self._update_progress(
                stage="enhance_frames",
                progress=progress,
                frames_completed=frames_completed,
                frames_total=total_frames,
            )

            # Check disk space periodically
            if i % 100 == 0:
                self._check_disk_space()

            # Sample VRAM usage
            if self._vram_monitor and i % 10 == 0:
                self._vram_monitor.sample()

        return enhanced_count

    def _enhance_frames_parallel(
        self,
        frames: List[Path],
        tile_size: int,
        tile_sequence: List[int],
        error_report: 'ErrorReport',
    ) -> int:
        """Enhance frames in parallel using ThreadPoolExecutor.

        Provides 2-4x speedup on multi-GPU systems or when GPU is
        not fully utilized.

        Args:
            frames: List of frame paths to enhance
            tile_size: Initial tile size
            tile_sequence: Sequence of fallback tile sizes
            error_report: Error report to update

        Returns:
            Number of successfully enhanced frames
        """
        import threading
        from ..errors import EnhancementError

        num_workers = self.config.parallel_frames
        enhanced_count = 0
        completed = 0
        total_frames = len(frames)
        lock = threading.Lock()
        current_tile_size = tile_size

        # Reset timing for this stage
        self._reset_stage_timing("enhance_frames")

        def process_frame(frame_path: Path) -> Tuple[Path, bool, Optional[str], float]:
            """Process a single frame with retry logic.

            Returns:
                Tuple of (output_path, success, error_message, processing_time)
            """
            nonlocal current_tile_size
            frame_start = time.time()

            retry_count = 0
            while retry_count <= self.config.max_retries:
                output_path, success, error_msg = self._enhance_single_frame(
                    frame_path,
                    self.config.enhanced_dir,
                    current_tile_size
                )

                if success:
                    return output_path, True, None, time.time() - frame_start

                # Check if VRAM error - reduce tile size for all workers
                if error_msg and ("vram" in error_msg.lower() or "memory" in error_msg.lower()):
                    with lock:
                        for smaller_tile in tile_sequence:
                            if smaller_tile < current_tile_size:
                                logger.info(f"VRAM error, reducing tile size to {smaller_tile}")
                                current_tile_size = smaller_tile
                                break
                        else:
                            return output_path, False, "Exhausted tile size options", time.time() - frame_start

                retry_count += 1
                if retry_count <= self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** retry_count)
                    time.sleep(delay)

            return output_path, False, error_msg, time.time() - frame_start

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all frames for processing
            future_to_frame = {
                executor.submit(process_frame, frame): frame
                for frame in frames
            }

            # Collect results as they complete
            for future in as_completed(future_to_frame):
                frame_path = future_to_frame[future]

                try:
                    output_path, success, error_msg, frame_time = future.result()

                    with lock:
                        completed += 1

                        # Record frame processing time for ETA
                        self._record_frame_time(frame_time)

                        progress = completed / total_frames
                        self._update_progress(
                            stage="enhance_frames",
                            progress=progress,
                            frames_completed=completed,
                            frames_total=total_frames,
                        )

                        if success:
                            enhanced_count += 1
                            error_report.add_success()

                            # Update checkpoint
                            if self.checkpoint_manager:
                                frame_num = int(frame_path.stem.split("_")[-1])
                                self.checkpoint_manager.update_frame(
                                    frame_number=frame_num,
                                    input_path=frame_path,
                                    output_path=output_path,
                                )
                        else:
                            error_report.add_error(
                                frame_path.name,
                                EnhancementError(error_msg or "Unknown error"),
                            )
                            if not self.config.continue_on_error:
                                # Cancel remaining futures
                                for f in future_to_frame:
                                    f.cancel()
                                raise EnhancementError(
                                    f"Failed to enhance frame {frame_path.name}: {error_msg}"
                                )
                            else:
                                # Copy original frame to output when enhancement fails
                                try:
                                    shutil.copy2(frame_path, output_path)
                                    logger.warning(
                                        f"Frame {frame_path.name} enhancement failed, using original. "
                                        f"Error: {error_msg}"
                                    )
                                    enhanced_count += 1  # Count as processed
                                except Exception as copy_err:
                                    logger.error(f"Could not copy original frame: {copy_err}")

                        # Check disk space periodically
                        if completed % 100 == 0:
                            self._check_disk_space()

                        # Sample VRAM usage
                        if self._vram_monitor and completed % 10 == 0:
                            self._vram_monitor.sample()

                except Exception as e:
                    if not isinstance(e, EnhancementError):
                        logger.error(f"Unexpected error processing {frame_path.name}: {e}")
                        error_report.add_error(frame_path.name, e)

        return enhanced_count
