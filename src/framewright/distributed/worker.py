"""Render worker for distributed processing."""

import json
import logging
import os
import platform
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import shutil

from .job import JobStatus
from .discovery import NodeDiscovery, NodeInfo, DiscoveryMethod

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for render worker."""
    # Identity
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: Optional[str] = None

    # Network
    coordinator_address: Optional[str] = None
    coordinator_port: int = 8764
    listen_port: int = 8765

    # Discovery
    discovery_method: DiscoveryMethod = DiscoveryMethod.MULTICAST
    announce_interval: float = 10.0

    # Processing
    max_concurrent_chunks: int = 1
    gpu_device: int = 0

    # Storage
    work_dir: Path = field(default_factory=lambda: Path.home() / ".framewright" / "worker")
    cache_dir: Optional[Path] = None

    # Heartbeat
    heartbeat_interval: float = 5.0

    def __post_init__(self):
        self.work_dir = Path(self.work_dir)
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)


class SystemInfo:
    """Gather system information for worker capabilities."""

    @staticmethod
    def get_hostname() -> str:
        """Get system hostname."""
        return socket.gethostname()

    @staticmethod
    def get_cpu_cores() -> int:
        """Get CPU core count."""
        return os.cpu_count() or 1

    @staticmethod
    def get_ram_gb() -> float:
        """Get total RAM in GB."""
        try:
            if platform.system() == "Windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulong = ctypes.c_ulong
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', c_ulong),
                        ('dwMemoryLoad', c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return stat.ullTotalPhys / (1024 ** 3)
            else:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            kb = int(line.split()[1])
                            return kb / (1024 ** 2)
        except Exception:
            pass
        return 8.0  # Default

    @staticmethod
    def get_gpu_info() -> tuple:
        """Get GPU information (count, names, total memory)."""
        try:
            # Try nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                gpus = []
                total_memory = 0.0

                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            gpus.append(parts[0].strip())
                            total_memory += float(parts[1]) / 1024  # MB to GB

                return len(gpus), gpus, total_memory

        except Exception:
            pass

        return 0, [], 0.0

    @staticmethod
    def get_local_ip() -> str:
        """Get local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"


class ChunkProcessor:
    """Processes individual render chunks."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self._ffmpeg_path = shutil.which("ffmpeg")

    def process_chunk(
        self,
        chunk_data: Dict[str, Any],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """Process a chunk and return result."""
        chunk_id = chunk_data["chunk_id"]
        job_id = chunk_data["job_id"]
        input_path = Path(chunk_data["input_path"])
        work_dir = Path(chunk_data["work_dir"])
        frame_start = chunk_data["frame_start"]
        frame_end = chunk_data["frame_end"]
        settings = chunk_data.get("settings", {})
        preset = chunk_data.get("preset", "balanced")

        logger.info(f"Processing chunk {chunk_id}: frames {frame_start}-{frame_end}")

        output_path = work_dir / f"{chunk_id}.mp4"
        start_time = time.time()

        try:
            # Extract frames for this chunk
            frames_processed = self._process_frames(
                input_path,
                output_path,
                frame_start,
                frame_end,
                settings,
                preset,
                progress_callback,
            )

            elapsed = time.time() - start_time
            fps = frames_processed / elapsed if elapsed > 0 else 0

            return {
                "success": True,
                "chunk_id": chunk_id,
                "output_path": str(output_path),
                "frames_processed": frames_processed,
                "elapsed_seconds": elapsed,
                "fps": fps,
                "metrics": {
                    "fps": fps,
                    "elapsed": elapsed,
                },
            }

        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            return {
                "success": False,
                "chunk_id": chunk_id,
                "error": str(e),
            }

    def _process_frames(
        self,
        input_path: Path,
        output_path: Path,
        frame_start: int,
        frame_end: int,
        settings: Dict[str, Any],
        preset: str,
        progress_callback: Optional[Callable[[float], None]],
    ) -> int:
        """Process frames using AI restoration pipeline.

        Extracts frames from the video chunk, applies AI models (Real-ESRGAN,
        denoising, colorization, face restoration), then re-encodes.
        """
        if not self._ffmpeg_path:
            raise RuntimeError("FFmpeg not found")

        import tempfile
        from framewright.config import Config, RestoreOptions
        from framewright.processors import RealESRGANProcessor

        total_frames = frame_end - frame_start + 1
        fps = self._get_fps(input_path)
        start_time = frame_start / fps
        duration = total_frames / fps

        # Create temporary directories for frame extraction and processing
        with tempfile.TemporaryDirectory(prefix="worker_chunk_") as temp_dir:
            temp_path = Path(temp_dir)
            extract_dir = temp_path / "extracted"
            processed_dir = temp_path / "processed"
            extract_dir.mkdir()
            processed_dir.mkdir()

            # Step 1: Extract frames from video chunk
            logger.info(f"Extracting frames {frame_start}-{frame_end} from {input_path}")
            extract_cmd = [
                self._ffmpeg_path,
                "-ss", str(start_time),
                "-i", str(input_path),
                "-t", str(duration),
                "-qscale:v", "1",
                str(extract_dir / "frame_%05d.png")
            ]
            result = subprocess.run(extract_cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"Frame extraction failed: {result.stderr.decode()}")

            # Get extracted frame paths
            frame_files = sorted(extract_dir.glob("frame_*.png"))
            if not frame_files:
                raise RuntimeError("No frames extracted")

            logger.info(f"Extracted {len(frame_files)} frames, processing with AI models...")

            # Step 2: Process each frame with AI restoration
            # NO GRACEFUL FALLBACKS - FAIL HARD IF MODELS UNAVAILABLE
            try:
                from framewright.processors import RealESRGANProcessor
            except ImportError as e:
                raise RuntimeError(
                    f"CRITICAL: RealESRGANProcessor not available. Cannot perform AI restoration.\n"
                    f"Import error: {e}\n"
                    f"Install required dependencies: pip install realesrgan basicsr"
                ) from e

            # Initialize processors based on settings
            processors_pipeline = []

            # Denoising (if enabled)
            if settings.get("denoise", False):
                try:
                    from framewright.processors import TemporalDenoiser
                    denoiser = TemporalDenoiser(strength=settings.get("denoise_strength", 0.5))
                    processors_pipeline.append(("denoise", denoiser))
                except ImportError as e:
                    raise RuntimeError(
                        f"CRITICAL: Denoising requested but TemporalDenoiser not available.\n"
                        f"Import error: {e}\n"
                        f"Either disable denoising or install required dependencies."
                    ) from e

            # Upscaling (Real-ESRGAN) - REQUIRED
            scale_factor = settings.get("scale_factor", 2)
            if scale_factor > 1:
                model_name = settings.get("model", "realesr-general-x4v3")
                try:
                    upscaler = RealESRGANProcessor(
                        model_name=model_name,
                        scale_factor=scale_factor,
                        gpu_device=self.config.gpu_device,
                    )
                    processors_pipeline.append(("upscale", upscaler))
                except Exception as e:
                    raise RuntimeError(
                        f"CRITICAL: Failed to initialize Real-ESRGAN upscaler.\n"
                        f"Model: {model_name}, Scale: {scale_factor}x, GPU: {self.config.gpu_device}\n"
                        f"Error: {e}\n"
                        f"Ensure model is downloaded and GPU is available."
                    ) from e

            # Face restoration (if enabled)
            if settings.get("face_enhance", False):
                try:
                    from framewright.processors import FaceRestorer
                    face_restorer = FaceRestorer(model="gfpgan-v1.4")
                    processors_pipeline.append(("face", face_restorer))
                except ImportError as e:
                    raise RuntimeError(
                        f"CRITICAL: Face enhancement requested but FaceRestorer not available.\n"
                        f"Import error: {e}\n"
                        f"Either disable face_enhance or install GFPGAN: pip install gfpgan"
                    ) from e

            # Colorization (if enabled)
            if settings.get("colorize", False):
                try:
                    from framewright.processors import DDColorProcessor
                    colorizer = DDColorProcessor()
                    processors_pipeline.append(("colorize", colorizer))
                except ImportError as e:
                    raise RuntimeError(
                        f"CRITICAL: Colorization requested but DDColorProcessor not available.\n"
                        f"Import error: {e}\n"
                        f"Either disable colorize or install ddcolor dependencies."
                    ) from e

            # Process frames through the pipeline - FAIL HARD ON ERRORS
            for idx, frame_file in enumerate(frame_files):
                current_frame = frame_file
                output_file = processed_dir / frame_file.name

                # Apply each processor in the pipeline
                for processor_name, processor in processors_pipeline:
                    temp_output = temp_path / f"temp_{processor_name}_{frame_file.name}"
                    try:
                        processor.process_frame(str(current_frame), str(temp_output))
                        current_frame = temp_output
                    except Exception as e:
                        raise RuntimeError(
                            f"CRITICAL: {processor_name.upper()} failed on frame {frame_file.name}\n"
                            f"Frame index: {idx + 1}/{len(frame_files)}\n"
                            f"Input: {current_frame}\n"
                            f"Error: {e}\n"
                            f"AI restoration pipeline ABORTED."
                        ) from e

                # Copy final result
                if current_frame != output_file:
                    shutil.copy(current_frame, output_file)

                if progress_callback:
                    progress_callback((idx + 1) / len(frame_files))

            # Step 3: Re-encode processed frames
            logger.info(f"Re-encoding {len(frame_files)} processed frames...")
            encode_cmd = [
                self._ffmpeg_path,
                "-framerate", str(fps),
                "-i", str(processed_dir / "frame_%05d.png"),
                "-c:v", "libx264",
                "-preset", settings.get("encoding_preset", "medium"),
                "-crf", str(settings.get("crf", 18)),
                "-pix_fmt", "yuv420p",
                "-an",  # No audio for chunks
                "-y",
                str(output_path)
            ]
            result = subprocess.run(encode_cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"Re-encoding failed: {result.stderr.decode()}")

        logger.info(f"Chunk processing complete: {total_frames} frames")
        return total_frames

    def _build_filters(self, settings: Dict[str, Any], preset: str) -> str:
        """Build FFmpeg filter string from settings."""
        filters = []

        # Scaling
        scale = settings.get("scale_factor", 1)
        if scale > 1:
            filters.append(f"scale=iw*{scale}:ih*{scale}:flags=lanczos")

        # Denoising
        if settings.get("denoise", False):
            strength = settings.get("denoise_strength", 0.5)
            filters.append(f"hqdn3d={strength * 4}:{strength * 3}:{strength * 6}:{strength * 4.5}")

        # Sharpening
        if settings.get("sharpen", False):
            strength = settings.get("sharpen_strength", 0.5)
            filters.append(f"unsharp=5:5:{strength}:5:5:{strength * 0.5}")

        # Color correction
        if settings.get("auto_color", False):
            filters.append("eq=saturation=1.1:contrast=1.05")

        return ",".join(filters) if filters else ""

    def _get_fps(self, video_path: Path) -> float:
        """Get video FPS."""
        try:
            ffprobe = self._ffmpeg_path.replace("ffmpeg", "ffprobe")
            cmd = [
                ffprobe,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "csv=p=0",
                str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                fps_str = result.stdout.strip()
                if "/" in fps_str:
                    num, denom = fps_str.split("/")
                    return float(num) / float(denom)
                return float(fps_str)

        except Exception:
            pass

        return 24.0  # Default


class RenderWorker:
    """Render worker node for distributed processing."""

    def __init__(self, config: Optional[WorkerConfig] = None):
        self.config = config or WorkerConfig()
        self.config.work_dir.mkdir(parents=True, exist_ok=True)

        # System info
        self._hostname = self.config.name or SystemInfo.get_hostname()
        self._address = SystemInfo.get_local_ip()
        gpu_count, gpu_names, gpu_memory = SystemInfo.get_gpu_info()

        self._node_info = NodeInfo(
            node_id=self.config.node_id,
            hostname=self._hostname,
            address=self._address,
            port=self.config.listen_port,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            gpu_memory_gb=gpu_memory,
            cpu_cores=SystemInfo.get_cpu_cores(),
            ram_gb=SystemInfo.get_ram_gb(),
        )

        # Discovery
        self.discovery = NodeDiscovery(
            method=self.config.discovery_method,
            announce_interval=self.config.announce_interval,
        )

        # Processor
        self.processor = ChunkProcessor(self.config)

        # State
        self._running = False
        self._current_chunks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Threads
        self._announce_thread: Optional[threading.Thread] = None
        self._work_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the worker."""
        if self._running:
            return

        self._running = True

        # Start discovery
        self.discovery.start()

        # Start announcement thread
        self._announce_thread = threading.Thread(target=self._announce_loop, daemon=True)
        self._announce_thread.start()

        # Start work polling thread
        self._work_thread = threading.Thread(target=self._work_loop, daemon=True)
        self._work_thread.start()

        logger.info(
            f"Render worker started: {self._hostname} "
            f"({self._node_info.gpu_count} GPUs, {self._node_info.gpu_memory_gb:.1f}GB VRAM)"
        )

    def stop(self) -> None:
        """Stop the worker."""
        self._running = False
        self.discovery.stop()
        logger.info("Render worker stopped")

    def _announce_loop(self) -> None:
        """Periodically announce presence to coordinators."""
        while self._running:
            # Update availability status
            with self._lock:
                self._node_info.is_available = len(self._current_chunks) < self.config.max_concurrent_chunks

            self.discovery.announce(self._node_info)
            time.sleep(self.config.announce_interval)

    def _work_loop(self) -> None:
        """Poll for work and process chunks."""
        assignments_dir = self.config.work_dir.parent / "coordinator" / "assignments"

        while self._running:
            try:
                # Check for work assignments
                if assignments_dir.exists():
                    for assignment_file in assignments_dir.glob("*.json"):
                        self._check_assignment(assignment_file)

            except Exception as e:
                logger.error(f"Work loop error: {e}")

            time.sleep(1.0)

    def _check_assignment(self, assignment_file: Path) -> None:
        """Check and potentially process an assignment."""
        try:
            with open(assignment_file) as f:
                assignment = json.load(f)

            chunk_id = assignment.get("chunk_id")
            worker_id = assignment.get("worker_id")

            # Check if this assignment is for us
            if worker_id != self.config.node_id:
                return

            # Check if already processing
            with self._lock:
                if chunk_id in self._current_chunks:
                    return

                if len(self._current_chunks) >= self.config.max_concurrent_chunks:
                    return

                self._current_chunks[chunk_id] = assignment

            # Remove assignment file
            assignment_file.unlink()

            # Process in new thread
            thread = threading.Thread(
                target=self._process_chunk_thread,
                args=(assignment,),
                daemon=True
            )
            thread.start()

        except Exception as e:
            logger.error(f"Assignment check failed: {e}")

    def _process_chunk_thread(self, chunk_data: Dict[str, Any]) -> None:
        """Process a chunk in a worker thread."""
        chunk_id = chunk_data["chunk_id"]

        try:
            # Update node info
            self._node_info.current_job = chunk_data.get("job_id")
            self._node_info.current_chunk = chunk_id

            def progress_callback(progress: float):
                # Could send progress updates to coordinator here
                pass

            # Process chunk
            result = self.processor.process_chunk(chunk_data, progress_callback)

            # Report result
            self._report_result(result)

        except Exception as e:
            logger.error(f"Chunk thread error: {e}")
            self._report_result({
                "success": False,
                "chunk_id": chunk_id,
                "error": str(e),
            })

        finally:
            with self._lock:
                self._current_chunks.pop(chunk_id, None)

            self._node_info.current_job = None
            self._node_info.current_chunk = None

    def _report_result(self, result: Dict[str, Any]) -> None:
        """Report chunk result to coordinator."""
        # Write result file for coordinator to pick up
        results_dir = self.config.work_dir.parent / "coordinator" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        result_file = results_dir / f"{result['chunk_id']}_result.json"
        result["worker_id"] = self.config.node_id
        result["timestamp"] = datetime.now().isoformat()

        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Reported result for chunk {result['chunk_id']}")

    def get_status(self) -> Dict[str, Any]:
        """Get worker status."""
        with self._lock:
            current_chunks = list(self._current_chunks.keys())

        return {
            "node_id": self.config.node_id,
            "hostname": self._hostname,
            "address": self._address,
            "port": self.config.listen_port,
            "running": self._running,
            "gpu_count": self._node_info.gpu_count,
            "gpu_memory_gb": self._node_info.gpu_memory_gb,
            "cpu_cores": self._node_info.cpu_cores,
            "ram_gb": self._node_info.ram_gb,
            "is_available": self._node_info.is_available,
            "current_chunks": current_chunks,
            "max_concurrent": self.config.max_concurrent_chunks,
        }


def start_worker_daemon(config: Optional[WorkerConfig] = None) -> RenderWorker:
    """Start a worker as a daemon process."""
    worker = RenderWorker(config)
    worker.start()
    return worker
