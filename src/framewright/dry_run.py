"""Dry-run analysis module for FrameWright.

Provides pre-processing analysis to estimate resources and time without
actually executing the restoration pipeline.
"""
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .utils.ffmpeg import probe_video, get_video_info, check_ffmpeg_installed
from .utils.gpu import get_gpu_memory_info, get_all_gpu_info, calculate_optimal_tile_size
from .utils.disk import estimate_required_space, get_disk_usage, SpaceEstimate, DiskUsage

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStep:
    """Represents a single step in the processing pipeline."""
    number: int
    name: str
    description: str
    estimated_time_seconds: float = 0.0
    estimated_disk_bytes: int = 0
    requires_gpu: bool = False
    vram_required_mb: int = 0


@dataclass
class HardwareRequirements:
    """Hardware requirements for processing."""
    vram_required_mb: int
    vram_available_mb: int
    vram_sufficient: bool
    recommended_tile_size: int
    disk_required_bytes: int
    disk_available_bytes: int
    disk_sufficient: bool
    gpu_name: str = "Unknown"
    gpu_count: int = 0
    has_gpu: bool = False


@dataclass
class TimeEstimate:
    """Processing time estimates."""
    extraction_seconds: float
    enhancement_seconds: float
    interpolation_seconds: float
    encoding_seconds: float
    total_seconds: float
    total_minutes: float
    time_range_str: str  # e.g., "45-60 minutes"

    @property
    def total_hours(self) -> float:
        return self.total_seconds / 3600


@dataclass
class DryRunResult:
    """Complete dry-run analysis result."""
    # Input video info
    input_path: str
    input_resolution: Tuple[int, int]
    input_fps: float
    input_duration_seconds: float
    input_codec: str
    has_audio: bool

    # Output info
    output_path: str
    output_resolution: Tuple[int, int]
    output_fps: float
    output_format: str

    # Frame counts
    input_frame_count: int
    output_frame_count: int
    detected_faces_estimate: int

    # Processing pipeline
    pipeline_steps: List[ProcessingStep]

    # Time estimates
    time_estimate: TimeEstimate

    # Space requirements
    temp_disk_usage_bytes: int
    output_disk_usage_bytes: int
    total_disk_required_bytes: int

    # Hardware requirements
    hardware: HardwareRequirements

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # Overall status
    can_proceed: bool = True
    blocking_issues: List[str] = field(default_factory=list)

    def format_summary(self, use_color: bool = True) -> str:
        """Generate human-readable summary with improved formatting.

        Args:
            use_color: Whether to use ANSI color codes
        """
        # ANSI color codes
        if use_color:
            BOLD = '\033[1m'
            GREEN = '\033[92m'
            YELLOW = '\033[93m'
            RED = '\033[91m'
            CYAN = '\033[96m'
            RESET = '\033[0m'
            DIM = '\033[2m'
        else:
            BOLD = GREEN = YELLOW = RED = CYAN = RESET = DIM = ''

        width, height = self.input_resolution
        out_width, out_height = self.output_resolution
        duration_str = self._format_duration(self.input_duration_seconds)

        # Calculate scaling factor for display
        scale_str = f"{self.output_resolution[0] // self.input_resolution[0]}x"

        lines = [
            "",
            f"{BOLD}+{'=' * 60}+{RESET}",
            f"{BOLD}|{'DRY RUN ANALYSIS':^60}|{RESET}",
            f"{BOLD}+{'=' * 60}+{RESET}",
            "",
            f"{BOLD}INPUT VIDEO{RESET}",
            f"  Path:       {self.input_path}",
            f"  Resolution: {width}x{height}",
            f"  FPS:        {self.input_fps:.2f}",
            f"  Duration:   {duration_str}",
            f"  Frames:     {self.input_frame_count:,}",
            f"  Codec:      {self.input_codec}",
            f"  Audio:      {'Yes' if self.has_audio else 'No'}",
            "",
            f"{BOLD}OUTPUT VIDEO{RESET}",
            f"  Path:       {self.output_path}",
            f"  Resolution: {out_width}x{out_height} ({scale_str} upscale)",
            f"  FPS:        {self.output_fps:.2f}",
            f"  Frames:     {self.output_frame_count:,}",
            f"  Format:     {self.output_format.upper()}",
            "",
            f"{BOLD}PROCESSING PIPELINE{RESET}",
        ]

        # Add pipeline steps with visual indicators
        for step in self.pipeline_steps:
            gpu_indicator = f" {CYAN}[GPU]{RESET}" if step.requires_gpu else ""
            lines.append(f"  {step.number}. {step.description}{gpu_indicator}")

        # Time estimate section
        lines.extend([
            "",
            f"{BOLD}TIME ESTIMATE{RESET}",
            f"  Total: {CYAN}{self.time_estimate.time_range_str}{RESET}",
        ])

        # Show breakdown if significant
        if self.time_estimate.total_minutes > 5:
            lines.append(f"  {DIM}Breakdown:{RESET}")
            lines.append(f"    {DIM}Extraction:   {self.time_estimate.extraction_seconds/60:.1f} min{RESET}")
            lines.append(f"    {DIM}Enhancement:  {self.time_estimate.enhancement_seconds/60:.1f} min{RESET}")
            if self.time_estimate.interpolation_seconds > 0:
                lines.append(f"    {DIM}Interpolate:  {self.time_estimate.interpolation_seconds/60:.1f} min{RESET}")
            lines.append(f"    {DIM}Encoding:     {self.time_estimate.encoding_seconds/60:.1f} min{RESET}")

        # Disk usage with visual bar
        temp_gb = self.temp_disk_usage_bytes / (1024**3)
        output_gb = self.output_disk_usage_bytes / (1024**3)
        total_gb = self.total_disk_required_bytes / (1024**3)
        avail_gb = self.hardware.disk_available_bytes / (1024**3)

        lines.extend([
            "",
            f"{BOLD}DISK USAGE{RESET}",
            f"  Temporary:  {temp_gb:.1f} GB",
            f"  Output:     {output_gb:.1f} GB",
            f"  Total:      {CYAN}{total_gb:.1f} GB{RESET}",
            f"  Available:  {avail_gb:.1f} GB",
        ])

        # Disk usage bar
        if avail_gb > 0:
            usage_pct = min(100, int((total_gb / avail_gb) * 100))
            bar_width = 30
            filled = int(bar_width * usage_pct / 100)
            bar_color = GREEN if usage_pct < 70 else (YELLOW if usage_pct < 90 else RED)
            bar = f"{bar_color}{'#' * filled}{RESET}{DIM}{'-' * (bar_width - filled)}{RESET}"
            lines.append(f"  [{bar}] {usage_pct}%")

        # Hardware section
        lines.extend([
            "",
            f"{BOLD}HARDWARE{RESET}",
        ])

        if self.hardware.has_gpu:
            vram_gb = self.hardware.vram_available_mb / 1024
            req_gb = self.hardware.vram_required_mb / 1024
            vram_ok = self.hardware.vram_sufficient
            status = f"{GREEN}OK{RESET}" if vram_ok else f"{YELLOW}Tiling{RESET}"
            lines.append(f"  GPU:        {self.hardware.gpu_name}")
            lines.append(f"  VRAM:       {vram_gb:.1f} GB available, {req_gb:.1f} GB required [{status}]")
            if not vram_ok and self.hardware.recommended_tile_size > 0:
                lines.append(f"  Tile Size:  {self.hardware.recommended_tile_size}px (auto-adjusted)")
        else:
            lines.append(f"  GPU:        {YELLOW}None detected{RESET}")
            lines.append(f"  {DIM}Processing will use CPU (significantly slower){RESET}")

        # Warnings section
        if self.warnings:
            lines.extend([
                "",
                f"{YELLOW}{BOLD}WARNINGS{RESET}",
            ])
            for warning in self.warnings:
                lines.append(f"  {YELLOW}!{RESET} {warning}")

        # Blocking issues section
        if self.blocking_issues:
            lines.extend([
                "",
                f"{RED}{BOLD}BLOCKING ISSUES{RESET}",
            ])
            for issue in self.blocking_issues:
                lines.append(f"  {RED}X{RESET} {issue}")

        # Final status
        lines.append("")
        if self.can_proceed:
            lines.append(f"{GREEN}{BOLD}Status: Ready to process{RESET}")
        else:
            lines.append(f"{RED}{BOLD}Status: Cannot proceed - resolve blocking issues{RESET}")

        lines.append("")

        return "\n".join(lines)

    def _format_duration(self, seconds: float) -> str:
        """Format duration as H:MM:SS or M:SS."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "input": {
                "path": self.input_path,
                "resolution": list(self.input_resolution),
                "fps": self.input_fps,
                "duration_seconds": self.input_duration_seconds,
                "codec": self.input_codec,
                "has_audio": self.has_audio,
                "frame_count": self.input_frame_count,
            },
            "output": {
                "path": self.output_path,
                "resolution": list(self.output_resolution),
                "fps": self.output_fps,
                "format": self.output_format,
                "frame_count": self.output_frame_count,
            },
            "pipeline_steps": [
                {
                    "number": s.number,
                    "name": s.name,
                    "description": s.description,
                    "estimated_time_seconds": s.estimated_time_seconds,
                }
                for s in self.pipeline_steps
            ],
            "time_estimate": {
                "extraction_seconds": self.time_estimate.extraction_seconds,
                "enhancement_seconds": self.time_estimate.enhancement_seconds,
                "interpolation_seconds": self.time_estimate.interpolation_seconds,
                "encoding_seconds": self.time_estimate.encoding_seconds,
                "total_seconds": self.time_estimate.total_seconds,
                "time_range": self.time_estimate.time_range_str,
            },
            "disk_usage": {
                "temp_bytes": self.temp_disk_usage_bytes,
                "output_bytes": self.output_disk_usage_bytes,
                "total_bytes": self.total_disk_required_bytes,
            },
            "hardware": {
                "vram_required_mb": self.hardware.vram_required_mb,
                "vram_available_mb": self.hardware.vram_available_mb,
                "vram_sufficient": self.hardware.vram_sufficient,
                "disk_sufficient": self.hardware.disk_sufficient,
                "gpu_name": self.hardware.gpu_name,
                "has_gpu": self.hardware.has_gpu,
            },
            "warnings": self.warnings,
            "can_proceed": self.can_proceed,
            "blocking_issues": self.blocking_issues,
        }


class DryRunAnalyzer:
    """Analyzes video and estimates processing requirements without execution.

    Provides detailed estimates of:
    - Processing time based on hardware
    - Disk space requirements (temp + output)
    - VRAM requirements
    - Frame counts
    - Processing pipeline steps
    """

    # Benchmark constants (frames per second at various resolutions)
    # Based on empirical testing with different hardware configurations
    BENCHMARK_FPS = {
        "enhancement": {
            "high_end_gpu": 2.5,      # RTX 3080/4080 level
            "mid_gpu": 1.0,           # RTX 3060/2080 level
            "low_gpu": 0.3,           # GTX 1060 level
            "cpu": 0.05,              # CPU-only
        },
        "interpolation": {
            "high_end_gpu": 30.0,
            "mid_gpu": 15.0,
            "low_gpu": 5.0,
            "cpu": 0.5,
        },
        "extraction": 100.0,  # Frames per second for ffmpeg extraction
        "encoding": 30.0,     # Frames per second for x265 encoding
    }

    # VRAM requirements per output megapixel
    VRAM_PER_MEGAPIXEL = {
        "realesrgan-x4plus": 450,
        "realesrgan-x4plus-anime": 400,
        "realesrgan-x2plus": 250,
        "realesr-animevideov3": 350,
    }

    # Typical face density in video (percentage of frames with faces)
    TYPICAL_FACE_RATIO = 0.15  # 15% of frames typically have faces

    def __init__(
        self,
        scale_factor: int = 4,
        model_name: str = "realesrgan-x4plus",
        crf: int = 18,
        output_format: str = "mkv",
        enable_interpolation: bool = False,
        target_fps: Optional[float] = None,
        enable_auto_enhance: bool = False,
        enable_face_restore: bool = True,
    ):
        """Initialize the dry-run analyzer.

        Args:
            scale_factor: Upscaling factor (2 or 4)
            model_name: Real-ESRGAN model name
            crf: Constant rate factor for encoding
            output_format: Output video format
            enable_interpolation: Whether RIFE interpolation will be used
            target_fps: Target FPS for interpolation (None = 2x source)
            enable_auto_enhance: Whether auto-enhancement is enabled
            enable_face_restore: Whether face restoration is enabled
        """
        self.scale_factor = scale_factor
        self.model_name = model_name
        self.crf = crf
        self.output_format = output_format
        self.enable_interpolation = enable_interpolation
        self.target_fps = target_fps
        self.enable_auto_enhance = enable_auto_enhance
        self.enable_face_restore = enable_face_restore

    def analyze(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
    ) -> DryRunResult:
        """Perform dry-run analysis on a video.

        Args:
            video_path: Path to input video
            output_path: Optional output path (used for display)

        Returns:
            DryRunResult with complete analysis
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Get video metadata
        metadata = probe_video(video_path)
        width = metadata.get("width", 1920)
        height = metadata.get("height", 1080)
        fps = metadata.get("framerate", 24.0)
        duration = metadata.get("duration", 0.0)
        codec = metadata.get("codec", "unknown")
        has_audio = metadata.get("has_audio", False)

        # Calculate frame counts
        input_frame_count = int(duration * fps)

        # Determine output FPS
        output_fps = fps
        if self.enable_interpolation:
            if self.target_fps:
                output_fps = self.target_fps
            else:
                output_fps = fps * 2  # Default 2x

        # Calculate output frame count
        output_frame_count = int(duration * output_fps)

        # Calculate output resolution
        out_width = width * self.scale_factor
        out_height = height * self.scale_factor

        # Determine output path
        if output_path is None:
            output_path = video_path.parent / f"restored_video.{self.output_format}"

        # Estimate faces (rough estimate based on typical content)
        detected_faces_estimate = int(input_frame_count * self.TYPICAL_FACE_RATIO) if self.enable_face_restore else 0

        # Build pipeline steps
        pipeline_steps = self._build_pipeline_steps(
            input_frame_count,
            output_frame_count,
            (width, height),
            (out_width, out_height),
            detected_faces_estimate,
        )

        # Estimate time
        time_estimate = self._estimate_time(
            input_frame_count,
            output_frame_count,
            (width, height),
        )

        # Estimate disk usage
        disk_estimates = self._estimate_disk_usage(
            video_path,
            input_frame_count,
            output_frame_count,
            (width, height),
            (out_width, out_height),
        )

        # Check hardware requirements
        hardware = self._check_hardware_requirements(
            (width, height),
            (out_width, out_height),
            disk_estimates["total"],
        )

        # Build warnings and blocking issues
        warnings = []
        blocking_issues = []

        if not hardware.vram_sufficient:
            if hardware.has_gpu:
                warnings.append(
                    f"VRAM may be insufficient ({hardware.vram_required_mb}MB needed, "
                    f"{hardware.vram_available_mb}MB available). "
                    f"Will use tile size {hardware.recommended_tile_size} for processing."
                )
            else:
                warnings.append(
                    "No GPU detected. Processing will use CPU and be significantly slower."
                )

        if not hardware.disk_sufficient:
            blocking_issues.append(
                f"Insufficient disk space. Required: {disk_estimates['total'] / (1024**3):.1f}GB, "
                f"Available: {hardware.disk_available_bytes / (1024**3):.1f}GB"
            )

        if duration > 3600:  # > 1 hour
            warnings.append(
                f"Long video detected ({duration/3600:.1f} hours). "
                "Consider using streaming mode for better memory management."
            )

        if out_width > 7680 or out_height > 4320:  # > 8K
            warnings.append(
                f"Output resolution ({out_width}x{out_height}) exceeds 8K. "
                "Some players may not support this resolution."
            )

        can_proceed = len(blocking_issues) == 0

        return DryRunResult(
            input_path=str(video_path),
            input_resolution=(width, height),
            input_fps=fps,
            input_duration_seconds=duration,
            input_codec=codec,
            has_audio=has_audio,
            output_path=str(output_path),
            output_resolution=(out_width, out_height),
            output_fps=output_fps,
            output_format=self.output_format,
            input_frame_count=input_frame_count,
            output_frame_count=output_frame_count,
            detected_faces_estimate=detected_faces_estimate,
            pipeline_steps=pipeline_steps,
            time_estimate=time_estimate,
            temp_disk_usage_bytes=disk_estimates["temp"],
            output_disk_usage_bytes=disk_estimates["output"],
            total_disk_required_bytes=disk_estimates["total"],
            hardware=hardware,
            warnings=warnings,
            can_proceed=can_proceed,
            blocking_issues=blocking_issues,
        )

    def _build_pipeline_steps(
        self,
        input_frames: int,
        output_frames: int,
        input_res: Tuple[int, int],
        output_res: Tuple[int, int],
    detected_faces: int,
    ) -> List[ProcessingStep]:
        """Build list of processing steps."""
        steps = []
        step_num = 1
        out_w, out_h = output_res

        # Step 1: Extract frames
        steps.append(ProcessingStep(
            number=step_num,
            name="extract_frames",
            description=f"Extract {input_frames:,} frames to ./frames/",
        ))
        step_num += 1

        # Step 2: Upscale
        steps.append(ProcessingStep(
            number=step_num,
            name="upscale",
            description=f"Upscale {self.scale_factor}x with {self.model_name} (-> {out_w}x{out_h})",
            requires_gpu=True,
            vram_required_mb=self._estimate_vram(input_res, self.scale_factor),
        ))
        step_num += 1

        # Step 3: Face restoration (if enabled)
        if self.enable_face_restore and detected_faces > 0:
            steps.append(ProcessingStep(
                number=step_num,
                name="face_restoration",
                description=f"Face restoration on ~{detected_faces:,} detected faces",
                requires_gpu=True,
            ))
            step_num += 1

        # Step 4: Interpolation (if enabled)
        if self.enable_interpolation:
            steps.append(ProcessingStep(
                number=step_num,
                name="interpolate",
                description=f"Interpolate to {self.target_fps or 'auto'}fps (-> {output_frames:,} frames)",
                requires_gpu=True,
            ))
            step_num += 1

        # Step 5: Encode
        steps.append(ProcessingStep(
            number=step_num,
            name="encode",
            description=f"Encode to {self.output_format.upper()} (CRF {self.crf})",
        ))

        return steps

    def _estimate_time(
        self,
        input_frames: int,
        output_frames: int,
        input_res: Tuple[int, int],
    ) -> TimeEstimate:
        """Estimate processing time based on hardware."""
        # Detect hardware tier
        gpus = get_all_gpu_info()
        if gpus:
            total_vram = sum(g.total_memory_mb for g in gpus)
            if total_vram >= 16000:
                tier = "high_end_gpu"
            elif total_vram >= 8000:
                tier = "mid_gpu"
            else:
                tier = "low_gpu"
        else:
            tier = "cpu"

        # Resolution scaling factor (based on 1080p baseline)
        width, height = input_res
        res_factor = (width * height) / (1920 * 1080)
        scale_factor_multiplier = self.scale_factor / 4.0  # Baseline is 4x

        # Calculate time for each stage
        extraction_fps = self.BENCHMARK_FPS["extraction"]
        enhancement_fps = self.BENCHMARK_FPS["enhancement"][tier]
        interpolation_fps = self.BENCHMARK_FPS["interpolation"][tier]
        encoding_fps = self.BENCHMARK_FPS["encoding"]

        # Adjust for resolution and scale factor
        enhancement_fps = enhancement_fps / (res_factor * scale_factor_multiplier)

        # Calculate times
        extraction_seconds = input_frames / extraction_fps
        enhancement_seconds = input_frames / enhancement_fps

        interpolation_seconds = 0.0
        if self.enable_interpolation:
            interpolation_seconds = output_frames / interpolation_fps

        encoding_seconds = output_frames / encoding_fps

        total_seconds = extraction_seconds + enhancement_seconds + interpolation_seconds + encoding_seconds

        # Calculate time range (pessimistic to optimistic)
        low_multiplier = 0.8
        high_multiplier = 1.5

        low_minutes = int((total_seconds * low_multiplier) / 60)
        high_minutes = int((total_seconds * high_multiplier) / 60)

        if high_minutes >= 120:
            time_range = f"{low_minutes/60:.1f}-{high_minutes/60:.1f} hours"
        else:
            time_range = f"{low_minutes}-{high_minutes} minutes"

        return TimeEstimate(
            extraction_seconds=extraction_seconds,
            enhancement_seconds=enhancement_seconds,
            interpolation_seconds=interpolation_seconds,
            encoding_seconds=encoding_seconds,
            total_seconds=total_seconds,
            total_minutes=total_seconds / 60,
            time_range_str=time_range,
        )

    def _estimate_disk_usage(
        self,
        video_path: Path,
        input_frames: int,
        output_frames: int,
        input_res: Tuple[int, int],
        output_res: Tuple[int, int],
    ) -> Dict[str, int]:
        """Estimate disk space requirements."""
        width, height = input_res
        out_w, out_h = output_res

        # PNG frame size estimates
        # Roughly 3 bytes per pixel for PNG (compressed)
        input_frame_size = int(width * height * 3 * 0.5)  # ~50% compression
        output_frame_size = int(out_w * out_h * 3 * 0.5)

        # Frame storage
        input_frames_bytes = input_frames * input_frame_size
        enhanced_frames_bytes = input_frames * output_frame_size

        # Interpolated frames (if applicable)
        interpolated_bytes = 0
        if self.enable_interpolation:
            interpolated_bytes = output_frames * output_frame_size

        # Audio extraction (~10MB per minute at 48kHz/24-bit)
        video_info = probe_video(video_path)
        duration = video_info.get("duration", 0)
        audio_bytes = int((duration / 60) * 10 * 1024 * 1024) if video_info.get("has_audio") else 0

        # Output video estimate (based on CRF and resolution)
        # Higher CRF = smaller file, higher resolution = larger file
        bitrate_estimate = self._estimate_bitrate(out_w, out_h, self.crf)
        output_bytes = int(duration * bitrate_estimate / 8)

        # Temporary storage (frames + audio)
        temp_bytes = input_frames_bytes + enhanced_frames_bytes + interpolated_bytes + audio_bytes

        # Total with 20% safety margin
        total_bytes = int((temp_bytes + output_bytes) * 1.2)

        return {
            "input_frames": input_frames_bytes,
            "enhanced_frames": enhanced_frames_bytes,
            "interpolated_frames": interpolated_bytes,
            "audio": audio_bytes,
            "output": output_bytes,
            "temp": temp_bytes,
            "total": total_bytes,
        }

    def _estimate_bitrate(self, width: int, height: int, crf: int) -> int:
        """Estimate video bitrate based on resolution and CRF."""
        # Base bitrate for 1080p at CRF 18 (approximately 8 Mbps)
        base_bitrate = 8_000_000
        base_pixels = 1920 * 1080
        base_crf = 18

        # Scale by pixel count
        pixel_ratio = (width * height) / base_pixels

        # CRF adjustment (each 6 CRF = ~2x size change)
        crf_factor = 2 ** ((base_crf - crf) / 6)

        return int(base_bitrate * pixel_ratio * crf_factor)

    def _estimate_vram(self, input_res: Tuple[int, int], scale_factor: int) -> int:
        """Estimate VRAM requirements in MB."""
        width, height = input_res
        out_w = width * scale_factor
        out_h = height * scale_factor
        output_megapixels = (out_w * out_h) / 1_000_000

        coeff = self.VRAM_PER_MEGAPIXEL.get(self.model_name, 450)
        estimated_vram = int(output_megapixels * coeff)

        # Add base overhead for model loading (~500MB)
        return estimated_vram + 500

    def _check_hardware_requirements(
        self,
        input_res: Tuple[int, int],
        output_res: Tuple[int, int],
        required_disk_bytes: int,
    ) -> HardwareRequirements:
        """Check hardware against requirements."""
        # GPU info
        gpus = get_all_gpu_info()
        has_gpu = len(gpus) > 0
        gpu_name = gpus[0].name if gpus else "No GPU"
        gpu_count = len(gpus)

        # VRAM
        vram_required = self._estimate_vram(input_res, self.scale_factor)
        vram_available = gpus[0].free_memory_mb if gpus else 0
        vram_sufficient = vram_available >= vram_required

        # Calculate recommended tile size
        recommended_tile = 0
        if not vram_sufficient and has_gpu:
            recommended_tile = calculate_optimal_tile_size(
                frame_resolution=input_res,
                scale_factor=self.scale_factor,
                available_vram_mb=vram_available,
                model_name=self.model_name,
            )

        # Disk space
        try:
            disk_usage = get_disk_usage(Path.cwd())
            disk_available = disk_usage.free_bytes
        except Exception:
            disk_available = 0

        disk_sufficient = disk_available >= required_disk_bytes

        return HardwareRequirements(
            vram_required_mb=vram_required,
            vram_available_mb=vram_available,
            vram_sufficient=vram_sufficient,
            recommended_tile_size=recommended_tile,
            disk_required_bytes=required_disk_bytes,
            disk_available_bytes=disk_available,
            disk_sufficient=disk_sufficient,
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            has_gpu=has_gpu,
        )


def perform_dry_run(
    video_path: Path,
    output_path: Optional[Path] = None,
    scale_factor: int = 4,
    model_name: str = "realesrgan-x4plus",
    crf: int = 18,
    output_format: str = "mkv",
    enable_interpolation: bool = False,
    target_fps: Optional[float] = None,
    enable_auto_enhance: bool = False,
    enable_face_restore: bool = True,
) -> DryRunResult:
    """Convenience function to perform dry-run analysis.

    Args:
        video_path: Path to input video
        output_path: Optional output path
        scale_factor: Upscaling factor (2 or 4)
        model_name: Real-ESRGAN model name
        crf: Constant rate factor
        output_format: Output video format
        enable_interpolation: Enable RIFE interpolation
        target_fps: Target FPS for interpolation
        enable_auto_enhance: Enable auto-enhancement
        enable_face_restore: Enable face restoration

    Returns:
        DryRunResult with complete analysis
    """
    analyzer = DryRunAnalyzer(
        scale_factor=scale_factor,
        model_name=model_name,
        crf=crf,
        output_format=output_format,
        enable_interpolation=enable_interpolation,
        target_fps=target_fps,
        enable_auto_enhance=enable_auto_enhance,
        enable_face_restore=enable_face_restore,
    )

    return analyzer.analyze(video_path, output_path)
