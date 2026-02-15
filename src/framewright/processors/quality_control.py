"""Quality Control and A/B Testing for FrameWright.

Provides:
- Quality gating: Pause processing if quality drops
- A/B testing: Compare different settings on same content
- Artifact detection: Flag potential processing issues

Example:
    >>> # Quality gating
    >>> gate = QualityGate(min_ssim=0.85, min_sharpness=100)
    >>> result = gate.check(original_frame, processed_frame)
    >>> if result.passed:
    ...     save_frame(processed_frame)
    ... else:
    ...     print(f"Quality issue: {result.issues}")

    >>> # A/B testing
    >>> tester = ABTester()
    >>> result = tester.compare(
    ...     video_path="test.mp4",
    ...     config_a={"preset": "quality"},
    ...     config_b={"preset": "ultimate"},
    ...     sample_frames=5
    ... )
"""

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


# =============================================================================
# Quality Gating
# =============================================================================

@dataclass
class QualityCheckResult:
    """Result of quality check."""
    passed: bool = True
    ssim: float = 1.0
    psnr: float = 100.0
    sharpness: float = 0.0
    noise_level: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class QualityGate:
    """Quality gating to ensure processing meets minimum standards.

    Pauses or alerts when processed frames don't meet quality thresholds.
    Useful for catching processing failures early.
    """

    def __init__(
        self,
        min_ssim: float = 0.8,
        min_psnr: float = 25.0,
        min_sharpness: float = 50.0,
        max_noise: float = 20.0,
        on_failure: Optional[Callable[[QualityCheckResult], None]] = None,
    ):
        """Initialize quality gate.

        Args:
            min_ssim: Minimum SSIM score (0-1)
            min_psnr: Minimum PSNR in dB
            min_sharpness: Minimum sharpness (Laplacian variance)
            max_noise: Maximum noise level
            on_failure: Callback when quality check fails
        """
        self.min_ssim = min_ssim
        self.min_psnr = min_psnr
        self.min_sharpness = min_sharpness
        self.max_noise = max_noise
        self.on_failure = on_failure

    def check(
        self,
        original: "np.ndarray",
        processed: "np.ndarray",
    ) -> QualityCheckResult:
        """Check quality of processed frame against original.

        Args:
            original: Original frame
            processed: Processed frame

        Returns:
            QualityCheckResult
        """
        if not HAS_OPENCV:
            return QualityCheckResult(passed=True)

        result = QualityCheckResult()

        # Resize processed to original size for comparison
        if original.shape[:2] != processed.shape[:2]:
            processed_resized = cv2.resize(
                processed,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_LANCZOS4
            )
        else:
            processed_resized = processed

        # Calculate SSIM
        result.ssim = self._calculate_ssim(original, processed_resized)
        if result.ssim < self.min_ssim:
            result.passed = False
            result.issues.append(f"SSIM too low: {result.ssim:.3f} < {self.min_ssim}")
            result.recommendations.append("Try reducing processing intensity")

        # Calculate PSNR
        result.psnr = self._calculate_psnr(original, processed_resized)
        if result.psnr < self.min_psnr:
            result.passed = False
            result.issues.append(f"PSNR too low: {result.psnr:.1f}dB < {self.min_psnr}dB")

        # Calculate sharpness
        result.sharpness = self._calculate_sharpness(processed)
        if result.sharpness < self.min_sharpness:
            result.passed = False
            result.issues.append(f"Image too blurry: sharpness {result.sharpness:.1f} < {self.min_sharpness}")
            result.recommendations.append("Check if denoising is too aggressive")

        # Estimate noise
        result.noise_level = self._estimate_noise(processed)
        if result.noise_level > self.max_noise:
            result.passed = False
            result.issues.append(f"Noise too high: {result.noise_level:.1f} > {self.max_noise}")
            result.recommendations.append("Try enabling denoising")

        # Callback on failure
        if not result.passed and self.on_failure:
            self.on_failure(result)

        return result

    def check_absolute(
        self,
        frame: "np.ndarray",
    ) -> QualityCheckResult:
        """Check absolute quality of a frame (no original comparison).

        Args:
            frame: Frame to check

        Returns:
            QualityCheckResult
        """
        if not HAS_OPENCV:
            return QualityCheckResult(passed=True)

        result = QualityCheckResult()

        result.sharpness = self._calculate_sharpness(frame)
        if result.sharpness < self.min_sharpness:
            result.passed = False
            result.issues.append(f"Image too blurry: {result.sharpness:.1f}")

        result.noise_level = self._estimate_noise(frame)
        if result.noise_level > self.max_noise:
            result.passed = False
            result.issues.append(f"Noise too high: {result.noise_level:.1f}")

        # Check for artifacts
        artifacts = self._detect_artifacts(frame)
        if artifacts:
            result.issues.extend(artifacts)
            if len(artifacts) > 2:
                result.passed = False

        return result

    def _calculate_ssim(self, img1: "np.ndarray", img2: "np.ndarray") -> float:
        """Calculate SSIM between two images."""
        try:
            from ..metrics import calculate_ssim
            return calculate_ssim(img1, img2)
        except:
            # Fallback
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            mu1 = np.mean(gray1)
            mu2 = np.mean(gray2)
            sigma1 = np.var(gray1)
            sigma2 = np.var(gray2)
            sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))

            return ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))

    def _calculate_psnr(self, img1: "np.ndarray", img2: "np.ndarray") -> float:
        """Calculate PSNR between two images."""
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))

    def _calculate_sharpness(self, img: "np.ndarray") -> float:
        """Calculate sharpness using Laplacian variance."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _estimate_noise(self, img: "np.ndarray") -> float:
        """Estimate noise level."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blurred = cv2.GaussianBlur(gray.astype(float), (5, 5), 0)
        noise = gray.astype(float) - blurred
        return np.std(noise)

    def _detect_artifacts(self, img: "np.ndarray") -> List[str]:
        """Detect common processing artifacts."""
        artifacts = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Check for banding (common in over-processed images)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        zero_bins = np.sum(hist == 0)
        if zero_bins > 200:  # Many empty histogram bins = banding
            artifacts.append("Possible color banding detected")

        # Check for blocking artifacts
        h, w = gray.shape
        block_size = 8
        dct_energy = []

        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size].astype(float)
                dct = cv2.dct(block)
                dct_energy.append(np.abs(dct[0, 0]))

        if dct_energy:
            energy_var = np.var(dct_energy)
            if energy_var > 1000:  # High variance in DC coefficients
                artifacts.append("Possible blocking artifacts")

        # Check for ringing
        edges = cv2.Canny(gray, 100, 200)
        dilated = cv2.dilate(edges, None, iterations=2)
        edge_region = dilated > 0

        if np.any(edge_region):
            edge_values = gray[edge_region]
            if np.std(edge_values) > 50:
                artifacts.append("Possible ringing artifacts near edges")

        return artifacts


# =============================================================================
# A/B Testing
# =============================================================================

@dataclass
class ABTestResult:
    """Result of A/B test comparison."""
    config_a_name: str = "Config A"
    config_b_name: str = "Config B"
    winner: str = ""  # "A", "B", or "tie"
    a_metrics: Dict[str, float] = field(default_factory=dict)
    b_metrics: Dict[str, float] = field(default_factory=dict)
    comparison_frames: List[Path] = field(default_factory=list)
    summary: str = ""


class ABTester:
    """Compare different processing configurations on same content.

    Useful for:
    - Finding optimal settings for specific content
    - Validating new models/algorithms
    - Quality assurance
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize A/B tester.

        Args:
            output_dir: Directory for comparison outputs
        """
        self.output_dir = output_dir or Path(tempfile.mkdtemp(prefix="framewright_ab_"))

    def compare(
        self,
        video_path: Path,
        config_a: Dict[str, Any],
        config_b: Dict[str, Any],
        config_a_name: str = "Config A",
        config_b_name: str = "Config B",
        sample_frames: int = 5,
        create_comparison_images: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ABTestResult:
        """Compare two configurations on same video.

        Args:
            video_path: Input video
            config_a: First configuration
            config_b: Second configuration
            config_a_name: Name for first config
            config_b_name: Name for second config
            sample_frames: Number of frames to sample
            create_comparison_images: Create side-by-side images
            progress_callback: Progress callback

        Returns:
            ABTestResult with comparison metrics
        """
        result = ABTestResult(
            config_a_name=config_a_name,
            config_b_name=config_b_name,
        )

        if not HAS_OPENCV:
            logger.error("OpenCV required for A/B testing")
            return result

        video_path = Path(video_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return result

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frame indices
        sample_indices = np.linspace(
            total_frames * 0.1,  # Skip first 10%
            total_frames * 0.9,  # Skip last 10%
            sample_frames,
            dtype=int
        )

        a_scores = {"ssim": [], "psnr": [], "sharpness": []}
        b_scores = {"ssim": [], "psnr": [], "sharpness": []}

        quality_gate = QualityGate()

        for i, frame_idx in enumerate(sample_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, original = cap.read()
            if not ret:
                continue

            # Process with config A
            processed_a = self._process_frame(original, config_a)

            # Process with config B
            processed_b = self._process_frame(original, config_b)

            # Calculate metrics
            if processed_a is not None:
                check_a = quality_gate.check(original, processed_a)
                a_scores["ssim"].append(check_a.ssim)
                a_scores["psnr"].append(check_a.psnr)
                a_scores["sharpness"].append(check_a.sharpness)

            if processed_b is not None:
                check_b = quality_gate.check(original, processed_b)
                b_scores["ssim"].append(check_b.ssim)
                b_scores["psnr"].append(check_b.psnr)
                b_scores["sharpness"].append(check_b.sharpness)

            # Create comparison image
            if create_comparison_images and processed_a is not None and processed_b is not None:
                comparison_path = self.output_dir / f"comparison_{i+1}.png"
                self._create_comparison_image(
                    original, processed_a, processed_b,
                    config_a_name, config_b_name,
                    comparison_path
                )
                result.comparison_frames.append(comparison_path)

            if progress_callback:
                progress_callback((i + 1) / len(sample_indices))

        cap.release()

        # Calculate averages
        result.a_metrics = {k: np.mean(v) if v else 0 for k, v in a_scores.items()}
        result.b_metrics = {k: np.mean(v) if v else 0 for k, v in b_scores.items()}

        # Determine winner
        a_total = sum(result.a_metrics.values())
        b_total = sum(result.b_metrics.values())

        if a_total > b_total * 1.05:  # 5% margin
            result.winner = "A"
        elif b_total > a_total * 1.05:
            result.winner = "B"
        else:
            result.winner = "tie"

        # Generate summary
        result.summary = self._generate_summary(result)

        return result

    def _process_frame(
        self,
        frame: "np.ndarray",
        config: Dict[str, Any],
    ) -> Optional["np.ndarray"]:
        """Process a single frame with given config."""
        try:
            # Simple processing based on config
            result = frame.copy()

            # Apply scale factor
            scale = config.get("scale_factor", 1)
            if scale > 1:
                result = cv2.resize(
                    result,
                    None,
                    fx=scale, fy=scale,
                    interpolation=cv2.INTER_LANCZOS4
                )

            # Apply denoising
            if config.get("enable_tap_denoise"):
                result = cv2.fastNlMeansDenoisingColored(result)

            # Apply sharpening
            if config.get("sr_model") in ["hat", "vrt"]:
                kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
                result = cv2.filter2D(result, -1, kernel)

            return result

        except Exception as e:
            logger.debug(f"Frame processing failed: {e}")
            return None

    def _create_comparison_image(
        self,
        original: "np.ndarray",
        processed_a: "np.ndarray",
        processed_b: "np.ndarray",
        name_a: str,
        name_b: str,
        output_path: Path,
    ):
        """Create side-by-side comparison image."""
        # Resize all to same height
        target_height = 720

        def resize_to_height(img, h):
            scale = h / img.shape[0]
            return cv2.resize(img, None, fx=scale, fy=scale)

        original_resized = resize_to_height(original, target_height)
        a_resized = resize_to_height(processed_a, target_height)
        b_resized = resize_to_height(processed_b, target_height)

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX

        def add_label(img, text):
            cv2.putText(img, text, (10, 30), font, 1, (255, 255, 255), 3)
            cv2.putText(img, text, (10, 30), font, 1, (0, 0, 0), 1)
            return img

        original_labeled = add_label(original_resized.copy(), "Original")
        a_labeled = add_label(a_resized.copy(), name_a)
        b_labeled = add_label(b_resized.copy(), name_b)

        # Stack horizontally
        comparison = cv2.hconcat([original_labeled, a_labeled, b_labeled])
        cv2.imwrite(str(output_path), comparison)

    def _generate_summary(self, result: ABTestResult) -> str:
        """Generate human-readable summary."""
        lines = [
            f"A/B Test Results",
            f"================",
            f"",
            f"{result.config_a_name}:",
            f"  SSIM: {result.a_metrics.get('ssim', 0):.3f}",
            f"  PSNR: {result.a_metrics.get('psnr', 0):.1f} dB",
            f"  Sharpness: {result.a_metrics.get('sharpness', 0):.1f}",
            f"",
            f"{result.config_b_name}:",
            f"  SSIM: {result.b_metrics.get('ssim', 0):.3f}",
            f"  PSNR: {result.b_metrics.get('psnr', 0):.1f} dB",
            f"  Sharpness: {result.b_metrics.get('sharpness', 0):.1f}",
            f"",
            f"Winner: {result.winner if result.winner != 'tie' else 'Tie (no significant difference)'}",
        ]

        return "\n".join(lines)


# =============================================================================
# Metadata Preservation
# =============================================================================

@dataclass
class VideoMetadata:
    """Video metadata to preserve."""
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    year: Optional[str] = None
    comment: Optional[str] = None
    chapters: List[Dict[str, Any]] = field(default_factory=list)
    subtitles: List[Dict[str, Any]] = field(default_factory=list)
    audio_tracks: List[Dict[str, Any]] = field(default_factory=list)


class MetadataPreserver:
    """Preserve and restore video metadata through processing.

    Extracts metadata, subtitles, chapters, and multiple audio tracks
    before processing, then restores them to the output.
    """

    def __init__(self):
        """Initialize metadata preserver."""
        pass

    def extract(self, video_path: Path) -> VideoMetadata:
        """Extract metadata from video.

        Args:
            video_path: Input video path

        Returns:
            VideoMetadata object
        """
        import subprocess
        import json

        metadata = VideoMetadata()

        try:
            # Use ffprobe to get metadata
            result = subprocess.run([
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_format', '-show_streams', '-show_chapters',
                str(video_path)
            ], capture_output=True, text=True)

            data = json.loads(result.stdout)

            # Extract format metadata
            fmt = data.get("format", {})
            tags = fmt.get("tags", {})
            metadata.title = tags.get("title")
            metadata.artist = tags.get("artist")
            metadata.album = tags.get("album")
            metadata.year = tags.get("date", tags.get("year"))
            metadata.comment = tags.get("comment")

            # Extract chapters
            for chapter in data.get("chapters", []):
                metadata.chapters.append({
                    "start": chapter.get("start_time", 0),
                    "end": chapter.get("end_time", 0),
                    "title": chapter.get("tags", {}).get("title", ""),
                })

            # Extract streams
            for stream in data.get("streams", []):
                codec_type = stream.get("codec_type")

                if codec_type == "subtitle":
                    metadata.subtitles.append({
                        "index": stream.get("index"),
                        "codec": stream.get("codec_name"),
                        "language": stream.get("tags", {}).get("language"),
                    })

                elif codec_type == "audio":
                    metadata.audio_tracks.append({
                        "index": stream.get("index"),
                        "codec": stream.get("codec_name"),
                        "language": stream.get("tags", {}).get("language"),
                        "channels": stream.get("channels"),
                    })

            logger.info(f"Extracted metadata: {len(metadata.chapters)} chapters, "
                       f"{len(metadata.subtitles)} subtitle tracks, "
                       f"{len(metadata.audio_tracks)} audio tracks")

        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")

        return metadata

    def extract_subtitles(
        self,
        video_path: Path,
        output_dir: Path,
    ) -> List[Path]:
        """Extract embedded subtitle tracks to files.

        Args:
            video_path: Input video
            output_dir: Output directory for subtitle files

        Returns:
            List of extracted subtitle file paths
        """
        import subprocess

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        extracted = []
        metadata = self.extract(video_path)

        for i, sub in enumerate(metadata.subtitles):
            lang = sub.get("language", "und")
            output_path = output_dir / f"subtitle_{i}_{lang}.srt"

            try:
                subprocess.run([
                    'ffmpeg', '-y',
                    '-i', str(video_path),
                    '-map', f'0:{sub["index"]}',
                    str(output_path)
                ], capture_output=True, check=True)

                extracted.append(output_path)
                logger.info(f"Extracted subtitle track: {output_path.name}")

            except Exception as e:
                logger.warning(f"Failed to extract subtitle {i}: {e}")

        return extracted

    def apply(
        self,
        video_path: Path,
        output_path: Path,
        metadata: VideoMetadata,
        subtitle_files: Optional[List[Path]] = None,
        audio_from: Optional[Path] = None,
    ) -> bool:
        """Apply preserved metadata to output video.

        Args:
            video_path: Processed video (video only)
            output_path: Final output path
            metadata: Metadata to apply
            subtitle_files: Subtitle files to embed
            audio_from: Source video for audio tracks

        Returns:
            True if successful
        """
        import subprocess
        import tempfile

        try:
            cmd = ['ffmpeg', '-y', '-i', str(video_path)]

            # Add audio from original
            if audio_from:
                cmd.extend(['-i', str(audio_from)])

            # Add subtitle files
            if subtitle_files:
                for sub in subtitle_files:
                    cmd.extend(['-i', str(sub)])

            # Map streams
            cmd.extend(['-map', '0:v'])  # Video from processed

            if audio_from:
                cmd.extend(['-map', '1:a?'])  # Audio from original

            # Add subtitles
            if subtitle_files:
                for i in range(len(subtitle_files)):
                    stream_idx = 2 + i if audio_from else 1 + i
                    cmd.extend(['-map', f'{stream_idx}:0'])

            # Codecs
            cmd.extend(['-c:v', 'copy'])
            if audio_from:
                cmd.extend(['-c:a', 'copy'])
            if subtitle_files:
                cmd.extend(['-c:s', 'mov_text'])

            # Add metadata
            if metadata.title:
                cmd.extend(['-metadata', f'title={metadata.title}'])
            if metadata.artist:
                cmd.extend(['-metadata', f'artist={metadata.artist}'])
            if metadata.year:
                cmd.extend(['-metadata', f'date={metadata.year}'])

            # Add chapters
            if metadata.chapters:
                chapters_file = Path(tempfile.mktemp(suffix='.txt'))
                with open(chapters_file, 'w') as f:
                    f.write(";FFMETADATA1\n")
                    for ch in metadata.chapters:
                        f.write(f"[CHAPTER]\n")
                        f.write(f"TIMEBASE=1/1000\n")
                        f.write(f"START={int(float(ch['start']) * 1000)}\n")
                        f.write(f"END={int(float(ch['end']) * 1000)}\n")
                        if ch.get('title'):
                            f.write(f"title={ch['title']}\n")

                cmd = ['ffmpeg', '-y',
                       '-i', str(video_path),
                       '-i', str(chapters_file),
                       '-map_metadata', '1',
                       ] + cmd[3:]  # Rebuild command with chapters

            cmd.append(str(output_path))

            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"Applied metadata to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to apply metadata: {e}")
            return False
