"""Scene-aware processing for intelligent video restoration.

This module provides scene detection and adaptive processing capabilities:
- Scene change detection using histogram comparison and SSIM
- Content-type analysis for each scene (faces, action, static, etc.)
- Per-scene enhancement parameter optimization
- Quality-based frame skipping for efficient processing
"""

import logging
import subprocess
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable

logger = logging.getLogger(__name__)


class SceneType(Enum):
    """Types of scenes for adaptive processing."""
    STATIC = auto()       # Little motion, can use higher quality settings
    ACTION = auto()       # Fast motion, may need deblurring
    DIALOG = auto()       # Face-focused, prioritize face restoration
    TRANSITION = auto()   # Scene transition, handle carefully
    LOW_QUALITY = auto()  # Needs aggressive enhancement
    UNKNOWN = auto()      # Default/unclassified


class TransitionType(Enum):
    """Types of scene transitions."""
    HARD_CUT = auto()      # Instant scene change
    FADE = auto()          # Gradual fade in/out
    DISSOLVE = auto()      # Cross-dissolve between scenes
    WIPE = auto()          # Wipe transition
    UNKNOWN = auto()       # Unidentified transition


@dataclass
class Scene:
    """Represents a detected scene in the video.

    Attributes:
        start_frame: First frame of the scene (0-indexed)
        end_frame: Last frame of the scene (inclusive)
        duration_frames: Number of frames in the scene
        scene_type: Classification of scene content
        complexity: Complexity score (0.0-1.0, higher = more complex)
        transition_in: Type of transition entering this scene
        transition_out: Type of transition exiting this scene
        avg_brightness: Average brightness level (0-255)
        avg_motion: Average motion intensity (0.0-1.0)
        has_faces: Whether faces were detected in the scene
        face_ratio: Ratio of frames with faces (0.0-1.0)
        quality_score: Estimated quality score (0.0-1.0)
    """
    start_frame: int
    end_frame: int
    duration_frames: int
    scene_type: SceneType = SceneType.UNKNOWN
    complexity: float = 0.5
    transition_in: TransitionType = TransitionType.HARD_CUT
    transition_out: TransitionType = TransitionType.HARD_CUT
    avg_brightness: float = 128.0
    avg_motion: float = 0.0
    has_faces: bool = False
    face_ratio: float = 0.0
    quality_score: float = 0.5

    def __post_init__(self):
        """Validate and compute derived fields."""
        if self.duration_frames <= 0:
            self.duration_frames = self.end_frame - self.start_frame + 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "scene_type": self.scene_type.name,
            "complexity": self.complexity,
            "transition_in": self.transition_in.name,
            "transition_out": self.transition_out.name,
            "avg_brightness": self.avg_brightness,
            "avg_motion": self.avg_motion,
            "has_faces": self.has_faces,
            "face_ratio": self.face_ratio,
            "quality_score": self.quality_score,
        }


@dataclass
class SceneEnhancementParams:
    """Per-scene enhancement parameters.

    Attributes:
        denoise_strength: Denoising intensity (0.0-1.0)
        sharpness: Sharpening intensity (0.0-1.0)
        model_override: Override model name for this scene (None = use default)
        face_restore_strength: Face restoration intensity (0.0-1.0)
        deblur_strength: Motion deblurring strength (0.0-1.0)
        contrast_boost: Contrast enhancement factor (0.0-2.0, 1.0 = no change)
        brightness_adjust: Brightness adjustment (-128 to 128)
        skip_processing: Whether to skip enhancement for this scene
        tile_size_override: Override tile size for this scene (None = auto)
    """
    denoise_strength: float = 0.3
    sharpness: float = 0.5
    model_override: Optional[str] = None
    face_restore_strength: float = 0.8
    deblur_strength: float = 0.0
    contrast_boost: float = 1.0
    brightness_adjust: float = 0.0
    skip_processing: bool = False
    tile_size_override: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "denoise_strength": self.denoise_strength,
            "sharpness": self.sharpness,
            "model_override": self.model_override,
            "face_restore_strength": self.face_restore_strength,
            "deblur_strength": self.deblur_strength,
            "contrast_boost": self.contrast_boost,
            "brightness_adjust": self.brightness_adjust,
            "skip_processing": self.skip_processing,
            "tile_size_override": self.tile_size_override,
        }


@dataclass
class SceneAnalysisResult:
    """Complete scene analysis results for a video.

    Attributes:
        scenes: List of detected scenes
        total_scenes: Total number of scenes
        avg_scene_length: Average scene length in frames
        dominant_scene_type: Most common scene type
        total_frames: Total frame count in video
        hard_cut_count: Number of hard cuts detected
        transition_count: Number of gradual transitions detected
        processing_recommendations: General recommendations
    """
    scenes: List[Scene] = field(default_factory=list)
    total_scenes: int = 0
    avg_scene_length: float = 0.0
    dominant_scene_type: SceneType = SceneType.UNKNOWN
    total_frames: int = 0
    hard_cut_count: int = 0
    transition_count: int = 0
    processing_recommendations: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute derived statistics."""
        if self.scenes and self.total_scenes == 0:
            self.total_scenes = len(self.scenes)
        if self.scenes and self.avg_scene_length == 0.0:
            self.avg_scene_length = sum(s.duration_frames for s in self.scenes) / len(self.scenes)
        if self.scenes and self.total_frames == 0:
            self.total_frames = self.scenes[-1].end_frame + 1 if self.scenes else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scenes": [s.to_dict() for s in self.scenes],
            "total_scenes": self.total_scenes,
            "avg_scene_length": self.avg_scene_length,
            "dominant_scene_type": self.dominant_scene_type.name,
            "total_frames": self.total_frames,
            "hard_cut_count": self.hard_cut_count,
            "transition_count": self.transition_count,
            "processing_recommendations": self.processing_recommendations,
        }


class SceneDetector:
    """Detects scene changes in video frames.

    Uses multiple methods for scene detection:
    - Histogram comparison for color distribution changes
    - SSIM (Structural Similarity) for structural changes
    - Motion estimation for action detection
    """

    def __init__(
        self,
        histogram_threshold: float = 0.3,
        ssim_threshold: float = 0.7,
        min_scene_length: int = 15,
        detect_transitions: bool = True,
        transition_window: int = 10,
    ):
        """Initialize the scene detector.

        Args:
            histogram_threshold: Threshold for histogram difference (0-1)
                                Lower = more sensitive to color changes
            ssim_threshold: Threshold for SSIM (0-1)
                           Lower = more sensitive to structural changes
            min_scene_length: Minimum frames for a valid scene
            detect_transitions: Whether to detect gradual transitions
            transition_window: Number of frames to analyze for transitions
        """
        self.histogram_threshold = histogram_threshold
        self.ssim_threshold = ssim_threshold
        self.min_scene_length = min_scene_length
        self.detect_transitions = detect_transitions
        self.transition_window = transition_window

    def detect_scenes(
        self,
        frames_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Scene]:
        """Detect scenes from a directory of frames.

        Args:
            frames_dir: Directory containing frame images
            progress_callback: Optional callback for progress updates (0.0-1.0)

        Returns:
            List of detected Scene objects
        """
        logger.info(f"Detecting scenes in: {frames_dir}")

        # Get sorted list of frames
        frames = self._get_sorted_frames(frames_dir)
        if len(frames) < 2:
            logger.warning("Not enough frames for scene detection")
            if frames:
                return [Scene(
                    start_frame=0,
                    end_frame=len(frames) - 1,
                    duration_frames=len(frames),
                    scene_type=SceneType.UNKNOWN,
                )]
            return []

        total_frames = len(frames)
        scene_boundaries = []
        transition_info = {}

        # Compare consecutive frames
        for i in range(1, total_frames):
            if progress_callback:
                progress_callback(i / total_frames)

            prev_frame = frames[i - 1]
            curr_frame = frames[i]

            # Calculate similarity metrics
            hist_diff = self._calculate_histogram_difference(prev_frame, curr_frame)
            ssim_score = self._calculate_ssim(prev_frame, curr_frame)

            # Detect scene change
            is_scene_change = (
                hist_diff > self.histogram_threshold or
                ssim_score < self.ssim_threshold
            )

            if is_scene_change:
                # Determine transition type
                if self.detect_transitions and i >= self.transition_window:
                    transition_type = self._detect_transition_type(
                        frames, i, hist_diff, ssim_score
                    )
                else:
                    transition_type = TransitionType.HARD_CUT

                scene_boundaries.append(i)
                transition_info[i] = transition_type

        # Build scenes from boundaries
        scenes = self._build_scenes_from_boundaries(
            scene_boundaries, transition_info, total_frames
        )

        # Filter out scenes that are too short
        scenes = self._merge_short_scenes(scenes)

        logger.info(f"Detected {len(scenes)} scenes in {total_frames} frames")
        return scenes

    def detect_scenes_ffmpeg(
        self,
        video_path: Path,
        threshold: float = 0.4,
    ) -> List[Scene]:
        """Detect scenes using FFmpeg's scene detection filter.

        This is faster than frame-by-frame analysis but less accurate
        for gradual transitions.

        Args:
            video_path: Path to video file
            threshold: Scene detection threshold (0-1)

        Returns:
            List of detected Scene objects
        """
        logger.info(f"Detecting scenes with FFmpeg: {video_path}")

        # Use FFmpeg's select filter with scene detection
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-show_frames',
            '-print_format', 'json',
            '-f', 'lavfi',
            f"movie='{video_path}',select='gt(scene,{threshold})'",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )

            if result.returncode != 0:
                # Fallback to simpler detection
                return self._detect_scenes_ffmpeg_simple(video_path, threshold)

            data = json.loads(result.stdout) if result.stdout.strip() else {}
            frames_data = data.get('frames', [])

            # Get total frame count
            total_frames = self._get_frame_count(video_path)

            # Extract scene boundaries
            boundaries = [0]
            for frame_info in frames_data:
                frame_num = int(frame_info.get('coded_picture_number', 0))
                if frame_num > 0:
                    boundaries.append(frame_num)

            if boundaries[-1] != total_frames - 1:
                boundaries.append(total_frames - 1)

            # Build scenes
            scenes = []
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1] - 1 if i < len(boundaries) - 2 else boundaries[i + 1]

                scenes.append(Scene(
                    start_frame=start,
                    end_frame=end,
                    duration_frames=end - start + 1,
                ))

            return scenes

        except Exception as e:
            logger.warning(f"FFmpeg scene detection failed: {e}")
            return self._detect_scenes_ffmpeg_simple(video_path, threshold)

    def _detect_scenes_ffmpeg_simple(
        self,
        video_path: Path,
        threshold: float,
    ) -> List[Scene]:
        """Simplified FFmpeg scene detection using scdet filter."""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-select_streams', 'v:0',
            '-show_entries', 'frame=pkt_pts_time',
            '-of', 'json',
            '-f', 'lavfi',
            f"movie='{video_path}',scdet=threshold={threshold}"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Get total frames and FPS
            total_frames = self._get_frame_count(video_path)
            fps = self._get_fps(video_path)

            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                timestamps = []
                for frame in data.get('frames', []):
                    pts_time = float(frame.get('pkt_pts_time', 0))
                    timestamps.append(pts_time)

                # Convert timestamps to frames
                boundaries = [int(ts * fps) for ts in timestamps]
            else:
                boundaries = []

            # Ensure we have start and end
            if not boundaries or boundaries[0] != 0:
                boundaries.insert(0, 0)
            if boundaries[-1] != total_frames - 1:
                boundaries.append(total_frames - 1)

            # Build scenes
            scenes = []
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1] - 1 if i < len(boundaries) - 2 else boundaries[i + 1]

                if end - start + 1 >= self.min_scene_length:
                    scenes.append(Scene(
                        start_frame=start,
                        end_frame=end,
                        duration_frames=end - start + 1,
                    ))

            # If no valid scenes, return entire video as one scene
            if not scenes:
                scenes = [Scene(
                    start_frame=0,
                    end_frame=total_frames - 1,
                    duration_frames=total_frames,
                )]

            return scenes

        except Exception as e:
            logger.warning(f"Simple scene detection failed: {e}")
            total_frames = self._get_frame_count(video_path)
            return [Scene(
                start_frame=0,
                end_frame=max(0, total_frames - 1),
                duration_frames=total_frames or 1,
            )]

    def _get_sorted_frames(self, frames_dir: Path) -> List[Path]:
        """Get sorted list of frame files."""
        frames = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            frames.extend(frames_dir.glob(ext))
        return sorted(frames)

    def _calculate_histogram_difference(
        self,
        frame1: Path,
        frame2: Path,
    ) -> float:
        """Calculate histogram difference between two frames.

        Uses FFmpeg to compare color histograms.

        Returns:
            Difference score (0-1, higher = more different)
        """
        # Use FFmpeg signalstats to compare
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-f', 'lavfi',
            '-i', f"movie='{frame1}',movie='{frame2}'[b];[0][b]blend=all_mode=difference,signalstats",
            '-show_entries', 'frame_tags=lavfi.signalstats.YAVG',
            '-print_format', 'json',
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                frames = data.get('frames', [{}])
                if frames:
                    tags = frames[0].get('tags', {})
                    diff_avg = float(tags.get('lavfi.signalstats.YAVG', 0))
                    # Normalize to 0-1 range (max diff would be ~255)
                    return min(1.0, diff_avg / 64.0)
        except Exception as e:
            logger.debug(f"Histogram comparison failed: {e}")

        # Fallback: use file size difference as rough estimate
        try:
            size1 = frame1.stat().st_size
            size2 = frame2.stat().st_size
            avg_size = (size1 + size2) / 2
            size_diff = abs(size1 - size2) / avg_size if avg_size > 0 else 0
            return min(1.0, size_diff * 2)  # Scale for sensitivity
        except Exception:
            return 0.5

    def _calculate_ssim(
        self,
        frame1: Path,
        frame2: Path,
    ) -> float:
        """Calculate SSIM between two frames.

        Returns:
            SSIM score (0-1, higher = more similar)
        """
        cmd = [
            'ffmpeg', '-i', str(frame1), '-i', str(frame2),
            '-lavfi', 'ssim=stats_file=-',
            '-f', 'null', '-'
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )

            # Parse SSIM from stderr
            output = result.stderr
            if 'All:' in output:
                # Format: "All:0.987654 (23.456789)"
                ssim_part = output.split('All:')[-1].split()[0]
                return float(ssim_part)
        except Exception as e:
            logger.debug(f"SSIM calculation failed: {e}")

        # Fallback: inverse of histogram difference
        hist_diff = self._calculate_histogram_difference(frame1, frame2)
        return 1.0 - hist_diff

    def _detect_transition_type(
        self,
        frames: List[Path],
        change_frame: int,
        hist_diff: float,
        ssim_score: float,
    ) -> TransitionType:
        """Detect the type of scene transition.

        Args:
            frames: List of all frames
            change_frame: Frame index where change was detected
            hist_diff: Histogram difference at change point
            ssim_score: SSIM score at change point

        Returns:
            Detected transition type
        """
        # Analyze frames around the transition
        start_idx = max(0, change_frame - self.transition_window)
        end_idx = min(len(frames), change_frame + self.transition_window)

        # Calculate differences for the window
        window_diffs = []
        for i in range(start_idx + 1, end_idx):
            diff = self._calculate_histogram_difference(frames[i-1], frames[i])
            window_diffs.append(diff)

        if not window_diffs:
            return TransitionType.HARD_CUT

        avg_diff = sum(window_diffs) / len(window_diffs)
        max_diff = max(window_diffs)

        # Hard cut: sudden large change
        if hist_diff > 0.6 and hist_diff / max(avg_diff, 0.01) > 3.0:
            return TransitionType.HARD_CUT

        # Fade: gradual brightness change
        if avg_diff > 0.1 and max_diff < 0.4:
            # Check for consistent gradual change (fade pattern)
            increasing = all(window_diffs[i] <= window_diffs[i+1]
                           for i in range(len(window_diffs)-1))
            decreasing = all(window_diffs[i] >= window_diffs[i+1]
                           for i in range(len(window_diffs)-1))
            if increasing or decreasing:
                return TransitionType.FADE

        # Dissolve: overlapping images
        if 0.2 < avg_diff < 0.5 and ssim_score > 0.4:
            return TransitionType.DISSOLVE

        # Wipe: directional transition
        # Would need more sophisticated edge detection
        # For now, classify as unknown
        if avg_diff > 0.15 and max_diff > 0.3:
            return TransitionType.UNKNOWN

        return TransitionType.HARD_CUT

    def _build_scenes_from_boundaries(
        self,
        boundaries: List[int],
        transition_info: Dict[int, TransitionType],
        total_frames: int,
    ) -> List[Scene]:
        """Build Scene objects from detected boundaries."""
        scenes = []

        # Add implicit start if not present
        if not boundaries or boundaries[0] != 0:
            boundaries = [0] + boundaries

        # Add implicit end
        if boundaries[-1] != total_frames - 1:
            boundaries.append(total_frames)

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1] - 1

            # Get transitions
            trans_in = transition_info.get(start, TransitionType.HARD_CUT)
            trans_out = transition_info.get(boundaries[i + 1], TransitionType.HARD_CUT)

            scene = Scene(
                start_frame=start,
                end_frame=end,
                duration_frames=end - start + 1,
                transition_in=trans_in if i > 0 else TransitionType.HARD_CUT,
                transition_out=trans_out,
            )
            scenes.append(scene)

        return scenes

    def _merge_short_scenes(self, scenes: List[Scene]) -> List[Scene]:
        """Merge scenes that are too short."""
        if not scenes:
            return scenes

        merged = []
        current = scenes[0]

        for next_scene in scenes[1:]:
            if current.duration_frames < self.min_scene_length:
                # Merge with next scene
                current = Scene(
                    start_frame=current.start_frame,
                    end_frame=next_scene.end_frame,
                    duration_frames=next_scene.end_frame - current.start_frame + 1,
                    transition_in=current.transition_in,
                    transition_out=next_scene.transition_out,
                )
            else:
                merged.append(current)
                current = next_scene

        # Don't forget the last scene
        if current.duration_frames >= self.min_scene_length:
            merged.append(current)
        elif merged:
            # Merge last short scene with previous
            prev = merged[-1]
            merged[-1] = Scene(
                start_frame=prev.start_frame,
                end_frame=current.end_frame,
                duration_frames=current.end_frame - prev.start_frame + 1,
                transition_in=prev.transition_in,
                transition_out=current.transition_out,
            )
        else:
            # Only one scene, keep it regardless of length
            merged.append(current)

        return merged

    def _get_frame_count(self, video_path: Path) -> int:
        """Get total frame count from video."""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-count_frames',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_read_frames',
            '-print_format', 'json',
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                streams = data.get('streams', [{}])
                if streams:
                    return int(streams[0].get('nb_read_frames', 0))
        except Exception as e:
            logger.debug(f"Frame count detection failed: {e}")

        # Fallback: estimate from duration and FPS
        fps = self._get_fps(video_path)
        duration = self._get_duration(video_path)
        return int(duration * fps)

    def _get_fps(self, video_path: Path) -> float:
        """Get FPS from video."""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-print_format', 'json',
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                streams = data.get('streams', [{}])
                if streams:
                    fps_str = streams[0].get('r_frame_rate', '24/1')
                    if '/' in fps_str:
                        num, den = map(int, fps_str.split('/'))
                        return num / den if den else 24.0
                    return float(fps_str)
        except Exception:
            pass
        return 24.0

    def _get_duration(self, video_path: Path) -> float:
        """Get duration from video in seconds."""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-print_format', 'json',
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data.get('format', {}).get('duration', 0))
        except Exception:
            pass
        return 0.0


class SceneAnalyzer:
    """Analyzes scenes for content type and enhancement parameters."""

    def __init__(
        self,
        face_detection_enabled: bool = True,
        motion_analysis_enabled: bool = True,
        quality_threshold: float = 0.8,
    ):
        """Initialize the scene analyzer.

        Args:
            face_detection_enabled: Whether to detect faces
            motion_analysis_enabled: Whether to analyze motion
            quality_threshold: Threshold for "high quality" classification
        """
        self.face_detection_enabled = face_detection_enabled
        self.motion_analysis_enabled = motion_analysis_enabled
        self.quality_threshold = quality_threshold

    def analyze_scene(
        self,
        scene: Scene,
        frames_dir: Path,
    ) -> SceneEnhancementParams:
        """Analyze a scene and determine enhancement parameters.

        Args:
            scene: Scene to analyze
            frames_dir: Directory containing frame images

        Returns:
            SceneEnhancementParams optimized for this scene
        """
        logger.debug(f"Analyzing scene: frames {scene.start_frame}-{scene.end_frame}")

        # Get sample frames from the scene
        sample_frames = self._get_scene_samples(scene, frames_dir)

        if not sample_frames:
            return SceneEnhancementParams()

        # Analyze scene characteristics
        brightness = self._analyze_brightness(sample_frames)
        motion = self._analyze_motion(sample_frames)
        faces = self._detect_faces(sample_frames) if self.face_detection_enabled else (False, 0.0)
        quality = self._estimate_quality(sample_frames)

        # Update scene metadata
        scene.avg_brightness = brightness
        scene.avg_motion = motion
        scene.has_faces = faces[0]
        scene.face_ratio = faces[1]
        scene.quality_score = quality

        # Determine scene type
        scene.scene_type = self._classify_scene_type(
            brightness, motion, faces[0], faces[1], quality
        )

        # Generate enhancement parameters
        params = self._generate_enhancement_params(scene)

        logger.debug(f"Scene type: {scene.scene_type.name}, quality: {quality:.2f}")
        return params

    def analyze_all_scenes(
        self,
        scenes: List[Scene],
        frames_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> SceneAnalysisResult:
        """Analyze all scenes and generate comprehensive results.

        Args:
            scenes: List of scenes to analyze
            frames_dir: Directory containing frame images
            progress_callback: Optional callback for progress updates

        Returns:
            SceneAnalysisResult with all scenes analyzed
        """
        logger.info(f"Analyzing {len(scenes)} scenes")

        result = SceneAnalysisResult(scenes=scenes)
        scene_type_counts: Dict[SceneType, int] = {}

        for i, scene in enumerate(scenes):
            if progress_callback:
                progress_callback((i + 1) / len(scenes))

            # Analyze each scene
            self.analyze_scene(scene, frames_dir)

            # Count scene types
            scene_type_counts[scene.scene_type] = scene_type_counts.get(
                scene.scene_type, 0
            ) + 1

            # Count transitions
            if scene.transition_in == TransitionType.HARD_CUT:
                result.hard_cut_count += 1
            elif scene.transition_in != TransitionType.UNKNOWN:
                result.transition_count += 1

        # Determine dominant scene type
        if scene_type_counts:
            result.dominant_scene_type = max(
                scene_type_counts, key=scene_type_counts.get
            )

        # Update statistics
        result.total_scenes = len(scenes)
        if scenes:
            result.avg_scene_length = sum(s.duration_frames for s in scenes) / len(scenes)
            result.total_frames = scenes[-1].end_frame + 1

        # Generate recommendations
        result.processing_recommendations = self._generate_recommendations(result)

        return result

    def get_adaptive_params(
        self,
        frame_path: Path,
        scene: Scene,
    ) -> Dict[str, Any]:
        """Get adaptive parameters for a specific frame within a scene.

        Args:
            frame_path: Path to the frame image
            scene: Scene containing the frame

        Returns:
            Dictionary of adaptive processing parameters
        """
        params = {}

        # Base parameters from scene type
        if scene.scene_type == SceneType.STATIC:
            params['denoise'] = 0.2
            params['sharpness'] = 0.7
            params['quality_level'] = 'high'
        elif scene.scene_type == SceneType.ACTION:
            params['denoise'] = 0.3
            params['sharpness'] = 0.5
            params['deblur'] = 0.4
            params['quality_level'] = 'medium'
        elif scene.scene_type == SceneType.DIALOG:
            params['denoise'] = 0.25
            params['sharpness'] = 0.6
            params['face_enhance'] = True
            params['quality_level'] = 'high'
        elif scene.scene_type == SceneType.TRANSITION:
            params['denoise'] = 0.15
            params['sharpness'] = 0.4
            params['blend_mode'] = True
            params['quality_level'] = 'medium'
        elif scene.scene_type == SceneType.LOW_QUALITY:
            params['denoise'] = 0.6
            params['sharpness'] = 0.8
            params['aggressive_enhance'] = True
            params['quality_level'] = 'low'
        else:
            params['denoise'] = 0.3
            params['sharpness'] = 0.5
            params['quality_level'] = 'medium'

        # Adjust for scene characteristics
        if scene.avg_brightness < 80:
            params['brightness_boost'] = 20
        elif scene.avg_brightness > 200:
            params['brightness_reduce'] = 15

        if scene.complexity > 0.7:
            params['tile_size'] = 256  # Smaller tiles for complex scenes

        return params

    def should_skip_frame(
        self,
        frame_path: Path,
        quality_threshold: float = 0.85,
    ) -> bool:
        """Determine if a frame is high-quality enough to skip processing.

        Args:
            frame_path: Path to the frame image
            quality_threshold: Minimum quality score to skip (0-1)

        Returns:
            True if frame should skip enhancement processing
        """
        quality = self._estimate_frame_quality(frame_path)

        skip = quality >= quality_threshold
        if skip:
            logger.debug(f"Skipping high-quality frame: {frame_path.name} (quality={quality:.2f})")

        return skip

    def _get_scene_samples(
        self,
        scene: Scene,
        frames_dir: Path,
        max_samples: int = 5,
    ) -> List[Path]:
        """Get sample frames from a scene."""
        frames = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            frames.extend(frames_dir.glob(ext))
        frames = sorted(frames)

        # Filter to scene range
        scene_frames = []
        for frame in frames:
            # Extract frame number from filename
            try:
                frame_num = int(frame.stem.split('_')[-1])
                if scene.start_frame <= frame_num <= scene.end_frame:
                    scene_frames.append(frame)
            except ValueError:
                # If we can't parse, include based on index
                idx = frames.index(frame)
                if scene.start_frame <= idx <= scene.end_frame:
                    scene_frames.append(frame)

        # Sample evenly across the scene
        if len(scene_frames) <= max_samples:
            return scene_frames

        step = len(scene_frames) // max_samples
        return scene_frames[::step][:max_samples]

    def _analyze_brightness(self, frames: List[Path]) -> float:
        """Analyze average brightness of frames."""
        brightnesses = []

        for frame in frames:
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-f', 'lavfi',
                '-i', f"movie='{frame}',signalstats",
                '-show_entries', 'frame_tags=lavfi.signalstats.YAVG',
                '-print_format', 'json'
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    data = json.loads(result.stdout)
                    frames_data = data.get('frames', [{}])
                    if frames_data:
                        brightness = float(
                            frames_data[0].get('tags', {}).get(
                                'lavfi.signalstats.YAVG', 128
                            )
                        )
                        brightnesses.append(brightness)
            except Exception:
                brightnesses.append(128.0)

        return sum(brightnesses) / len(brightnesses) if brightnesses else 128.0

    def _analyze_motion(self, frames: List[Path]) -> float:
        """Analyze motion between consecutive frames."""
        if len(frames) < 2:
            return 0.0

        motion_scores = []
        for i in range(1, len(frames)):
            # Use PSNR as inverse motion indicator
            cmd = [
                'ffmpeg', '-i', str(frames[i-1]), '-i', str(frames[i]),
                '-lavfi', 'psnr=stats_file=-',
                '-f', 'null', '-'
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                output = result.stderr
                if 'average:' in output:
                    psnr_part = output.split('average:')[-1].split()[0]
                    psnr = float(psnr_part)
                    # Lower PSNR = more difference = more motion
                    # Normalize: 40+ = no motion, <20 = high motion
                    motion = max(0.0, min(1.0, (40 - psnr) / 30))
                    motion_scores.append(motion)
            except Exception:
                motion_scores.append(0.3)  # Default moderate motion

        return sum(motion_scores) / len(motion_scores) if motion_scores else 0.0

    def _detect_faces(self, frames: List[Path]) -> Tuple[bool, float]:
        """Detect faces in frames.

        Returns:
            Tuple of (has_faces, face_ratio)
        """
        face_count = 0

        for frame in frames:
            # Use FFmpeg's facedetect filter
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-f', 'lavfi',
                '-i', f"movie='{frame}',facedetect",
                '-show_entries', 'frame_tags',
                '-print_format', 'json'
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0 and result.stdout.strip():
                    data = json.loads(result.stdout)
                    frames_data = data.get('frames', [{}])
                    if frames_data:
                        tags = frames_data[0].get('tags', {})
                        # Count face-related tags
                        has_face = any('face' in k.lower() for k in tags.keys())
                        if has_face:
                            face_count += 1
            except Exception:
                pass

        face_ratio = face_count / len(frames) if frames else 0.0
        return (face_ratio > 0.0, face_ratio)

    def _estimate_quality(self, frames: List[Path]) -> float:
        """Estimate overall quality of frames."""
        quality_scores = []

        for frame in frames:
            quality_scores.append(self._estimate_frame_quality(frame))

        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.5

    def _estimate_frame_quality(self, frame: Path) -> float:
        """Estimate quality of a single frame."""
        # Use multiple signals for quality estimation

        # 1. File size (larger = more detail)
        try:
            size = frame.stat().st_size
            # Assume ~1MB for high quality 1080p PNG
            size_score = min(1.0, size / (1024 * 1024))
        except Exception:
            size_score = 0.5

        # 2. Sharpness via Laplacian variance
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-f', 'lavfi',
            '-i', f"movie='{frame}',signalstats",
            '-show_entries', 'frame_tags',
            '-print_format', 'json'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                frames_data = data.get('frames', [{}])
                if frames_data:
                    tags = frames_data[0].get('tags', {})
                    # Higher YHIGH suggests more detail
                    yhigh = float(tags.get('lavfi.signalstats.YHIGH', 200))
                    detail_score = min(1.0, yhigh / 250)
                else:
                    detail_score = 0.5
            else:
                detail_score = 0.5
        except Exception:
            detail_score = 0.5

        # Combine scores
        quality = (size_score * 0.3 + detail_score * 0.7)
        return quality

    def _classify_scene_type(
        self,
        brightness: float,
        motion: float,
        has_faces: bool,
        face_ratio: float,
        quality: float,
    ) -> SceneType:
        """Classify scene based on analyzed characteristics."""
        # Low quality takes priority
        if quality < 0.3:
            return SceneType.LOW_QUALITY

        # Dialog scenes (face-focused)
        if has_faces and face_ratio > 0.6:
            return SceneType.DIALOG

        # Action scenes (high motion)
        if motion > 0.5:
            return SceneType.ACTION

        # Static scenes (low motion)
        if motion < 0.15:
            return SceneType.STATIC

        # Default to unknown
        return SceneType.UNKNOWN

    def _generate_enhancement_params(self, scene: Scene) -> SceneEnhancementParams:
        """Generate enhancement parameters based on scene analysis."""
        params = SceneEnhancementParams()

        # Scene type specific parameters
        if scene.scene_type == SceneType.STATIC:
            params.denoise_strength = 0.2
            params.sharpness = 0.7
            params.deblur_strength = 0.0

        elif scene.scene_type == SceneType.ACTION:
            params.denoise_strength = 0.3
            params.sharpness = 0.5
            params.deblur_strength = 0.5

        elif scene.scene_type == SceneType.DIALOG:
            params.denoise_strength = 0.25
            params.sharpness = 0.6
            params.face_restore_strength = 1.0

        elif scene.scene_type == SceneType.TRANSITION:
            params.denoise_strength = 0.15
            params.sharpness = 0.4

        elif scene.scene_type == SceneType.LOW_QUALITY:
            params.denoise_strength = 0.7
            params.sharpness = 0.8
            params.deblur_strength = 0.3

        else:
            params.denoise_strength = 0.3
            params.sharpness = 0.5

        # Brightness adjustments
        if scene.avg_brightness < 60:
            params.brightness_adjust = min(40, 80 - scene.avg_brightness)
        elif scene.avg_brightness > 220:
            params.brightness_adjust = max(-40, 200 - scene.avg_brightness)

        # Quality-based model selection
        if scene.quality_score > 0.8:
            params.skip_processing = True  # High quality, skip enhancement
        elif scene.quality_score < 0.4:
            params.model_override = "realesrgan-x4plus"  # Use best model

        # Complexity-based tile size
        if scene.complexity > 0.7:
            params.tile_size_override = 256

        return params

    def _generate_recommendations(
        self,
        result: SceneAnalysisResult,
    ) -> Dict[str, Any]:
        """Generate processing recommendations from analysis."""
        recommendations = {}

        # Model recommendation based on dominant content
        if result.dominant_scene_type == SceneType.DIALOG:
            recommendations['model'] = 'realesrgan-x4plus'
            recommendations['face_restoration'] = True
        elif result.dominant_scene_type == SceneType.ACTION:
            recommendations['model'] = 'realesrgan-x4plus'
            recommendations['deblur'] = True
        elif result.dominant_scene_type == SceneType.STATIC:
            recommendations['model'] = 'realesrgan-x4plus'
            recommendations['high_quality'] = True
        else:
            recommendations['model'] = 'realesrgan-x4plus'

        # Processing approach based on scene count
        if result.total_scenes > 20:
            recommendations['parallel_processing'] = True
            recommendations['batch_size'] = 4
        else:
            recommendations['sequential_processing'] = True

        # Transition handling
        if result.transition_count > 5:
            recommendations['careful_transitions'] = True
            recommendations['blend_frames'] = True

        return recommendations


def detect_and_analyze_scenes(
    frames_dir: Path,
    histogram_threshold: float = 0.3,
    ssim_threshold: float = 0.7,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> SceneAnalysisResult:
    """Convenience function to detect and analyze scenes in one call.

    Args:
        frames_dir: Directory containing frame images
        histogram_threshold: Threshold for scene detection
        ssim_threshold: SSIM threshold for scene detection
        progress_callback: Callback(stage, progress) for updates

    Returns:
        SceneAnalysisResult with detected and analyzed scenes
    """
    detector = SceneDetector(
        histogram_threshold=histogram_threshold,
        ssim_threshold=ssim_threshold,
    )

    analyzer = SceneAnalyzer()

    # Detect scenes
    if progress_callback:
        progress_callback("detection", 0.0)

    def detection_progress(p):
        if progress_callback:
            progress_callback("detection", p)

    scenes = detector.detect_scenes(frames_dir, detection_progress)

    # Analyze scenes
    if progress_callback:
        progress_callback("analysis", 0.0)

    def analysis_progress(p):
        if progress_callback:
            progress_callback("analysis", p)

    result = analyzer.analyze_all_scenes(scenes, frames_dir, analysis_progress)

    return result
