"""
Frame Quality Scoring - Identify problem frames for manual review.

Analyzes each frame for quality issues like blur, noise, artifacts,
and exposure problems to help identify frames needing attention.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum
import json

import cv2
import numpy as np


class QualityIssue(Enum):
    """Types of quality issues detected."""
    BLUR = "blur"
    NOISE = "noise"
    OVEREXPOSED = "overexposed"
    UNDEREXPOSED = "underexposed"
    BLOCKING = "blocking"  # Compression artifacts
    BANDING = "banding"  # Color banding
    INTERLACING = "interlacing"
    DUPLICATE = "duplicate"
    BLACK_FRAME = "black_frame"
    CORRUPTED = "corrupted"


@dataclass
class FrameScore:
    """Quality score for a single frame."""
    frame_number: int
    timestamp: float
    overall_score: float  # 0-100, higher is better
    sharpness_score: float
    noise_score: float
    exposure_score: float
    artifact_score: float
    issues: List[QualityIssue] = field(default_factory=list)
    issue_details: Dict[str, float] = field(default_factory=dict)

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def is_problematic(self) -> bool:
        """Frame needs attention if score below 50 or has critical issues."""
        critical = {QualityIssue.CORRUPTED, QualityIssue.BLACK_FRAME}
        return self.overall_score < 50 or any(i in critical for i in self.issues)


@dataclass
class QualityReport:
    """Complete quality analysis report."""
    video_path: str
    total_frames: int
    analyzed_frames: int
    average_score: float
    min_score: float
    max_score: float
    problem_frames: List[FrameScore]
    issue_counts: Dict[str, int]
    score_distribution: Dict[str, int]  # score ranges -> count
    recommendations: List[str]

    def get_worst_frames(self, n: int = 10) -> List[FrameScore]:
        """Get n worst scoring frames."""
        return sorted(self.problem_frames, key=lambda f: f.overall_score)[:n]

    def get_frames_with_issue(self, issue: QualityIssue) -> List[FrameScore]:
        """Get all frames with a specific issue."""
        return [f for f in self.problem_frames if issue in f.issues]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "analyzed_frames": self.analyzed_frames,
            "average_score": self.average_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "problem_frame_count": len(self.problem_frames),
            "problem_frames": [
                {
                    "frame": f.frame_number,
                    "timestamp": f.timestamp,
                    "score": f.overall_score,
                    "issues": [i.value for i in f.issues]
                }
                for f in self.problem_frames
            ],
            "issue_counts": self.issue_counts,
            "score_distribution": self.score_distribution,
            "recommendations": self.recommendations
        }

    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class FrameQualityScorer:
    """
    Analyze video frames for quality issues.

    Scores each frame on multiple quality metrics and identifies
    problem frames that may need manual attention or special processing.
    """

    # Thresholds for issue detection
    BLUR_THRESHOLD = 100  # Laplacian variance below this = blurry
    NOISE_THRESHOLD = 15  # Noise level above this = noisy
    OVEREXPOSED_THRESHOLD = 250  # Mean brightness above this
    UNDEREXPOSED_THRESHOLD = 20  # Mean brightness below this
    BLACK_FRAME_THRESHOLD = 5  # Mean brightness below this
    DUPLICATE_THRESHOLD = 0.99  # Correlation above this = duplicate

    def __init__(
        self,
        sample_rate: int = 1,  # Analyze every Nth frame
        problem_threshold: float = 60.0,  # Score below this = problem
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Initialize scorer.

        Args:
            sample_rate: Analyze every Nth frame (1 = all frames)
            problem_threshold: Frames scoring below this are flagged
            progress_callback: Called with (current_frame, total_frames)
        """
        self.sample_rate = max(1, sample_rate)
        self.problem_threshold = problem_threshold
        self.progress_callback = progress_callback
        self._prev_frame_hash = None

    def _report_progress(self, current: int, total: int) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(current, total)

    def _calculate_sharpness(self, gray: np.ndarray) -> Tuple[float, bool]:
        """
        Calculate sharpness score using Laplacian variance.

        Returns:
            (score 0-100, is_blurry)
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Normalize to 0-100 (typical variance range 0-2000+)
        score = min(100, (variance / 500) * 100)
        is_blurry = variance < self.BLUR_THRESHOLD

        return score, is_blurry

    def _calculate_noise(self, gray: np.ndarray) -> Tuple[float, bool]:
        """
        Estimate noise level.

        Returns:
            (score 0-100 where higher is cleaner, is_noisy)
        """
        # Use high-pass filter to isolate noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blurred)
        noise_level = np.std(noise)

        # Normalize (typical noise std 0-50)
        score = max(0, 100 - (noise_level / 30) * 100)
        is_noisy = noise_level > self.NOISE_THRESHOLD

        return score, is_noisy

    def _calculate_exposure(self, gray: np.ndarray) -> Tuple[float, bool, bool]:
        """
        Analyze exposure.

        Returns:
            (score 0-100, is_overexposed, is_underexposed)
        """
        mean_brightness = np.mean(gray)

        # Check for black frame
        if mean_brightness < self.BLACK_FRAME_THRESHOLD:
            return 0, False, True

        is_overexposed = mean_brightness > self.OVEREXPOSED_THRESHOLD
        is_underexposed = mean_brightness < self.UNDEREXPOSED_THRESHOLD

        # Best exposure around 128 (middle gray)
        deviation = abs(mean_brightness - 128) / 128
        score = 100 * (1 - deviation)

        # Penalize clipping
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        clipped_dark = np.sum(hist[:5]) / gray.size
        clipped_bright = np.sum(hist[251:]) / gray.size

        if clipped_dark > 0.05 or clipped_bright > 0.05:
            score *= 0.8

        return max(0, score), is_overexposed, is_underexposed

    def _detect_blocking(self, gray: np.ndarray) -> Tuple[float, bool]:
        """
        Detect compression blocking artifacts.

        Returns:
            (score 0-100 where higher is cleaner, has_blocking)
        """
        # Look for 8x8 block boundaries (common in JPEG/MPEG)
        h, w = gray.shape

        # Calculate horizontal and vertical differences at 8-pixel intervals
        block_diff = 0
        count = 0

        for i in range(8, h - 8, 8):
            diff = np.abs(gray[i, :].astype(float) - gray[i-1, :].astype(float))
            block_diff += np.mean(diff)
            count += 1

        for j in range(8, w - 8, 8):
            diff = np.abs(gray[:, j].astype(float) - gray[:, j-1].astype(float))
            block_diff += np.mean(diff)
            count += 1

        if count > 0:
            block_diff /= count

        # Compare to overall edge strength
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        overall_edges = np.mean(np.abs(edges))

        # Blocking ratio - higher means more visible blocks
        if overall_edges > 0:
            blocking_ratio = block_diff / overall_edges
        else:
            blocking_ratio = 0

        has_blocking = blocking_ratio > 1.5
        score = max(0, 100 - (blocking_ratio - 1) * 50) if blocking_ratio > 1 else 100

        return score, has_blocking

    def _detect_banding(self, frame: np.ndarray) -> Tuple[float, bool]:
        """
        Detect color banding (posterization).

        Returns:
            (score 0-100 where higher is smoother, has_banding)
        """
        # Convert to LAB for perceptual color analysis
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # Look for flat regions with abrupt transitions
        gradient = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)

        # Banding shows as sparse but strong gradients
        gradient_abs = np.abs(gradient)
        strong_edges = np.sum(gradient_abs > 30)
        weak_edges = np.sum((gradient_abs > 5) & (gradient_abs <= 30))

        if weak_edges > 0:
            banding_ratio = strong_edges / weak_edges
        else:
            banding_ratio = 0

        has_banding = banding_ratio > 2
        score = max(0, 100 - (banding_ratio - 1) * 25) if banding_ratio > 1 else 100

        return score, has_banding

    def _detect_interlacing(self, gray: np.ndarray) -> Tuple[float, bool]:
        """
        Detect interlacing artifacts (combing).

        Returns:
            (score 0-100, has_interlacing)
        """
        # Calculate horizontal differences between adjacent rows
        row_diff = np.abs(gray[::2, :].astype(float) - gray[1::2, :].astype(float))

        # Strong horizontal patterns suggest interlacing
        mean_diff = np.mean(row_diff)

        # Compare to vertical smoothness
        col_diff = np.abs(gray[:, ::2].astype(float) - gray[:, 1::2].astype(float))
        mean_col_diff = np.mean(col_diff)

        if mean_col_diff > 0:
            interlace_ratio = mean_diff / mean_col_diff
        else:
            interlace_ratio = 0

        has_interlacing = interlace_ratio > 1.5 and mean_diff > 10
        score = max(0, 100 - (interlace_ratio - 1) * 30) if interlace_ratio > 1 else 100

        return score, has_interlacing

    def _check_duplicate(self, gray: np.ndarray) -> bool:
        """Check if frame is duplicate of previous."""
        # Use perceptual hash for speed
        small = cv2.resize(gray, (32, 32))
        frame_hash = small.flatten()

        if self._prev_frame_hash is not None:
            # Calculate correlation
            corr = np.corrcoef(frame_hash, self._prev_frame_hash)[0, 1]
            is_dup = corr > self.DUPLICATE_THRESHOLD
        else:
            is_dup = False

        self._prev_frame_hash = frame_hash
        return is_dup

    def _check_corrupted(self, frame: np.ndarray) -> bool:
        """Basic corruption detection."""
        # Check for unusual patterns that suggest corruption
        if frame is None or frame.size == 0:
            return True

        # Check for solid color (often indicates decode error)
        std = np.std(frame)
        if std < 1:
            return True

        # Check for extreme values only
        unique_vals = len(np.unique(frame))
        if unique_vals < 10:
            return True

        return False

    def score_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> FrameScore:
        """
        Score a single frame.

        Args:
            frame: BGR frame
            frame_number: Frame index
            fps: Video FPS for timestamp calculation

        Returns:
            FrameScore with all metrics
        """
        timestamp = frame_number / fps if fps > 0 else 0
        issues = []
        issue_details = {}

        # Check for corruption first
        if self._check_corrupted(frame):
            return FrameScore(
                frame_number=frame_number,
                timestamp=timestamp,
                overall_score=0,
                sharpness_score=0,
                noise_score=0,
                exposure_score=0,
                artifact_score=0,
                issues=[QualityIssue.CORRUPTED]
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate all metrics
        sharpness_score, is_blurry = self._calculate_sharpness(gray)
        noise_score, is_noisy = self._calculate_noise(gray)
        exposure_score, is_over, is_under = self._calculate_exposure(gray)
        blocking_score, has_blocking = self._detect_blocking(gray)
        banding_score, has_banding = self._detect_banding(frame)
        interlace_score, has_interlacing = self._detect_interlacing(gray)
        is_duplicate = self._check_duplicate(gray)

        # Collect issues
        if is_blurry:
            issues.append(QualityIssue.BLUR)
            issue_details["blur_variance"] = sharpness_score
        if is_noisy:
            issues.append(QualityIssue.NOISE)
            issue_details["noise_level"] = 100 - noise_score
        if is_over:
            issues.append(QualityIssue.OVEREXPOSED)
        if is_under:
            issues.append(QualityIssue.UNDEREXPOSED)
            if np.mean(gray) < self.BLACK_FRAME_THRESHOLD:
                issues.append(QualityIssue.BLACK_FRAME)
        if has_blocking:
            issues.append(QualityIssue.BLOCKING)
        if has_banding:
            issues.append(QualityIssue.BANDING)
        if has_interlacing:
            issues.append(QualityIssue.INTERLACING)
        if is_duplicate:
            issues.append(QualityIssue.DUPLICATE)

        # Calculate artifact score (average of artifact-related scores)
        artifact_score = (blocking_score + banding_score + interlace_score) / 3

        # Overall score - weighted average
        overall_score = (
            sharpness_score * 0.3 +
            noise_score * 0.25 +
            exposure_score * 0.25 +
            artifact_score * 0.2
        )

        return FrameScore(
            frame_number=frame_number,
            timestamp=timestamp,
            overall_score=overall_score,
            sharpness_score=sharpness_score,
            noise_score=noise_score,
            exposure_score=exposure_score,
            artifact_score=artifact_score,
            issues=issues,
            issue_details=issue_details
        )

    def analyze_video(
        self,
        video_path: Path,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> QualityReport:
        """
        Analyze entire video for quality issues.

        Args:
            video_path: Path to video file
            start_frame: First frame to analyze
            end_frame: Last frame (None = all frames)

        Returns:
            QualityReport with all findings
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame is None:
            end_frame = total_frames

        # Reset duplicate detection
        self._prev_frame_hash = None

        all_scores = []
        problem_frames = []
        issue_counts = {issue.value: 0 for issue in QualityIssue}

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_num = start_frame
        analyzed_count = 0

        while frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Only analyze sampled frames
            if (frame_num - start_frame) % self.sample_rate == 0:
                score = self.score_frame(frame, frame_num, fps)
                all_scores.append(score.overall_score)
                analyzed_count += 1

                # Track issues
                for issue in score.issues:
                    issue_counts[issue.value] += 1

                # Flag problem frames
                if score.overall_score < self.problem_threshold or score.has_issues:
                    problem_frames.append(score)

                self._report_progress(frame_num, end_frame)

            frame_num += 1

        cap.release()

        # Calculate distribution
        score_distribution = {
            "excellent (90-100)": 0,
            "good (70-89)": 0,
            "fair (50-69)": 0,
            "poor (30-49)": 0,
            "bad (0-29)": 0
        }

        for score in all_scores:
            if score >= 90:
                score_distribution["excellent (90-100)"] += 1
            elif score >= 70:
                score_distribution["good (70-89)"] += 1
            elif score >= 50:
                score_distribution["fair (50-69)"] += 1
            elif score >= 30:
                score_distribution["poor (30-49)"] += 1
            else:
                score_distribution["bad (0-29)"] += 1

        # Generate recommendations
        recommendations = self._generate_recommendations(
            all_scores, issue_counts, problem_frames
        )

        return QualityReport(
            video_path=str(video_path),
            total_frames=total_frames,
            analyzed_frames=analyzed_count,
            average_score=np.mean(all_scores) if all_scores else 0,
            min_score=min(all_scores) if all_scores else 0,
            max_score=max(all_scores) if all_scores else 0,
            problem_frames=problem_frames,
            issue_counts=issue_counts,
            score_distribution=score_distribution,
            recommendations=recommendations
        )

    def _generate_recommendations(
        self,
        scores: List[float],
        issue_counts: Dict[str, int],
        problem_frames: List[FrameScore]
    ) -> List[str]:
        """Generate processing recommendations based on analysis."""
        recommendations = []

        if not scores:
            return ["Unable to analyze video"]

        avg_score = np.mean(scores)

        # Overall quality assessment
        if avg_score >= 80:
            recommendations.append("Video quality is good - light processing recommended")
        elif avg_score >= 50:
            recommendations.append("Video has moderate quality issues - standard processing recommended")
        else:
            recommendations.append("Video has significant quality issues - aggressive processing recommended")

        # Specific issue recommendations
        total_issues = sum(issue_counts.values())

        if issue_counts.get("blur", 0) > total_issues * 0.1:
            recommendations.append(
                "Significant blur detected - consider sharpening or AI upscaling"
            )

        if issue_counts.get("noise", 0) > total_issues * 0.1:
            recommendations.append(
                "High noise levels - recommend denoising (temporal if available)"
            )

        if issue_counts.get("blocking", 0) > total_issues * 0.1:
            recommendations.append(
                "Compression artifacts detected - consider deblocking filter"
            )

        if issue_counts.get("interlacing", 0) > total_issues * 0.05:
            recommendations.append(
                "Interlacing detected - apply deinterlacing (YADIF or QTGMC)"
            )

        if issue_counts.get("banding", 0) > total_issues * 0.1:
            recommendations.append(
                "Color banding present - consider debanding or bit depth increase"
            )

        if issue_counts.get("duplicate", 0) > len(scores) * 0.1:
            recommendations.append(
                "Many duplicate frames - video may be telecined or have frame rate issues"
            )

        if issue_counts.get("black_frame", 0) > 0:
            recommendations.append(
                f"Found {issue_counts['black_frame']} black frames - may need manual review"
            )

        return recommendations
