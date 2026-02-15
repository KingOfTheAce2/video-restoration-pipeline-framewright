"""Progress tracking and checkpoint management for crash recovery."""

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .job_store import JobStore, JobRecord, FrameRecord, JobState, FrameState

logger = logging.getLogger(__name__)


@dataclass
class ProgressSnapshot:
    """Snapshot of job progress at a point in time."""
    job_id: str
    timestamp: datetime

    # Frame counts
    total_frames: int = 0
    frames_completed: int = 0
    frames_failed: int = 0
    frames_pending: int = 0

    # Current position
    current_frame: int = 0
    current_stage: str = ""

    # Timing
    elapsed_seconds: float = 0.0
    avg_frame_time_ms: float = 0.0
    eta_seconds: Optional[float] = None

    # Quality
    avg_psnr: Optional[float] = None
    avg_ssim: Optional[float] = None

    # Resource usage
    vram_used_mb: float = 0.0
    ram_used_mb: float = 0.0
    cpu_percent: float = 0.0

    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.total_frames == 0:
            return 0.0
        return (self.frames_completed / self.total_frames) * 100

    @property
    def eta_formatted(self) -> str:
        """Get ETA as formatted string."""
        if self.eta_seconds is None or self.eta_seconds < 0:
            return "Unknown"

        eta = timedelta(seconds=int(self.eta_seconds))
        if eta.days > 0:
            return f"{eta.days}d {eta.seconds // 3600}h"
        elif eta.seconds >= 3600:
            return f"{eta.seconds // 3600}h {(eta.seconds % 3600) // 60}m"
        elif eta.seconds >= 60:
            return f"{eta.seconds // 60}m {eta.seconds % 60}s"
        else:
            return f"{eta.seconds}s"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "timestamp": self.timestamp.isoformat(),
            "total_frames": self.total_frames,
            "frames_completed": self.frames_completed,
            "frames_failed": self.frames_failed,
            "frames_pending": self.frames_pending,
            "current_frame": self.current_frame,
            "current_stage": self.current_stage,
            "progress_percent": self.progress_percent,
            "elapsed_seconds": self.elapsed_seconds,
            "avg_frame_time_ms": self.avg_frame_time_ms,
            "eta_seconds": self.eta_seconds,
            "eta_formatted": self.eta_formatted,
            "avg_psnr": self.avg_psnr,
            "avg_ssim": self.avg_ssim,
            "vram_used_mb": self.vram_used_mb,
            "ram_used_mb": self.ram_used_mb,
            "cpu_percent": self.cpu_percent,
        }


class ProgressTracker:
    """Tracks and persists job progress with auto-save."""

    def __init__(
        self,
        store: JobStore,
        auto_save_interval: float = 5.0,
        history_size: int = 100,
    ):
        self.store = store
        self.auto_save_interval = auto_save_interval
        self.history_size = history_size

        # Tracking state
        self._active_jobs: Dict[str, JobRecord] = {}
        self._frame_times: Dict[str, List[float]] = {}  # job_id -> recent frame times
        self._quality_scores: Dict[str, List[tuple]] = {}  # job_id -> (psnr, ssim) list
        self._start_times: Dict[str, datetime] = {}
        self._history: Dict[str, List[ProgressSnapshot]] = {}

        # Callbacks
        self._progress_callbacks: List[Callable[[ProgressSnapshot], None]] = []

        # Auto-save thread
        self._lock = threading.Lock()
        self._running = False
        self._save_thread: Optional[threading.Thread] = None

    def add_progress_callback(self, callback: Callable[[ProgressSnapshot], None]) -> None:
        """Add callback for progress updates."""
        self._progress_callbacks.append(callback)

    def _notify_progress(self, snapshot: ProgressSnapshot) -> None:
        """Notify callbacks of progress update."""
        for callback in self._progress_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def start_tracking(self, job: JobRecord) -> None:
        """Start tracking a job."""
        with self._lock:
            self._active_jobs[job.job_id] = job
            self._frame_times[job.job_id] = []
            self._quality_scores[job.job_id] = []
            self._start_times[job.job_id] = datetime.now()
            self._history[job.job_id] = []

        # Start auto-save if not running
        if not self._running:
            self._start_auto_save()

        logger.info(f"Started tracking job: {job.job_id}")

    def stop_tracking(self, job_id: str) -> None:
        """Stop tracking a job."""
        with self._lock:
            self._active_jobs.pop(job_id, None)
            self._frame_times.pop(job_id, None)
            self._quality_scores.pop(job_id, None)
            self._start_times.pop(job_id, None)
            # Keep history

        # Stop auto-save if no active jobs
        if not self._active_jobs:
            self._stop_auto_save()

        logger.info(f"Stopped tracking job: {job_id}")

    def update_frame(
        self,
        job_id: str,
        frame_number: int,
        state: FrameState,
        processing_time_ms: float = 0,
        psnr: Optional[float] = None,
        ssim: Optional[float] = None,
        output_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update frame progress."""
        with self._lock:
            if job_id not in self._active_jobs:
                return

            job = self._active_jobs[job_id]

            # Update frame in store
            self.store.update_frame_state(job_id, frame_number, state, output_path, error)

            # Track timing
            if processing_time_ms > 0:
                self._frame_times[job_id].append(processing_time_ms)
                # Keep only recent times
                if len(self._frame_times[job_id]) > 100:
                    self._frame_times[job_id] = self._frame_times[job_id][-100:]

            # Track quality
            if psnr is not None or ssim is not None:
                self._quality_scores[job_id].append((psnr, ssim))
                if len(self._quality_scores[job_id]) > 100:
                    self._quality_scores[job_id] = self._quality_scores[job_id][-100:]

            # Update job progress
            if state == FrameState.COMPLETED:
                job.frames_processed += 1
                job.current_frame = frame_number
            elif state == FrameState.FAILED:
                job.frames_failed += 1
                job.last_error_frame = frame_number
                job.error_message = error

            # Calculate averages
            if self._frame_times[job_id]:
                job.avg_frame_time_ms = sum(self._frame_times[job_id]) / len(self._frame_times[job_id])

    def update_stage(self, job_id: str, stage: str) -> None:
        """Update current processing stage."""
        with self._lock:
            if job_id in self._active_jobs:
                # Could extend JobRecord to track stage
                pass

        # Update in store
        state_map = {
            "extracting": JobState.EXTRACTING,
            "processing": JobState.PROCESSING,
            "encoding": JobState.ENCODING,
            "finalizing": JobState.FINALIZING,
        }
        if stage in state_map:
            self.store.update_job_state(job_id, state_map[stage])

    def get_snapshot(self, job_id: str) -> Optional[ProgressSnapshot]:
        """Get current progress snapshot."""
        with self._lock:
            if job_id not in self._active_jobs:
                return None

            job = self._active_jobs[job_id]
            start_time = self._start_times.get(job_id, datetime.now())
            frame_times = self._frame_times.get(job_id, [])
            quality_scores = self._quality_scores.get(job_id, [])

            # Calculate ETA
            elapsed = (datetime.now() - start_time).total_seconds()
            frames_remaining = job.total_frames - job.frames_processed
            eta = None
            if frame_times and frames_remaining > 0:
                avg_time = sum(frame_times) / len(frame_times)
                eta = (frames_remaining * avg_time) / 1000  # Convert ms to seconds

            # Calculate quality averages
            avg_psnr = None
            avg_ssim = None
            if quality_scores:
                psnr_values = [p for p, _ in quality_scores if p is not None]
                ssim_values = [s for _, s in quality_scores if s is not None]
                if psnr_values:
                    avg_psnr = sum(psnr_values) / len(psnr_values)
                if ssim_values:
                    avg_ssim = sum(ssim_values) / len(ssim_values)

            # Get resource usage
            vram_mb, ram_mb, cpu_pct = self._get_resource_usage()

            snapshot = ProgressSnapshot(
                job_id=job_id,
                timestamp=datetime.now(),
                total_frames=job.total_frames,
                frames_completed=job.frames_processed,
                frames_failed=job.frames_failed,
                frames_pending=job.total_frames - job.frames_processed - job.frames_failed,
                current_frame=job.current_frame,
                current_stage=job.state.value,
                elapsed_seconds=elapsed,
                avg_frame_time_ms=job.avg_frame_time_ms,
                eta_seconds=eta,
                avg_psnr=avg_psnr,
                avg_ssim=avg_ssim,
                vram_used_mb=vram_mb,
                ram_used_mb=ram_mb,
                cpu_percent=cpu_pct,
            )

            return snapshot

    def _get_resource_usage(self) -> tuple:
        """Get current resource usage (VRAM MB, RAM MB, CPU %)."""
        vram_mb = 0.0
        ram_mb = 0.0
        cpu_pct = 0.0

        try:
            import torch
            if torch.cuda.is_available():
                vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        except ImportError:
            pass

        try:
            import psutil
            process = psutil.Process()
            ram_mb = process.memory_info().rss / (1024 * 1024)
            cpu_pct = process.cpu_percent()
        except ImportError:
            pass

        return vram_mb, ram_mb, cpu_pct

    def get_history(self, job_id: str) -> List[ProgressSnapshot]:
        """Get progress history for a job."""
        return self._history.get(job_id, [])

    def _start_auto_save(self) -> None:
        """Start auto-save thread."""
        if self._running:
            return

        self._running = True
        self._save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self._save_thread.start()

    def _stop_auto_save(self) -> None:
        """Stop auto-save thread."""
        self._running = False
        if self._save_thread:
            self._save_thread.join(timeout=2.0)
            self._save_thread = None

    def _auto_save_loop(self) -> None:
        """Auto-save loop."""
        while self._running:
            time.sleep(self.auto_save_interval)

            with self._lock:
                for job_id, job in list(self._active_jobs.items()):
                    try:
                        # Save to store
                        self.store.update_job(job)

                        # Take snapshot
                        snapshot = self.get_snapshot(job_id)
                        if snapshot:
                            # Add to history
                            if job_id not in self._history:
                                self._history[job_id] = []
                            self._history[job_id].append(snapshot)

                            # Trim history
                            if len(self._history[job_id]) > self.history_size:
                                self._history[job_id] = self._history[job_id][-self.history_size:]

                            # Notify callbacks
                            self._notify_progress(snapshot)

                    except Exception as e:
                        logger.error(f"Auto-save error for job {job_id}: {e}")


class CheckpointManager:
    """Manages checkpoints for crash recovery."""

    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        job_id: str,
        frame_number: int,
        state: Dict[str, Any],
    ) -> Path:
        """Save a checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{job_id}_frame{frame_number:08d}_{timestamp}.ckpt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save state
        checkpoint_data = {
            "job_id": job_id,
            "frame_number": frame_number,
            "timestamp": datetime.now().isoformat(),
            "state": state,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(job_id)

        logger.debug(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def load_latest_checkpoint(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint for a job."""
        checkpoints = self._get_checkpoints(job_id)
        if not checkpoints:
            return None

        latest = checkpoints[-1]
        try:
            with open(latest, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded checkpoint: {latest}")
            return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def get_resume_frame(self, job_id: str) -> Optional[int]:
        """Get frame number to resume from."""
        checkpoint = self.load_latest_checkpoint(job_id)
        if checkpoint:
            return checkpoint.get("frame_number", 0)
        return None

    def _get_checkpoints(self, job_id: str) -> List[Path]:
        """Get all checkpoints for a job, sorted by time."""
        pattern = f"{job_id}_frame*.ckpt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        return sorted(checkpoints, key=lambda p: p.stat().st_mtime)

    def _cleanup_old_checkpoints(self, job_id: str) -> None:
        """Remove old checkpoints, keeping only the most recent."""
        checkpoints = self._get_checkpoints(job_id)
        while len(checkpoints) > self.max_checkpoints:
            old = checkpoints.pop(0)
            try:
                old.unlink()
                logger.debug(f"Removed old checkpoint: {old}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint: {e}")

    def delete_checkpoints(self, job_id: str) -> None:
        """Delete all checkpoints for a job."""
        for checkpoint in self._get_checkpoints(job_id):
            try:
                checkpoint.unlink()
            except Exception:
                pass

    def list_all_checkpoints(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all checkpoints grouped by job."""
        result: Dict[str, List[Dict[str, Any]]] = {}

        for checkpoint in self.checkpoint_dir.glob("*.ckpt"):
            try:
                with open(checkpoint, "r") as f:
                    data = json.load(f)
                job_id = data.get("job_id", "unknown")
                if job_id not in result:
                    result[job_id] = []
                result[job_id].append({
                    "path": str(checkpoint),
                    "frame": data.get("frame_number"),
                    "timestamp": data.get("timestamp"),
                })
            except Exception:
                pass

        return result
