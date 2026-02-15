"""Job definitions for distributed rendering."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNING = "assigning"
    PROCESSING = "processing"
    MERGING = "merging"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3
    CRITICAL = 4


@dataclass
class FrameRange:
    """Represents a range of frames to process."""
    start: int
    end: int
    step: int = 1

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError(f"End frame {self.end} must be >= start frame {self.start}")
        if self.step < 1:
            raise ValueError("Step must be at least 1")

    def frame_count(self) -> int:
        """Get total number of frames in range."""
        return (self.end - self.start) // self.step + 1

    def split(self, num_chunks: int) -> List["FrameRange"]:
        """Split range into multiple chunks."""
        total = self.frame_count()
        if num_chunks >= total:
            return [FrameRange(f, f, self.step) for f in range(self.start, self.end + 1, self.step)]

        chunk_size = total // num_chunks
        remainder = total % num_chunks

        chunks = []
        current = self.start

        for i in range(num_chunks):
            size = chunk_size + (1 if i < remainder else 0)
            end = current + (size - 1) * self.step

            if end > self.end:
                end = self.end

            if current <= self.end:
                chunks.append(FrameRange(current, end, self.step))

            current = end + self.step

        return chunks

    def contains(self, frame: int) -> bool:
        """Check if frame is in range."""
        if frame < self.start or frame > self.end:
            return False
        return (frame - self.start) % self.step == 0

    def __iter__(self):
        """Iterate over frames in range."""
        for frame in range(self.start, self.end + 1, self.step):
            yield frame

    def __len__(self):
        return self.frame_count()


@dataclass
class ChunkAssignment:
    """Assignment of a frame chunk to a worker."""
    chunk_id: str
    frame_range: FrameRange
    worker_id: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    output_path: Optional[Path] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def duration_seconds(self) -> Optional[float]:
        """Get processing duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class RenderJob:
    """Distributed render job."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""

    # Input/Output
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    work_dir: Optional[Path] = None

    # Frame range
    frame_range: Optional[FrameRange] = None
    total_frames: int = 0

    # Settings
    settings: Dict[str, Any] = field(default_factory=dict)
    preset: str = "balanced"

    # Priority and scheduling
    priority: JobPriority = JobPriority.NORMAL
    max_workers: int = 0  # 0 = unlimited
    chunk_size: int = 100  # Frames per chunk
    max_retries: int = 3

    # State
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Chunks
    chunks: List[ChunkAssignment] = field(default_factory=list)

    # Progress tracking
    frames_completed: int = 0
    frames_failed: int = 0

    # Results
    output_files: List[Path] = field(default_factory=list)
    final_output: Optional[Path] = None
    error_message: Optional[str] = None

    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    def initialize_chunks(self) -> None:
        """Create chunk assignments from frame range."""
        if not self.frame_range:
            raise ValueError("Frame range not set")

        self.chunks.clear()

        # Determine number of chunks
        total_frames = self.frame_range.frame_count()
        num_chunks = (total_frames + self.chunk_size - 1) // self.chunk_size

        # Split frame range
        chunk_ranges = self.frame_range.split(num_chunks)

        for i, frame_range in enumerate(chunk_ranges):
            chunk = ChunkAssignment(
                chunk_id=f"{self.job_id}_chunk_{i:04d}",
                frame_range=frame_range,
            )
            self.chunks.append(chunk)

        logger.info(f"Job {self.job_id}: Created {len(self.chunks)} chunks")

    def get_pending_chunks(self) -> List[ChunkAssignment]:
        """Get chunks that need assignment."""
        return [c for c in self.chunks if c.status == JobStatus.PENDING]

    def get_processing_chunks(self) -> List[ChunkAssignment]:
        """Get chunks currently being processed."""
        return [c for c in self.chunks if c.status == JobStatus.PROCESSING]

    def get_completed_chunks(self) -> List[ChunkAssignment]:
        """Get completed chunks."""
        return [c for c in self.chunks if c.status == JobStatus.COMPLETED]

    def get_failed_chunks(self) -> List[ChunkAssignment]:
        """Get failed chunks."""
        return [c for c in self.chunks if c.status == JobStatus.FAILED]

    def progress(self) -> float:
        """Get overall progress percentage."""
        if not self.chunks:
            return 0.0

        total_frames = sum(c.frame_range.frame_count() for c in self.chunks)
        if total_frames == 0:
            return 0.0

        completed_frames = sum(
            c.frame_range.frame_count()
            for c in self.chunks
            if c.status == JobStatus.COMPLETED
        )

        processing_frames = sum(
            int(c.frame_range.frame_count() * c.progress)
            for c in self.chunks
            if c.status == JobStatus.PROCESSING
        )

        return (completed_frames + processing_frames) / total_frames * 100

    def is_complete(self) -> bool:
        """Check if all chunks are complete."""
        return all(c.status == JobStatus.COMPLETED for c in self.chunks)

    def has_failed(self) -> bool:
        """Check if any chunks failed beyond retry limit."""
        return any(
            c.status == JobStatus.FAILED and c.retry_count >= self.max_retries
            for c in self.chunks
        )

    def assign_chunk(self, chunk_id: str, worker_id: str) -> Optional[ChunkAssignment]:
        """Assign a chunk to a worker."""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id and chunk.status == JobStatus.PENDING:
                chunk.worker_id = worker_id
                chunk.status = JobStatus.ASSIGNING
                chunk.assigned_at = datetime.now()
                return chunk
        return None

    def start_chunk(self, chunk_id: str) -> Optional[ChunkAssignment]:
        """Mark chunk as started processing."""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                chunk.status = JobStatus.PROCESSING
                chunk.started_at = datetime.now()
                return chunk
        return None

    def complete_chunk(
        self,
        chunk_id: str,
        output_path: Path,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Optional[ChunkAssignment]:
        """Mark chunk as completed."""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                chunk.status = JobStatus.COMPLETED
                chunk.completed_at = datetime.now()
                chunk.progress = 1.0
                chunk.output_path = output_path
                if metrics:
                    chunk.metrics.update(metrics)

                self.frames_completed += chunk.frame_range.frame_count()
                return chunk
        return None

    def fail_chunk(self, chunk_id: str, error: str) -> Optional[ChunkAssignment]:
        """Mark chunk as failed."""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                chunk.retry_count += 1

                if chunk.retry_count < self.max_retries:
                    # Reset for retry
                    chunk.status = JobStatus.PENDING
                    chunk.worker_id = None
                    chunk.error_message = error
                    logger.warning(
                        f"Chunk {chunk_id} failed, retry {chunk.retry_count}/{self.max_retries}"
                    )
                else:
                    chunk.status = JobStatus.FAILED
                    chunk.error_message = error
                    self.frames_failed += chunk.frame_range.frame_count()
                    logger.error(f"Chunk {chunk_id} failed permanently: {error}")

                return chunk
        return None

    def update_chunk_progress(self, chunk_id: str, progress: float) -> None:
        """Update progress of a chunk."""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                chunk.progress = max(0.0, min(1.0, progress))
                break

    def estimated_time_remaining(self, frames_per_second: float) -> Optional[float]:
        """Estimate remaining time in seconds."""
        if frames_per_second <= 0:
            return None

        remaining_frames = sum(
            c.frame_range.frame_count() * (1 - c.progress)
            for c in self.chunks
            if c.status in (JobStatus.PENDING, JobStatus.PROCESSING, JobStatus.QUEUED)
        )

        return remaining_frames / frames_per_second

    def to_dict(self) -> Dict[str, Any]:
        """Serialize job to dictionary."""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "input_path": str(self.input_path) if self.input_path else None,
            "output_path": str(self.output_path) if self.output_path else None,
            "work_dir": str(self.work_dir) if self.work_dir else None,
            "frame_range": {
                "start": self.frame_range.start,
                "end": self.frame_range.end,
                "step": self.frame_range.step,
            } if self.frame_range else None,
            "total_frames": self.total_frames,
            "settings": self.settings,
            "preset": self.preset,
            "priority": self.priority.value,
            "max_workers": self.max_workers,
            "chunk_size": self.chunk_size,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress(),
            "frames_completed": self.frames_completed,
            "frames_failed": self.frames_failed,
            "chunk_count": len(self.chunks),
            "chunks_completed": len(self.get_completed_chunks()),
            "chunks_processing": len(self.get_processing_chunks()),
            "chunks_pending": len(self.get_pending_chunks()),
            "chunks_failed": len(self.get_failed_chunks()),
            "error_message": self.error_message,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RenderJob":
        """Deserialize job from dictionary."""
        job = cls(
            job_id=data["job_id"],
            name=data.get("name", ""),
            input_path=Path(data["input_path"]) if data.get("input_path") else None,
            output_path=Path(data["output_path"]) if data.get("output_path") else None,
            work_dir=Path(data["work_dir"]) if data.get("work_dir") else None,
            total_frames=data.get("total_frames", 0),
            settings=data.get("settings", {}),
            preset=data.get("preset", "balanced"),
            priority=JobPriority(data.get("priority", 1)),
            max_workers=data.get("max_workers", 0),
            chunk_size=data.get("chunk_size", 100),
            max_retries=data.get("max_retries", 3),
            status=JobStatus(data.get("status", "pending")),
        )

        if data.get("frame_range"):
            fr = data["frame_range"]
            job.frame_range = FrameRange(fr["start"], fr["end"], fr.get("step", 1))

        if data.get("created_at"):
            job.created_at = datetime.fromisoformat(data["created_at"])

        return job
