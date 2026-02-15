"""SQLite-based job state persistence for crash recovery."""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


class JobState(Enum):
    """Job execution state."""
    CREATED = "created"
    ANALYZING = "analyzing"
    EXTRACTING = "extracting"
    PROCESSING = "processing"
    ENCODING = "encoding"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class FrameState(Enum):
    """Frame processing state."""
    PENDING = "pending"
    EXTRACTED = "extracted"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class FrameRecord:
    """Record for a single frame."""
    frame_number: int
    state: FrameState = FrameState.PENDING

    # Paths
    extracted_path: Optional[str] = None
    processed_path: Optional[str] = None

    # Processing info
    processor_name: Optional[str] = None
    processing_time_ms: int = 0
    retry_count: int = 0

    # Quality metrics
    psnr: Optional[float] = None
    ssim: Optional[float] = None

    # Error info
    error_message: Optional[str] = None

    # Timestamps
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame_number": self.frame_number,
            "state": self.state.value,
            "extracted_path": self.extracted_path,
            "processed_path": self.processed_path,
            "processor_name": self.processor_name,
            "processing_time_ms": self.processing_time_ms,
            "retry_count": self.retry_count,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameRecord":
        """Create from dictionary."""
        return cls(
            frame_number=data["frame_number"],
            state=FrameState(data["state"]),
            extracted_path=data.get("extracted_path"),
            processed_path=data.get("processed_path"),
            processor_name=data.get("processor_name"),
            processing_time_ms=data.get("processing_time_ms", 0),
            retry_count=data.get("retry_count", 0),
            psnr=data.get("psnr"),
            ssim=data.get("ssim"),
            error_message=data.get("error_message"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )


@dataclass
class JobRecord:
    """Record for a restoration job."""
    job_id: str
    name: str
    state: JobState = JobState.CREATED

    # Input/Output
    input_path: str = ""
    output_path: str = ""
    project_dir: str = ""

    # Video info
    total_frames: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    duration_seconds: float = 0.0

    # Progress
    frames_extracted: int = 0
    frames_processed: int = 0
    frames_failed: int = 0
    current_frame: int = 0

    # Settings
    preset: str = "balanced"
    config_json: str = "{}"

    # Timing
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_completion: Optional[str] = None

    # Performance
    avg_frame_time_ms: float = 0.0
    total_processing_time_seconds: float = 0.0

    # Error info
    error_message: Optional[str] = None
    last_error_frame: Optional[int] = None

    # Quality
    avg_psnr: Optional[float] = None
    avg_ssim: Optional[float] = None

    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.total_frames == 0:
            return 0.0
        return (self.frames_processed / self.total_frames) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "state": self.state.value,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "project_dir": self.project_dir,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "frames_extracted": self.frames_extracted,
            "frames_processed": self.frames_processed,
            "frames_failed": self.frames_failed,
            "current_frame": self.current_frame,
            "preset": self.preset,
            "config_json": self.config_json,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "estimated_completion": self.estimated_completion,
            "avg_frame_time_ms": self.avg_frame_time_ms,
            "total_processing_time_seconds": self.total_processing_time_seconds,
            "error_message": self.error_message,
            "last_error_frame": self.last_error_frame,
            "avg_psnr": self.avg_psnr,
            "avg_ssim": self.avg_ssim,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobRecord":
        """Create from dictionary."""
        return cls(
            job_id=data["job_id"],
            name=data["name"],
            state=JobState(data["state"]),
            input_path=data.get("input_path", ""),
            output_path=data.get("output_path", ""),
            project_dir=data.get("project_dir", ""),
            total_frames=data.get("total_frames", 0),
            fps=data.get("fps", 0.0),
            width=data.get("width", 0),
            height=data.get("height", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            frames_extracted=data.get("frames_extracted", 0),
            frames_processed=data.get("frames_processed", 0),
            frames_failed=data.get("frames_failed", 0),
            current_frame=data.get("current_frame", 0),
            preset=data.get("preset", "balanced"),
            config_json=data.get("config_json", "{}"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            estimated_completion=data.get("estimated_completion"),
            avg_frame_time_ms=data.get("avg_frame_time_ms", 0.0),
            total_processing_time_seconds=data.get("total_processing_time_seconds", 0.0),
            error_message=data.get("error_message"),
            last_error_frame=data.get("last_error_frame"),
            avg_psnr=data.get("avg_psnr"),
            avg_ssim=data.get("avg_ssim"),
        )


class JobStore:
    """SQLite-based storage for job state and frame progress."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize job store.

        Args:
            db_path: Path to SQLite database. If None, uses default location.
        """
        if db_path is None:
            db_path = Path.home() / ".framewright" / "jobs.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                isolation_level=None,  # Autocommit
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for transactions."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("BEGIN")
        try:
            yield cursor
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                state TEXT NOT NULL,
                input_path TEXT,
                output_path TEXT,
                project_dir TEXT,
                total_frames INTEGER DEFAULT 0,
                fps REAL DEFAULT 0,
                width INTEGER DEFAULT 0,
                height INTEGER DEFAULT 0,
                duration_seconds REAL DEFAULT 0,
                frames_extracted INTEGER DEFAULT 0,
                frames_processed INTEGER DEFAULT 0,
                frames_failed INTEGER DEFAULT 0,
                current_frame INTEGER DEFAULT 0,
                preset TEXT DEFAULT 'balanced',
                config_json TEXT DEFAULT '{}',
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                estimated_completion TEXT,
                avg_frame_time_ms REAL DEFAULT 0,
                total_processing_time_seconds REAL DEFAULT 0,
                error_message TEXT,
                last_error_frame INTEGER,
                avg_psnr REAL,
                avg_ssim REAL
            )
        """)

        # Frames table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                job_id TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                state TEXT NOT NULL,
                extracted_path TEXT,
                processed_path TEXT,
                processor_name TEXT,
                processing_time_ms INTEGER DEFAULT 0,
                retry_count INTEGER DEFAULT 0,
                psnr REAL,
                ssim REAL,
                error_message TEXT,
                started_at TEXT,
                completed_at TEXT,
                PRIMARY KEY (job_id, frame_number),
                FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_frames_state ON frames(job_id, state)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state)")

        # Schema version
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        cursor.execute(
            "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
            ("version", str(self.SCHEMA_VERSION))
        )

    # Job operations

    def create_job(self, job: JobRecord) -> None:
        """Create a new job record."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO jobs (
                job_id, name, state, input_path, output_path, project_dir,
                total_frames, fps, width, height, duration_seconds,
                preset, config_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job.job_id, job.name, job.state.value, job.input_path, job.output_path,
            job.project_dir, job.total_frames, job.fps, job.width, job.height,
            job.duration_seconds, job.preset, job.config_json, job.created_at
        ))

        logger.debug(f"Created job: {job.job_id}")

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Get job by ID."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return JobRecord.from_dict(dict(row))

    def update_job(self, job: JobRecord) -> None:
        """Update job record."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE jobs SET
                name = ?, state = ?, input_path = ?, output_path = ?, project_dir = ?,
                total_frames = ?, fps = ?, width = ?, height = ?, duration_seconds = ?,
                frames_extracted = ?, frames_processed = ?, frames_failed = ?,
                current_frame = ?, preset = ?, config_json = ?,
                started_at = ?, completed_at = ?, estimated_completion = ?,
                avg_frame_time_ms = ?, total_processing_time_seconds = ?,
                error_message = ?, last_error_frame = ?, avg_psnr = ?, avg_ssim = ?
            WHERE job_id = ?
        """, (
            job.name, job.state.value, job.input_path, job.output_path, job.project_dir,
            job.total_frames, job.fps, job.width, job.height, job.duration_seconds,
            job.frames_extracted, job.frames_processed, job.frames_failed,
            job.current_frame, job.preset, job.config_json,
            job.started_at, job.completed_at, job.estimated_completion,
            job.avg_frame_time_ms, job.total_processing_time_seconds,
            job.error_message, job.last_error_frame, job.avg_psnr, job.avg_ssim,
            job.job_id
        ))

    def update_job_state(self, job_id: str, state: JobState, error: Optional[str] = None) -> None:
        """Update job state."""
        conn = self._get_conn()
        cursor = conn.cursor()

        if state == JobState.COMPLETED:
            cursor.execute(
                "UPDATE jobs SET state = ?, completed_at = ?, error_message = ? WHERE job_id = ?",
                (state.value, datetime.now().isoformat(), error, job_id)
            )
        elif state in (JobState.FAILED, JobState.CANCELLED):
            cursor.execute(
                "UPDATE jobs SET state = ?, error_message = ? WHERE job_id = ?",
                (state.value, error, job_id)
            )
        else:
            cursor.execute(
                "UPDATE jobs SET state = ? WHERE job_id = ?",
                (state.value, job_id)
            )

    def update_job_progress(
        self,
        job_id: str,
        frames_processed: int,
        current_frame: int,
        avg_frame_time_ms: float,
    ) -> None:
        """Update job progress."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE jobs SET
                frames_processed = ?,
                current_frame = ?,
                avg_frame_time_ms = ?
            WHERE job_id = ?
        """, (frames_processed, current_frame, avg_frame_time_ms, job_id))

    def delete_job(self, job_id: str) -> None:
        """Delete job and all associated frames."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM frames WHERE job_id = ?", (job_id,))
        cursor.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))

        logger.debug(f"Deleted job: {job_id}")

    def list_jobs(
        self,
        state: Optional[JobState] = None,
        limit: int = 100,
    ) -> List[JobRecord]:
        """List jobs, optionally filtered by state."""
        conn = self._get_conn()
        cursor = conn.cursor()

        if state:
            cursor.execute(
                "SELECT * FROM jobs WHERE state = ? ORDER BY created_at DESC LIMIT ?",
                (state.value, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )

        return [JobRecord.from_dict(dict(row)) for row in cursor.fetchall()]

    def get_resumable_jobs(self) -> List[JobRecord]:
        """Get jobs that can be resumed after crash."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM jobs
            WHERE state IN (?, ?, ?, ?)
            ORDER BY created_at DESC
        """, (
            JobState.PROCESSING.value,
            JobState.EXTRACTING.value,
            JobState.ENCODING.value,
            JobState.PAUSED.value,
        ))

        return [JobRecord.from_dict(dict(row)) for row in cursor.fetchall()]

    # Frame operations

    def create_frames(self, job_id: str, total_frames: int) -> None:
        """Create frame records for a job."""
        with self._transaction() as cursor:
            for frame_num in range(total_frames):
                cursor.execute("""
                    INSERT OR IGNORE INTO frames (job_id, frame_number, state)
                    VALUES (?, ?, ?)
                """, (job_id, frame_num, FrameState.PENDING.value))

        logger.debug(f"Created {total_frames} frame records for job {job_id}")

    def get_frame(self, job_id: str, frame_number: int) -> Optional[FrameRecord]:
        """Get frame record."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM frames WHERE job_id = ? AND frame_number = ?",
            (job_id, frame_number)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return FrameRecord.from_dict(dict(row))

    def update_frame(self, job_id: str, frame: FrameRecord) -> None:
        """Update frame record."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE frames SET
                state = ?, extracted_path = ?, processed_path = ?,
                processor_name = ?, processing_time_ms = ?, retry_count = ?,
                psnr = ?, ssim = ?, error_message = ?,
                started_at = ?, completed_at = ?
            WHERE job_id = ? AND frame_number = ?
        """, (
            frame.state.value, frame.extracted_path, frame.processed_path,
            frame.processor_name, frame.processing_time_ms, frame.retry_count,
            frame.psnr, frame.ssim, frame.error_message,
            frame.started_at, frame.completed_at,
            job_id, frame.frame_number
        ))

    def update_frame_state(
        self,
        job_id: str,
        frame_number: int,
        state: FrameState,
        path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Quick update of frame state."""
        conn = self._get_conn()
        cursor = conn.cursor()

        if state == FrameState.COMPLETED and path:
            cursor.execute("""
                UPDATE frames SET state = ?, processed_path = ?, completed_at = ?
                WHERE job_id = ? AND frame_number = ?
            """, (state.value, path, datetime.now().isoformat(), job_id, frame_number))
        elif state == FrameState.FAILED:
            cursor.execute("""
                UPDATE frames SET state = ?, error_message = ?, retry_count = retry_count + 1
                WHERE job_id = ? AND frame_number = ?
            """, (state.value, error, job_id, frame_number))
        else:
            cursor.execute(
                "UPDATE frames SET state = ? WHERE job_id = ? AND frame_number = ?",
                (state.value, job_id, frame_number)
            )

    def get_pending_frames(self, job_id: str, limit: int = 100) -> List[FrameRecord]:
        """Get frames that need processing."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM frames
            WHERE job_id = ? AND state = ?
            ORDER BY frame_number
            LIMIT ?
        """, (job_id, FrameState.PENDING.value, limit))

        return [FrameRecord.from_dict(dict(row)) for row in cursor.fetchall()]

    def get_failed_frames(self, job_id: str, max_retries: int = 3) -> List[FrameRecord]:
        """Get frames that failed but can be retried."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM frames
            WHERE job_id = ? AND state = ? AND retry_count < ?
            ORDER BY frame_number
        """, (job_id, FrameState.FAILED.value, max_retries))

        return [FrameRecord.from_dict(dict(row)) for row in cursor.fetchall()]

    def get_frame_stats(self, job_id: str) -> Dict[str, int]:
        """Get frame statistics for a job."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT state, COUNT(*) as count
            FROM frames WHERE job_id = ?
            GROUP BY state
        """, (job_id,))

        stats = {state.value: 0 for state in FrameState}
        for row in cursor.fetchall():
            stats[row["state"]] = row["count"]

        return stats

    def get_next_frame_to_process(self, job_id: str) -> Optional[int]:
        """Get next frame number that needs processing."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT frame_number FROM frames
            WHERE job_id = ? AND state = ?
            ORDER BY frame_number
            LIMIT 1
        """, (job_id, FrameState.PENDING.value))

        row = cursor.fetchone()
        return row["frame_number"] if row else None

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
