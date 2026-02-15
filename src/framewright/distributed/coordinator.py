"""Render farm coordinator for distributed processing."""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import subprocess
import shutil

from .job import RenderJob, JobStatus, JobPriority, FrameRange, ChunkAssignment
from .discovery import NodeDiscovery, NodeInfo, DiscoveryMethod

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorConfig:
    """Configuration for render coordinator."""
    # Network
    bind_address: str = "0.0.0.0"
    port: int = 8764

    # Discovery
    discovery_method: DiscoveryMethod = DiscoveryMethod.MULTICAST
    static_workers: List[str] = field(default_factory=list)

    # Job management
    max_concurrent_jobs: int = 10
    default_chunk_size: int = 100
    max_retries: int = 3

    # Scheduling
    scheduling_interval: float = 1.0
    heartbeat_interval: float = 10.0
    worker_timeout: float = 60.0

    # Storage
    work_dir: Path = field(default_factory=lambda: Path.home() / ".framewright" / "coordinator")
    shared_storage: Optional[Path] = None  # Network shared storage

    # Merge
    merge_method: str = "ffmpeg"  # ffmpeg, concat, copy

    def __post_init__(self):
        self.work_dir = Path(self.work_dir)
        if self.shared_storage:
            self.shared_storage = Path(self.shared_storage)


class RenderCoordinator:
    """Coordinates distributed rendering across multiple workers."""

    def __init__(self, config: Optional[CoordinatorConfig] = None):
        self.config = config or CoordinatorConfig()
        self.config.work_dir.mkdir(parents=True, exist_ok=True)

        # Job management
        self._jobs: Dict[str, RenderJob] = {}
        self._job_queue: List[str] = []  # Job IDs in priority order
        self._lock = threading.Lock()

        # Node discovery
        self.discovery = NodeDiscovery(
            method=self.config.discovery_method,
            static_nodes=self.config.static_workers,
            stale_timeout=self.config.worker_timeout,
        )
        self.discovery.add_callback(self._on_node_event)

        # State
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None

        # Callbacks
        self._job_callbacks: List[Callable[[RenderJob, str], None]] = []

        # FFmpeg for merging
        self._ffmpeg_path = shutil.which("ffmpeg")

    def add_job_callback(self, callback: Callable[[RenderJob, str], None]) -> None:
        """Add callback for job events (job, event_type)."""
        self._job_callbacks.append(callback)

    def _notify_job(self, job: RenderJob, event: str) -> None:
        """Notify callbacks of job event."""
        for callback in self._job_callbacks:
            try:
                callback(job, event)
            except Exception as e:
                logger.error(f"Job callback error: {e}")

    def start(self) -> None:
        """Start the coordinator."""
        if self._running:
            return

        self._running = True

        # Start node discovery
        self.discovery.start()

        # Start scheduler
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()

        logger.info(f"Render coordinator started on port {self.config.port}")

    def stop(self) -> None:
        """Stop the coordinator."""
        self._running = False
        self.discovery.stop()
        logger.info("Render coordinator stopped")

    def submit_job(
        self,
        input_path: Path,
        output_path: Path,
        frame_range: Optional[FrameRange] = None,
        settings: Optional[Dict[str, Any]] = None,
        preset: str = "balanced",
        priority: JobPriority = JobPriority.NORMAL,
        name: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> RenderJob:
        """Submit a new render job."""
        # Create job
        job = RenderJob(
            name=name or input_path.stem,
            input_path=input_path,
            output_path=output_path,
            settings=settings or {},
            preset=preset,
            priority=priority,
            chunk_size=chunk_size or self.config.default_chunk_size,
            max_retries=self.config.max_retries,
        )

        # Get frame count if not specified
        if frame_range:
            job.frame_range = frame_range
        else:
            total_frames = self._get_video_frame_count(input_path)
            job.frame_range = FrameRange(0, total_frames - 1)

        job.total_frames = job.frame_range.frame_count()

        # Create work directory
        job.work_dir = self.config.work_dir / job.job_id
        job.work_dir.mkdir(parents=True, exist_ok=True)

        # Initialize chunks
        job.initialize_chunks()

        # Add to queue
        with self._lock:
            self._jobs[job.job_id] = job
            self._insert_job_priority(job.job_id)

        job.status = JobStatus.QUEUED
        self._notify_job(job, "submitted")
        logger.info(f"Job submitted: {job.job_id} ({job.name}) - {len(job.chunks)} chunks")

        return job

    def _insert_job_priority(self, job_id: str) -> None:
        """Insert job into queue based on priority."""
        job = self._jobs[job_id]

        # Find insertion point
        insert_idx = len(self._job_queue)
        for i, existing_id in enumerate(self._job_queue):
            existing = self._jobs.get(existing_id)
            if existing and job.priority.value > existing.priority.value:
                insert_idx = i
                break

        self._job_queue.insert(insert_idx, job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        with self._lock:
            if job_id not in self._jobs:
                return False

            job = self._jobs[job_id]
            if job.status in (JobStatus.COMPLETED, JobStatus.CANCELLED):
                return False

            job.status = JobStatus.CANCELLED

            # Cancel all pending chunks
            for chunk in job.chunks:
                if chunk.status in (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.ASSIGNING):
                    chunk.status = JobStatus.CANCELLED

            if job_id in self._job_queue:
                self._job_queue.remove(job_id)

        self._notify_job(job, "cancelled")
        logger.info(f"Job cancelled: {job_id}")
        return True

    def pause_job(self, job_id: str) -> bool:
        """Pause a job."""
        with self._lock:
            if job_id not in self._jobs:
                return False

            job = self._jobs[job_id]
            if job.status != JobStatus.PROCESSING:
                return False

            job.status = JobStatus.PAUSED

        self._notify_job(job, "paused")
        logger.info(f"Job paused: {job_id}")
        return True

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        with self._lock:
            if job_id not in self._jobs:
                return False

            job = self._jobs[job_id]
            if job.status != JobStatus.PAUSED:
                return False

            job.status = JobStatus.PROCESSING

        self._notify_job(job, "resumed")
        logger.info(f"Job resumed: {job_id}")
        return True

    def get_job(self, job_id: str) -> Optional[RenderJob]:
        """Get job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_all_jobs(self) -> List[RenderJob]:
        """Get all jobs."""
        with self._lock:
            return list(self._jobs.values())

    def get_active_jobs(self) -> List[RenderJob]:
        """Get active (non-completed) jobs."""
        with self._lock:
            return [
                job for job in self._jobs.values()
                if job.status not in (JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.FAILED)
            ]

    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                self._schedule_chunks()
                self._check_job_completion()
            except Exception as e:
                logger.error(f"Scheduler error: {e}")

            time.sleep(self.config.scheduling_interval)

    def _schedule_chunks(self) -> None:
        """Assign pending chunks to available workers."""
        available_workers = self.discovery.get_available_nodes()
        if not available_workers:
            return

        with self._lock:
            for job_id in self._job_queue:
                job = self._jobs.get(job_id)
                if not job or job.status not in (JobStatus.QUEUED, JobStatus.PROCESSING):
                    continue

                # Get pending chunks
                pending = job.get_pending_chunks()
                if not pending:
                    continue

                # Start job if first assignment
                if job.status == JobStatus.QUEUED:
                    job.status = JobStatus.PROCESSING
                    job.started_at = datetime.now()
                    self._notify_job(job, "started")

                # Assign chunks to workers
                for chunk in pending:
                    if not available_workers:
                        break

                    # Select best worker
                    worker = self._select_worker(available_workers, job, chunk)
                    if not worker:
                        continue

                    # Assign chunk
                    job.assign_chunk(chunk.chunk_id, worker.node_id)

                    # Mark worker busy
                    self.discovery.update_node_status(
                        worker.node_id,
                        is_available=False,
                        current_job=job.job_id,
                        current_chunk=chunk.chunk_id,
                    )

                    # Send work to worker
                    self._dispatch_chunk(worker, job, chunk)

                    # Remove from available
                    available_workers = [w for w in available_workers if w.node_id != worker.node_id]

                    logger.debug(f"Assigned chunk {chunk.chunk_id} to {worker.hostname}")

    def _select_worker(
        self,
        workers: List[NodeInfo],
        job: RenderJob,
        chunk: ChunkAssignment
    ) -> Optional[NodeInfo]:
        """Select best worker for a chunk."""
        if not workers:
            return None

        # Simple selection: prefer worker with most GPU memory
        workers_sorted = sorted(
            workers,
            key=lambda w: (w.gpu_memory_gb, w.estimated_fps),
            reverse=True
        )

        return workers_sorted[0]

    def _dispatch_chunk(
        self,
        worker: NodeInfo,
        job: RenderJob,
        chunk: ChunkAssignment
    ) -> None:
        """Send chunk work to a worker."""
        # In a full implementation, this would send via network
        # For now, we store the assignment for the worker to poll

        assignment_file = self.config.work_dir / "assignments" / f"{chunk.chunk_id}.json"
        assignment_file.parent.mkdir(parents=True, exist_ok=True)

        assignment_data = {
            "chunk_id": chunk.chunk_id,
            "job_id": job.job_id,
            "input_path": str(job.input_path),
            "work_dir": str(job.work_dir),
            "frame_start": chunk.frame_range.start,
            "frame_end": chunk.frame_range.end,
            "frame_step": chunk.frame_range.step,
            "settings": job.settings,
            "preset": job.preset,
            "worker_id": worker.node_id,
            "assigned_at": datetime.now().isoformat(),
        }

        with open(assignment_file, "w") as f:
            json.dump(assignment_data, f, indent=2)

        job.start_chunk(chunk.chunk_id)

    def receive_chunk_result(
        self,
        chunk_id: str,
        worker_id: str,
        success: bool,
        output_path: Optional[Path] = None,
        error_message: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Receive result from a worker."""
        with self._lock:
            # Find job for chunk
            job = None
            for j in self._jobs.values():
                for c in j.chunks:
                    if c.chunk_id == chunk_id:
                        job = j
                        break
                if job:
                    break

            if not job:
                logger.warning(f"Received result for unknown chunk: {chunk_id}")
                return

            # Update chunk status
            if success and output_path:
                job.complete_chunk(chunk_id, output_path, metrics)
                logger.info(f"Chunk completed: {chunk_id}")
            else:
                job.fail_chunk(chunk_id, error_message or "Unknown error")

            # Free worker
            self.discovery.update_node_status(
                worker_id,
                is_available=True,
                current_job="",
                current_chunk="",
                estimated_fps=metrics.get("fps", 1.0) if metrics else None,
            )

    def _check_job_completion(self) -> None:
        """Check if any jobs are complete and merge outputs."""
        with self._lock:
            for job_id in list(self._job_queue):
                job = self._jobs.get(job_id)
                if not job:
                    continue

                if job.is_complete():
                    # All chunks done, merge outputs
                    self._job_queue.remove(job_id)
                    job.status = JobStatus.MERGING

        # Merge outside lock
        for job in self._jobs.values():
            if job.status == JobStatus.MERGING:
                self._merge_job_outputs(job)

    def _merge_job_outputs(self, job: RenderJob) -> None:
        """Merge chunk outputs into final video."""
        logger.info(f"Merging job outputs: {job.job_id}")

        try:
            # Collect output files in order
            chunk_outputs = []
            for chunk in sorted(job.chunks, key=lambda c: c.frame_range.start):
                if chunk.output_path and chunk.output_path.exists():
                    chunk_outputs.append(chunk.output_path)

            if not chunk_outputs:
                job.status = JobStatus.FAILED
                job.error_message = "No chunk outputs to merge"
                self._notify_job(job, "failed")
                return

            # Merge using FFmpeg concat
            if len(chunk_outputs) == 1:
                # Single chunk, just copy
                shutil.copy(chunk_outputs[0], job.output_path)
            else:
                self._ffmpeg_concat(chunk_outputs, job.output_path)

            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.final_output = job.output_path
            self._notify_job(job, "completed")
            logger.info(f"Job completed: {job.job_id}")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            self._notify_job(job, "failed")
            logger.error(f"Job merge failed: {job.job_id} - {e}")

    def _ffmpeg_concat(self, inputs: List[Path], output: Path) -> None:
        """Concatenate video files using FFmpeg."""
        if not self._ffmpeg_path:
            raise RuntimeError("FFmpeg not found")

        # Create concat file
        concat_file = output.parent / f"{output.stem}_concat.txt"
        with open(concat_file, "w") as f:
            for input_path in inputs:
                f.write(f"file '{input_path}'\n")

        try:
            cmd = [
                self._ffmpeg_path,
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                "-y",
                str(output)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

        finally:
            concat_file.unlink(missing_ok=True)

    def _get_video_frame_count(self, video_path: Path) -> int:
        """Get frame count from video file."""
        if not self._ffmpeg_path:
            return 1000  # Default

        try:
            ffprobe = self._ffmpeg_path.replace("ffmpeg", "ffprobe")
            cmd = [
                ffprobe,
                "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0",
                str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip())

        except Exception as e:
            logger.warning(f"Failed to get frame count: {e}")

        return 1000  # Default fallback

    def _on_node_event(self, node: NodeInfo, event: str) -> None:
        """Handle node discovery events."""
        if event == "joined":
            logger.info(f"Worker joined: {node.hostname} ({node.gpu_count} GPUs, {node.gpu_memory_gb:.1f}GB VRAM)")
        elif event == "left":
            logger.warning(f"Worker left: {node.hostname}")

            # Reassign any chunks from this worker
            with self._lock:
                for job in self._jobs.values():
                    for chunk in job.chunks:
                        if chunk.worker_id == node.node_id and chunk.status == JobStatus.PROCESSING:
                            logger.warning(f"Reassigning chunk {chunk.chunk_id} from lost worker")
                            chunk.status = JobStatus.PENDING
                            chunk.worker_id = None

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status."""
        cluster_stats = self.discovery.get_cluster_stats()

        with self._lock:
            jobs_summary = {
                "total": len(self._jobs),
                "queued": len([j for j in self._jobs.values() if j.status == JobStatus.QUEUED]),
                "processing": len([j for j in self._jobs.values() if j.status == JobStatus.PROCESSING]),
                "completed": len([j for j in self._jobs.values() if j.status == JobStatus.COMPLETED]),
                "failed": len([j for j in self._jobs.values() if j.status == JobStatus.FAILED]),
            }

            total_chunks = sum(len(j.chunks) for j in self._jobs.values())
            completed_chunks = sum(len(j.get_completed_chunks()) for j in self._jobs.values())
            processing_chunks = sum(len(j.get_processing_chunks()) for j in self._jobs.values())

        return {
            "coordinator": {
                "address": f"{self.config.bind_address}:{self.config.port}",
                "running": self._running,
            },
            "cluster": cluster_stats,
            "jobs": jobs_summary,
            "chunks": {
                "total": total_chunks,
                "completed": completed_chunks,
                "processing": processing_chunks,
                "pending": total_chunks - completed_chunks - processing_chunks,
            },
        }
