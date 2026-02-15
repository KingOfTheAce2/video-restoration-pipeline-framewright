"""Cloud burst scaling for automatic offloading when local capacity is exceeded."""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import CloudProvider, JobStatus, JobState, ProcessingConfig

logger = logging.getLogger(__name__)


class ScalingTrigger(Enum):
    """Triggers for cloud burst scaling."""
    QUEUE_DEPTH = auto()  # Too many frames waiting
    PROCESSING_TIME = auto()  # Taking too long locally
    VRAM_PRESSURE = auto()  # Running out of VRAM
    DEADLINE = auto()  # Need to meet a deadline
    MANUAL = auto()  # User requested


@dataclass
class BurstConfig:
    """Configuration for cloud burst scaling."""
    # Scaling triggers
    enable_auto_burst: bool = True
    queue_depth_threshold: int = 100  # Frames queued to trigger burst
    processing_time_threshold: float = 60.0  # Seconds per frame to trigger
    vram_usage_threshold: float = 0.95  # VRAM usage to trigger

    # Cloud settings
    provider: str = "runpod"  # runpod, vastai
    max_cloud_instances: int = 4
    min_cloud_instances: int = 0
    instance_gpu_type: str = "RTX 4090"
    max_cost_per_hour: float = 10.0  # Max $/hour

    # Frame distribution
    frames_per_cloud_batch: int = 50
    cloud_batch_timeout: float = 300.0  # Seconds

    # Cost management
    max_total_cost: float = 100.0  # Maximum spend
    prefer_spot_instances: bool = True

    # Timing
    scale_up_cooldown: float = 60.0  # Seconds between scale ups
    scale_down_delay: float = 300.0  # Seconds idle before scaling down


@dataclass
class CloudInstance:
    """Represents a cloud GPU instance."""
    instance_id: str
    provider: str
    gpu_type: str
    status: str = "initializing"
    cost_per_hour: float = 0.0
    started_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    frames_processed: int = 0
    total_cost: float = 0.0


@dataclass
class BurstStats:
    """Statistics for burst scaling."""
    total_frames: int = 0
    local_frames: int = 0
    cloud_frames: int = 0
    cloud_instances_used: int = 0
    cloud_cost: float = 0.0
    time_saved_seconds: float = 0.0
    avg_local_time_ms: float = 0.0
    avg_cloud_time_ms: float = 0.0


class CloudBurstManager:
    """Manages automatic cloud burst scaling."""

    def __init__(
        self,
        config: Optional[BurstConfig] = None,
        provider: Optional[CloudProvider] = None,
    ):
        self.config = config or BurstConfig()
        self._provider = provider

        # State
        self._active_instances: Dict[str, CloudInstance] = {}
        self._pending_batches: queue.Queue = queue.Queue()
        self._completed_frames: Dict[int, Any] = {}
        self._stats = BurstStats()

        # Scaling state
        self._last_scale_up: Optional[datetime] = None
        self._last_activity: Optional[datetime] = None
        self._scaling_lock = threading.Lock()

        # Monitoring
        self._local_times: List[float] = []
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

    def set_provider(self, provider: CloudProvider) -> None:
        """Set the cloud provider."""
        self._provider = provider

    def start(self) -> None:
        """Start burst scaling manager."""
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="BurstMonitor",
        )
        self._monitor_thread.start()
        logger.info("Cloud burst manager started")

    def stop(self) -> None:
        """Stop burst scaling and terminate instances."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

        # Terminate all instances
        self._scale_down_all()
        logger.info("Cloud burst manager stopped")

    def should_burst(
        self,
        queue_depth: int,
        avg_frame_time_ms: float,
        vram_usage: float,
    ) -> Tuple[bool, ScalingTrigger]:
        """Check if burst scaling should be triggered.

        Args:
            queue_depth: Number of frames waiting
            avg_frame_time_ms: Average local processing time
            vram_usage: VRAM usage ratio (0-1)

        Returns:
            (should_burst, trigger_reason)
        """
        if not self.config.enable_auto_burst:
            return False, ScalingTrigger.MANUAL

        # Check cost limit
        if self._stats.cloud_cost >= self.config.max_total_cost:
            logger.warning("Cloud cost limit reached, no burst")
            return False, ScalingTrigger.MANUAL

        # Check cooldown
        if self._last_scale_up:
            elapsed = (datetime.now() - self._last_scale_up).total_seconds()
            if elapsed < self.config.scale_up_cooldown:
                return False, ScalingTrigger.MANUAL

        # Check triggers
        if queue_depth >= self.config.queue_depth_threshold:
            return True, ScalingTrigger.QUEUE_DEPTH

        if avg_frame_time_ms / 1000 >= self.config.processing_time_threshold:
            return True, ScalingTrigger.PROCESSING_TIME

        if vram_usage >= self.config.vram_usage_threshold:
            return True, ScalingTrigger.VRAM_PRESSURE

        return False, ScalingTrigger.MANUAL

    def request_burst(
        self,
        num_instances: int = 1,
        reason: ScalingTrigger = ScalingTrigger.MANUAL,
    ) -> List[CloudInstance]:
        """Request cloud instances for burst processing.

        Args:
            num_instances: Number of instances to request
            reason: Reason for scaling

        Returns:
            List of launched instances
        """
        with self._scaling_lock:
            # Check limits
            current = len(self._active_instances)
            available = self.config.max_cloud_instances - current
            to_launch = min(num_instances, available)

            if to_launch <= 0:
                logger.info("Max cloud instances reached")
                return []

            if not self._provider:
                logger.error("No cloud provider configured")
                return []

            logger.info(f"Launching {to_launch} cloud instances (reason: {reason.name})")

            instances = []
            for _ in range(to_launch):
                try:
                    instance = self._launch_instance()
                    if instance:
                        instances.append(instance)
                except Exception as e:
                    logger.error(f"Failed to launch instance: {e}")

            self._last_scale_up = datetime.now()
            return instances

    def _launch_instance(self) -> Optional[CloudInstance]:
        """Launch a single cloud instance."""
        if not self._provider:
            return None

        try:
            # Create processing config
            config = ProcessingConfig(
                gpu_type=self.config.instance_gpu_type,
                prefer_spot=self.config.prefer_spot_instances,
            )

            # Submit job to get instance
            job_id = self._provider.submit_job(
                input_path="",  # Will be set per batch
                output_path="",
                config=config,
            )

            instance = CloudInstance(
                instance_id=job_id,
                provider=self.config.provider,
                gpu_type=self.config.instance_gpu_type,
                status="running",
                started_at=datetime.now(),
                last_activity=datetime.now(),
            )

            self._active_instances[job_id] = instance
            self._stats.cloud_instances_used += 1

            logger.info(f"Launched cloud instance: {job_id}")
            return instance

        except Exception as e:
            logger.error(f"Instance launch failed: {e}")
            return None

    def submit_batch(
        self,
        frames: List[Tuple[int, Path]],
        callback: Callable[[int, Any], None],
    ) -> str:
        """Submit a batch of frames to cloud processing.

        Args:
            frames: List of (frame_number, frame_path) tuples
            callback: Called with (frame_number, result) for each frame

        Returns:
            Batch ID
        """
        batch_id = f"batch_{int(time.time() * 1000)}"

        batch = {
            "id": batch_id,
            "frames": frames,
            "callback": callback,
            "submitted_at": datetime.now(),
        }

        self._pending_batches.put(batch)
        self._last_activity = datetime.now()

        logger.debug(f"Submitted batch {batch_id} with {len(frames)} frames")
        return batch_id

    def record_local_time(self, frame_time_ms: float) -> None:
        """Record local processing time for adaptive scaling."""
        self._local_times.append(frame_time_ms)
        if len(self._local_times) > 100:
            self._local_times = self._local_times[-100:]
        self._stats.avg_local_time_ms = sum(self._local_times) / len(self._local_times)

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self._process_pending_batches()
                self._check_instance_health()
                self._check_scale_down()
                self._update_costs()
            except Exception as e:
                logger.error(f"Monitor error: {e}")

            time.sleep(5.0)

    def _process_pending_batches(self) -> None:
        """Process pending batch submissions."""
        if not self._active_instances:
            return

        while not self._pending_batches.empty():
            try:
                batch = self._pending_batches.get_nowait()

                # Find available instance
                for instance_id, instance in self._active_instances.items():
                    if instance.status == "running":
                        self._send_batch_to_instance(instance, batch)
                        break
                else:
                    # No available instance, put back
                    self._pending_batches.put(batch)
                    break

            except queue.Empty:
                break

    def _send_batch_to_instance(
        self,
        instance: CloudInstance,
        batch: Dict[str, Any],
    ) -> None:
        """Send batch to cloud instance for processing."""
        instance.last_activity = datetime.now()

        # In a real implementation, this would:
        # 1. Upload frames to cloud storage
        # 2. Trigger processing on the instance
        # 3. Download results and call callback

        logger.debug(f"Sending batch {batch['id']} to instance {instance.instance_id}")

        # Simulate processing for now
        for frame_num, frame_path in batch["frames"]:
            instance.frames_processed += 1
            self._stats.cloud_frames += 1

            # Call callback (would be called with actual result)
            batch["callback"](frame_num, None)

    def _check_instance_health(self) -> None:
        """Check health of cloud instances."""
        if not self._provider:
            return

        for instance_id, instance in list(self._active_instances.items()):
            try:
                status = self._provider.get_job_status(instance_id)
                instance.status = status.state.value

                if status.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
                    logger.info(f"Instance {instance_id} ended: {status.state.value}")
                    del self._active_instances[instance_id]

            except Exception as e:
                logger.warning(f"Health check failed for {instance_id}: {e}")

    def _check_scale_down(self) -> None:
        """Check if instances should be scaled down."""
        if not self._active_instances:
            return

        now = datetime.now()
        idle_threshold = timedelta(seconds=self.config.scale_down_delay)

        for instance_id, instance in list(self._active_instances.items()):
            if instance.last_activity:
                idle_time = now - instance.last_activity
                if idle_time > idle_threshold:
                    logger.info(f"Scaling down idle instance: {instance_id}")
                    self._terminate_instance(instance_id)

    def _scale_down_all(self) -> None:
        """Terminate all cloud instances."""
        for instance_id in list(self._active_instances.keys()):
            self._terminate_instance(instance_id)

    def _terminate_instance(self, instance_id: str) -> None:
        """Terminate a cloud instance."""
        if instance_id not in self._active_instances:
            return

        try:
            if self._provider:
                self._provider.cancel_job(instance_id)
        except Exception as e:
            logger.warning(f"Failed to terminate {instance_id}: {e}")

        instance = self._active_instances.pop(instance_id, None)
        if instance and instance.started_at:
            runtime = (datetime.now() - instance.started_at).total_seconds() / 3600
            instance.total_cost = runtime * instance.cost_per_hour
            self._stats.cloud_cost += instance.total_cost

        logger.info(f"Terminated instance: {instance_id}")

    def _update_costs(self) -> None:
        """Update running cost estimates."""
        for instance in self._active_instances.values():
            if instance.started_at:
                runtime = (datetime.now() - instance.started_at).total_seconds() / 3600
                instance.total_cost = runtime * instance.cost_per_hour

    def get_stats(self) -> BurstStats:
        """Get current burst statistics."""
        return self._stats

    def get_active_instances(self) -> List[CloudInstance]:
        """Get list of active cloud instances."""
        return list(self._active_instances.values())

    def estimate_cost(
        self,
        total_frames: int,
        local_fps: float,
        deadline_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Estimate cloud burst cost.

        Args:
            total_frames: Total frames to process
            local_fps: Local processing speed
            deadline_seconds: Optional deadline

        Returns:
            Cost estimation
        """
        # Local only time
        local_time = total_frames / local_fps if local_fps > 0 else float('inf')

        estimate = {
            "total_frames": total_frames,
            "local_only_time_hours": local_time / 3600,
            "local_only_time_formatted": self._format_time(local_time),
            "burst_recommended": False,
            "estimated_instances": 0,
            "estimated_cost": 0.0,
            "estimated_time_hours": local_time / 3600,
        }

        if deadline_seconds and local_time > deadline_seconds:
            # Need to burst to meet deadline
            cloud_fps = 10.0  # Assume cloud processing speed

            # Calculate instances needed
            required_fps = total_frames / deadline_seconds
            cloud_instances = max(1, int((required_fps - local_fps) / cloud_fps))
            cloud_instances = min(cloud_instances, self.config.max_cloud_instances)

            # Estimate cost
            cost_per_instance = 2.0  # $/hour assumption
            hours = deadline_seconds / 3600
            cost = cloud_instances * cost_per_instance * hours

            estimate["burst_recommended"] = True
            estimate["estimated_instances"] = cloud_instances
            estimate["estimated_cost"] = cost
            estimate["estimated_time_hours"] = deadline_seconds / 3600

        return estimate

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"
