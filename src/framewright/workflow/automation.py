"""Workflow Automation for FrameWright.

Provides automated processing features:
- Watch folder: Monitor directory for new videos
- Queue system: Add videos to processing queue
- Profiles: Save and load named configurations
- Scene detection: Detect scenes for varied processing

Example:
    >>> # Watch folder mode
    >>> watcher = FolderWatcher("./incoming", "./done")
    >>> watcher.start()  # Runs until stopped

    >>> # Queue system
    >>> queue = ProcessingQueue()
    >>> queue.add("video1.mp4", priority=1)
    >>> queue.add("video2.mp4", priority=2)
    >>> queue.process_all()

    >>> # Profiles
    >>> profiles = ProfileManager()
    >>> profiles.save("my_settings", {"sr_model": "hat", "scale_factor": 4})
    >>> config = profiles.load("my_settings")
"""

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from queue import PriorityQueue

logger = logging.getLogger(__name__)


# =============================================================================
# Watch Folder
# =============================================================================

@dataclass
class WatchEvent:
    """Event from file watcher."""
    event_type: str  # "created", "modified", "deleted"
    path: Path
    timestamp: datetime


class FolderWatcher:
    """Monitor a folder for new videos and auto-process them.

    Example:
        >>> watcher = FolderWatcher(
        ...     watch_dir="./incoming",
        ...     output_dir="./processed",
        ...     config={"preset": "best"}
        ... )
        >>> watcher.start()
        # Now copy videos to ./incoming and they'll be processed
        >>> watcher.stop()
    """

    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.webm', '.m4v', '.flv'}

    def __init__(
        self,
        watch_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None,
        check_interval: float = 5.0,
        stable_time: float = 2.0,  # Wait for file to stop changing
        on_complete: Optional[Callable[[Path, Path], None]] = None,
        on_error: Optional[Callable[[Path, Exception], None]] = None,
    ):
        """Initialize folder watcher.

        Args:
            watch_dir: Directory to watch
            output_dir: Output directory for processed videos
            config: Processing configuration
            check_interval: Seconds between directory scans
            stable_time: Seconds file must be unchanged before processing
            on_complete: Callback when processing completes
            on_error: Callback on processing error
        """
        self.watch_dir = Path(watch_dir)
        self.output_dir = Path(output_dir)
        self.config = config or {}
        self.check_interval = check_interval
        self.stable_time = stable_time
        self.on_complete = on_complete
        self.on_error = on_error

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._processed: set = set()
        self._pending: Dict[Path, float] = {}  # path -> last_modified_time

    def start(self, blocking: bool = True):
        """Start watching folder.

        Args:
            blocking: If True, blocks until stop() is called
        """
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._running = True
        logger.info(f"Watching folder: {self.watch_dir}")
        logger.info(f"Output folder: {self.output_dir}")

        if blocking:
            self._watch_loop()
        else:
            self._thread = threading.Thread(target=self._watch_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop watching folder."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Folder watcher stopped")

    def _watch_loop(self):
        """Main watch loop."""
        while self._running:
            try:
                self._scan_directory()
                self._process_stable_files()
            except Exception as e:
                logger.error(f"Watch loop error: {e}")

            time.sleep(self.check_interval)

    def _scan_directory(self):
        """Scan directory for new videos."""
        for path in self.watch_dir.iterdir():
            if not path.is_file():
                continue

            if path.suffix.lower() not in self.VIDEO_EXTENSIONS:
                continue

            if path in self._processed:
                continue

            # Track file modification time
            try:
                mtime = path.stat().st_mtime
                if path not in self._pending or self._pending[path] != mtime:
                    self._pending[path] = mtime
                    logger.debug(f"Detected: {path.name}")
            except OSError:
                pass

    def _process_stable_files(self):
        """Process files that have been stable (not modified) for stable_time."""
        current_time = time.time()
        to_process = []

        for path, mtime in list(self._pending.items()):
            # Check if file is stable
            if current_time - mtime >= self.stable_time:
                to_process.append(path)

        for path in to_process:
            del self._pending[path]
            self._process_file(path)

    def _process_file(self, path: Path):
        """Process a single video file."""
        logger.info(f"Processing: {path.name}")

        try:
            from ..cli_simple import run_best_restore
            from ..ui.terminal import create_console

            output_path = self.output_dir / f"{path.stem}_restored.mp4"
            console = create_console(quiet=True)

            run_best_restore(
                path,
                output_path,
                console,
                **self.config
            )

            self._processed.add(path)
            logger.info(f"Completed: {output_path.name}")

            if self.on_complete:
                self.on_complete(path, output_path)

            # Move original to processed subfolder
            processed_dir = self.watch_dir / "processed"
            processed_dir.mkdir(exist_ok=True)
            shutil.move(str(path), str(processed_dir / path.name))

        except Exception as e:
            logger.error(f"Failed to process {path.name}: {e}")
            if self.on_error:
                self.on_error(path, e)

            # Move to failed subfolder
            failed_dir = self.watch_dir / "failed"
            failed_dir.mkdir(exist_ok=True)
            shutil.move(str(path), str(failed_dir / path.name))


# =============================================================================
# Processing Queue
# =============================================================================

class QueueItemStatus(Enum):
    """Status of queue item."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueueItem:
    """Item in processing queue."""
    id: str
    video_path: Path
    output_path: Optional[Path]
    config: Dict[str, Any]
    priority: int = 5  # 1 = highest, 10 = lowest
    status: QueueItemStatus = QueueItemStatus.PENDING
    added_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    progress: float = 0.0

    def __lt__(self, other):
        """Compare for priority queue (lower priority value = higher priority)."""
        return self.priority < other.priority


class ProcessingQueue:
    """Queue system for batch video processing.

    Supports:
    - Priority-based processing
    - Pause/resume
    - Progress tracking
    - Persistent queue state

    Example:
        >>> queue = ProcessingQueue()
        >>> queue.add("video1.mp4", priority=1)  # Process first
        >>> queue.add("video2.mp4", priority=5)  # Process after
        >>> queue.start()
    """

    def __init__(
        self,
        max_concurrent: int = 1,
        state_file: Optional[Path] = None,
    ):
        """Initialize processing queue.

        Args:
            max_concurrent: Maximum concurrent processing jobs
            state_file: File to persist queue state
        """
        self.max_concurrent = max_concurrent
        self.state_file = state_file or (Path.home() / '.framewright' / 'queue.json')

        self._queue: PriorityQueue = PriorityQueue()
        self._items: Dict[str, QueueItem] = {}
        self._running = False
        self._paused = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        self._load_state()

    def add(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        priority: int = 5,
    ) -> str:
        """Add video to queue.

        Args:
            video_path: Input video path
            output_path: Output path (auto-generated if None)
            config: Processing configuration
            priority: Priority (1=highest, 10=lowest)

        Returns:
            Queue item ID
        """
        import uuid

        video_path = Path(video_path)
        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_restored.mp4"

        item_id = str(uuid.uuid4())[:8]
        item = QueueItem(
            id=item_id,
            video_path=video_path,
            output_path=output_path,
            config=config or {},
            priority=priority,
        )

        with self._lock:
            self._items[item_id] = item
            self._queue.put((priority, item_id))

        self._save_state()
        logger.info(f"Added to queue: {video_path.name} (priority={priority})")

        return item_id

    def remove(self, item_id: str) -> bool:
        """Remove item from queue.

        Args:
            item_id: Item ID to remove

        Returns:
            True if removed
        """
        with self._lock:
            if item_id in self._items:
                item = self._items[item_id]
                if item.status == QueueItemStatus.PENDING:
                    item.status = QueueItemStatus.CANCELLED
                    self._save_state()
                    return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get queue status.

        Returns:
            Status dictionary
        """
        with self._lock:
            pending = sum(1 for i in self._items.values() if i.status == QueueItemStatus.PENDING)
            processing = sum(1 for i in self._items.values() if i.status == QueueItemStatus.PROCESSING)
            completed = sum(1 for i in self._items.values() if i.status == QueueItemStatus.COMPLETED)
            failed = sum(1 for i in self._items.values() if i.status == QueueItemStatus.FAILED)

            return {
                "running": self._running,
                "paused": self._paused,
                "pending": pending,
                "processing": processing,
                "completed": completed,
                "failed": failed,
                "total": len(self._items),
            }

    def list_items(self) -> List[Dict[str, Any]]:
        """List all queue items.

        Returns:
            List of item info dictionaries
        """
        with self._lock:
            return [
                {
                    "id": item.id,
                    "video": str(item.video_path.name),
                    "status": item.status.value,
                    "priority": item.priority,
                    "progress": item.progress,
                    "error": item.error,
                }
                for item in sorted(self._items.values(), key=lambda x: (x.priority, x.added_at))
            ]

    def start(self, blocking: bool = False):
        """Start processing queue.

        Args:
            blocking: If True, blocks until queue is empty
        """
        self._running = True
        self._paused = False

        if blocking:
            self._process_loop()
        else:
            self._thread = threading.Thread(target=self._process_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop processing queue."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=30)

    def pause(self):
        """Pause processing."""
        self._paused = True
        logger.info("Queue paused")

    def resume(self):
        """Resume processing."""
        self._paused = False
        logger.info("Queue resumed")

    def clear_completed(self) -> int:
        """Remove completed, failed, and cancelled items from queue.

        Returns:
            Number of items cleared
        """
        cleared = 0
        with self._lock:
            to_remove = [
                item_id for item_id, item in self._items.items()
                if item.status in (
                    QueueItemStatus.COMPLETED,
                    QueueItemStatus.FAILED,
                    QueueItemStatus.CANCELLED,
                )
            ]
            for item_id in to_remove:
                del self._items[item_id]
                cleared += 1

        if cleared:
            self._save_state()
            logger.info(f"Cleared {cleared} completed/failed items")

        return cleared

    def _process_loop(self):
        """Main processing loop."""
        while self._running:
            if self._paused:
                time.sleep(1)
                continue

            # Get next item
            item = self._get_next_item()
            if item is None:
                time.sleep(1)
                continue

            # Process item
            self._process_item(item)

    def _get_next_item(self) -> Optional[QueueItem]:
        """Get next item to process."""
        try:
            _, item_id = self._queue.get_nowait()
            with self._lock:
                item = self._items.get(item_id)
                if item and item.status == QueueItemStatus.PENDING:
                    return item
        except:
            pass
        return None

    def _process_item(self, item: QueueItem):
        """Process a queue item."""
        item.status = QueueItemStatus.PROCESSING
        item.started_at = datetime.now()
        self._save_state()

        logger.info(f"Processing: {item.video_path.name}")

        try:
            from ..cli_simple import run_best_restore
            from ..ui.terminal import create_console

            console = create_console(quiet=True)

            run_best_restore(
                item.video_path,
                item.output_path,
                console,
                **item.config
            )

            item.status = QueueItemStatus.COMPLETED
            item.completed_at = datetime.now()
            item.progress = 1.0
            logger.info(f"Completed: {item.video_path.name}")

        except Exception as e:
            item.status = QueueItemStatus.FAILED
            item.error = str(e)
            logger.error(f"Failed: {item.video_path.name} - {e}")

        self._save_state()

    def _save_state(self):
        """Save queue state to file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    item_id: {
                        "video_path": str(item.video_path),
                        "output_path": str(item.output_path) if item.output_path else None,
                        "config": item.config,
                        "priority": item.priority,
                        "status": item.status.value,
                        "error": item.error,
                    }
                    for item_id, item in self._items.items()
                }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save queue state: {e}")

    def _load_state(self):
        """Load queue state from file."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file) as f:
                data = json.load(f)

            for item_id, item_data in data.items():
                status = QueueItemStatus(item_data.get("status", "pending"))
                if status == QueueItemStatus.PROCESSING:
                    status = QueueItemStatus.PENDING  # Restart interrupted jobs

                item = QueueItem(
                    id=item_id,
                    video_path=Path(item_data["video_path"]),
                    output_path=Path(item_data["output_path"]) if item_data.get("output_path") else None,
                    config=item_data.get("config", {}),
                    priority=item_data.get("priority", 5),
                    status=status,
                    error=item_data.get("error"),
                )
                self._items[item_id] = item

                if item.status == QueueItemStatus.PENDING:
                    self._queue.put((item.priority, item_id))

            logger.info(f"Loaded {len(self._items)} items from queue state")

        except Exception as e:
            logger.debug(f"Failed to load queue state: {e}")


# =============================================================================
# Profiles
# =============================================================================

@dataclass
class Profile:
    """A saved configuration profile."""
    name: str
    config: Dict[str, Any]
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None


class ProfileManager:
    """Manage saved configuration profiles.

    Example:
        >>> profiles = ProfileManager()
        >>> profiles.save("archive_4k", {
        ...     "preset": "ultimate",
        ...     "scale_factor": 4,
        ...     "enable_frame_generation": True,
        ... }, description="For 4K archive restoration")
        >>> config = profiles.load("archive_4k")
    """

    def __init__(self, profiles_dir: Optional[Path] = None):
        """Initialize profile manager.

        Args:
            profiles_dir: Directory to store profiles
        """
        self.profiles_dir = profiles_dir or (Path.home() / '.framewright' / 'profiles')
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        config: Dict[str, Any],
        description: str = "",
    ) -> Path:
        """Save a configuration profile.

        Args:
            name: Profile name (alphanumeric + underscores)
            config: Configuration dictionary
            description: Profile description

        Returns:
            Path to saved profile
        """
        # Sanitize name
        name = "".join(c if c.isalnum() or c == '_' else '_' for c in name)

        profile_path = self.profiles_dir / f"{name}.json"

        data = {
            "name": name,
            "description": description,
            "config": config,
            "created_at": datetime.now().isoformat(),
        }

        with open(profile_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved profile: {name}")
        return profile_path

    def load(self, name: str) -> Dict[str, Any]:
        """Load a configuration profile.

        Args:
            name: Profile name

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If profile doesn't exist
        """
        profile_path = self.profiles_dir / f"{name}.json"

        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {name}")

        with open(profile_path) as f:
            data = json.load(f)

        logger.info(f"Loaded profile: {name}")
        return data.get("config", {})

    def delete(self, name: str) -> bool:
        """Delete a profile.

        Args:
            name: Profile name

        Returns:
            True if deleted
        """
        profile_path = self.profiles_dir / f"{name}.json"

        if profile_path.exists():
            profile_path.unlink()
            logger.info(f"Deleted profile: {name}")
            return True

        return False

    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all saved profiles.

        Returns:
            List of profile info dictionaries
        """
        profiles = []

        for path in self.profiles_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                profiles.append({
                    "name": data.get("name", path.stem),
                    "description": data.get("description", ""),
                    "created_at": data.get("created_at"),
                })
            except Exception:
                pass

        return profiles

    def get_builtin_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get built-in profile templates.

        Returns:
            Dictionary of built-in profiles
        """
        return {
            "archive_4k": {
                "description": "Maximum quality 4K restoration for archive footage",
                "config": {
                    "preset": "ultimate",
                    "scale_factor": 4,
                    "enable_frame_generation": True,
                    "enable_tap_denoise": True,
                    "enable_temporal_colorization": True,
                    "frame_generation_model": "svd",
                }
            },
            "quick_cleanup": {
                "description": "Fast cleanup for modern videos",
                "config": {
                    "preset": "fast",
                    "scale_factor": 2,
                    "enable_tap_denoise": True,
                }
            },
            "colorize_1940s": {
                "description": "Optimized for 1940s-era black and white footage",
                "config": {
                    "preset": "quality",
                    "scale_factor": 4,
                    "enable_colorization": True,
                    "enable_temporal_colorization": True,
                    "enable_frame_generation": True,
                    "temporal_method": "hybrid",
                }
            },
            "home_movie": {
                "description": "For VHS/8mm home movies",
                "config": {
                    "preset": "quality",
                    "scale_factor": 2,
                    "enable_tap_denoise": True,
                    "auto_face_restore": True,
                    "enable_audio_enhance": True,
                }
            },
            "youtube_ready": {
                "description": "Optimized for YouTube upload",
                "config": {
                    "preset": "balanced",
                    "scale_factor": 2,
                    "export_preset": "youtube",
                }
            },
        }


# =============================================================================
# Scene Detection
# =============================================================================

@dataclass
class Scene:
    """A detected scene in video."""
    index: int
    start_frame: int
    end_frame: int
    start_time: float  # seconds
    end_time: float
    scene_type: str  # "dialogue", "action", "landscape", "unknown"
    has_faces: bool = False
    motion_level: float = 0.0  # 0-1
    brightness: float = 0.5


class SceneDetector:
    """Detect scene boundaries and types in video.

    Used for applying different settings to different scenes:
    - Dialogue scenes: Prioritize face enhancement
    - Action scenes: Prioritize temporal consistency
    - Landscapes: Prioritize detail/sharpness
    """

    def __init__(
        self,
        threshold: float = 30.0,
        min_scene_length: int = 24,  # frames
    ):
        """Initialize scene detector.

        Args:
            threshold: Scene change threshold
            min_scene_length: Minimum scene length in frames
        """
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self._cv2 = None

    def _ensure_cv2(self) -> bool:
        """Ensure OpenCV is available."""
        try:
            import cv2
            self._cv2 = cv2
            return True
        except ImportError:
            return False

    def detect_scenes(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Scene]:
        """Detect scenes in video.

        Args:
            video_path: Path to video
            progress_callback: Progress callback

        Returns:
            List of detected scenes
        """
        if not self._ensure_cv2():
            logger.warning("OpenCV not available for scene detection")
            return []

        cv2 = self._cv2
        import numpy as np

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        scenes = []
        scene_boundaries = [0]  # Start of first scene

        prev_frame = None
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)

                # Detect scene change
                if mean_diff > self.threshold:
                    if frame_num - scene_boundaries[-1] >= self.min_scene_length:
                        scene_boundaries.append(frame_num)

            prev_frame = gray
            frame_num += 1

            if progress_callback and frame_num % 100 == 0:
                progress_callback(frame_num / total_frames)

        # Add end of video
        scene_boundaries.append(total_frames)

        # Analyze each scene
        for i in range(len(scene_boundaries) - 1):
            start_frame = scene_boundaries[i]
            end_frame = scene_boundaries[i + 1]

            scene = Scene(
                index=i + 1,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_frame / fps,
                end_time=end_frame / fps,
                scene_type="unknown",
            )

            # Analyze scene content
            scene_info = self._analyze_scene(cap, start_frame, end_frame, fps)
            scene.scene_type = scene_info.get("type", "unknown")
            scene.has_faces = scene_info.get("has_faces", False)
            scene.motion_level = scene_info.get("motion", 0.0)
            scene.brightness = scene_info.get("brightness", 0.5)

            scenes.append(scene)

        cap.release()
        return scenes

    def _analyze_scene(
        self,
        cap,
        start_frame: int,
        end_frame: int,
        fps: float,
    ) -> Dict[str, Any]:
        """Analyze scene content."""
        cv2 = self._cv2
        import numpy as np

        # Sample frames from scene
        num_samples = min(5, (end_frame - start_frame) // int(fps))
        sample_frames = np.linspace(start_frame, end_frame - 1, num_samples, dtype=int)

        motion_scores = []
        brightness_scores = []
        face_counts = []

        # Try to load face detector
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except:
            face_cascade = None

        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Brightness
            brightness_scores.append(np.mean(gray) / 255.0)

            # Motion (edge density as proxy)
            edges = cv2.Canny(gray, 50, 150)
            motion_scores.append(np.sum(edges > 0) / edges.size)

            # Face detection
            if face_cascade is not None:
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                face_counts.append(len(faces))

        # Determine scene type
        avg_motion = np.mean(motion_scores) if motion_scores else 0
        avg_brightness = np.mean(brightness_scores) if brightness_scores else 0.5
        has_faces = any(c > 0 for c in face_counts) if face_counts else False

        if has_faces and avg_motion < 0.1:
            scene_type = "dialogue"
        elif avg_motion > 0.2:
            scene_type = "action"
        elif avg_brightness > 0.6 and avg_motion < 0.05:
            scene_type = "landscape"
        else:
            scene_type = "unknown"

        return {
            "type": scene_type,
            "has_faces": has_faces,
            "motion": avg_motion,
            "brightness": avg_brightness,
        }

    def get_config_for_scene(self, scene: Scene) -> Dict[str, Any]:
        """Get recommended config adjustments for scene type.

        Args:
            scene: Scene to get config for

        Returns:
            Config adjustments for this scene
        """
        adjustments = {}

        if scene.scene_type == "dialogue":
            adjustments["auto_face_restore"] = True
            adjustments["face_model"] = "aesrgan"
            adjustments["temporal_window"] = 8  # Less aggressive temporal

        elif scene.scene_type == "action":
            adjustments["temporal_window"] = 16  # More temporal smoothing
            adjustments["enable_raft_flow"] = True

        elif scene.scene_type == "landscape":
            adjustments["sr_model"] = "hat"  # Maximum detail
            adjustments["auto_face_restore"] = False

        return adjustments
