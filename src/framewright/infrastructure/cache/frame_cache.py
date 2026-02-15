"""Frame caching infrastructure for video restoration.

Provides high-performance frame caching with LRU eviction, memory-mapped
file backing for large videos, and thread-safe access.
"""

import logging
import mmap
import os
import struct
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    from .eviction import (
        EvictionCandidate,
        EvictionPolicy,
        LRUEviction,
        LFUEviction,
        SizeAwareEviction,
        CompositeEviction,
        create_eviction_policy,
    )
except ImportError:
    from eviction import (
        EvictionCandidate,
        EvictionPolicy,
        LRUEviction,
        LFUEviction,
        SizeAwareEviction,
        CompositeEviction,
        create_eviction_policy,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FrameCacheConfig:
    """Configuration for frame cache.

    Attributes:
        max_memory_mb: Maximum memory usage in megabytes
        max_frames: Maximum number of frames to cache (0 = unlimited)
        eviction_policy: Eviction policy name ("lru", "lfu", "fifo", "size")
        use_memory_mapping: Use memory-mapped files for large caches
        mmap_threshold_mb: Size threshold for switching to memory mapping
        disk_cache_dir: Directory for disk cache spillover
        enable_disk_spillover: Enable spilling to disk when memory full
        compression_level: Compression level (0=none, 1-9=zlib levels)
        prefetch_count: Number of frames to prefetch ahead
    """
    max_memory_mb: int = 2048
    max_frames: int = 0
    eviction_policy: str = "lru"
    use_memory_mapping: bool = True
    mmap_threshold_mb: int = 512
    disk_cache_dir: Optional[Path] = None
    enable_disk_spillover: bool = True
    compression_level: int = 0
    prefetch_count: int = 5

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.compression_level < 0 or self.compression_level > 9:
            raise ValueError("compression_level must be 0-9")
        if self.disk_cache_dir is not None:
            self.disk_cache_dir = Path(self.disk_cache_dir)


@dataclass
class CachedFrame:
    """Represents a cached frame.

    Attributes:
        frame_id: Unique identifier for the frame
        data: Frame data (numpy array or bytes)
        width: Frame width in pixels
        height: Frame height in pixels
        channels: Number of color channels
        dtype: Data type string
        size_bytes: Size in bytes
        created_at: Unix timestamp when cached
        last_accessed: Unix timestamp of last access
        access_count: Number of times accessed
        is_compressed: Whether data is compressed
        is_on_disk: Whether data is stored on disk
        disk_path: Path to disk file if on disk
    """
    frame_id: Union[int, str]
    data: Optional[Union[np.ndarray, bytes]] = None
    width: int = 0
    height: int = 0
    channels: int = 3
    dtype: str = "uint8"
    size_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    is_compressed: bool = False
    is_on_disk: bool = False
    disk_path: Optional[Path] = None

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1

    def get_array(self) -> Optional[np.ndarray]:
        """Get frame data as numpy array.

        Returns:
            Frame data as numpy array, or None if not available
        """
        if self.data is None:
            return None

        if isinstance(self.data, np.ndarray):
            return self.data

        # Decompress if needed
        if self.is_compressed:
            import zlib
            decompressed = zlib.decompress(self.data)
            arr = np.frombuffer(decompressed, dtype=self.dtype)
        else:
            arr = np.frombuffer(self.data, dtype=self.dtype)

        return arr.reshape((self.height, self.width, self.channels))


@dataclass
class FrameCacheStats:
    """Statistics for frame cache.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        evictions: Number of evictions
        disk_reads: Number of reads from disk cache
        disk_writes: Number of writes to disk cache
        memory_used_bytes: Current memory usage
        frames_cached: Number of frames in cache
        hit_ratio: Cache hit ratio
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    disk_reads: int = 0
    disk_writes: int = 0
    memory_used_bytes: int = 0
    frames_cached: int = 0
    hit_ratio: float = 0.0

    def update_hit_ratio(self) -> None:
        """Recalculate hit ratio."""
        total = self.hits + self.misses
        self.hit_ratio = self.hits / total if total > 0 else 0.0


# =============================================================================
# Memory-Mapped Frame Storage
# =============================================================================

class MemoryMappedStorage:
    """Memory-mapped file storage for efficient large frame caching.

    Uses memory-mapped files to allow OS-level memory management and
    efficient access to frame data without loading everything into RAM.
    """

    # Header format: magic (4B) + version (2B) + frame_count (4B) + reserved (22B)
    HEADER_SIZE = 32
    HEADER_MAGIC = b"FWFC"  # FrameWright Frame Cache
    HEADER_VERSION = 1

    # Frame entry format: frame_id (8B) + offset (8B) + size (8B) + flags (4B)
    ENTRY_SIZE = 28

    def __init__(
        self,
        path: Path,
        max_size_bytes: int,
        create: bool = True,
    ):
        """Initialize memory-mapped storage.

        Args:
            path: Path to the memory-mapped file
            max_size_bytes: Maximum file size
            create: Whether to create the file if it doesn't exist
        """
        self.path = Path(path)
        self.max_size_bytes = max_size_bytes
        self._file = None
        self._mmap: Optional[mmap.mmap] = None
        self._lock = threading.RLock()
        self._frame_index: Dict[Union[int, str], Tuple[int, int]] = {}  # frame_id -> (offset, size)
        self._write_offset = self.HEADER_SIZE

        if create and not self.path.exists():
            self._create_file()
        elif self.path.exists():
            self._open_file()

    def _create_file(self) -> None:
        """Create and initialize the memory-mapped file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Create file with initial size
        with open(self.path, "wb") as f:
            # Write header
            header = struct.pack(
                "<4sHI22s",
                self.HEADER_MAGIC,
                self.HEADER_VERSION,
                0,  # frame count
                b"\x00" * 22,  # reserved
            )
            f.write(header)
            # Extend file to max size
            f.seek(self.max_size_bytes - 1)
            f.write(b"\x00")

        self._open_file()

    def _open_file(self) -> None:
        """Open the memory-mapped file."""
        self._file = open(self.path, "r+b")
        self._mmap = mmap.mmap(self._file.fileno(), 0)
        self._read_header()

    def _read_header(self) -> None:
        """Read and validate file header."""
        if self._mmap is None:
            return

        self._mmap.seek(0)
        header_data = self._mmap.read(self.HEADER_SIZE)

        magic, version, frame_count, _ = struct.unpack("<4sHI22s", header_data)

        if magic != self.HEADER_MAGIC:
            raise ValueError(f"Invalid cache file: {self.path}")

        if version != self.HEADER_VERSION:
            raise ValueError(f"Unsupported cache version: {version}")

        self._write_offset = self.HEADER_SIZE

    def write_frame(
        self,
        frame_id: Union[int, str],
        data: bytes,
    ) -> Tuple[int, int]:
        """Write frame data to storage.

        Args:
            frame_id: Frame identifier
            data: Frame data as bytes

        Returns:
            Tuple of (offset, size) where data was written
        """
        with self._lock:
            if self._mmap is None:
                raise RuntimeError("Storage not initialized")

            size = len(data)
            if self._write_offset + size > self.max_size_bytes:
                raise MemoryError("Memory-mapped storage full")

            offset = self._write_offset
            self._mmap.seek(offset)
            self._mmap.write(data)

            self._frame_index[frame_id] = (offset, size)
            self._write_offset += size

            return offset, size

    def read_frame(self, frame_id: Union[int, str]) -> Optional[bytes]:
        """Read frame data from storage.

        Args:
            frame_id: Frame identifier

        Returns:
            Frame data as bytes, or None if not found
        """
        with self._lock:
            if self._mmap is None:
                return None

            if frame_id not in self._frame_index:
                return None

            offset, size = self._frame_index[frame_id]
            self._mmap.seek(offset)
            return self._mmap.read(size)

    def remove_frame(self, frame_id: Union[int, str]) -> None:
        """Remove frame from index (does not reclaim space).

        Args:
            frame_id: Frame identifier
        """
        with self._lock:
            self._frame_index.pop(frame_id, None)

    def contains(self, frame_id: Union[int, str]) -> bool:
        """Check if frame is in storage.

        Args:
            frame_id: Frame identifier

        Returns:
            True if frame exists
        """
        return frame_id in self._frame_index

    def get_usage(self) -> Dict[str, Any]:
        """Get storage usage statistics.

        Returns:
            Dictionary with usage information
        """
        return {
            "frames": len(self._frame_index),
            "used_bytes": self._write_offset - self.HEADER_SIZE,
            "max_bytes": self.max_size_bytes,
            "utilization": (self._write_offset - self.HEADER_SIZE) / self.max_size_bytes,
        }

    def compact(self) -> int:
        """Compact storage by removing gaps (expensive operation).

        Returns:
            Number of bytes reclaimed
        """
        # For simplicity, this implementation doesn't support true compaction
        # A full implementation would rewrite the file
        logger.warning("Compaction not implemented - recreate cache for space recovery")
        return 0

    def close(self) -> None:
        """Close the memory-mapped file."""
        with self._lock:
            if self._mmap is not None:
                self._mmap.close()
                self._mmap = None
            if self._file is not None:
                self._file.close()
                self._file = None


# =============================================================================
# Frame Cache
# =============================================================================

class FrameCache:
    """High-performance frame cache with LRU eviction.

    Provides thread-safe caching of video frames with configurable
    eviction policies, optional memory mapping for large caches,
    and disk spillover capability.

    Example:
        >>> config = FrameCacheConfig(max_memory_mb=1024)
        >>> cache = FrameCache(config)
        >>>
        >>> # Cache a frame
        >>> frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        >>> cache.put(0, frame)
        >>>
        >>> # Retrieve frame
        >>> cached = cache.get(0)
        >>> assert cached is not None
        >>>
        >>> # Get range of frames
        >>> frames = cache.get_range(0, 10)
    """

    def __init__(self, config: Optional[FrameCacheConfig] = None):
        """Initialize frame cache.

        Args:
            config: Cache configuration
        """
        self.config = config or FrameCacheConfig()
        self._frames: OrderedDict[Union[int, str], CachedFrame] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = FrameCacheStats()

        # Initialize eviction policy
        self._eviction_policy = create_eviction_policy(self.config.eviction_policy)

        # Memory tracking
        self._memory_used = 0
        self._max_memory = self.config.max_memory_mb * 1024 * 1024

        # Memory-mapped storage (initialized lazily)
        self._mmap_storage: Optional[MemoryMappedStorage] = None

        # Disk cache
        self._disk_cache_dir: Optional[Path] = None
        if self.config.enable_disk_spillover:
            self._init_disk_cache()

        logger.info(
            f"FrameCache initialized: max_memory={self.config.max_memory_mb}MB, "
            f"policy={self.config.eviction_policy}"
        )

    def _init_disk_cache(self) -> None:
        """Initialize disk cache directory."""
        if self.config.disk_cache_dir:
            self._disk_cache_dir = self.config.disk_cache_dir
        else:
            self._disk_cache_dir = Path(tempfile.gettempdir()) / "framewright_cache"

        self._disk_cache_dir.mkdir(parents=True, exist_ok=True)

    def _init_mmap_storage(self) -> None:
        """Initialize memory-mapped storage lazily."""
        if self._mmap_storage is not None:
            return

        if not self.config.use_memory_mapping:
            return

        if self._memory_used < self.config.mmap_threshold_mb * 1024 * 1024:
            return

        mmap_path = (self._disk_cache_dir or Path(tempfile.gettempdir())) / "frame_cache.mmap"
        self._mmap_storage = MemoryMappedStorage(
            mmap_path,
            max_size_bytes=self._max_memory * 2,  # Allow 2x for spillover
        )
        logger.info(f"Initialized memory-mapped storage: {mmap_path}")

    def get(self, frame_id: Union[int, str]) -> Optional[np.ndarray]:
        """Get a cached frame.

        Args:
            frame_id: Frame identifier (typically frame number)

        Returns:
            Frame data as numpy array, or None if not cached
        """
        with self._lock:
            if frame_id not in self._frames:
                self._stats.misses += 1
                self._stats.update_hit_ratio()
                return None

            cached = self._frames[frame_id]
            cached.touch()
            self._eviction_policy.on_access(frame_id)
            self._frames.move_to_end(frame_id)

            self._stats.hits += 1
            self._stats.update_hit_ratio()

            # Load from disk if needed
            if cached.is_on_disk and cached.disk_path:
                return self._load_from_disk(cached)

            return cached.get_array()

    def put(
        self,
        frame_id: Union[int, str],
        frame: np.ndarray,
        priority: int = 0,
    ) -> bool:
        """Cache a frame.

        Args:
            frame_id: Frame identifier
            frame: Frame data as numpy array
            priority: Priority level (higher = less likely to evict)

        Returns:
            True if frame was cached successfully
        """
        with self._lock:
            # Calculate frame size
            size_bytes = frame.nbytes

            # Check if we need to evict
            while (self._memory_used + size_bytes > self._max_memory or
                   (self.config.max_frames > 0 and len(self._frames) >= self.config.max_frames)):
                if not self._evict_one():
                    # Can't evict anything, try disk spillover
                    if self.config.enable_disk_spillover:
                        if not self._spill_to_disk():
                            logger.warning(f"Cannot cache frame {frame_id}: cache full")
                            return False
                    else:
                        logger.warning(f"Cannot cache frame {frame_id}: cache full")
                        return False

            # Compress if configured
            data: Union[np.ndarray, bytes] = frame
            is_compressed = False

            if self.config.compression_level > 0:
                import zlib
                data = zlib.compress(frame.tobytes(), self.config.compression_level)
                size_bytes = len(data)
                is_compressed = True

            # Create cached frame entry
            cached = CachedFrame(
                frame_id=frame_id,
                data=data,
                width=frame.shape[1],
                height=frame.shape[0],
                channels=frame.shape[2] if len(frame.shape) > 2 else 1,
                dtype=str(frame.dtype),
                size_bytes=size_bytes,
                is_compressed=is_compressed,
            )

            # Update existing or add new
            if frame_id in self._frames:
                old_cached = self._frames[frame_id]
                self._memory_used -= old_cached.size_bytes
                self._eviction_policy.on_remove(frame_id)

            self._frames[frame_id] = cached
            self._memory_used += size_bytes
            self._eviction_policy.on_insert(frame_id, size_bytes)

            self._stats.frames_cached = len(self._frames)
            self._stats.memory_used_bytes = self._memory_used

            return True

    def get_or_compute(
        self,
        frame_id: Union[int, str],
        compute_fn: Callable[[], np.ndarray],
        priority: int = 0,
    ) -> np.ndarray:
        """Get frame from cache or compute if not cached.

        Args:
            frame_id: Frame identifier
            compute_fn: Function to compute frame if not cached (no arguments)
            priority: Priority level for caching (higher = less likely to evict)

        Returns:
            Frame data as numpy array

        Example:
            >>> cache = FrameCache()
            >>> frame = cache.get_or_compute(
            ...     key=f"denoised_{frame_id}",
            ...     compute_fn=lambda: denoise_frame(original_frame)
            ... )
        """
        # Try to get from cache
        cached_frame = self.get(frame_id)
        if cached_frame is not None:
            return cached_frame

        # Not in cache, compute it
        computed_frame = compute_fn()

        # Cache the result
        self.put(frame_id, computed_frame, priority)

        return computed_frame

    def get_range(
        self,
        start: int,
        end: int,
    ) -> Dict[int, np.ndarray]:
        """Get a range of cached frames.

        Args:
            start: Start frame number (inclusive)
            end: End frame number (exclusive)

        Returns:
            Dictionary mapping frame numbers to frame data
        """
        result: Dict[int, np.ndarray] = {}

        for frame_id in range(start, end):
            frame = self.get(frame_id)
            if frame is not None:
                result[frame_id] = frame

        return result

    def contains(self, frame_id: Union[int, str]) -> bool:
        """Check if a frame is cached.

        Args:
            frame_id: Frame identifier

        Returns:
            True if frame is in cache
        """
        return frame_id in self._frames

    def remove(self, frame_id: Union[int, str]) -> bool:
        """Remove a frame from cache.

        Args:
            frame_id: Frame identifier

        Returns:
            True if frame was removed
        """
        with self._lock:
            if frame_id not in self._frames:
                return False

            cached = self._frames.pop(frame_id)
            self._memory_used -= cached.size_bytes
            self._eviction_policy.on_remove(frame_id)

            # Clean up disk file if exists
            if cached.is_on_disk and cached.disk_path and cached.disk_path.exists():
                cached.disk_path.unlink()

            self._stats.frames_cached = len(self._frames)
            self._stats.memory_used_bytes = self._memory_used

            return True

    def _evict_one(self) -> bool:
        """Evict one frame from cache.

        Returns:
            True if a frame was evicted
        """
        if not self._frames:
            return False

        # Build eviction candidates
        candidates = [
            EvictionCandidate(
                key=frame_id,
                score=0,
                size_bytes=cached.size_bytes,
                metadata={"access_count": cached.access_count},
            )
            for frame_id, cached in self._frames.items()
            if not cached.is_on_disk  # Don't evict already-spilled frames
        ]

        if not candidates:
            return False

        victim = self._eviction_policy.select_victim(candidates)
        if victim is None:
            return False

        # Remove the victim
        self.remove(victim.key)
        self._stats.evictions += 1

        logger.debug(f"Evicted frame {victim.key} ({victim.size_bytes} bytes)")
        return True

    def _spill_to_disk(self) -> bool:
        """Spill least recently used frame to disk.

        Returns:
            True if a frame was spilled
        """
        if self._disk_cache_dir is None:
            return False

        # Find LRU frame that's not already on disk
        for frame_id, cached in self._frames.items():
            if not cached.is_on_disk and cached.data is not None:
                # Write to disk
                disk_path = self._disk_cache_dir / f"frame_{frame_id}.npy"

                frame_data = cached.get_array()
                if frame_data is not None:
                    np.save(disk_path, frame_data)

                    # Update cached entry
                    self._memory_used -= cached.size_bytes
                    cached.data = None
                    cached.is_on_disk = True
                    cached.disk_path = disk_path

                    self._stats.disk_writes += 1
                    self._stats.memory_used_bytes = self._memory_used

                    logger.debug(f"Spilled frame {frame_id} to disk")
                    return True

        return False

    def _load_from_disk(self, cached: CachedFrame) -> Optional[np.ndarray]:
        """Load a frame from disk cache.

        Args:
            cached: CachedFrame with disk_path set

        Returns:
            Frame data as numpy array
        """
        if cached.disk_path is None or not cached.disk_path.exists():
            return None

        frame = np.load(cached.disk_path)
        self._stats.disk_reads += 1

        return frame

    def clear(self) -> None:
        """Clear all cached frames."""
        with self._lock:
            # Clean up disk files
            for cached in self._frames.values():
                if cached.is_on_disk and cached.disk_path and cached.disk_path.exists():
                    cached.disk_path.unlink()

            self._frames.clear()
            self._memory_used = 0
            self._stats = FrameCacheStats()

            logger.info("Frame cache cleared")

    def get_stats(self) -> FrameCacheStats:
        """Get cache statistics.

        Returns:
            FrameCacheStats with current metrics
        """
        with self._lock:
            self._stats.frames_cached = len(self._frames)
            self._stats.memory_used_bytes = self._memory_used
            self._stats.update_hit_ratio()
            return self._stats

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information.

        Returns:
            Dictionary with memory usage details
        """
        with self._lock:
            return {
                "used_bytes": self._memory_used,
                "max_bytes": self._max_memory,
                "utilization": self._memory_used / self._max_memory,
                "frames_in_memory": sum(1 for f in self._frames.values() if not f.is_on_disk),
                "frames_on_disk": sum(1 for f in self._frames.values() if f.is_on_disk),
            }

    def prefetch(
        self,
        frame_ids: List[Union[int, str]],
        loader: Callable[[Union[int, str]], np.ndarray],
    ) -> int:
        """Prefetch frames into cache.

        Args:
            frame_ids: List of frame IDs to prefetch
            loader: Function to load a frame by ID

        Returns:
            Number of frames successfully prefetched
        """
        count = 0
        for frame_id in frame_ids:
            if frame_id not in self._frames:
                try:
                    frame = loader(frame_id)
                    if self.put(frame_id, frame):
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to prefetch frame {frame_id}: {e}")
        return count

    def close(self) -> None:
        """Close cache and clean up resources."""
        with self._lock:
            if self._mmap_storage is not None:
                self._mmap_storage.close()
                self._mmap_storage = None

            self.clear()
            logger.info("Frame cache closed")


# =============================================================================
# Disk Frame Cache
# =============================================================================

class DiskFrameCache:
    """Disk-based frame cache for persistence across sessions.

    Stores frames on disk with an in-memory index for fast lookups.
    Useful for caching processed frames that are expensive to regenerate.

    Example:
        >>> cache = DiskFrameCache(Path("./frame_cache"))
        >>> cache.put("frame_001", processed_frame)
        >>> loaded = cache.get("frame_001")
    """

    def __init__(
        self,
        cache_dir: Path,
        max_size_gb: float = 10.0,
        eviction_policy: str = "lru",
    ):
        """Initialize disk frame cache.

        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in gigabytes
            eviction_policy: Eviction policy for when cache is full
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self._eviction_policy = create_eviction_policy(eviction_policy)

        self._index: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self._current_size = 0

        # Load existing index
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index from disk."""
        index_path = self.cache_dir / "index.json"
        if index_path.exists():
            import json
            try:
                with open(index_path) as f:
                    data = json.load(f)
                    self._index = OrderedDict(data.get("entries", {}))
                    self._current_size = data.get("total_size", 0)

                # Verify files exist
                for frame_id in list(self._index.keys()):
                    frame_path = self.cache_dir / f"{frame_id}.npy"
                    if not frame_path.exists():
                        del self._index[frame_id]

            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")

    def _save_index(self) -> None:
        """Save cache index to disk."""
        import json
        index_path = self.cache_dir / "index.json"
        try:
            with open(index_path, "w") as f:
                json.dump({
                    "entries": dict(self._index),
                    "total_size": self._current_size,
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")

    def get(self, frame_id: str) -> Optional[np.ndarray]:
        """Get a cached frame from disk.

        Args:
            frame_id: Frame identifier

        Returns:
            Frame data as numpy array, or None if not cached
        """
        with self._lock:
            if frame_id not in self._index:
                return None

            frame_path = self.cache_dir / f"{frame_id}.npy"
            if not frame_path.exists():
                del self._index[frame_id]
                return None

            # Update access metadata
            self._index[frame_id]["last_accessed"] = time.time()
            self._index[frame_id]["access_count"] = self._index[frame_id].get("access_count", 0) + 1
            self._index.move_to_end(frame_id)
            self._eviction_policy.on_access(frame_id)

            return np.load(frame_path)

    def put(self, frame_id: str, frame: np.ndarray) -> bool:
        """Store a frame to disk cache.

        Args:
            frame_id: Frame identifier
            frame: Frame data as numpy array

        Returns:
            True if stored successfully
        """
        with self._lock:
            size_bytes = frame.nbytes

            # Evict if needed
            while self._current_size + size_bytes > self.max_size_bytes:
                if not self._evict_one():
                    logger.warning(f"Cannot cache frame {frame_id}: disk cache full")
                    return False

            # Save frame
            frame_path = self.cache_dir / f"{frame_id}.npy"
            np.save(frame_path, frame)

            # Update index
            if frame_id in self._index:
                self._current_size -= self._index[frame_id]["size_bytes"]

            self._index[frame_id] = {
                "size_bytes": size_bytes,
                "width": frame.shape[1],
                "height": frame.shape[0],
                "dtype": str(frame.dtype),
                "created_at": time.time(),
                "last_accessed": time.time(),
                "access_count": 0,
            }
            self._current_size += size_bytes
            self._eviction_policy.on_insert(frame_id, size_bytes)

            self._save_index()
            return True

    def _evict_one(self) -> bool:
        """Evict one frame from disk cache.

        Returns:
            True if a frame was evicted
        """
        if not self._index:
            return False

        # Build candidates
        candidates = [
            EvictionCandidate(
                key=frame_id,
                score=0,
                size_bytes=info["size_bytes"],
            )
            for frame_id, info in self._index.items()
        ]

        victim = self._eviction_policy.select_victim(candidates)
        if victim is None:
            return False

        # Remove victim
        frame_path = self.cache_dir / f"{victim.key}.npy"
        if frame_path.exists():
            frame_path.unlink()

        self._current_size -= self._index[victim.key]["size_bytes"]
        del self._index[victim.key]
        self._eviction_policy.on_remove(victim.key)

        return True

    def contains(self, frame_id: str) -> bool:
        """Check if frame is in cache.

        Args:
            frame_id: Frame identifier

        Returns:
            True if cached
        """
        return frame_id in self._index

    def remove(self, frame_id: str) -> bool:
        """Remove a frame from cache.

        Args:
            frame_id: Frame identifier

        Returns:
            True if removed
        """
        with self._lock:
            if frame_id not in self._index:
                return False

            frame_path = self.cache_dir / f"{frame_id}.npy"
            if frame_path.exists():
                frame_path.unlink()

            self._current_size -= self._index[frame_id]["size_bytes"]
            del self._index[frame_id]
            self._eviction_policy.on_remove(frame_id)

            self._save_index()
            return True

    def clear(self) -> None:
        """Clear entire disk cache."""
        with self._lock:
            for frame_id in list(self._index.keys()):
                frame_path = self.cache_dir / f"{frame_id}.npy"
                if frame_path.exists():
                    frame_path.unlink()

            self._index.clear()
            self._current_size = 0
            self._save_index()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "frames_cached": len(self._index),
            "size_bytes": self._current_size,
            "max_size_bytes": self.max_size_bytes,
            "utilization": self._current_size / self.max_size_bytes if self.max_size_bytes > 0 else 0,
        }
