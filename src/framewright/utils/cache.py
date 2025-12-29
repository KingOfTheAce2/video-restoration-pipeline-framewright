"""Intelligent frame caching module for FrameWright video restoration pipeline.

Provides content-based frame caching with LRU eviction, persistence across
sessions, and multi-level caching (memory + disk) for optimal performance.
"""
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Cache Configuration and Data Classes
# =============================================================================

@dataclass
class CacheConfig:
    """Configuration for the frame cache system.

    Attributes:
        max_size_gb: Maximum cache size in gigabytes
        ttl_days: Time-to-live for cache entries in days
        enable_persistence: Enable disk persistence of cache metadata
        cache_dir: Directory for storing cached frames
        memory_cache_size: Number of entries to keep in memory LRU cache
        enable_compression: Enable compression for cached frames
        hash_algorithm: Algorithm for perceptual hashing ('phash', 'dhash', 'average')
    """
    max_size_gb: float = 10.0
    ttl_days: int = 30
    enable_persistence: bool = True
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".framewright" / "cache")
    memory_cache_size: int = 100
    enable_compression: bool = False
    hash_algorithm: str = "phash"

    def __post_init__(self) -> None:
        """Validate configuration and convert paths."""
        if not isinstance(self.cache_dir, Path):
            self.cache_dir = Path(self.cache_dir)

        if self.max_size_gb <= 0:
            raise ValueError("max_size_gb must be positive")

        if self.ttl_days <= 0:
            raise ValueError("ttl_days must be positive")

        if self.memory_cache_size < 0:
            raise ValueError("memory_cache_size must be non-negative")

        valid_algorithms = {"phash", "dhash", "average", "md5"}
        if self.hash_algorithm not in valid_algorithms:
            raise ValueError(f"hash_algorithm must be one of {valid_algorithms}")


@dataclass
class CacheEntry:
    """Represents a cached frame entry.

    Attributes:
        source_hash: Perceptual hash of the source frame
        enhanced_path: Path to the enhanced/cached frame
        config_hash: Hash of the processing configuration
        timestamp: Unix timestamp when entry was created
        size_bytes: Size of the cached file in bytes
        source_path: Original source file path (for reference)
        access_count: Number of times this entry was accessed
        last_accessed: Unix timestamp of last access
    """
    source_hash: str
    enhanced_path: Path
    config_hash: str
    timestamp: float
    size_bytes: int
    source_path: Optional[str] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Convert paths to Path objects."""
        if not isinstance(self.enhanced_path, Path):
            self.enhanced_path = Path(self.enhanced_path)

    @property
    def age_days(self) -> float:
        """Calculate age of entry in days."""
        return (time.time() - self.timestamp) / 86400

    @property
    def is_valid(self) -> bool:
        """Check if the cached file still exists."""
        return self.enhanced_path.exists()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_hash": self.source_hash,
            "enhanced_path": str(self.enhanced_path),
            "config_hash": self.config_hash,
            "timestamp": self.timestamp,
            "size_bytes": self.size_bytes,
            "source_path": self.source_path,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        return cls(
            source_hash=data["source_hash"],
            enhanced_path=Path(data["enhanced_path"]),
            config_hash=data["config_hash"],
            timestamp=data["timestamp"],
            size_bytes=data["size_bytes"],
            source_path=data.get("source_path"),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed", data["timestamp"]),
        )


@dataclass
class CacheStats:
    """Statistics about cache usage.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        hit_ratio: Ratio of hits to total requests
        total_size_mb: Total size of cached files in megabytes
        entry_count: Number of entries in cache
        evictions: Number of entries evicted
        oldest_entry_age_days: Age of oldest entry in days
        memory_entries: Number of entries in memory cache
    """
    hits: int = 0
    misses: int = 0
    hit_ratio: float = 0.0
    total_size_mb: float = 0.0
    entry_count: int = 0
    evictions: int = 0
    oldest_entry_age_days: float = 0.0
    memory_entries: int = 0

    def update_hit_ratio(self) -> None:
        """Recalculate hit ratio."""
        total = self.hits + self.misses
        self.hit_ratio = self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# Perceptual Hashing
# =============================================================================

class PerceptualHasher:
    """Computes perceptual hashes for frame images.

    Supports multiple algorithms:
    - phash: Perceptual hash using DCT
    - dhash: Difference hash (simpler, faster)
    - average: Average hash
    - md5: Content hash (not perceptual, but reliable)
    """

    def __init__(self, algorithm: str = "phash", hash_size: int = 16):
        """Initialize hasher.

        Args:
            algorithm: Hash algorithm to use
            hash_size: Size of the hash (bits per side for image hashes)
        """
        self.algorithm = algorithm
        self.hash_size = hash_size
        self._pil_available = self._check_pil()

    def _check_pil(self) -> bool:
        """Check if PIL/Pillow is available."""
        try:
            from PIL import Image
            return True
        except ImportError:
            return False

    def compute_hash(self, image_path: Path) -> str:
        """Compute perceptual hash for an image.

        Args:
            image_path: Path to the image file

        Returns:
            Hexadecimal hash string
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.algorithm == "md5":
            return self._compute_md5(image_path)

        if not self._pil_available:
            logger.warning("PIL not available, falling back to MD5 hash")
            return self._compute_md5(image_path)

        try:
            if self.algorithm == "phash":
                return self._compute_phash(image_path)
            elif self.algorithm == "dhash":
                return self._compute_dhash(image_path)
            elif self.algorithm == "average":
                return self._compute_average_hash(image_path)
            else:
                return self._compute_md5(image_path)
        except Exception as e:
            logger.warning(f"Perceptual hash failed, falling back to MD5: {e}")
            return self._compute_md5(image_path)

    def _compute_md5(self, image_path: Path) -> str:
        """Compute MD5 hash of file content."""
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _compute_phash(self, image_path: Path) -> str:
        """Compute perceptual hash using DCT."""
        from PIL import Image
        import numpy as np

        # Load and resize image
        img = Image.open(image_path).convert("L")
        img = img.resize((self.hash_size * 4, self.hash_size * 4), Image.Resampling.LANCZOS)

        # Convert to numpy array
        pixels = np.array(img, dtype=np.float64)

        # Compute DCT (simplified version without scipy)
        # Use a simple frequency analysis
        dct_low = self._simple_dct(pixels, self.hash_size)

        # Compute median and generate hash
        median = np.median(dct_low)
        diff = dct_low > median

        # Convert to hex string
        hash_bits = diff.flatten()
        hash_int = int("".join(str(int(b)) for b in hash_bits), 2)
        return format(hash_int, f"0{self.hash_size * self.hash_size // 4}x")

    def _simple_dct(self, pixels: "np.ndarray", size: int) -> "np.ndarray":
        """Simple DCT-like frequency analysis for perceptual hashing."""
        import numpy as np

        # Resize to target size
        h, w = pixels.shape
        result = np.zeros((size, size))

        step_h = h // size
        step_w = w // size

        for i in range(size):
            for j in range(size):
                block = pixels[i*step_h:(i+1)*step_h, j*step_w:(j+1)*step_w]
                # Use mean as a simple frequency component
                result[i, j] = np.mean(block)

        return result

    def _compute_dhash(self, image_path: Path) -> str:
        """Compute difference hash (horizontal gradient)."""
        from PIL import Image

        # Load and resize
        img = Image.open(image_path).convert("L")
        img = img.resize((self.hash_size + 1, self.hash_size), Image.Resampling.LANCZOS)

        pixels = list(img.getdata())
        width = self.hash_size + 1

        # Compute horizontal differences
        diff_bits = []
        for row in range(self.hash_size):
            for col in range(self.hash_size):
                left = pixels[row * width + col]
                right = pixels[row * width + col + 1]
                diff_bits.append(1 if left > right else 0)

        # Convert to hex
        hash_int = int("".join(str(b) for b in diff_bits), 2)
        return format(hash_int, f"0{self.hash_size * self.hash_size // 4}x")

    def _compute_average_hash(self, image_path: Path) -> str:
        """Compute average hash."""
        from PIL import Image

        # Load and resize
        img = Image.open(image_path).convert("L")
        img = img.resize((self.hash_size, self.hash_size), Image.Resampling.LANCZOS)

        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)

        # Generate hash bits
        hash_bits = [1 if p > avg else 0 for p in pixels]
        hash_int = int("".join(str(b) for b in hash_bits), 2)
        return format(hash_int, f"0{self.hash_size * self.hash_size // 4}x")


# =============================================================================
# Frame Cache
# =============================================================================

class FrameCache:
    """LRU cache for enhanced frames with content-based identification.

    Uses perceptual hashing to identify frames, allowing cache hits even
    when the same frame content appears in different files. Implements
    LRU eviction to manage memory usage.

    Example:
        >>> cache = FrameCache(CacheConfig(max_size_gb=5.0))
        >>> cache.initialize()
        >>>
        >>> # Check for cached version
        >>> source_hash = cache.compute_frame_hash(source_frame)
        >>> config_hash = cache.compute_config_hash(config)
        >>> cached = cache.get_cached(source_hash, config_hash)
        >>>
        >>> if cached:
        ...     # Use cached frame
        ...     enhanced = cached
        ... else:
        ...     # Process and cache
        ...     enhanced = enhance_frame(source_frame)
        ...     cache.store(source_frame, enhanced, config_hash)
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize frame cache.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self.config = config or CacheConfig()
        self._hasher = PerceptualHasher(algorithm=self.config.hash_algorithm)
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = threading.RLock()
        self._db_path: Optional[Path] = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize cache directories and load persisted state."""
        with self._lock:
            if self._initialized:
                return

            # Create cache directory
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectory for cached frames
            frames_dir = self.config.cache_dir / "frames"
            frames_dir.mkdir(exist_ok=True)

            # Initialize database if persistence enabled
            if self.config.enable_persistence:
                self._db_path = self.config.cache_dir / "cache.db"
                self._init_database()
                self._load_from_database()

            self._initialized = True
            logger.info(
                f"Frame cache initialized: {self.config.cache_dir} "
                f"(max: {self.config.max_size_gb}GB, entries: {len(self._memory_cache)})"
            )

    def _init_database(self) -> None:
        """Initialize SQLite database for cache metadata."""
        with self._get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    source_hash TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    enhanced_path TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    source_path TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL NOT NULL,
                    PRIMARY KEY (source_hash, config_hash)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed
                ON cache_entries(last_accessed)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON cache_entries(timestamp)
            """)
            conn.commit()

    @contextmanager
    def _get_db_connection(self) -> Iterator[sqlite3.Connection]:
        """Get database connection with proper handling."""
        if self._db_path is None:
            raise RuntimeError("Database not initialized")

        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _load_from_database(self) -> None:
        """Load cache entries from database into memory cache."""
        if not self._db_path or not self._db_path.exists():
            return

        try:
            with self._get_db_connection() as conn:
                # Load most recently accessed entries up to memory limit
                cursor = conn.execute("""
                    SELECT * FROM cache_entries
                    ORDER BY last_accessed DESC
                    LIMIT ?
                """, (self.config.memory_cache_size,))

                for row in cursor:
                    entry = CacheEntry(
                        source_hash=row["source_hash"],
                        enhanced_path=Path(row["enhanced_path"]),
                        config_hash=row["config_hash"],
                        timestamp=row["timestamp"],
                        size_bytes=row["size_bytes"],
                        source_path=row["source_path"],
                        access_count=row["access_count"],
                        last_accessed=row["last_accessed"],
                    )

                    # Only add if file still exists
                    if entry.is_valid:
                        key = self._make_key(entry.source_hash, entry.config_hash)
                        self._memory_cache[key] = entry

                # Update stats
                self._update_stats_from_db(conn)

        except Exception as e:
            logger.warning(f"Failed to load cache from database: {e}")

    def _update_stats_from_db(self, conn: sqlite3.Connection) -> None:
        """Update statistics from database."""
        cursor = conn.execute("""
            SELECT
                COUNT(*) as entry_count,
                SUM(size_bytes) as total_size,
                MIN(timestamp) as oldest
            FROM cache_entries
        """)
        row = cursor.fetchone()

        if row:
            self._stats.entry_count = row["entry_count"] or 0
            self._stats.total_size_mb = (row["total_size"] or 0) / (1024 * 1024)
            if row["oldest"]:
                self._stats.oldest_entry_age_days = (time.time() - row["oldest"]) / 86400

        self._stats.memory_entries = len(self._memory_cache)

    def _make_key(self, source_hash: str, config_hash: str) -> str:
        """Create composite key from source and config hashes."""
        return f"{source_hash}:{config_hash}"

    def compute_frame_hash(self, frame_path: Path) -> str:
        """Compute perceptual hash for a frame.

        Args:
            frame_path: Path to the frame image

        Returns:
            Hexadecimal hash string
        """
        return self._hasher.compute_hash(frame_path)

    def compute_config_hash(self, config: Any) -> str:
        """Compute hash for processing configuration.

        Args:
            config: Configuration object or dictionary

        Returns:
            Hexadecimal hash string
        """
        if hasattr(config, "get_hash"):
            return config.get_hash()

        if hasattr(config, "to_dict"):
            config_data = config.to_dict()
        elif isinstance(config, dict):
            config_data = config
        else:
            config_data = str(config)

        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get_cached(
        self,
        source_hash: str,
        config_hash: str
    ) -> Optional[Path]:
        """Get cached enhanced frame if available.

        Args:
            source_hash: Perceptual hash of source frame
            config_hash: Hash of processing configuration

        Returns:
            Path to cached enhanced frame, or None if not cached
        """
        with self._lock:
            key = self._make_key(source_hash, config_hash)

            # Check memory cache first
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if entry.is_valid:
                    # Update access info
                    entry.access_count += 1
                    entry.last_accessed = time.time()
                    # Move to end (most recently used)
                    self._memory_cache.move_to_end(key)

                    self._stats.hits += 1
                    self._stats.update_hit_ratio()

                    logger.debug(f"Cache hit (memory): {source_hash[:12]}")
                    return entry.enhanced_path
                else:
                    # Invalid entry, remove it
                    del self._memory_cache[key]

            # Check disk cache
            if self.config.enable_persistence and self._db_path:
                entry = self._get_from_database(source_hash, config_hash)
                if entry and entry.is_valid:
                    # Update access info in database
                    self._update_access_in_database(source_hash, config_hash)

                    # Add to memory cache
                    self._add_to_memory_cache(entry)

                    self._stats.hits += 1
                    self._stats.update_hit_ratio()

                    logger.debug(f"Cache hit (disk): {source_hash[:12]}")
                    return entry.enhanced_path

            self._stats.misses += 1
            self._stats.update_hit_ratio()
            return None

    def _get_from_database(
        self,
        source_hash: str,
        config_hash: str
    ) -> Optional[CacheEntry]:
        """Get entry from database."""
        if not self._db_path:
            return None

        try:
            with self._get_db_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM cache_entries
                    WHERE source_hash = ? AND config_hash = ?
                """, (source_hash, config_hash))
                row = cursor.fetchone()

                if row:
                    return CacheEntry(
                        source_hash=row["source_hash"],
                        enhanced_path=Path(row["enhanced_path"]),
                        config_hash=row["config_hash"],
                        timestamp=row["timestamp"],
                        size_bytes=row["size_bytes"],
                        source_path=row["source_path"],
                        access_count=row["access_count"],
                        last_accessed=row["last_accessed"],
                    )
        except Exception as e:
            logger.warning(f"Database read error: {e}")

        return None

    def _update_access_in_database(
        self,
        source_hash: str,
        config_hash: str
    ) -> None:
        """Update access info in database."""
        if not self._db_path:
            return

        try:
            with self._get_db_connection() as conn:
                conn.execute("""
                    UPDATE cache_entries
                    SET access_count = access_count + 1,
                        last_accessed = ?
                    WHERE source_hash = ? AND config_hash = ?
                """, (time.time(), source_hash, config_hash))
                conn.commit()
        except Exception as e:
            logger.warning(f"Database update error: {e}")

    def _add_to_memory_cache(self, entry: CacheEntry) -> None:
        """Add entry to memory cache with LRU eviction."""
        key = self._make_key(entry.source_hash, entry.config_hash)

        # Evict oldest entries if at capacity
        while len(self._memory_cache) >= self.config.memory_cache_size:
            evicted_key, evicted_entry = self._memory_cache.popitem(last=False)
            logger.debug(f"Evicted from memory cache: {evicted_key[:24]}")

        self._memory_cache[key] = entry
        self._stats.memory_entries = len(self._memory_cache)

    def store(
        self,
        source_path: Path,
        enhanced_path: Path,
        config_hash: str,
        source_hash: Optional[str] = None
    ) -> CacheEntry:
        """Store an enhanced frame in the cache.

        Args:
            source_path: Path to source frame
            enhanced_path: Path to enhanced frame
            config_hash: Hash of processing configuration
            source_hash: Optional pre-computed source hash

        Returns:
            Created cache entry
        """
        with self._lock:
            # Compute source hash if not provided
            if source_hash is None:
                source_hash = self.compute_frame_hash(source_path)

            # Copy enhanced frame to cache directory
            cache_subdir = self.config.cache_dir / "frames" / source_hash[:2]
            cache_subdir.mkdir(parents=True, exist_ok=True)

            cached_filename = f"{source_hash[:16]}_{config_hash[:8]}.png"
            cached_path = cache_subdir / cached_filename

            # Copy the enhanced frame
            if enhanced_path != cached_path:
                shutil.copy2(enhanced_path, cached_path)

            # Get file size
            size_bytes = cached_path.stat().st_size

            # Check if we need to evict entries
            self._enforce_size_limit(size_bytes)

            # Create entry
            entry = CacheEntry(
                source_hash=source_hash,
                enhanced_path=cached_path,
                config_hash=config_hash,
                timestamp=time.time(),
                size_bytes=size_bytes,
                source_path=str(source_path),
                access_count=0,
                last_accessed=time.time(),
            )

            # Add to memory cache
            self._add_to_memory_cache(entry)

            # Persist to database
            if self.config.enable_persistence and self._db_path:
                self._save_to_database(entry)

            # Update stats
            self._stats.total_size_mb += size_bytes / (1024 * 1024)
            self._stats.entry_count += 1

            logger.debug(f"Cached frame: {source_hash[:12]} ({size_bytes / 1024:.1f} KB)")
            return entry

    def _save_to_database(self, entry: CacheEntry) -> None:
        """Save entry to database."""
        if not self._db_path:
            return

        try:
            with self._get_db_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries
                    (source_hash, config_hash, enhanced_path, timestamp,
                     size_bytes, source_path, access_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.source_hash,
                    entry.config_hash,
                    str(entry.enhanced_path),
                    entry.timestamp,
                    entry.size_bytes,
                    entry.source_path,
                    entry.access_count,
                    entry.last_accessed,
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Database write error: {e}")

    def _enforce_size_limit(self, new_entry_size: int) -> None:
        """Evict entries if cache exceeds size limit."""
        max_bytes = self.config.max_size_gb * 1024 * 1024 * 1024
        current_bytes = self._stats.total_size_mb * 1024 * 1024

        if current_bytes + new_entry_size <= max_bytes:
            return

        # Need to evict entries
        logger.info(
            f"Cache size limit reached ({self._stats.total_size_mb:.1f}MB / "
            f"{self.config.max_size_gb * 1024:.1f}MB), evicting old entries"
        )

        target_bytes = max_bytes * 0.8  # Evict down to 80% capacity
        bytes_to_free = (current_bytes + new_entry_size) - target_bytes

        self._evict_entries(int(bytes_to_free))

    def _evict_entries(self, bytes_to_free: int) -> int:
        """Evict entries to free specified amount of space.

        Uses LRU strategy (least recently accessed first).

        Args:
            bytes_to_free: Number of bytes to free

        Returns:
            Number of entries evicted
        """
        evicted_count = 0
        freed_bytes = 0

        if self.config.enable_persistence and self._db_path:
            # Get LRU entries from database
            try:
                with self._get_db_connection() as conn:
                    cursor = conn.execute("""
                        SELECT source_hash, config_hash, enhanced_path, size_bytes
                        FROM cache_entries
                        ORDER BY last_accessed ASC
                    """)

                    entries_to_delete = []
                    for row in cursor:
                        if freed_bytes >= bytes_to_free:
                            break

                        entries_to_delete.append((
                            row["source_hash"],
                            row["config_hash"],
                            row["enhanced_path"],
                            row["size_bytes"]
                        ))
                        freed_bytes += row["size_bytes"]

                    # Delete entries
                    for source_hash, config_hash, enhanced_path, size_bytes in entries_to_delete:
                        # Delete file
                        try:
                            Path(enhanced_path).unlink(missing_ok=True)
                        except Exception:
                            pass

                        # Delete from database
                        conn.execute("""
                            DELETE FROM cache_entries
                            WHERE source_hash = ? AND config_hash = ?
                        """, (source_hash, config_hash))

                        # Remove from memory cache
                        key = self._make_key(source_hash, config_hash)
                        self._memory_cache.pop(key, None)

                        evicted_count += 1
                        self._stats.evictions += 1

                    conn.commit()

            except Exception as e:
                logger.warning(f"Error during eviction: {e}")

        else:
            # Memory-only eviction
            while freed_bytes < bytes_to_free and self._memory_cache:
                key, entry = self._memory_cache.popitem(last=False)
                freed_bytes += entry.size_bytes
                evicted_count += 1
                self._stats.evictions += 1

                # Delete file
                try:
                    entry.enhanced_path.unlink(missing_ok=True)
                except Exception:
                    pass

        # Update stats
        self._stats.total_size_mb -= freed_bytes / (1024 * 1024)
        self._stats.entry_count -= evicted_count
        self._stats.memory_entries = len(self._memory_cache)

        logger.info(f"Evicted {evicted_count} entries, freed {freed_bytes / (1024 * 1024):.1f} MB")
        return evicted_count

    def cleanup_stale(self, max_age_days: Optional[int] = None) -> int:
        """Remove cache entries older than specified age.

        Args:
            max_age_days: Maximum age in days (default: config.ttl_days)

        Returns:
            Number of entries removed
        """
        with self._lock:
            if max_age_days is None:
                max_age_days = self.config.ttl_days

            cutoff_time = time.time() - (max_age_days * 86400)
            removed_count = 0
            freed_bytes = 0

            if self.config.enable_persistence and self._db_path:
                try:
                    with self._get_db_connection() as conn:
                        # Get stale entries
                        cursor = conn.execute("""
                            SELECT source_hash, config_hash, enhanced_path, size_bytes
                            FROM cache_entries
                            WHERE timestamp < ?
                        """, (cutoff_time,))

                        for row in cursor:
                            # Delete file
                            try:
                                Path(row["enhanced_path"]).unlink(missing_ok=True)
                            except Exception:
                                pass

                            freed_bytes += row["size_bytes"]
                            removed_count += 1

                            # Remove from memory cache
                            key = self._make_key(row["source_hash"], row["config_hash"])
                            self._memory_cache.pop(key, None)

                        # Delete from database
                        conn.execute("""
                            DELETE FROM cache_entries WHERE timestamp < ?
                        """, (cutoff_time,))
                        conn.commit()

                except Exception as e:
                    logger.warning(f"Error during stale cleanup: {e}")

            else:
                # Memory-only cleanup
                stale_keys = []
                for key, entry in self._memory_cache.items():
                    if entry.timestamp < cutoff_time:
                        stale_keys.append(key)
                        freed_bytes += entry.size_bytes

                for key in stale_keys:
                    entry = self._memory_cache.pop(key)
                    try:
                        entry.enhanced_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    removed_count += 1

            # Update stats
            self._stats.total_size_mb -= freed_bytes / (1024 * 1024)
            self._stats.entry_count -= removed_count
            self._stats.memory_entries = len(self._memory_cache)

            if removed_count > 0:
                logger.info(
                    f"Cleaned up {removed_count} stale entries "
                    f"(freed {freed_bytes / (1024 * 1024):.1f} MB)"
                )

            return removed_count

    def get_stats(self) -> CacheStats:
        """Get current cache statistics.

        Returns:
            CacheStats with current metrics
        """
        with self._lock:
            self._stats.memory_entries = len(self._memory_cache)
            return self._stats

    def clear(self) -> None:
        """Clear all cache entries and files."""
        with self._lock:
            # Clear memory cache
            self._memory_cache.clear()

            # Clear database
            if self.config.enable_persistence and self._db_path:
                try:
                    with self._get_db_connection() as conn:
                        conn.execute("DELETE FROM cache_entries")
                        conn.commit()
                except Exception as e:
                    logger.warning(f"Error clearing database: {e}")

            # Remove cached files
            frames_dir = self.config.cache_dir / "frames"
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
                frames_dir.mkdir()

            # Reset stats
            self._stats = CacheStats()

            logger.info("Cache cleared")

    def close(self) -> None:
        """Close cache and persist state."""
        with self._lock:
            logger.info(
                f"Closing cache: {self._stats.hits} hits, "
                f"{self._stats.misses} misses, "
                f"{self._stats.hit_ratio:.1%} hit ratio"
            )
            self._initialized = False


# =============================================================================
# Cache Manager
# =============================================================================

class CacheManager:
    """High-level cache manager for the video restoration pipeline.

    Manages cache operations, statistics, and integrates with the
    restoration workflow.

    Example:
        >>> manager = CacheManager()
        >>> manager.initialize()
        >>>
        >>> # Process frames with caching
        >>> for frame in frames:
        ...     cached = manager.get_cached_frame(frame, config)
        ...     if cached:
        ...         enhanced = cached
        ...     else:
        ...         enhanced = enhance(frame)
        ...         manager.cache_frame(frame, enhanced, config)
        >>>
        >>> # Log performance
        >>> stats = manager.get_stats()
        >>> print(f"Hit ratio: {stats.hit_ratio:.1%}")
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize cache manager.

        Args:
            config: Cache configuration
            cache_dir: Override for cache directory
        """
        if config is None:
            config = CacheConfig()

        if cache_dir is not None:
            config.cache_dir = Path(cache_dir)

        self.config = config
        self._cache = FrameCache(config)
        self._processing_config_hash: Optional[str] = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the cache manager."""
        if self._initialized:
            return

        self._cache.initialize()
        self._initialized = True

        # Perform maintenance
        self._cache.cleanup_stale()

        logger.info(f"Cache manager initialized: {self.config.cache_dir}")

    def set_processing_config(self, config: Any) -> str:
        """Set the current processing configuration.

        The config hash is used to ensure cached frames were
        processed with the same settings.

        Args:
            config: Processing configuration object

        Returns:
            Configuration hash string
        """
        self._processing_config_hash = self._cache.compute_config_hash(config)
        logger.debug(f"Processing config hash: {self._processing_config_hash}")
        return self._processing_config_hash

    def get_cached_frame(
        self,
        source_path: Path,
        config: Optional[Any] = None
    ) -> Optional[Path]:
        """Get cached enhanced frame if available.

        Args:
            source_path: Path to source frame
            config: Optional config (uses set config if None)

        Returns:
            Path to cached frame, or None
        """
        if not self._initialized:
            self.initialize()

        # Get config hash
        if config is not None:
            config_hash = self._cache.compute_config_hash(config)
        elif self._processing_config_hash is not None:
            config_hash = self._processing_config_hash
        else:
            logger.warning("No config hash set, cache lookup skipped")
            return None

        # Compute source hash
        source_hash = self._cache.compute_frame_hash(source_path)

        return self._cache.get_cached(source_hash, config_hash)

    def cache_frame(
        self,
        source_path: Path,
        enhanced_path: Path,
        config: Optional[Any] = None
    ) -> CacheEntry:
        """Cache an enhanced frame.

        Args:
            source_path: Path to source frame
            enhanced_path: Path to enhanced frame
            config: Optional config (uses set config if None)

        Returns:
            Created cache entry
        """
        if not self._initialized:
            self.initialize()

        # Get config hash
        if config is not None:
            config_hash = self._cache.compute_config_hash(config)
        elif self._processing_config_hash is not None:
            config_hash = self._processing_config_hash
        else:
            raise ValueError("No config hash available")

        return self._cache.store(
            source_path=source_path,
            enhanced_path=enhanced_path,
            config_hash=config_hash
        )

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with current metrics
        """
        return self._cache.get_stats()

    def log_stats(self) -> None:
        """Log cache statistics."""
        stats = self.get_stats()
        logger.info(
            f"Cache stats: {stats.hits} hits, {stats.misses} misses "
            f"({stats.hit_ratio:.1%} hit ratio), "
            f"{stats.entry_count} entries ({stats.total_size_mb:.1f} MB)"
        )

    def invalidate_config(self, config: Any) -> int:
        """Invalidate all cache entries for a specific config.

        Useful when config changes require reprocessing.

        Args:
            config: Configuration to invalidate

        Returns:
            Number of entries invalidated
        """
        config_hash = self._cache.compute_config_hash(config)

        # This would require iterating all entries
        # For now, we don't support partial invalidation
        # A full clear is recommended when config changes significantly
        logger.warning(
            "Config-specific invalidation not supported. "
            "Consider clearing cache if config changed significantly."
        )
        return 0

    def cleanup(self, max_age_days: Optional[int] = None) -> int:
        """Clean up stale cache entries.

        Args:
            max_age_days: Maximum age (default: config TTL)

        Returns:
            Number of entries removed
        """
        return self._cache.cleanup_stale(max_age_days)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def close(self) -> None:
        """Close cache manager."""
        self._cache.close()


# =============================================================================
# Helper Functions
# =============================================================================

def compute_frame_hash(frame_path: Path, algorithm: str = "phash") -> str:
    """Compute perceptual hash for a frame image.

    Convenience function for computing frame hashes without
    instantiating a full cache.

    Args:
        frame_path: Path to the frame image
        algorithm: Hash algorithm ('phash', 'dhash', 'average', 'md5')

    Returns:
        Hexadecimal hash string
    """
    hasher = PerceptualHasher(algorithm=algorithm)
    return hasher.compute_hash(frame_path)


def get_cache_manager(
    cache_dir: Optional[Path] = None,
    max_size_gb: float = 10.0,
    ttl_days: int = 30
) -> CacheManager:
    """Get a configured cache manager instance.

    Convenience function for quickly setting up caching.

    Args:
        cache_dir: Cache directory (default: ~/.framewright/cache)
        max_size_gb: Maximum cache size in GB
        ttl_days: Time-to-live for entries in days

    Returns:
        Initialized CacheManager
    """
    config = CacheConfig(
        max_size_gb=max_size_gb,
        ttl_days=ttl_days,
        cache_dir=cache_dir or Path.home() / ".framewright" / "cache",
    )

    manager = CacheManager(config)
    manager.initialize()
    return manager


# Singleton instance for global cache access
_global_cache_manager: Optional[CacheManager] = None


def get_global_cache() -> CacheManager:
    """Get the global cache manager instance.

    Creates and initializes a default cache manager on first call.

    Returns:
        Global CacheManager instance
    """
    global _global_cache_manager

    if _global_cache_manager is None:
        _global_cache_manager = get_cache_manager()

    return _global_cache_manager


def shutdown_cache() -> None:
    """Shutdown the global cache manager."""
    global _global_cache_manager

    if _global_cache_manager is not None:
        _global_cache_manager.close()
        _global_cache_manager = None
