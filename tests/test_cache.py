"""Tests for the intelligent frame caching module."""
import hashlib
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from framewright.utils.cache import (
    CacheConfig,
    CacheEntry,
    CacheManager,
    CacheStats,
    FrameCache,
    PerceptualHasher,
    compute_frame_hash,
    get_cache_manager,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def cache_config(temp_dir):
    """Create a test cache configuration."""
    return CacheConfig(
        max_size_gb=0.1,  # 100 MB for testing
        ttl_days=7,
        enable_persistence=True,
        cache_dir=temp_dir / "cache",
        memory_cache_size=10,
    )


@pytest.fixture
def frame_cache(cache_config):
    """Create a test frame cache."""
    cache = FrameCache(cache_config)
    cache.initialize()
    yield cache
    cache.close()


@pytest.fixture
def cache_manager(cache_config):
    """Create a test cache manager."""
    manager = CacheManager(cache_config)
    manager.initialize()
    yield manager
    manager.close()


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample PNG image for testing."""
    # Create a simple PNG file (1x1 pixel)
    image_path = temp_dir / "sample.png"

    # PNG header and minimal valid PNG content
    png_data = (
        b'\x89PNG\r\n\x1a\n'  # PNG signature
        b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde'  # 1x1 RGB
        b'\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00'
        b'\x03\x00\x01\x00\x05\xfe\xd4'  # Compressed data
        b'\x00\x00\x00\x00IEND\xaeB`\x82'  # End chunk
    )

    with open(image_path, 'wb') as f:
        f.write(png_data)

    return image_path


@pytest.fixture
def sample_enhanced_image(temp_dir):
    """Create a sample enhanced PNG image."""
    image_path = temp_dir / "enhanced.png"

    # Slightly different PNG content
    png_data = (
        b'\x89PNG\r\n\x1a\n'
        b'\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02'
        b'\x08\x02\x00\x00\x00\xfd\xd4\x9as'  # 2x2 RGB
        b'\x00\x00\x00\x12IDATx\x9cc\xfc\xff\xff?\x03\x10'
        b'\x00\x00\xff\xff\x03\x00\x05\xfe\x02\xfe\xa6\xba\xb0\xb0'
        b'\x00\x00\x00\x00IEND\xaeB`\x82'
    )

    with open(image_path, 'wb') as f:
        f.write(png_data)

    return image_path


# =============================================================================
# CacheConfig Tests
# =============================================================================

class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()

        assert config.max_size_gb == 10.0
        assert config.ttl_days == 30
        assert config.enable_persistence is True
        assert config.memory_cache_size == 100
        assert config.hash_algorithm == "phash"
        assert isinstance(config.cache_dir, Path)

    def test_custom_config(self, temp_dir):
        """Test custom configuration."""
        config = CacheConfig(
            max_size_gb=5.0,
            ttl_days=14,
            cache_dir=temp_dir / "custom_cache",
            memory_cache_size=50,
            hash_algorithm="dhash",
        )

        assert config.max_size_gb == 5.0
        assert config.ttl_days == 14
        assert config.cache_dir == temp_dir / "custom_cache"
        assert config.memory_cache_size == 50
        assert config.hash_algorithm == "dhash"

    def test_invalid_max_size(self):
        """Test validation of max_size_gb."""
        with pytest.raises(ValueError, match="max_size_gb must be positive"):
            CacheConfig(max_size_gb=0)

        with pytest.raises(ValueError, match="max_size_gb must be positive"):
            CacheConfig(max_size_gb=-1.0)

    def test_invalid_ttl(self):
        """Test validation of ttl_days."""
        with pytest.raises(ValueError, match="ttl_days must be positive"):
            CacheConfig(ttl_days=0)

    def test_invalid_algorithm(self):
        """Test validation of hash_algorithm."""
        with pytest.raises(ValueError, match="hash_algorithm must be one of"):
            CacheConfig(hash_algorithm="invalid")

    def test_path_conversion(self, temp_dir):
        """Test that string paths are converted to Path objects."""
        config = CacheConfig(cache_dir=str(temp_dir))
        assert isinstance(config.cache_dir, Path)


# =============================================================================
# CacheEntry Tests
# =============================================================================

class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_entry_creation(self, temp_dir):
        """Test creating a cache entry."""
        entry = CacheEntry(
            source_hash="abc123",
            enhanced_path=temp_dir / "enhanced.png",
            config_hash="cfg456",
            timestamp=time.time(),
            size_bytes=1024,
        )

        assert entry.source_hash == "abc123"
        assert entry.config_hash == "cfg456"
        assert entry.size_bytes == 1024
        assert entry.access_count == 0

    def test_entry_age(self):
        """Test age calculation."""
        old_timestamp = time.time() - (2 * 86400)  # 2 days ago
        entry = CacheEntry(
            source_hash="abc",
            enhanced_path=Path("/tmp/test.png"),
            config_hash="cfg",
            timestamp=old_timestamp,
            size_bytes=100,
        )

        assert 1.9 < entry.age_days < 2.1

    def test_entry_validity(self, sample_image, temp_dir):
        """Test is_valid property."""
        # Valid entry - file exists
        entry = CacheEntry(
            source_hash="abc",
            enhanced_path=sample_image,
            config_hash="cfg",
            timestamp=time.time(),
            size_bytes=100,
        )
        assert entry.is_valid is True

        # Invalid entry - file doesn't exist
        entry2 = CacheEntry(
            source_hash="abc",
            enhanced_path=temp_dir / "nonexistent.png",
            config_hash="cfg",
            timestamp=time.time(),
            size_bytes=100,
        )
        assert entry2.is_valid is False

    def test_entry_serialization(self, temp_dir):
        """Test to_dict and from_dict."""
        entry = CacheEntry(
            source_hash="abc123",
            enhanced_path=temp_dir / "test.png",
            config_hash="cfg456",
            timestamp=1234567890.0,
            size_bytes=2048,
            source_path="/original/frame.png",
            access_count=5,
        )

        # Serialize
        data = entry.to_dict()
        assert data["source_hash"] == "abc123"
        assert data["config_hash"] == "cfg456"
        assert data["size_bytes"] == 2048

        # Deserialize
        restored = CacheEntry.from_dict(data)
        assert restored.source_hash == entry.source_hash
        assert restored.config_hash == entry.config_hash
        assert restored.size_bytes == entry.size_bytes


# =============================================================================
# CacheStats Tests
# =============================================================================

class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_default_stats(self):
        """Test default statistics."""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_ratio == 0.0
        assert stats.entry_count == 0

    def test_hit_ratio_calculation(self):
        """Test hit ratio updates correctly."""
        stats = CacheStats(hits=8, misses=2)
        stats.update_hit_ratio()

        assert stats.hit_ratio == 0.8

    def test_hit_ratio_with_zero_requests(self):
        """Test hit ratio with no requests."""
        stats = CacheStats()
        stats.update_hit_ratio()

        assert stats.hit_ratio == 0.0


# =============================================================================
# PerceptualHasher Tests
# =============================================================================

class TestPerceptualHasher:
    """Tests for PerceptualHasher."""

    def test_md5_hash(self, sample_image):
        """Test MD5 hashing."""
        hasher = PerceptualHasher(algorithm="md5")
        hash_result = hasher.compute_hash(sample_image)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 32  # MD5 hex length

    def test_hash_consistency(self, sample_image):
        """Test that same file produces same hash."""
        hasher = PerceptualHasher(algorithm="md5")

        hash1 = hasher.compute_hash(sample_image)
        hash2 = hasher.compute_hash(sample_image)

        assert hash1 == hash2

    def test_different_files_different_hashes(self, sample_image, sample_enhanced_image):
        """Test that different files produce different hashes."""
        hasher = PerceptualHasher(algorithm="md5")

        hash1 = hasher.compute_hash(sample_image)
        hash2 = hasher.compute_hash(sample_enhanced_image)

        assert hash1 != hash2

    def test_file_not_found(self, temp_dir):
        """Test error handling for missing files."""
        hasher = PerceptualHasher()

        with pytest.raises(FileNotFoundError):
            hasher.compute_hash(temp_dir / "nonexistent.png")


# =============================================================================
# FrameCache Tests
# =============================================================================

class TestFrameCache:
    """Tests for FrameCache."""

    def test_cache_initialization(self, cache_config):
        """Test cache initialization."""
        cache = FrameCache(cache_config)
        cache.initialize()

        assert cache._initialized is True
        assert cache_config.cache_dir.exists()
        assert (cache_config.cache_dir / "frames").exists()

        cache.close()

    def test_compute_frame_hash(self, frame_cache, sample_image):
        """Test computing frame hash."""
        hash_result = frame_cache.compute_frame_hash(sample_image)

        assert isinstance(hash_result, str)
        assert len(hash_result) > 0

    def test_compute_config_hash_dict(self, frame_cache):
        """Test computing config hash from dict."""
        config = {"scale": 4, "model": "realesrgan"}
        hash1 = frame_cache.compute_config_hash(config)
        hash2 = frame_cache.compute_config_hash(config)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_compute_config_hash_object(self, frame_cache):
        """Test computing config hash from object with get_hash method."""
        mock_config = MagicMock()
        mock_config.get_hash.return_value = "custom_hash_123"

        result = frame_cache.compute_config_hash(mock_config)
        assert result == "custom_hash_123"

    def test_store_and_retrieve(
        self,
        frame_cache,
        sample_image,
        sample_enhanced_image
    ):
        """Test storing and retrieving cached frames."""
        config_hash = "test_config_123"

        # Store frame
        entry = frame_cache.store(
            source_path=sample_image,
            enhanced_path=sample_enhanced_image,
            config_hash=config_hash,
        )

        assert entry.source_hash is not None
        assert entry.config_hash == config_hash
        assert entry.enhanced_path.exists()

        # Retrieve frame
        source_hash = frame_cache.compute_frame_hash(sample_image)
        cached = frame_cache.get_cached(source_hash, config_hash)

        assert cached is not None
        assert cached.exists()

    def test_cache_miss(self, frame_cache, sample_image):
        """Test cache miss returns None."""
        source_hash = frame_cache.compute_frame_hash(sample_image)
        result = frame_cache.get_cached(source_hash, "nonexistent_config")

        assert result is None

    def test_cache_statistics(
        self,
        frame_cache,
        sample_image,
        sample_enhanced_image
    ):
        """Test cache statistics tracking."""
        config_hash = "stats_test"
        source_hash = frame_cache.compute_frame_hash(sample_image)

        # Initial stats
        stats = frame_cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0

        # Cache miss
        frame_cache.get_cached(source_hash, config_hash)
        stats = frame_cache.get_stats()
        assert stats.misses == 1

        # Store and hit
        frame_cache.store(sample_image, sample_enhanced_image, config_hash)
        frame_cache.get_cached(source_hash, config_hash)

        stats = frame_cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_ratio == 0.5

    def test_lru_eviction(self, cache_config, temp_dir, sample_enhanced_image):
        """Test LRU eviction from memory cache."""
        # Small memory cache
        cache_config.memory_cache_size = 3
        cache = FrameCache(cache_config)
        cache.initialize()

        # Create multiple source images
        source_images = []
        for i in range(5):
            img_path = temp_dir / f"source_{i}.png"
            shutil.copy(sample_enhanced_image, img_path)
            source_images.append(img_path)

        # Store more than memory cache size
        for img in source_images:
            cache.store(
                source_path=img,
                enhanced_path=sample_enhanced_image,
                config_hash="test_config",
            )

        # Memory cache should only have last 3 entries
        assert len(cache._memory_cache) <= 3

        cache.close()

    def test_cleanup_stale(
        self,
        frame_cache,
        sample_image,
        sample_enhanced_image
    ):
        """Test cleanup of stale entries."""
        # Store with old timestamp
        entry = frame_cache.store(
            sample_image,
            sample_enhanced_image,
            "old_config"
        )

        # Manually age the entry in database
        if frame_cache._db_path:
            old_timestamp = time.time() - (40 * 86400)  # 40 days ago
            with frame_cache._get_db_connection() as conn:
                conn.execute(
                    "UPDATE cache_entries SET timestamp = ?",
                    (old_timestamp,)
                )
                conn.commit()

        # Cleanup entries older than 30 days
        removed = frame_cache.cleanup_stale(max_age_days=30)

        assert removed >= 1

    def test_clear_cache(
        self,
        frame_cache,
        sample_image,
        sample_enhanced_image
    ):
        """Test clearing all cache entries."""
        # Store some entries
        frame_cache.store(sample_image, sample_enhanced_image, "config1")
        frame_cache.store(sample_image, sample_enhanced_image, "config2")

        stats_before = frame_cache.get_stats()
        assert stats_before.entry_count >= 2

        # Clear
        frame_cache.clear()

        stats_after = frame_cache.get_stats()
        assert stats_after.entry_count == 0
        assert len(frame_cache._memory_cache) == 0


# =============================================================================
# CacheManager Tests
# =============================================================================

class TestCacheManager:
    """Tests for CacheManager."""

    def test_manager_initialization(self, cache_config):
        """Test cache manager initialization."""
        manager = CacheManager(cache_config)
        manager.initialize()

        assert manager._initialized is True

        manager.close()

    def test_set_processing_config(self, cache_manager):
        """Test setting processing configuration."""
        config = {"scale": 4, "model": "realesrgan-x4plus"}
        config_hash = cache_manager.set_processing_config(config)

        assert config_hash is not None
        assert len(config_hash) == 16

    def test_cache_frame_workflow(
        self,
        cache_manager,
        sample_image,
        sample_enhanced_image
    ):
        """Test complete caching workflow."""
        config = {"scale": 4, "crf": 18}
        cache_manager.set_processing_config(config)

        # First access - cache miss
        cached = cache_manager.get_cached_frame(sample_image)
        assert cached is None

        # Store frame
        entry = cache_manager.cache_frame(sample_image, sample_enhanced_image)
        assert entry is not None

        # Second access - cache hit
        cached = cache_manager.get_cached_frame(sample_image)
        assert cached is not None
        assert cached.exists()

    def test_get_stats(self, cache_manager, sample_image, sample_enhanced_image):
        """Test getting statistics."""
        cache_manager.set_processing_config({"test": "config"})
        cache_manager.get_cached_frame(sample_image)  # Miss
        cache_manager.cache_frame(sample_image, sample_enhanced_image)
        cache_manager.get_cached_frame(sample_image)  # Hit

        stats = cache_manager.get_stats()

        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.entry_count >= 1

    def test_cleanup(self, cache_manager, sample_image, sample_enhanced_image):
        """Test cleanup functionality."""
        cache_manager.set_processing_config({"test": "config"})
        cache_manager.cache_frame(sample_image, sample_enhanced_image)

        # Cleanup with long TTL (entries are very new)
        removed = cache_manager.cleanup(max_age_days=30)

        # Entry was just created, so it shouldn't be removed with 30 day TTL
        assert removed == 0


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_compute_frame_hash_function(self, sample_image):
        """Test compute_frame_hash convenience function."""
        hash_result = compute_frame_hash(sample_image, algorithm="md5")

        assert isinstance(hash_result, str)
        assert len(hash_result) == 32

    def test_get_cache_manager_function(self, temp_dir):
        """Test get_cache_manager convenience function."""
        manager = get_cache_manager(
            cache_dir=temp_dir / "test_cache",
            max_size_gb=0.1,
            ttl_days=7,
        )

        assert manager._initialized is True
        assert manager.config.max_size_gb == 0.1
        assert manager.config.ttl_days == 7

        manager.close()


# =============================================================================
# Integration Tests
# =============================================================================

class TestCacheIntegration:
    """Integration tests for the caching system."""

    def test_persistence_across_instances(
        self,
        cache_config,
        sample_image,
        sample_enhanced_image
    ):
        """Test that cache persists across cache instances."""
        config_hash = "persist_test"

        # First instance - store
        cache1 = FrameCache(cache_config)
        cache1.initialize()
        source_hash = cache1.compute_frame_hash(sample_image)
        cache1.store(sample_image, sample_enhanced_image, config_hash)
        cache1.close()

        # Second instance - retrieve
        cache2 = FrameCache(cache_config)
        cache2.initialize()
        cached = cache2.get_cached(source_hash, config_hash)

        assert cached is not None
        assert cached.exists()

        cache2.close()

    def test_concurrent_access(
        self,
        frame_cache,
        sample_image,
        sample_enhanced_image
    ):
        """Test concurrent cache access (basic thread safety)."""
        import concurrent.futures
        import threading

        config_hash = "concurrent_test"
        errors = []

        def cache_operation(i):
            try:
                source_hash = frame_cache.compute_frame_hash(sample_image)

                # Alternate between read and write
                if i % 2 == 0:
                    frame_cache.store(
                        sample_image,
                        sample_enhanced_image,
                        f"{config_hash}_{i}"
                    )
                else:
                    frame_cache.get_cached(source_hash, config_hash)
            except Exception as e:
                errors.append(e)

        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cache_operation, i) for i in range(20)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"

    def test_size_limit_enforcement(self, temp_dir):
        """Test that size limits are enforced."""
        # Very small cache
        config = CacheConfig(
            max_size_gb=0.0001,  # ~100KB
            cache_dir=temp_dir / "small_cache",
            ttl_days=30,
        )

        cache = FrameCache(config)
        cache.initialize()

        # Create large-ish test files
        for i in range(10):
            source = temp_dir / f"source_{i}.png"
            enhanced = temp_dir / f"enhanced_{i}.png"

            # Create files with some content
            with open(source, 'wb') as f:
                f.write(b'0' * 20000)  # 20KB
            with open(enhanced, 'wb') as f:
                f.write(b'0' * 20000)  # 20KB

            cache.store(source, enhanced, f"config_{i}")

        # Stats should show evictions occurred
        stats = cache.get_stats()
        assert stats.evictions > 0 or stats.entry_count < 10

        cache.close()
