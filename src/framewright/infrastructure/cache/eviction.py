"""Eviction policies for cache management.

Provides pluggable eviction strategies for frame and model caches,
including LRU, LFU, FIFO, size-aware, and composite policies.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Tuple, TypeVar

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar("T")  # Type for cache keys
V = TypeVar("V")  # Type for cache values


class CacheEntry(Protocol):
    """Protocol for cache entries that can be evicted."""

    @property
    def size_bytes(self) -> int:
        """Size of the entry in bytes."""
        ...

    @property
    def last_accessed(self) -> float:
        """Unix timestamp of last access."""
        ...

    @property
    def access_count(self) -> int:
        """Number of times this entry was accessed."""
        ...

    @property
    def created_at(self) -> float:
        """Unix timestamp when entry was created."""
        ...


@dataclass
class EvictionCandidate:
    """Represents a candidate for eviction.

    Attributes:
        key: Cache key for the entry
        score: Eviction score (lower = more likely to be evicted)
        size_bytes: Size of the entry
        metadata: Additional metadata about the entry
    """
    key: Any
    score: float
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvictionResult(Enum):
    """Result of an eviction decision."""
    EVICT = "evict"          # Entry should be evicted
    KEEP = "keep"            # Entry should be kept
    DEFER = "defer"          # Decision deferred (check again later)


@dataclass
class EvictionStats:
    """Statistics about eviction operations.

    Attributes:
        total_evictions: Total number of items evicted
        bytes_freed: Total bytes freed through eviction
        eviction_time_ms: Total time spent on eviction decisions
        policy_name: Name of the eviction policy
    """
    total_evictions: int = 0
    bytes_freed: int = 0
    eviction_time_ms: float = 0.0
    policy_name: str = ""

    def record_eviction(self, bytes_freed: int, duration_ms: float) -> None:
        """Record an eviction event."""
        self.total_evictions += 1
        self.bytes_freed += bytes_freed
        self.eviction_time_ms += duration_ms


# =============================================================================
# Base Eviction Policy
# =============================================================================

class EvictionPolicy(ABC, Generic[T]):
    """Abstract base class for cache eviction policies.

    Eviction policies determine which items to remove when the cache
    is full or needs to free space. Subclasses implement different
    strategies like LRU, LFU, FIFO, etc.

    Example:
        >>> policy = LRUEviction()
        >>> policy.on_access("key1")
        >>> policy.on_access("key2")
        >>> policy.on_access("key1")  # key1 more recently used
        >>> victim = policy.select_victim(candidates)
        >>> print(victim.key)  # key2 (least recently used)
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize eviction policy.

        Args:
            name: Optional name for the policy (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self._stats = EvictionStats(policy_name=self.name)

    @abstractmethod
    def on_access(self, key: T) -> None:
        """Called when an item is accessed (read or write).

        Args:
            key: The key of the accessed item
        """
        pass

    @abstractmethod
    def on_insert(self, key: T, size_bytes: int = 0) -> None:
        """Called when a new item is inserted.

        Args:
            key: The key of the inserted item
            size_bytes: Size of the item in bytes
        """
        pass

    @abstractmethod
    def on_remove(self, key: T) -> None:
        """Called when an item is removed.

        Args:
            key: The key of the removed item
        """
        pass

    @abstractmethod
    def select_victim(
        self,
        candidates: List[EvictionCandidate]
    ) -> Optional[EvictionCandidate]:
        """Select the best candidate for eviction.

        Args:
            candidates: List of potential eviction candidates

        Returns:
            The candidate to evict, or None if no suitable candidate
        """
        pass

    def select_victims(
        self,
        candidates: List[EvictionCandidate],
        target_bytes: int
    ) -> List[EvictionCandidate]:
        """Select multiple candidates to free target amount of space.

        Args:
            candidates: List of potential eviction candidates
            target_bytes: Target number of bytes to free

        Returns:
            List of candidates to evict
        """
        victims: List[EvictionCandidate] = []
        freed_bytes = 0
        remaining = list(candidates)

        while freed_bytes < target_bytes and remaining:
            victim = self.select_victim(remaining)
            if victim is None:
                break

            victims.append(victim)
            freed_bytes += victim.size_bytes
            remaining.remove(victim)

        return victims

    def compute_score(self, candidate: EvictionCandidate) -> float:
        """Compute eviction score for a candidate.

        Lower scores are more likely to be evicted.

        Args:
            candidate: The candidate to score

        Returns:
            Eviction score (lower = evict first)
        """
        return candidate.score

    def get_stats(self) -> EvictionStats:
        """Get eviction statistics.

        Returns:
            EvictionStats with current metrics
        """
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = EvictionStats(policy_name=self.name)


# =============================================================================
# LRU Eviction (Least Recently Used)
# =============================================================================

class LRUEviction(EvictionPolicy[T]):
    """Least Recently Used eviction policy.

    Evicts items that haven't been accessed recently. Uses an OrderedDict
    to track access order efficiently.

    This is generally a good default policy for most caching scenarios
    as it keeps frequently accessed items in cache.

    Attributes:
        _access_order: OrderedDict tracking key access order
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize LRU policy."""
        super().__init__(name or "LRU")
        self._access_order: OrderedDict[T, float] = OrderedDict()

    def on_access(self, key: T) -> None:
        """Record access and move key to end (most recently used)."""
        self._access_order[key] = time.time()
        self._access_order.move_to_end(key)

    def on_insert(self, key: T, size_bytes: int = 0) -> None:
        """Record insertion as an access."""
        self.on_access(key)

    def on_remove(self, key: T) -> None:
        """Remove key from access tracking."""
        self._access_order.pop(key, None)

    def select_victim(
        self,
        candidates: List[EvictionCandidate]
    ) -> Optional[EvictionCandidate]:
        """Select least recently used candidate."""
        if not candidates:
            return None

        # Score candidates by last access time (older = lower score)
        candidate_keys = {c.key for c in candidates}

        # Find oldest accessed key that's in candidates
        for key in self._access_order:
            if key in candidate_keys:
                for candidate in candidates:
                    if candidate.key == key:
                        return candidate

        # Fallback: return first candidate
        return candidates[0] if candidates else None

    def compute_score(self, candidate: EvictionCandidate) -> float:
        """Score based on last access time (older = lower score)."""
        access_time = self._access_order.get(candidate.key, 0)
        return access_time

    def get_access_order(self) -> List[T]:
        """Get keys in access order (oldest first).

        Returns:
            List of keys from oldest to newest access
        """
        return list(self._access_order.keys())


# =============================================================================
# LFU Eviction (Least Frequently Used)
# =============================================================================

class LFUEviction(EvictionPolicy[T]):
    """Least Frequently Used eviction policy.

    Evicts items that have been accessed the fewest number of times.
    Includes optional decay to prevent old popular items from staying
    forever.

    Attributes:
        _access_counts: Dictionary of access counts per key
        _decay_factor: Optional decay factor applied over time
        _last_decay_time: Timestamp of last decay application
    """

    def __init__(
        self,
        name: Optional[str] = None,
        decay_factor: float = 0.0,
        decay_interval_seconds: float = 3600.0,
    ):
        """Initialize LFU policy.

        Args:
            name: Optional policy name
            decay_factor: Factor to decay counts (0 = no decay)
            decay_interval_seconds: How often to apply decay
        """
        super().__init__(name or "LFU")
        self._access_counts: Dict[T, int] = {}
        self._decay_factor = decay_factor
        self._decay_interval = decay_interval_seconds
        self._last_decay_time = time.time()

    def on_access(self, key: T) -> None:
        """Increment access count for key."""
        self._maybe_apply_decay()
        self._access_counts[key] = self._access_counts.get(key, 0) + 1

    def on_insert(self, key: T, size_bytes: int = 0) -> None:
        """Initialize access count for new key."""
        self._access_counts[key] = 1

    def on_remove(self, key: T) -> None:
        """Remove key from access tracking."""
        self._access_counts.pop(key, None)

    def _maybe_apply_decay(self) -> None:
        """Apply decay to access counts if interval has passed."""
        if self._decay_factor <= 0:
            return

        now = time.time()
        if now - self._last_decay_time >= self._decay_interval:
            decay = 1.0 - self._decay_factor
            self._access_counts = {
                k: max(1, int(v * decay))
                for k, v in self._access_counts.items()
            }
            self._last_decay_time = now

    def select_victim(
        self,
        candidates: List[EvictionCandidate]
    ) -> Optional[EvictionCandidate]:
        """Select least frequently used candidate."""
        if not candidates:
            return None

        # Find candidate with lowest access count
        min_count = float('inf')
        victim = None

        for candidate in candidates:
            count = self._access_counts.get(candidate.key, 0)
            if count < min_count:
                min_count = count
                victim = candidate

        return victim

    def compute_score(self, candidate: EvictionCandidate) -> float:
        """Score based on access frequency (lower count = lower score)."""
        return float(self._access_counts.get(candidate.key, 0))

    def get_access_counts(self) -> Dict[T, int]:
        """Get access counts for all tracked keys.

        Returns:
            Dictionary mapping keys to access counts
        """
        return dict(self._access_counts)


# =============================================================================
# FIFO Eviction (First In First Out)
# =============================================================================

class FIFOEviction(EvictionPolicy[T]):
    """First In First Out eviction policy.

    Evicts oldest inserted items regardless of access pattern.
    Simple and predictable, good for time-based caching scenarios.

    Attributes:
        _insert_order: List tracking insertion order
        _insert_times: Dictionary of insertion timestamps
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize FIFO policy."""
        super().__init__(name or "FIFO")
        self._insert_order: List[T] = []
        self._insert_times: Dict[T, float] = {}

    def on_access(self, key: T) -> None:
        """FIFO ignores access patterns."""
        pass

    def on_insert(self, key: T, size_bytes: int = 0) -> None:
        """Track insertion order."""
        if key not in self._insert_times:
            self._insert_order.append(key)
            self._insert_times[key] = time.time()

    def on_remove(self, key: T) -> None:
        """Remove key from tracking."""
        if key in self._insert_order:
            self._insert_order.remove(key)
        self._insert_times.pop(key, None)

    def select_victim(
        self,
        candidates: List[EvictionCandidate]
    ) -> Optional[EvictionCandidate]:
        """Select oldest inserted candidate."""
        if not candidates:
            return None

        candidate_keys = {c.key for c in candidates}

        # Find oldest key that's in candidates
        for key in self._insert_order:
            if key in candidate_keys:
                for candidate in candidates:
                    if candidate.key == key:
                        return candidate

        return candidates[0] if candidates else None

    def compute_score(self, candidate: EvictionCandidate) -> float:
        """Score based on insertion time (older = lower score)."""
        return self._insert_times.get(candidate.key, 0)

    def get_insert_order(self) -> List[T]:
        """Get keys in insertion order (oldest first).

        Returns:
            List of keys from oldest to newest insertion
        """
        return list(self._insert_order)


# =============================================================================
# Size-Aware Eviction
# =============================================================================

class SizeAwareEviction(EvictionPolicy[T]):
    """Size-aware eviction policy.

    Evicts largest items first to maximize space freed with minimum
    evictions. Can be combined with other policies using CompositeEviction.

    Attributes:
        _sizes: Dictionary tracking sizes of cached items
        _prefer_large: If True, evict large items; if False, evict small
    """

    def __init__(
        self,
        name: Optional[str] = None,
        prefer_large: bool = True,
    ):
        """Initialize size-aware policy.

        Args:
            name: Optional policy name
            prefer_large: If True, evict largest items first
        """
        super().__init__(name or "SizeAware")
        self._sizes: Dict[T, int] = {}
        self._prefer_large = prefer_large

    def on_access(self, key: T) -> None:
        """Size-aware policy ignores access patterns."""
        pass

    def on_insert(self, key: T, size_bytes: int = 0) -> None:
        """Track item size."""
        self._sizes[key] = size_bytes

    def on_remove(self, key: T) -> None:
        """Remove size tracking."""
        self._sizes.pop(key, None)

    def update_size(self, key: T, size_bytes: int) -> None:
        """Update the tracked size for a key.

        Args:
            key: The key to update
            size_bytes: New size in bytes
        """
        self._sizes[key] = size_bytes

    def select_victim(
        self,
        candidates: List[EvictionCandidate]
    ) -> Optional[EvictionCandidate]:
        """Select candidate based on size."""
        if not candidates:
            return None

        # Sort by size
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.size_bytes or self._sizes.get(c.key, 0),
            reverse=self._prefer_large
        )

        return sorted_candidates[0] if sorted_candidates else None

    def compute_score(self, candidate: EvictionCandidate) -> float:
        """Score based on size."""
        size = candidate.size_bytes or self._sizes.get(candidate.key, 0)
        if self._prefer_large:
            return -float(size)  # Negative so larger items have lower score
        return float(size)

    def get_sizes(self) -> Dict[T, int]:
        """Get tracked sizes.

        Returns:
            Dictionary mapping keys to sizes
        """
        return dict(self._sizes)

    def get_total_size(self) -> int:
        """Get total tracked size.

        Returns:
            Total size in bytes
        """
        return sum(self._sizes.values())


# =============================================================================
# Time-Based Eviction (TTL)
# =============================================================================

class TTLEviction(EvictionPolicy[T]):
    """Time-To-Live eviction policy.

    Evicts items that have exceeded their TTL regardless of other factors.
    Can be used standalone or combined with other policies.

    Attributes:
        _insert_times: Dictionary of insertion timestamps
        _ttl_seconds: Default TTL for items
        _custom_ttls: Per-key custom TTLs
    """

    def __init__(
        self,
        ttl_seconds: float,
        name: Optional[str] = None,
    ):
        """Initialize TTL policy.

        Args:
            ttl_seconds: Default time-to-live in seconds
            name: Optional policy name
        """
        super().__init__(name or "TTL")
        self._insert_times: Dict[T, float] = {}
        self._ttl_seconds = ttl_seconds
        self._custom_ttls: Dict[T, float] = {}

    def on_access(self, key: T) -> None:
        """TTL policy ignores access patterns."""
        pass

    def on_insert(self, key: T, size_bytes: int = 0) -> None:
        """Record insertion time."""
        self._insert_times[key] = time.time()

    def on_remove(self, key: T) -> None:
        """Remove TTL tracking."""
        self._insert_times.pop(key, None)
        self._custom_ttls.pop(key, None)

    def set_ttl(self, key: T, ttl_seconds: float) -> None:
        """Set custom TTL for a specific key.

        Args:
            key: The key to set TTL for
            ttl_seconds: TTL in seconds
        """
        self._custom_ttls[key] = ttl_seconds

    def is_expired(self, key: T) -> bool:
        """Check if a key has expired.

        Args:
            key: The key to check

        Returns:
            True if expired, False otherwise
        """
        if key not in self._insert_times:
            return False

        ttl = self._custom_ttls.get(key, self._ttl_seconds)
        age = time.time() - self._insert_times[key]
        return age > ttl

    def get_expired_keys(self) -> List[T]:
        """Get all expired keys.

        Returns:
            List of expired keys
        """
        return [key for key in self._insert_times if self.is_expired(key)]

    def select_victim(
        self,
        candidates: List[EvictionCandidate]
    ) -> Optional[EvictionCandidate]:
        """Select expired or oldest candidate."""
        if not candidates:
            return None

        # First, try to find an expired candidate
        for candidate in candidates:
            if self.is_expired(candidate.key):
                return candidate

        # If no expired, return the oldest
        oldest_time = float('inf')
        victim = None

        for candidate in candidates:
            insert_time = self._insert_times.get(candidate.key, time.time())
            if insert_time < oldest_time:
                oldest_time = insert_time
                victim = candidate

        return victim

    def compute_score(self, candidate: EvictionCandidate) -> float:
        """Score based on remaining TTL (less time = lower score)."""
        insert_time = self._insert_times.get(candidate.key, time.time())
        ttl = self._custom_ttls.get(candidate.key, self._ttl_seconds)
        remaining = ttl - (time.time() - insert_time)
        return remaining


# =============================================================================
# Composite Eviction
# =============================================================================

class CompositeEviction(EvictionPolicy[T]):
    """Composite eviction policy combining multiple strategies.

    Allows combining multiple eviction policies with configurable weights.
    The final eviction score is a weighted sum of individual policy scores.

    Example:
        >>> # 70% LRU, 30% size-aware
        >>> composite = CompositeEviction([
        ...     (LRUEviction(), 0.7),
        ...     (SizeAwareEviction(), 0.3),
        ... ])
        >>> victim = composite.select_victim(candidates)

    Attributes:
        _policies: List of (policy, weight) tuples
        _normalize_scores: Whether to normalize scores before combining
    """

    def __init__(
        self,
        policies: List[Tuple[EvictionPolicy[T], float]],
        name: Optional[str] = None,
        normalize_scores: bool = True,
    ):
        """Initialize composite policy.

        Args:
            policies: List of (policy, weight) tuples
            name: Optional policy name
            normalize_scores: Whether to normalize scores before combining
        """
        super().__init__(name or "Composite")
        self._policies = policies
        self._normalize_scores = normalize_scores

        # Validate weights
        total_weight = sum(w for _, w in policies)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Policy weights sum to {total_weight}, not 1.0")

    def on_access(self, key: T) -> None:
        """Forward to all policies."""
        for policy, _ in self._policies:
            policy.on_access(key)

    def on_insert(self, key: T, size_bytes: int = 0) -> None:
        """Forward to all policies."""
        for policy, _ in self._policies:
            policy.on_insert(key, size_bytes)

    def on_remove(self, key: T) -> None:
        """Forward to all policies."""
        for policy, _ in self._policies:
            policy.on_remove(key)

    def select_victim(
        self,
        candidates: List[EvictionCandidate]
    ) -> Optional[EvictionCandidate]:
        """Select victim using weighted scores from all policies."""
        if not candidates:
            return None

        # Compute weighted scores
        scored_candidates: List[Tuple[EvictionCandidate, float]] = []

        for candidate in candidates:
            total_score = 0.0

            for policy, weight in self._policies:
                score = policy.compute_score(candidate)

                if self._normalize_scores:
                    # Normalize score to 0-1 range for this policy
                    all_scores = [policy.compute_score(c) for c in candidates]
                    min_score = min(all_scores)
                    max_score = max(all_scores)
                    score_range = max_score - min_score
                    if score_range > 0:
                        score = (score - min_score) / score_range

                total_score += score * weight

            scored_candidates.append((candidate, total_score))

        # Return candidate with lowest score
        scored_candidates.sort(key=lambda x: x[1])
        return scored_candidates[0][0] if scored_candidates else None

    def compute_score(self, candidate: EvictionCandidate) -> float:
        """Compute weighted composite score."""
        total_score = 0.0
        for policy, weight in self._policies:
            total_score += policy.compute_score(candidate) * weight
        return total_score

    def add_policy(self, policy: EvictionPolicy[T], weight: float) -> None:
        """Add a policy to the composite.

        Args:
            policy: The policy to add
            weight: Weight for this policy
        """
        self._policies.append((policy, weight))

    def get_policies(self) -> List[Tuple[EvictionPolicy[T], float]]:
        """Get all policies and weights.

        Returns:
            List of (policy, weight) tuples
        """
        return list(self._policies)


# =============================================================================
# Adaptive Eviction
# =============================================================================

class AdaptiveEviction(EvictionPolicy[T]):
    """Adaptive eviction policy that adjusts strategy based on workload.

    Monitors cache access patterns and automatically adjusts the eviction
    strategy. For example, it might favor LRU for random access patterns
    but switch to LFU for skewed access patterns.

    Attributes:
        _lru: LRU policy for recency-based eviction
        _lfu: LFU policy for frequency-based eviction
        _current_strategy: Currently active strategy
        _access_history: Recent access history for pattern detection
    """

    def __init__(
        self,
        name: Optional[str] = None,
        window_size: int = 1000,
        adaptation_interval: int = 100,
    ):
        """Initialize adaptive policy.

        Args:
            name: Optional policy name
            window_size: Size of access history window
            adaptation_interval: How often to check for strategy adaptation
        """
        super().__init__(name or "Adaptive")
        self._lru = LRUEviction()
        self._lfu = LFUEviction()
        self._current_strategy: EvictionPolicy[T] = self._lru

        self._window_size = window_size
        self._adaptation_interval = adaptation_interval
        self._access_history: List[T] = []
        self._access_count = 0

    def on_access(self, key: T) -> None:
        """Track access and forward to strategies."""
        self._lru.on_access(key)
        self._lfu.on_access(key)

        # Track access history
        self._access_history.append(key)
        if len(self._access_history) > self._window_size:
            self._access_history.pop(0)

        self._access_count += 1
        if self._access_count % self._adaptation_interval == 0:
            self._adapt_strategy()

    def on_insert(self, key: T, size_bytes: int = 0) -> None:
        """Forward to all strategies."""
        self._lru.on_insert(key, size_bytes)
        self._lfu.on_insert(key, size_bytes)

    def on_remove(self, key: T) -> None:
        """Forward to all strategies."""
        self._lru.on_remove(key)
        self._lfu.on_remove(key)

    def _adapt_strategy(self) -> None:
        """Adapt strategy based on access patterns."""
        if len(self._access_history) < 100:
            return

        # Calculate access frequency distribution
        access_counts: Dict[T, int] = {}
        for key in self._access_history:
            access_counts[key] = access_counts.get(key, 0) + 1

        # Check if access is skewed (some items accessed much more)
        unique_keys = len(access_counts)
        if unique_keys == 0:
            return

        max_count = max(access_counts.values())
        avg_count = len(self._access_history) / unique_keys

        # If max is much higher than average, access is skewed -> use LFU
        skew_ratio = max_count / avg_count if avg_count > 0 else 1.0

        if skew_ratio > 3.0:
            # Skewed access pattern - favor LFU
            if self._current_strategy != self._lfu:
                logger.debug("Adaptive: Switching to LFU (skewed access)")
                self._current_strategy = self._lfu
        else:
            # More uniform access - favor LRU
            if self._current_strategy != self._lru:
                logger.debug("Adaptive: Switching to LRU (uniform access)")
                self._current_strategy = self._lru

    def select_victim(
        self,
        candidates: List[EvictionCandidate]
    ) -> Optional[EvictionCandidate]:
        """Select victim using current strategy."""
        return self._current_strategy.select_victim(candidates)

    def compute_score(self, candidate: EvictionCandidate) -> float:
        """Compute score using current strategy."""
        return self._current_strategy.compute_score(candidate)

    def get_current_strategy(self) -> str:
        """Get name of current strategy.

        Returns:
            Name of the active eviction strategy
        """
        return self._current_strategy.name


# =============================================================================
# Factory Functions
# =============================================================================

def create_eviction_policy(
    policy_type: str,
    **kwargs: Any
) -> EvictionPolicy[Any]:
    """Create an eviction policy by type name.

    Args:
        policy_type: Type of policy ("lru", "lfu", "fifo", "size", "ttl", "adaptive")
        **kwargs: Additional arguments for the policy

    Returns:
        Configured eviction policy

    Raises:
        ValueError: If policy_type is unknown
    """
    policy_map: Dict[str, type] = {
        "lru": LRUEviction,
        "lfu": LFUEviction,
        "fifo": FIFOEviction,
        "size": SizeAwareEviction,
        "ttl": TTLEviction,
        "adaptive": AdaptiveEviction,
    }

    if policy_type.lower() not in policy_map:
        available = ", ".join(policy_map.keys())
        raise ValueError(f"Unknown policy type: {policy_type}. Available: {available}")

    policy_class = policy_map[policy_type.lower()]

    # Handle TTL requiring ttl_seconds parameter
    if policy_type.lower() == "ttl" and "ttl_seconds" not in kwargs:
        kwargs["ttl_seconds"] = 3600.0  # Default 1 hour

    return policy_class(**kwargs)


def create_composite_policy(
    policies: List[Tuple[str, float]],
    **kwargs: Any
) -> CompositeEviction[Any]:
    """Create a composite eviction policy from policy type names.

    Args:
        policies: List of (policy_type, weight) tuples
        **kwargs: Additional arguments passed to CompositeEviction

    Returns:
        Configured composite eviction policy

    Example:
        >>> policy = create_composite_policy([
        ...     ("lru", 0.6),
        ...     ("size", 0.4),
        ... ])
    """
    policy_instances = [
        (create_eviction_policy(policy_type), weight)
        for policy_type, weight in policies
    ]
    return CompositeEviction(policy_instances, **kwargs)
