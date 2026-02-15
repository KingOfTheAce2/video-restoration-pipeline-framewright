"""Hook system for plugin extensibility."""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class HookPoint(Enum):
    """Available hook points in the pipeline."""
    # Job lifecycle
    JOB_START = auto()
    JOB_COMPLETE = auto()
    JOB_FAILED = auto()
    JOB_PROGRESS = auto()

    # Frame extraction
    PRE_EXTRACT = auto()
    POST_EXTRACT = auto()

    # Frame processing
    PRE_PROCESS_FRAME = auto()
    POST_PROCESS_FRAME = auto()
    PRE_PROCESS_BATCH = auto()
    POST_PROCESS_BATCH = auto()

    # Stage transitions
    PRE_DENOISE = auto()
    POST_DENOISE = auto()
    PRE_UPSCALE = auto()
    POST_UPSCALE = auto()
    PRE_FACE_RESTORE = auto()
    POST_FACE_RESTORE = auto()
    PRE_COLORIZE = auto()
    POST_COLORIZE = auto()
    PRE_INTERPOLATE = auto()
    POST_INTERPOLATE = auto()

    # Quality analysis
    PRE_QUALITY_CHECK = auto()
    POST_QUALITY_CHECK = auto()

    # Encoding
    PRE_ENCODE = auto()
    POST_ENCODE = auto()

    # Scene handling
    SCENE_DETECTED = auto()
    SCENE_TRANSITION = auto()


# Type alias for hook callbacks
HookCallback = Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]


@dataclass
class HookRegistration:
    """Registration information for a hook."""
    callback: HookCallback
    priority: int = 0
    name: str = ""
    plugin_name: Optional[str] = None
    enabled: bool = True


class HookManager:
    """Manages pipeline hooks for plugins."""

    def __init__(self):
        self._hooks: Dict[HookPoint, List[HookRegistration]] = {}
        self._global_hooks: List[Tuple[HookCallback, int, str]] = []

    def register(
        self,
        hook_point: HookPoint,
        callback: HookCallback,
        priority: int = 0,
        name: str = "",
        plugin_name: Optional[str] = None,
    ) -> str:
        """Register a hook callback.

        Args:
            hook_point: Where to call the hook
            callback: Function to call
            priority: Higher = called first
            name: Optional name for the registration
            plugin_name: Plugin that owns this hook

        Returns:
            Registration ID
        """
        if hook_point not in self._hooks:
            self._hooks[hook_point] = []

        reg = HookRegistration(
            callback=callback,
            priority=priority,
            name=name or f"hook_{id(callback)}",
            plugin_name=plugin_name,
        )

        self._hooks[hook_point].append(reg)
        # Sort by priority (descending)
        self._hooks[hook_point].sort(key=lambda r: -r.priority)

        logger.debug(f"Registered hook: {name} at {hook_point.name}")
        return reg.name

    def register_global(
        self,
        callback: HookCallback,
        priority: int = 0,
        name: str = "",
    ) -> None:
        """Register a global hook called for all hook points."""
        self._global_hooks.append((callback, priority, name))
        self._global_hooks.sort(key=lambda x: -x[1])

    def unregister(self, hook_point: HookPoint, name: str) -> bool:
        """Unregister a hook by name."""
        if hook_point not in self._hooks:
            return False

        original_len = len(self._hooks[hook_point])
        self._hooks[hook_point] = [
            r for r in self._hooks[hook_point] if r.name != name
        ]

        removed = len(self._hooks[hook_point]) < original_len
        if removed:
            logger.debug(f"Unregistered hook: {name} from {hook_point.name}")
        return removed

    def unregister_plugin(self, plugin_name: str) -> int:
        """Unregister all hooks from a plugin."""
        count = 0
        for hook_point in self._hooks:
            original_len = len(self._hooks[hook_point])
            self._hooks[hook_point] = [
                r for r in self._hooks[hook_point]
                if r.plugin_name != plugin_name
            ]
            count += original_len - len(self._hooks[hook_point])

        if count:
            logger.debug(f"Unregistered {count} hooks from plugin: {plugin_name}")
        return count

    def enable(self, hook_point: HookPoint, name: str) -> bool:
        """Enable a hook."""
        for reg in self._hooks.get(hook_point, []):
            if reg.name == name:
                reg.enabled = True
                return True
        return False

    def disable(self, hook_point: HookPoint, name: str) -> bool:
        """Disable a hook."""
        for reg in self._hooks.get(hook_point, []):
            if reg.name == name:
                reg.enabled = False
                return True
        return False

    def trigger(
        self,
        hook_point: HookPoint,
        context: Dict[str, Any],
        allow_modification: bool = True,
    ) -> Dict[str, Any]:
        """Trigger a hook point.

        Args:
            hook_point: The hook point to trigger
            context: Data passed to hooks
            allow_modification: If True, hooks can modify context

        Returns:
            Potentially modified context
        """
        result = dict(context)

        # Call global hooks first
        for callback, _, name in self._global_hooks:
            try:
                ret = callback({"hook_point": hook_point.name, **result})
                if allow_modification and ret is not None:
                    result.update(ret)
            except Exception as e:
                logger.error(f"Global hook error ({name}): {e}")

        # Call specific hooks
        for reg in self._hooks.get(hook_point, []):
            if not reg.enabled:
                continue

            try:
                ret = reg.callback(result)
                if allow_modification and ret is not None:
                    result.update(ret)
            except Exception as e:
                logger.error(f"Hook error ({reg.name} at {hook_point.name}): {e}")

        return result

    def trigger_frame_hook(
        self,
        hook_point: HookPoint,
        frame: np.ndarray,
        frame_number: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Trigger a frame processing hook.

        Args:
            hook_point: The hook point
            frame: Input frame
            frame_number: Frame index
            metadata: Additional metadata

        Returns:
            (potentially modified frame, metadata)
        """
        context = {
            "frame": frame,
            "frame_number": frame_number,
            "metadata": metadata or {},
        }

        result = self.trigger(hook_point, context)

        return result.get("frame", frame), result.get("metadata", {})

    def list_hooks(
        self,
        hook_point: Optional[HookPoint] = None,
    ) -> List[Dict[str, Any]]:
        """List registered hooks."""
        result = []

        if hook_point:
            points = [hook_point]
        else:
            points = list(self._hooks.keys())

        for point in points:
            for reg in self._hooks.get(point, []):
                result.append({
                    "hook_point": point.name,
                    "name": reg.name,
                    "priority": reg.priority,
                    "plugin": reg.plugin_name,
                    "enabled": reg.enabled,
                })

        return result

    def clear(self) -> None:
        """Clear all hooks."""
        self._hooks.clear()
        self._global_hooks.clear()


# Decorator for creating hooks
def hook(
    hook_point: HookPoint,
    priority: int = 0,
    name: Optional[str] = None,
):
    """Decorator for marking methods as hooks.

    Usage:
        class MyPlugin(ProcessorPlugin):
            @hook(HookPoint.PRE_PROCESS_FRAME)
            def on_pre_process(self, context):
                # Modify context
                return context
    """
    def decorator(func: Callable) -> Callable:
        func._hook_point = hook_point
        func._hook_priority = priority
        func._hook_name = name or func.__name__
        return func
    return decorator


class HookContext:
    """Context manager for hook execution."""

    def __init__(
        self,
        manager: HookManager,
        pre_hook: HookPoint,
        post_hook: HookPoint,
        context: Dict[str, Any],
    ):
        self.manager = manager
        self.pre_hook = pre_hook
        self.post_hook = post_hook
        self.context = context
        self.result = None

    def __enter__(self):
        self.context = self.manager.trigger(self.pre_hook, self.context)
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.context = self.manager.trigger(self.post_hook, self.context)
        self.result = self.context
        return False


# Convenience functions for common hooks
def create_logging_hook(
    logger_name: str = "framewright.hooks",
    level: int = logging.DEBUG,
) -> HookCallback:
    """Create a hook that logs all events."""
    hook_logger = logging.getLogger(logger_name)

    def log_hook(context: Dict[str, Any]) -> None:
        hook_point = context.get("hook_point", "unknown")
        frame_num = context.get("frame_number", "N/A")
        hook_logger.log(level, f"Hook: {hook_point} | Frame: {frame_num}")
        return None

    return log_hook


def create_timing_hook() -> Tuple[HookCallback, Callable[[], Dict[str, float]]]:
    """Create hooks for timing pipeline stages."""
    import time
    timings: Dict[str, float] = {}
    start_times: Dict[str, float] = {}

    def timing_hook(context: Dict[str, Any]) -> None:
        hook_point = context.get("hook_point", "")

        if hook_point.startswith("PRE_"):
            stage = hook_point[4:]
            start_times[stage] = time.time()
        elif hook_point.startswith("POST_"):
            stage = hook_point[5:]
            if stage in start_times:
                timings[stage] = time.time() - start_times[stage]
                del start_times[stage]

        return None

    def get_timings() -> Dict[str, float]:
        return dict(timings)

    return timing_hook, get_timings


def create_quality_tracking_hook() -> Tuple[HookCallback, Callable[[], List[Dict]]]:
    """Create hooks for tracking quality metrics."""
    metrics: List[Dict[str, Any]] = []

    def quality_hook(context: Dict[str, Any]) -> None:
        if "psnr" in context or "ssim" in context:
            metrics.append({
                "frame": context.get("frame_number"),
                "psnr": context.get("psnr"),
                "ssim": context.get("ssim"),
                "stage": context.get("hook_point"),
            })
        return None

    def get_metrics() -> List[Dict]:
        return list(metrics)

    return quality_hook, get_metrics
