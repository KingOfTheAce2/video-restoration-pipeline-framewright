"""
Progress Webhook - Send processing updates to external services.

Supports Discord, Slack, and generic HTTP webhooks for monitoring
long-running restoration jobs.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from urllib.parse import urlparse
import threading
import queue

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class WebhookType(Enum):
    """Supported webhook types."""
    DISCORD = "discord"
    SLACK = "slack"
    GENERIC = "generic"
    NTFY = "ntfy"  # ntfy.sh notification service


class EventType(Enum):
    """Types of events that can be sent."""
    JOB_STARTED = "job_started"
    JOB_PROGRESS = "job_progress"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_PAUSED = "job_paused"
    JOB_RESUMED = "job_resumed"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class WebhookConfig:
    """Configuration for a webhook endpoint."""
    url: str
    webhook_type: WebhookType = WebhookType.GENERIC
    enabled: bool = True
    events: List[EventType] = field(default_factory=lambda: list(EventType))
    rate_limit_seconds: float = 5.0  # Min time between messages
    include_thumbnails: bool = False
    custom_headers: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebhookConfig":
        """Create from dictionary."""
        return cls(
            url=data["url"],
            webhook_type=WebhookType(data.get("type", "generic")),
            enabled=data.get("enabled", True),
            events=[EventType(e) for e in data.get("events", [e.value for e in EventType])],
            rate_limit_seconds=data.get("rate_limit", 5.0),
            include_thumbnails=data.get("thumbnails", False),
            custom_headers=data.get("headers", {})
        )


@dataclass
class WebhookEvent:
    """An event to be sent via webhook."""
    event_type: EventType
    job_id: str
    job_name: str
    message: str
    progress: Optional[float] = None  # 0-100
    details: Dict[str, Any] = field(default_factory=dict)
    thumbnail_path: Optional[Path] = None
    timestamp: float = field(default_factory=time.time)


class ProgressWebhook:
    """
    Send processing updates to external services.

    Supports multiple webhook endpoints with rate limiting
    and async delivery.
    """

    def __init__(
        self,
        configs: Optional[List[WebhookConfig]] = None,
        async_delivery: bool = True
    ):
        """
        Initialize webhook manager.

        Args:
            configs: List of webhook configurations
            async_delivery: Send webhooks in background thread
        """
        if not HAS_REQUESTS:
            raise ImportError("requests package required for webhooks: pip install requests")

        self.configs = configs or []
        self.async_delivery = async_delivery
        self._last_sent: Dict[str, float] = {}  # url -> timestamp
        self._queue: Optional[queue.Queue] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if async_delivery:
            self._start_worker()

    def _start_worker(self) -> None:
        """Start background worker thread."""
        self._queue = queue.Queue()
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Background worker for async delivery."""
        while not self._stop_event.is_set():
            try:
                event, config = self._queue.get(timeout=1.0)
                self._send_webhook(event, config)
                self._queue.task_done()
            except queue.Empty:
                continue

    def stop(self) -> None:
        """Stop the background worker."""
        if self._worker_thread:
            self._stop_event.set()
            self._worker_thread.join(timeout=5.0)

    def add_config(self, config: WebhookConfig) -> None:
        """Add a webhook configuration."""
        self.configs.append(config)

    def add_discord(self, webhook_url: str, **kwargs) -> None:
        """Add a Discord webhook."""
        self.add_config(WebhookConfig(
            url=webhook_url,
            webhook_type=WebhookType.DISCORD,
            **kwargs
        ))

    def add_slack(self, webhook_url: str, **kwargs) -> None:
        """Add a Slack webhook."""
        self.add_config(WebhookConfig(
            url=webhook_url,
            webhook_type=WebhookType.SLACK,
            **kwargs
        ))

    def add_ntfy(self, topic: str, server: str = "https://ntfy.sh", **kwargs) -> None:
        """Add an ntfy.sh webhook."""
        self.add_config(WebhookConfig(
            url=f"{server}/{topic}",
            webhook_type=WebhookType.NTFY,
            **kwargs
        ))

    def _should_send(self, config: WebhookConfig, event: WebhookEvent) -> bool:
        """Check if event should be sent to this webhook."""
        if not config.enabled:
            return False

        if event.event_type not in config.events:
            return False

        # Rate limiting
        last_time = self._last_sent.get(config.url, 0)
        if time.time() - last_time < config.rate_limit_seconds:
            # Allow important events through
            if event.event_type not in [
                EventType.JOB_COMPLETED,
                EventType.JOB_FAILED,
                EventType.ERROR
            ]:
                return False

        return True

    def _format_discord(self, event: WebhookEvent) -> Dict[str, Any]:
        """Format event for Discord webhook."""
        # Color based on event type
        colors = {
            EventType.JOB_STARTED: 0x3498db,  # Blue
            EventType.JOB_COMPLETED: 0x2ecc71,  # Green
            EventType.JOB_FAILED: 0xe74c3c,  # Red
            EventType.WARNING: 0xf1c40f,  # Yellow
            EventType.ERROR: 0xe74c3c,  # Red
            EventType.JOB_PROGRESS: 0x9b59b6,  # Purple
        }

        color = colors.get(event.event_type, 0x95a5a6)  # Gray default

        # Build embed
        embed = {
            "title": f"🎬 {event.job_name}",
            "description": event.message,
            "color": color,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(event.timestamp)),
            "footer": {"text": f"Job ID: {event.job_id}"}
        }

        # Add progress bar if available
        if event.progress is not None:
            filled = int(event.progress / 5)
            bar = "█" * filled + "░" * (20 - filled)
            embed["fields"] = [
                {"name": "Progress", "value": f"`{bar}` {event.progress:.1f}%", "inline": False}
            ]

        # Add details
        if event.details:
            if "fields" not in embed:
                embed["fields"] = []
            for key, value in event.details.items():
                embed["fields"].append({
                    "name": key.replace("_", " ").title(),
                    "value": str(value),
                    "inline": True
                })

        return {"embeds": [embed]}

    def _format_slack(self, event: WebhookEvent) -> Dict[str, Any]:
        """Format event for Slack webhook."""
        # Emoji based on event type
        emojis = {
            EventType.JOB_STARTED: "🚀",
            EventType.JOB_COMPLETED: "✅",
            EventType.JOB_FAILED: "❌",
            EventType.WARNING: "⚠️",
            EventType.ERROR: "🔴",
            EventType.JOB_PROGRESS: "⏳",
        }

        emoji = emojis.get(event.event_type, "📋")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {event.job_name}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": event.message
                }
            }
        ]

        # Add progress if available
        if event.progress is not None:
            filled = int(event.progress / 5)
            bar = "█" * filled + "░" * (20 - filled)
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Progress:* `{bar}` {event.progress:.1f}%"
                }
            })

        # Add details
        if event.details:
            fields = []
            for key, value in event.details.items():
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*{key.replace('_', ' ').title()}:*\n{value}"
                })
            blocks.append({
                "type": "section",
                "fields": fields[:10]  # Slack limit
            })

        blocks.append({
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"Job ID: `{event.job_id}`"}
            ]
        })

        return {"blocks": blocks}

    def _format_ntfy(self, event: WebhookEvent) -> tuple:
        """Format event for ntfy.sh. Returns (headers, body)."""
        priority_map = {
            EventType.JOB_FAILED: "5",  # Max priority
            EventType.ERROR: "5",
            EventType.WARNING: "4",
            EventType.JOB_COMPLETED: "3",
            EventType.JOB_STARTED: "2",
            EventType.JOB_PROGRESS: "1",  # Min priority
        }

        headers = {
            "Title": f"FrameWright: {event.job_name}",
            "Priority": priority_map.get(event.event_type, "3"),
            "Tags": event.event_type.value.replace("_", ",")
        }

        body = event.message
        if event.progress is not None:
            body += f"\n\nProgress: {event.progress:.1f}%"

        return headers, body

    def _format_generic(self, event: WebhookEvent) -> Dict[str, Any]:
        """Format event for generic HTTP webhook."""
        return {
            "event": event.event_type.value,
            "job_id": event.job_id,
            "job_name": event.job_name,
            "message": event.message,
            "progress": event.progress,
            "details": event.details,
            "timestamp": event.timestamp
        }

    def _send_webhook(self, event: WebhookEvent, config: WebhookConfig) -> bool:
        """Send event to a single webhook endpoint."""
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(config.custom_headers)

            if config.webhook_type == WebhookType.DISCORD:
                payload = self._format_discord(event)
                response = requests.post(config.url, json=payload, headers=headers, timeout=10)

            elif config.webhook_type == WebhookType.SLACK:
                payload = self._format_slack(event)
                response = requests.post(config.url, json=payload, headers=headers, timeout=10)

            elif config.webhook_type == WebhookType.NTFY:
                ntfy_headers, body = self._format_ntfy(event)
                headers.update(ntfy_headers)
                headers["Content-Type"] = "text/plain"
                response = requests.post(config.url, data=body, headers=headers, timeout=10)

            else:  # Generic
                payload = self._format_generic(event)
                response = requests.post(config.url, json=payload, headers=headers, timeout=10)

            self._last_sent[config.url] = time.time()
            return response.status_code < 400

        except Exception as e:
            # Log error but don't fail the main process
            print(f"Webhook error ({config.url}): {e}")
            return False

    def send(self, event: WebhookEvent) -> int:
        """
        Send event to all configured webhooks.

        Args:
            event: Event to send

        Returns:
            Number of webhooks notified
        """
        sent_count = 0

        for config in self.configs:
            if not self._should_send(config, event):
                continue

            if self.async_delivery and self._queue:
                self._queue.put((event, config))
                sent_count += 1
            else:
                if self._send_webhook(event, config):
                    sent_count += 1

        return sent_count

    def notify_started(
        self,
        job_id: str,
        job_name: str,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """Send job started notification."""
        return self.send(WebhookEvent(
            event_type=EventType.JOB_STARTED,
            job_id=job_id,
            job_name=job_name,
            message=f"Processing started for: {job_name}",
            details=details or {}
        ))

    def notify_progress(
        self,
        job_id: str,
        job_name: str,
        progress: float,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """Send progress update."""
        return self.send(WebhookEvent(
            event_type=EventType.JOB_PROGRESS,
            job_id=job_id,
            job_name=job_name,
            message=message or f"Processing: {progress:.1f}% complete",
            progress=progress,
            details=details or {}
        ))

    def notify_completed(
        self,
        job_id: str,
        job_name: str,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """Send job completed notification."""
        return self.send(WebhookEvent(
            event_type=EventType.JOB_COMPLETED,
            job_id=job_id,
            job_name=job_name,
            message=f"Processing completed successfully: {job_name}",
            progress=100.0,
            details=details or {}
        ))

    def notify_failed(
        self,
        job_id: str,
        job_name: str,
        error: str,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """Send job failed notification."""
        return self.send(WebhookEvent(
            event_type=EventType.JOB_FAILED,
            job_id=job_id,
            job_name=job_name,
            message=f"Processing failed: {error}",
            details=details or {}
        ))

    def notify_warning(
        self,
        job_id: str,
        job_name: str,
        warning: str,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """Send warning notification."""
        return self.send(WebhookEvent(
            event_type=EventType.WARNING,
            job_id=job_id,
            job_name=job_name,
            message=f"Warning: {warning}",
            details=details or {}
        ))


class WebhookManager:
    """
    High-level manager for webhook notifications in processing pipeline.

    Provides context managers and decorators for easy integration.
    """

    def __init__(self, webhook: Optional[ProgressWebhook] = None):
        """Initialize with optional webhook instance."""
        self.webhook = webhook
        self._current_job_id: Optional[str] = None
        self._current_job_name: Optional[str] = None

    def configure_from_file(self, config_path: Path) -> None:
        """Load webhook configuration from JSON file."""
        with open(config_path) as f:
            data = json.load(f)

        configs = [WebhookConfig.from_dict(c) for c in data.get("webhooks", [])]
        self.webhook = ProgressWebhook(configs)

    def processing_job(self, job_id: str, job_name: str):
        """Context manager for a processing job."""
        return _JobContext(self, job_id, job_name)

    def update_progress(self, progress: float, message: Optional[str] = None) -> None:
        """Update progress for current job."""
        if self.webhook and self._current_job_id:
            self.webhook.notify_progress(
                self._current_job_id,
                self._current_job_name or "Unknown",
                progress,
                message
            )


class _JobContext:
    """Context manager for job notifications."""

    def __init__(self, manager: WebhookManager, job_id: str, job_name: str):
        self.manager = manager
        self.job_id = job_id
        self.job_name = job_name

    def __enter__(self):
        self.manager._current_job_id = self.job_id
        self.manager._current_job_name = self.job_name
        if self.manager.webhook:
            self.manager.webhook.notify_started(self.job_id, self.job_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.manager.webhook:
            if exc_type:
                self.manager.webhook.notify_failed(
                    self.job_id,
                    self.job_name,
                    str(exc_val)
                )
            else:
                self.manager.webhook.notify_completed(self.job_id, self.job_name)

        self.manager._current_job_id = None
        self.manager._current_job_name = None
        return False  # Don't suppress exceptions
