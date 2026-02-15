"""Webhook integration for external notifications and automation.

Supports Discord, Slack, HTTP callbacks, and custom integrations.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
import ssl

logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Types of events that can trigger webhooks."""
    # Job lifecycle
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_CANCELLED = "job.cancelled"

    # Progress
    PROGRESS_UPDATE = "progress.update"
    STAGE_STARTED = "stage.started"
    STAGE_COMPLETED = "stage.completed"

    # Quality
    QUALITY_CHECK_PASSED = "quality.passed"
    QUALITY_CHECK_FAILED = "quality.failed"
    QUALITY_REPORT_READY = "quality.report_ready"

    # Frames
    FRAME_RESTORED = "frame.restored"
    BATCH_COMPLETED = "batch.completed"

    # System
    ERROR = "error"
    WARNING = "warning"
    SYSTEM_STATUS = "system.status"


class WebhookFormat(Enum):
    """Webhook payload formats."""
    JSON = "json"
    DISCORD = "discord"
    SLACK = "slack"
    TEAMS = "teams"
    CUSTOM = "custom"


@dataclass
class WebhookConfig:
    """Configuration for a webhook endpoint."""
    name: str
    url: str
    format: WebhookFormat = WebhookFormat.JSON

    # Events to trigger on
    events: List[WebhookEvent] = field(default_factory=lambda: [WebhookEvent.JOB_COMPLETED])

    # Security
    secret: Optional[str] = None  # For HMAC signature
    headers: Dict[str, str] = field(default_factory=dict)

    # Behavior
    enabled: bool = True
    retry_count: int = 3
    retry_delay: float = 5.0  # seconds
    timeout: float = 30.0  # seconds

    # Filtering
    min_progress_interval: float = 10.0  # Min seconds between progress updates
    include_thumbnails: bool = False
    include_metrics: bool = True

    # Rate limiting
    rate_limit_per_minute: int = 60


@dataclass
class WebhookPayload:
    """Webhook payload data."""
    event: WebhookEvent
    timestamp: datetime
    job_id: Optional[str] = None
    job_name: Optional[str] = None

    # Progress info
    progress: Optional[float] = None
    stage: Optional[str] = None
    eta_seconds: Optional[float] = None

    # Quality info
    quality_score: Optional[float] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None

    # Error info
    error_message: Optional[str] = None
    error_type: Optional[str] = None

    # Output info
    output_path: Optional[str] = None
    output_size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None

    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    thumbnail_base64: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event": self.event.value,
            "timestamp": self.timestamp.isoformat(),
            "job_id": self.job_id,
            "job_name": self.job_name,
            "progress": self.progress,
            "stage": self.stage,
            "eta_seconds": self.eta_seconds,
            "quality": {
                "score": self.quality_score,
                "psnr": self.psnr,
                "ssim": self.ssim,
            } if any([self.quality_score, self.psnr, self.ssim]) else None,
            "error": {
                "message": self.error_message,
                "type": self.error_type,
            } if self.error_message else None,
            "output": {
                "path": self.output_path,
                "size_bytes": self.output_size_bytes,
                "duration_seconds": self.duration_seconds,
            } if self.output_path else None,
            "metadata": self.metadata if self.metadata else None,
            "thumbnail": self.thumbnail_base64,
        }


class WebhookFormatter:
    """Format payloads for different webhook services."""

    def format(self, payload: WebhookPayload, format: WebhookFormat) -> Dict[str, Any]:
        """Format payload for the specified service."""
        if format == WebhookFormat.JSON:
            return self._format_json(payload)
        elif format == WebhookFormat.DISCORD:
            return self._format_discord(payload)
        elif format == WebhookFormat.SLACK:
            return self._format_slack(payload)
        elif format == WebhookFormat.TEAMS:
            return self._format_teams(payload)
        else:
            return self._format_json(payload)

    def _format_json(self, payload: WebhookPayload) -> Dict[str, Any]:
        """Standard JSON format."""
        data = payload.to_dict()
        # Remove None values
        return {k: v for k, v in data.items() if v is not None}

    def _format_discord(self, payload: WebhookPayload) -> Dict[str, Any]:
        """Discord webhook format with embeds."""
        # Determine color based on event
        colors = {
            WebhookEvent.JOB_STARTED: 0x3498DB,  # Blue
            WebhookEvent.JOB_COMPLETED: 0x2ECC71,  # Green
            WebhookEvent.JOB_FAILED: 0xE74C3C,  # Red
            WebhookEvent.JOB_CANCELLED: 0x95A5A6,  # Gray
            WebhookEvent.QUALITY_CHECK_PASSED: 0x2ECC71,
            WebhookEvent.QUALITY_CHECK_FAILED: 0xE74C3C,
            WebhookEvent.ERROR: 0xE74C3C,
            WebhookEvent.WARNING: 0xF1C40F,  # Yellow
            WebhookEvent.PROGRESS_UPDATE: 0x9B59B6,  # Purple
        }

        color = colors.get(payload.event, 0x3498DB)

        # Build embed
        embed = {
            "title": self._get_title(payload),
            "color": color,
            "timestamp": payload.timestamp.isoformat(),
            "fields": [],
        }

        # Add fields
        if payload.job_name:
            embed["fields"].append({
                "name": "Job",
                "value": payload.job_name,
                "inline": True,
            })

        if payload.progress is not None:
            progress_bar = self._create_progress_bar(payload.progress)
            embed["fields"].append({
                "name": "Progress",
                "value": f"{progress_bar} {payload.progress:.1f}%",
                "inline": True,
            })

        if payload.stage:
            embed["fields"].append({
                "name": "Stage",
                "value": payload.stage,
                "inline": True,
            })

        if payload.eta_seconds:
            eta_str = self._format_duration(payload.eta_seconds)
            embed["fields"].append({
                "name": "ETA",
                "value": eta_str,
                "inline": True,
            })

        if payload.quality_score:
            embed["fields"].append({
                "name": "Quality Score",
                "value": f"{payload.quality_score:.2f}",
                "inline": True,
            })

        if payload.psnr:
            embed["fields"].append({
                "name": "PSNR",
                "value": f"{payload.psnr:.2f} dB",
                "inline": True,
            })

        if payload.ssim:
            embed["fields"].append({
                "name": "SSIM",
                "value": f"{payload.ssim:.4f}",
                "inline": True,
            })

        if payload.error_message:
            embed["fields"].append({
                "name": "Error",
                "value": payload.error_message[:1024],
                "inline": False,
            })

        if payload.output_path:
            embed["fields"].append({
                "name": "Output",
                "value": f"`{payload.output_path}`",
                "inline": False,
            })

        if payload.thumbnail_base64:
            embed["thumbnail"] = {
                "url": f"data:image/png;base64,{payload.thumbnail_base64}"
            }

        embed["footer"] = {
            "text": "FrameWright Video Restoration"
        }

        return {
            "embeds": [embed],
            "username": "FrameWright",
        }

    def _format_slack(self, payload: WebhookPayload) -> Dict[str, Any]:
        """Slack webhook format with blocks."""
        # Determine emoji based on event
        emojis = {
            WebhookEvent.JOB_STARTED: ":arrow_forward:",
            WebhookEvent.JOB_COMPLETED: ":white_check_mark:",
            WebhookEvent.JOB_FAILED: ":x:",
            WebhookEvent.JOB_CANCELLED: ":stop_button:",
            WebhookEvent.QUALITY_CHECK_PASSED: ":+1:",
            WebhookEvent.QUALITY_CHECK_FAILED: ":-1:",
            WebhookEvent.ERROR: ":warning:",
            WebhookEvent.WARNING: ":large_yellow_circle:",
            WebhookEvent.PROGRESS_UPDATE: ":hourglass_flowing_sand:",
        }

        emoji = emojis.get(payload.event, ":film_frames:")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {self._get_title(payload)}",
                    "emoji": True,
                }
            }
        ]

        # Add fields section
        fields = []

        if payload.job_name:
            fields.append({
                "type": "mrkdwn",
                "text": f"*Job:*\n{payload.job_name}"
            })

        if payload.progress is not None:
            progress_bar = self._create_progress_bar(payload.progress)
            fields.append({
                "type": "mrkdwn",
                "text": f"*Progress:*\n{progress_bar} {payload.progress:.1f}%"
            })

        if payload.stage:
            fields.append({
                "type": "mrkdwn",
                "text": f"*Stage:*\n{payload.stage}"
            })

        if payload.eta_seconds:
            fields.append({
                "type": "mrkdwn",
                "text": f"*ETA:*\n{self._format_duration(payload.eta_seconds)}"
            })

        if payload.quality_score:
            fields.append({
                "type": "mrkdwn",
                "text": f"*Quality:*\n{payload.quality_score:.2f}"
            })

        if fields:
            blocks.append({
                "type": "section",
                "fields": fields[:10]  # Slack limit
            })

        if payload.error_message:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":rotating_light: *Error:*\n```{payload.error_message[:2900]}```"
                }
            })

        if payload.output_path:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":file_folder: *Output:*\n`{payload.output_path}`"
                }
            })

        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"FrameWright | {payload.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                }
            ]
        })

        return {"blocks": blocks}

    def _format_teams(self, payload: WebhookPayload) -> Dict[str, Any]:
        """Microsoft Teams webhook format (Adaptive Cards)."""
        # Card color based on event
        colors = {
            WebhookEvent.JOB_COMPLETED: "Good",
            WebhookEvent.JOB_FAILED: "Attention",
            WebhookEvent.ERROR: "Attention",
            WebhookEvent.WARNING: "Warning",
        }

        theme_color = colors.get(payload.event, "Default")

        facts = []

        if payload.job_name:
            facts.append({"title": "Job", "value": payload.job_name})

        if payload.progress is not None:
            facts.append({"title": "Progress", "value": f"{payload.progress:.1f}%"})

        if payload.stage:
            facts.append({"title": "Stage", "value": payload.stage})

        if payload.quality_score:
            facts.append({"title": "Quality Score", "value": f"{payload.quality_score:.2f}"})

        if payload.error_message:
            facts.append({"title": "Error", "value": payload.error_message[:500]})

        if payload.output_path:
            facts.append({"title": "Output", "value": payload.output_path})

        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": theme_color,
            "summary": self._get_title(payload),
            "sections": [{
                "activityTitle": self._get_title(payload),
                "activitySubtitle": payload.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "facts": facts,
                "markdown": True,
            }],
        }

    def _get_title(self, payload: WebhookPayload) -> str:
        """Get human-readable title for event."""
        titles = {
            WebhookEvent.JOB_STARTED: "Restoration Job Started",
            WebhookEvent.JOB_COMPLETED: "Restoration Complete",
            WebhookEvent.JOB_FAILED: "Restoration Failed",
            WebhookEvent.JOB_CANCELLED: "Restoration Cancelled",
            WebhookEvent.PROGRESS_UPDATE: "Progress Update",
            WebhookEvent.STAGE_STARTED: "Stage Started",
            WebhookEvent.STAGE_COMPLETED: "Stage Completed",
            WebhookEvent.QUALITY_CHECK_PASSED: "Quality Check Passed",
            WebhookEvent.QUALITY_CHECK_FAILED: "Quality Check Failed",
            WebhookEvent.QUALITY_REPORT_READY: "Quality Report Ready",
            WebhookEvent.ERROR: "Error Occurred",
            WebhookEvent.WARNING: "Warning",
            WebhookEvent.SYSTEM_STATUS: "System Status",
        }
        return titles.get(payload.event, payload.event.value)

    def _create_progress_bar(self, progress: float, width: int = 10) -> str:
        """Create text-based progress bar."""
        filled = int(progress / 100 * width)
        empty = width - filled
        return "▓" * filled + "░" * empty

    def _format_duration(self, seconds: float) -> str:
        """Format seconds as human-readable duration."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


class WebhookSender:
    """Send webhook requests with retry and rate limiting."""

    def __init__(self):
        self.formatter = WebhookFormatter()
        self._rate_limits: Dict[str, List[float]] = {}  # endpoint -> timestamps
        self._last_progress: Dict[str, float] = {}  # endpoint -> last progress time

    def send(self, config: WebhookConfig, payload: WebhookPayload) -> bool:
        """Send webhook request."""
        if not config.enabled:
            return False

        # Check rate limit
        if not self._check_rate_limit(config):
            logger.debug(f"Rate limited: {config.name}")
            return False

        # Check progress interval
        if payload.event == WebhookEvent.PROGRESS_UPDATE:
            if not self._check_progress_interval(config):
                return False

        # Format payload
        data = self.formatter.format(payload, config.format)

        # Send with retry
        for attempt in range(config.retry_count):
            try:
                success = self._do_send(config, data)
                if success:
                    self._record_send(config, payload)
                    return True
            except Exception as e:
                logger.warning(f"Webhook attempt {attempt + 1} failed: {e}")
                if attempt < config.retry_count - 1:
                    time.sleep(config.retry_delay)

        logger.error(f"Webhook failed after {config.retry_count} attempts: {config.name}")
        return False

    def _do_send(self, config: WebhookConfig, data: Dict[str, Any]) -> bool:
        """Perform the actual HTTP request."""
        json_data = json.dumps(data).encode("utf-8")

        # Build request
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "FrameWright/2.0",
        }
        headers.update(config.headers)

        # Add signature if secret configured
        if config.secret:
            signature = hmac.new(
                config.secret.encode(),
                json_data,
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"

        req = Request(config.url, data=json_data, headers=headers, method="POST")

        # Create SSL context that handles most cases
        ctx = ssl.create_default_context()

        try:
            with urlopen(req, timeout=config.timeout, context=ctx) as response:
                status = response.status
                if 200 <= status < 300:
                    logger.debug(f"Webhook sent successfully: {config.name}")
                    return True
                else:
                    logger.warning(f"Webhook returned {status}: {config.name}")
                    return False
        except HTTPError as e:
            logger.warning(f"Webhook HTTP error {e.code}: {config.name}")
            return False
        except URLError as e:
            logger.warning(f"Webhook URL error: {e.reason}")
            return False

    def _check_rate_limit(self, config: WebhookConfig) -> bool:
        """Check if we're within rate limit."""
        now = time.time()
        window_start = now - 60

        if config.url not in self._rate_limits:
            self._rate_limits[config.url] = []

        # Clean old timestamps
        self._rate_limits[config.url] = [
            t for t in self._rate_limits[config.url] if t > window_start
        ]

        return len(self._rate_limits[config.url]) < config.rate_limit_per_minute

    def _check_progress_interval(self, config: WebhookConfig) -> bool:
        """Check if enough time has passed since last progress update."""
        now = time.time()
        last = self._last_progress.get(config.url, 0)
        return now - last >= config.min_progress_interval

    def _record_send(self, config: WebhookConfig, payload: WebhookPayload) -> None:
        """Record successful send for rate limiting."""
        now = time.time()

        if config.url not in self._rate_limits:
            self._rate_limits[config.url] = []

        self._rate_limits[config.url].append(now)

        if payload.event == WebhookEvent.PROGRESS_UPDATE:
            self._last_progress[config.url] = now


class WebhookManager:
    """Manage multiple webhook endpoints."""

    def __init__(self):
        self.configs: Dict[str, WebhookConfig] = {}
        self.sender = WebhookSender()
        self._async_queue: List[tuple] = []
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def add_webhook(self, config: WebhookConfig) -> None:
        """Add a webhook configuration."""
        self.configs[config.name] = config
        logger.info(f"Added webhook: {config.name} ({config.format.value})")

    def remove_webhook(self, name: str) -> None:
        """Remove a webhook configuration."""
        if name in self.configs:
            del self.configs[name]
            logger.info(f"Removed webhook: {name}")

    def get_webhook(self, name: str) -> Optional[WebhookConfig]:
        """Get webhook configuration by name."""
        return self.configs.get(name)

    def list_webhooks(self) -> List[str]:
        """List all webhook names."""
        return list(self.configs.keys())

    def trigger(
        self,
        event: WebhookEvent,
        job_id: Optional[str] = None,
        job_name: Optional[str] = None,
        progress: Optional[float] = None,
        stage: Optional[str] = None,
        eta_seconds: Optional[float] = None,
        quality_score: Optional[float] = None,
        psnr: Optional[float] = None,
        ssim: Optional[float] = None,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        output_path: Optional[str] = None,
        output_size_bytes: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        thumbnail_base64: Optional[str] = None,
        async_send: bool = True,
    ) -> None:
        """Trigger webhooks for an event."""
        payload = WebhookPayload(
            event=event,
            timestamp=datetime.now(),
            job_id=job_id,
            job_name=job_name,
            progress=progress,
            stage=stage,
            eta_seconds=eta_seconds,
            quality_score=quality_score,
            psnr=psnr,
            ssim=ssim,
            error_message=error_message,
            error_type=error_type,
            output_path=output_path,
            output_size_bytes=output_size_bytes,
            duration_seconds=duration_seconds,
            metadata=metadata or {},
            thumbnail_base64=thumbnail_base64,
        )

        # Find matching webhooks
        matching = [
            config for config in self.configs.values()
            if config.enabled and event in config.events
        ]

        if not matching:
            return

        if async_send:
            for config in matching:
                self._async_queue.append((config, payload))
            self._ensure_worker()
        else:
            for config in matching:
                self.sender.send(config, payload)

    def _ensure_worker(self) -> None:
        """Ensure async worker thread is running."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Worker thread for async sends."""
        while not self._stop_event.is_set():
            if self._async_queue:
                config, payload = self._async_queue.pop(0)
                try:
                    self.sender.send(config, payload)
                except Exception as e:
                    logger.error(f"Async webhook send failed: {e}")
            else:
                time.sleep(0.1)

    def stop(self) -> None:
        """Stop the async worker."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

    def load_from_config(self, config_path: Path) -> None:
        """Load webhook configurations from file."""
        if not config_path.exists():
            return

        try:
            with open(config_path) as f:
                data = json.load(f)

            for webhook_data in data.get("webhooks", []):
                config = WebhookConfig(
                    name=webhook_data["name"],
                    url=webhook_data["url"],
                    format=WebhookFormat(webhook_data.get("format", "json")),
                    events=[WebhookEvent(e) for e in webhook_data.get("events", ["job.completed"])],
                    secret=webhook_data.get("secret"),
                    headers=webhook_data.get("headers", {}),
                    enabled=webhook_data.get("enabled", True),
                    retry_count=webhook_data.get("retry_count", 3),
                    timeout=webhook_data.get("timeout", 30.0),
                    min_progress_interval=webhook_data.get("min_progress_interval", 10.0),
                    include_thumbnails=webhook_data.get("include_thumbnails", False),
                    rate_limit_per_minute=webhook_data.get("rate_limit_per_minute", 60),
                )
                self.add_webhook(config)

        except Exception as e:
            logger.error(f"Failed to load webhook config: {e}")

    def save_to_config(self, config_path: Path) -> None:
        """Save webhook configurations to file."""
        data = {
            "webhooks": [
                {
                    "name": config.name,
                    "url": config.url,
                    "format": config.format.value,
                    "events": [e.value for e in config.events],
                    "secret": config.secret,
                    "headers": config.headers,
                    "enabled": config.enabled,
                    "retry_count": config.retry_count,
                    "timeout": config.timeout,
                    "min_progress_interval": config.min_progress_interval,
                    "include_thumbnails": config.include_thumbnails,
                    "rate_limit_per_minute": config.rate_limit_per_minute,
                }
                for config in self.configs.values()
            ]
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved webhook config to {config_path}")


def create_discord_webhook(url: str, name: str = "Discord") -> WebhookConfig:
    """Create a Discord webhook configuration."""
    return WebhookConfig(
        name=name,
        url=url,
        format=WebhookFormat.DISCORD,
        events=[
            WebhookEvent.JOB_STARTED,
            WebhookEvent.JOB_COMPLETED,
            WebhookEvent.JOB_FAILED,
            WebhookEvent.ERROR,
        ],
    )


def create_slack_webhook(url: str, name: str = "Slack") -> WebhookConfig:
    """Create a Slack webhook configuration."""
    return WebhookConfig(
        name=name,
        url=url,
        format=WebhookFormat.SLACK,
        events=[
            WebhookEvent.JOB_STARTED,
            WebhookEvent.JOB_COMPLETED,
            WebhookEvent.JOB_FAILED,
            WebhookEvent.ERROR,
        ],
    )


def create_progress_webhook(url: str, name: str = "Progress") -> WebhookConfig:
    """Create a webhook for progress updates."""
    return WebhookConfig(
        name=name,
        url=url,
        format=WebhookFormat.JSON,
        events=[
            WebhookEvent.PROGRESS_UPDATE,
            WebhookEvent.STAGE_STARTED,
            WebhookEvent.STAGE_COMPLETED,
        ],
        min_progress_interval=30.0,  # Only send every 30 seconds
    )
