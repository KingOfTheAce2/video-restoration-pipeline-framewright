"""Email and SMS notification system for FrameWright.

Supports email notifications via SMTP and SMS via Twilio API.
Integrates with WebhookEvent for event-driven notifications.
"""

import logging
import smtplib
import ssl
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json

from .webhooks import WebhookEvent, WebhookPayload

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of notification channels."""
    EMAIL = "email"
    SMS = "sms"


@dataclass
class EmailConfig:
    """Configuration for email notifications via SMTP."""
    # SMTP server settings
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True
    use_ssl: bool = False

    # Authentication
    username: str = ""
    password: str = ""  # App password for Gmail

    # Sender settings
    from_address: str = ""
    from_name: str = "FrameWright"

    # Recipients
    to_addresses: List[str] = field(default_factory=list)
    cc_addresses: List[str] = field(default_factory=list)
    bcc_addresses: List[str] = field(default_factory=list)

    # Events to notify on
    events: List[WebhookEvent] = field(default_factory=lambda: [
        WebhookEvent.JOB_COMPLETED,
        WebhookEvent.JOB_FAILED,
    ])

    # Behavior
    enabled: bool = True
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 5.0

    # Content options
    include_metrics: bool = True
    include_timestamps: bool = True


@dataclass
class SMSConfig:
    """Configuration for SMS notifications via Twilio."""
    # Twilio credentials
    account_sid: str = ""
    auth_token: str = ""

    # Phone numbers
    from_number: str = ""  # Twilio phone number
    to_numbers: List[str] = field(default_factory=list)

    # Events to notify on
    events: List[WebhookEvent] = field(default_factory=lambda: [
        WebhookEvent.JOB_COMPLETED,
        WebhookEvent.JOB_FAILED,
    ])

    # Behavior
    enabled: bool = True
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 5.0

    # Rate limiting
    rate_limit_per_minute: int = 10

    # Content options
    max_message_length: int = 160  # Standard SMS length


@dataclass
class NotificationConfig:
    """Combined configuration for all notification channels."""
    name: str
    notification_type: NotificationType

    # Channel-specific configs
    email_config: Optional[EmailConfig] = None
    sms_config: Optional[SMSConfig] = None

    # General settings
    enabled: bool = True

    def get_events(self) -> List[WebhookEvent]:
        """Get events that this notification should trigger on."""
        if self.notification_type == NotificationType.EMAIL and self.email_config:
            return self.email_config.events
        elif self.notification_type == NotificationType.SMS and self.sms_config:
            return self.sms_config.events
        return []


class EmailSender:
    """Send email notifications via SMTP."""

    def __init__(self, config: EmailConfig):
        """Initialize email sender with configuration.

        Args:
            config: Email configuration with SMTP settings
        """
        self.config = config
        self._lock = threading.Lock()

    def configure(self, **kwargs) -> None:
        """Update configuration settings.

        Args:
            **kwargs: Configuration options to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def send(
        self,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        to_addresses: Optional[List[str]] = None,
    ) -> bool:
        """Send an email notification.

        Args:
            subject: Email subject line
            body: Plain text body
            html_body: Optional HTML body
            to_addresses: Override recipient list

        Returns:
            True if email was sent successfully
        """
        if not self.config.enabled:
            logger.debug("Email notifications disabled")
            return False

        recipients = to_addresses or self.config.to_addresses
        if not recipients:
            logger.warning("No email recipients configured")
            return False

        for attempt in range(self.config.retry_count):
            try:
                success = self._do_send(subject, body, html_body, recipients)
                if success:
                    return True
            except Exception as e:
                logger.warning(f"Email attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay)

        logger.error(f"Email failed after {self.config.retry_count} attempts")
        return False

    def _do_send(
        self,
        subject: str,
        body: str,
        html_body: Optional[str],
        recipients: List[str],
    ) -> bool:
        """Perform the actual email send."""
        with self._lock:
            # Create message
            if html_body:
                msg = MIMEMultipart("alternative")
                msg.attach(MIMEText(body, "plain"))
                msg.attach(MIMEText(html_body, "html"))
            else:
                msg = MIMEMultipart()
                msg.attach(MIMEText(body, "plain"))

            msg["Subject"] = subject
            msg["From"] = f"{self.config.from_name} <{self.config.from_address}>"
            msg["To"] = ", ".join(recipients)

            if self.config.cc_addresses:
                msg["Cc"] = ", ".join(self.config.cc_addresses)

            # Build full recipient list
            all_recipients = (
                recipients +
                self.config.cc_addresses +
                self.config.bcc_addresses
            )

            # Connect and send
            try:
                if self.config.use_ssl:
                    context = ssl.create_default_context()
                    server = smtplib.SMTP_SSL(
                        self.config.smtp_host,
                        self.config.smtp_port,
                        timeout=self.config.timeout,
                        context=context,
                    )
                else:
                    server = smtplib.SMTP(
                        self.config.smtp_host,
                        self.config.smtp_port,
                        timeout=self.config.timeout,
                    )
                    if self.config.use_tls:
                        context = ssl.create_default_context()
                        server.starttls(context=context)

                if self.config.username and self.config.password:
                    server.login(self.config.username, self.config.password)

                server.sendmail(
                    self.config.from_address,
                    all_recipients,
                    msg.as_string(),
                )
                server.quit()

                logger.info(f"Email sent successfully to {len(all_recipients)} recipients")
                return True

            except smtplib.SMTPException as e:
                logger.error(f"SMTP error: {e}")
                return False

    def send_event(self, payload: WebhookPayload) -> bool:
        """Send email notification for a webhook event.

        Args:
            payload: Event payload to convert to email

        Returns:
            True if email was sent successfully
        """
        subject = self._format_subject(payload)
        body = self._format_body(payload)
        html_body = self._format_html_body(payload)

        return self.send(subject, body, html_body)

    def _format_subject(self, payload: WebhookPayload) -> str:
        """Format email subject from payload."""
        titles = {
            WebhookEvent.JOB_STARTED: "Job Started",
            WebhookEvent.JOB_COMPLETED: "Job Complete",
            WebhookEvent.JOB_FAILED: "Job Failed",
            WebhookEvent.JOB_CANCELLED: "Job Cancelled",
            WebhookEvent.PROGRESS_UPDATE: "Progress Update",
            WebhookEvent.QUALITY_CHECK_PASSED: "Quality Check Passed",
            WebhookEvent.QUALITY_CHECK_FAILED: "Quality Check Failed",
            WebhookEvent.ERROR: "Error Occurred",
            WebhookEvent.WARNING: "Warning",
        }

        title = titles.get(payload.event, payload.event.value)

        if payload.job_name:
            return f"[FrameWright] {title}: {payload.job_name}"
        return f"[FrameWright] {title}"

    def _format_body(self, payload: WebhookPayload) -> str:
        """Format plain text email body from payload."""
        lines = [
            f"FrameWright Video Restoration Notification",
            f"",
            f"Event: {payload.event.value}",
        ]

        if self.config.include_timestamps:
            lines.append(f"Time: {payload.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        if payload.job_name:
            lines.append(f"Job: {payload.job_name}")

        if payload.job_id:
            lines.append(f"Job ID: {payload.job_id}")

        if payload.progress is not None:
            lines.append(f"Progress: {payload.progress:.1f}%")

        if payload.stage:
            lines.append(f"Stage: {payload.stage}")

        if payload.eta_seconds:
            lines.append(f"ETA: {self._format_duration(payload.eta_seconds)}")

        if self.config.include_metrics:
            if payload.quality_score:
                lines.append(f"Quality Score: {payload.quality_score:.2f}")
            if payload.psnr:
                lines.append(f"PSNR: {payload.psnr:.2f} dB")
            if payload.ssim:
                lines.append(f"SSIM: {payload.ssim:.4f}")

        if payload.error_message:
            lines.append(f"")
            lines.append(f"Error: {payload.error_message}")
            if payload.error_type:
                lines.append(f"Error Type: {payload.error_type}")

        if payload.output_path:
            lines.append(f"")
            lines.append(f"Output: {payload.output_path}")
            if payload.output_size_bytes:
                size_mb = payload.output_size_bytes / (1024 * 1024)
                lines.append(f"Size: {size_mb:.2f} MB")
            if payload.duration_seconds:
                lines.append(f"Duration: {self._format_duration(payload.duration_seconds)}")

        lines.append(f"")
        lines.append(f"---")
        lines.append(f"Sent by FrameWright Video Restoration Pipeline")

        return "\n".join(lines)

    def _format_html_body(self, payload: WebhookPayload) -> str:
        """Format HTML email body from payload."""
        # Determine status color
        colors = {
            WebhookEvent.JOB_COMPLETED: "#28a745",  # Green
            WebhookEvent.JOB_FAILED: "#dc3545",  # Red
            WebhookEvent.JOB_CANCELLED: "#6c757d",  # Gray
            WebhookEvent.ERROR: "#dc3545",
            WebhookEvent.WARNING: "#ffc107",  # Yellow
            WebhookEvent.QUALITY_CHECK_PASSED: "#28a745",
            WebhookEvent.QUALITY_CHECK_FAILED: "#dc3545",
        }

        status_color = colors.get(payload.event, "#007bff")  # Blue default

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: {status_color}; color: white; padding: 20px; text-align: center; }}
        .content {{ padding: 20px; background: #f9f9f9; }}
        .field {{ margin-bottom: 10px; }}
        .label {{ font-weight: bold; color: #555; }}
        .value {{ color: #333; }}
        .error {{ background: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; margin: 10px 0; }}
        .footer {{ text-align: center; padding: 20px; color: #888; font-size: 12px; }}
        .metrics {{ display: flex; gap: 20px; margin-top: 10px; }}
        .metric {{ background: white; padding: 10px; border-radius: 5px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FrameWright Notification</h1>
            <p>{payload.event.value.replace('.', ' ').title()}</p>
        </div>
        <div class="content">
"""

        if payload.job_name:
            html += f'<div class="field"><span class="label">Job:</span> <span class="value">{payload.job_name}</span></div>'

        if payload.job_id:
            html += f'<div class="field"><span class="label">Job ID:</span> <span class="value">{payload.job_id}</span></div>'

        if self.config.include_timestamps:
            html += f'<div class="field"><span class="label">Time:</span> <span class="value">{payload.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</span></div>'

        if payload.progress is not None:
            progress_bar = self._html_progress_bar(payload.progress)
            html += f'<div class="field"><span class="label">Progress:</span> {progress_bar} {payload.progress:.1f}%</div>'

        if payload.stage:
            html += f'<div class="field"><span class="label">Stage:</span> <span class="value">{payload.stage}</span></div>'

        if payload.eta_seconds:
            html += f'<div class="field"><span class="label">ETA:</span> <span class="value">{self._format_duration(payload.eta_seconds)}</span></div>'

        if self.config.include_metrics and any([payload.quality_score, payload.psnr, payload.ssim]):
            html += '<div class="metrics">'
            if payload.quality_score:
                html += f'<div class="metric"><div class="label">Quality</div><div class="value">{payload.quality_score:.2f}</div></div>'
            if payload.psnr:
                html += f'<div class="metric"><div class="label">PSNR</div><div class="value">{payload.psnr:.2f} dB</div></div>'
            if payload.ssim:
                html += f'<div class="metric"><div class="label">SSIM</div><div class="value">{payload.ssim:.4f}</div></div>'
            html += '</div>'

        if payload.error_message:
            html += f'<div class="error"><strong>Error:</strong> {payload.error_message}'
            if payload.error_type:
                html += f'<br><small>Type: {payload.error_type}</small>'
            html += '</div>'

        if payload.output_path:
            html += f'<div class="field"><span class="label">Output:</span> <span class="value">{payload.output_path}</span></div>'
            if payload.output_size_bytes:
                size_mb = payload.output_size_bytes / (1024 * 1024)
                html += f'<div class="field"><span class="label">Size:</span> <span class="value">{size_mb:.2f} MB</span></div>'
            if payload.duration_seconds:
                html += f'<div class="field"><span class="label">Processing Time:</span> <span class="value">{self._format_duration(payload.duration_seconds)}</span></div>'

        html += """
        </div>
        <div class="footer">
            <p>Sent by FrameWright Video Restoration Pipeline</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _html_progress_bar(self, progress: float, width: int = 200) -> str:
        """Create HTML progress bar."""
        filled_width = int(progress / 100 * width)
        return f'<span style="display: inline-block; width: {width}px; height: 10px; background: #e0e0e0; border-radius: 5px;"><span style="display: inline-block; width: {filled_width}px; height: 10px; background: #28a745; border-radius: 5px;"></span></span>'

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


class SMSSender:
    """Send SMS notifications via Twilio API.

    Requires twilio package: pip install twilio
    """

    def __init__(self, config: SMSConfig):
        """Initialize SMS sender with configuration.

        Args:
            config: SMS configuration with Twilio settings
        """
        self.config = config
        self._client = None
        self._rate_limits: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def configure(self, **kwargs) -> None:
        """Update configuration settings.

        Args:
            **kwargs: Configuration options to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        # Reset client if credentials changed
        if 'account_sid' in kwargs or 'auth_token' in kwargs:
            self._client = None

    def _get_client(self):
        """Get or create Twilio client (lazy initialization)."""
        if self._client is None:
            try:
                from twilio.rest import Client
                self._client = Client(
                    self.config.account_sid,
                    self.config.auth_token,
                )
            except ImportError:
                raise ImportError(
                    "Twilio package required for SMS notifications. "
                    "Install with: pip install twilio"
                )
        return self._client

    def send(
        self,
        message: str,
        to_numbers: Optional[List[str]] = None,
    ) -> bool:
        """Send an SMS notification.

        Args:
            message: SMS message text (will be truncated if too long)
            to_numbers: Override recipient phone numbers

        Returns:
            True if all SMS were sent successfully
        """
        if not self.config.enabled:
            logger.debug("SMS notifications disabled")
            return False

        recipients = to_numbers or self.config.to_numbers
        if not recipients:
            logger.warning("No SMS recipients configured")
            return False

        if not self.config.account_sid or not self.config.auth_token:
            logger.warning("Twilio credentials not configured")
            return False

        if not self.config.from_number:
            logger.warning("Twilio from number not configured")
            return False

        # Truncate message if needed
        if len(message) > self.config.max_message_length:
            message = message[:self.config.max_message_length - 3] + "..."

        all_success = True
        for number in recipients:
            if not self._check_rate_limit(number):
                logger.debug(f"Rate limited SMS to {number}")
                continue

            success = self._send_single(message, number)
            if success:
                self._record_send(number)
            else:
                all_success = False

        return all_success

    def _send_single(self, message: str, to_number: str) -> bool:
        """Send SMS to a single recipient with retry."""
        for attempt in range(self.config.retry_count):
            try:
                client = self._get_client()

                result = client.messages.create(
                    body=message,
                    from_=self.config.from_number,
                    to=to_number,
                )

                logger.info(f"SMS sent successfully to {to_number}: {result.sid}")
                return True

            except ImportError:
                logger.error("Twilio package not installed")
                return False
            except Exception as e:
                logger.warning(f"SMS attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay)

        logger.error(f"SMS failed after {self.config.retry_count} attempts to {to_number}")
        return False

    def _check_rate_limit(self, number: str) -> bool:
        """Check if we're within rate limit for a number."""
        with self._lock:
            now = time.time()
            window_start = now - 60

            if number not in self._rate_limits:
                self._rate_limits[number] = []

            # Clean old timestamps
            self._rate_limits[number] = [
                t for t in self._rate_limits[number] if t > window_start
            ]

            return len(self._rate_limits[number]) < self.config.rate_limit_per_minute

    def _record_send(self, number: str) -> None:
        """Record successful send for rate limiting."""
        with self._lock:
            now = time.time()
            if number not in self._rate_limits:
                self._rate_limits[number] = []
            self._rate_limits[number].append(now)

    def send_event(self, payload: WebhookPayload) -> bool:
        """Send SMS notification for a webhook event.

        Args:
            payload: Event payload to convert to SMS

        Returns:
            True if SMS was sent successfully
        """
        message = self._format_message(payload)
        return self.send(message)

    def _format_message(self, payload: WebhookPayload) -> str:
        """Format SMS message from payload."""
        # Keep it short for SMS
        titles = {
            WebhookEvent.JOB_STARTED: "Started",
            WebhookEvent.JOB_COMPLETED: "Complete",
            WebhookEvent.JOB_FAILED: "FAILED",
            WebhookEvent.JOB_CANCELLED: "Cancelled",
            WebhookEvent.PROGRESS_UPDATE: "Progress",
            WebhookEvent.QUALITY_CHECK_PASSED: "QC Pass",
            WebhookEvent.QUALITY_CHECK_FAILED: "QC Fail",
            WebhookEvent.ERROR: "ERROR",
            WebhookEvent.WARNING: "Warning",
        }

        title = titles.get(payload.event, payload.event.value)

        parts = [f"[FrameWright] {title}"]

        if payload.job_name:
            parts.append(f": {payload.job_name}")

        if payload.progress is not None:
            parts.append(f" ({payload.progress:.0f}%)")

        if payload.error_message:
            # Truncate error for SMS
            error = payload.error_message[:50]
            if len(payload.error_message) > 50:
                error += "..."
            parts.append(f" - {error}")

        if payload.quality_score:
            parts.append(f" Q:{payload.quality_score:.1f}")

        return "".join(parts)


class NotificationManager:
    """Manage multiple notification channels and send notifications on events."""

    def __init__(self):
        """Initialize notification manager."""
        self.email_senders: Dict[str, EmailSender] = {}
        self.sms_senders: Dict[str, SMSSender] = {}
        self.configs: Dict[str, NotificationConfig] = {}
        self._async_queue: List[tuple] = []
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def add_email(self, name: str, config: EmailConfig) -> None:
        """Add an email notification channel.

        Args:
            name: Unique name for this channel
            config: Email configuration
        """
        self.email_senders[name] = EmailSender(config)
        self.configs[name] = NotificationConfig(
            name=name,
            notification_type=NotificationType.EMAIL,
            email_config=config,
        )
        logger.info(f"Added email notification channel: {name}")

    def add_sms(self, name: str, config: SMSConfig) -> None:
        """Add an SMS notification channel.

        Args:
            name: Unique name for this channel
            config: SMS configuration
        """
        self.sms_senders[name] = SMSSender(config)
        self.configs[name] = NotificationConfig(
            name=name,
            notification_type=NotificationType.SMS,
            sms_config=config,
        )
        logger.info(f"Added SMS notification channel: {name}")

    def remove(self, name: str) -> None:
        """Remove a notification channel.

        Args:
            name: Name of the channel to remove
        """
        if name in self.email_senders:
            del self.email_senders[name]
        if name in self.sms_senders:
            del self.sms_senders[name]
        if name in self.configs:
            del self.configs[name]
        logger.info(f"Removed notification channel: {name}")

    def get(self, name: str) -> Optional[NotificationConfig]:
        """Get notification configuration by name.

        Args:
            name: Channel name

        Returns:
            NotificationConfig if found, None otherwise
        """
        return self.configs.get(name)

    def list_channels(self) -> List[str]:
        """List all notification channel names.

        Returns:
            List of channel names
        """
        return list(self.configs.keys())

    def configure(self, name: str, **kwargs) -> None:
        """Update configuration for a channel.

        Args:
            name: Channel name
            **kwargs: Configuration options to update
        """
        if name in self.email_senders:
            self.email_senders[name].configure(**kwargs)
        elif name in self.sms_senders:
            self.sms_senders[name].configure(**kwargs)

    def send(
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
        async_send: bool = True,
    ) -> None:
        """Send notifications for an event.

        Args:
            event: The event type
            job_id: Optional job identifier
            job_name: Optional job name
            progress: Optional progress percentage
            stage: Optional current stage
            eta_seconds: Optional ETA in seconds
            quality_score: Optional quality score
            psnr: Optional PSNR value
            ssim: Optional SSIM value
            error_message: Optional error message
            error_type: Optional error type
            output_path: Optional output file path
            output_size_bytes: Optional output size
            duration_seconds: Optional processing duration
            metadata: Optional additional metadata
            async_send: Whether to send asynchronously
        """
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
        )

        # Find matching channels
        for name, config in self.configs.items():
            if not config.enabled:
                continue

            events = config.get_events()
            if event not in events:
                continue

            if async_send:
                self._async_queue.append((name, payload))
            else:
                self._send_notification(name, payload)

        if async_send and self._async_queue:
            self._ensure_worker()

    def _send_notification(self, name: str, payload: WebhookPayload) -> bool:
        """Send notification through a specific channel.

        Args:
            name: Channel name
            payload: Event payload

        Returns:
            True if sent successfully
        """
        try:
            if name in self.email_senders:
                return self.email_senders[name].send_event(payload)
            elif name in self.sms_senders:
                return self.sms_senders[name].send_event(payload)
            return False
        except Exception as e:
            logger.error(f"Notification send failed for {name}: {e}")
            return False

    def _ensure_worker(self) -> None:
        """Ensure async worker thread is running."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
            )
            self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Worker thread for async sends."""
        while not self._stop_event.is_set():
            if self._async_queue:
                name, payload = self._async_queue.pop(0)
                try:
                    self._send_notification(name, payload)
                except Exception as e:
                    logger.error(f"Async notification send failed: {e}")
            else:
                time.sleep(0.1)

    def stop(self) -> None:
        """Stop the async worker."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

    def load_from_config(self, config_path: Path) -> None:
        """Load notification configurations from file.

        Args:
            config_path: Path to JSON configuration file
        """
        if not config_path.exists():
            return

        try:
            with open(config_path) as f:
                data = json.load(f)

            # Load email configs
            for email_data in data.get("email", []):
                config = EmailConfig(
                    smtp_host=email_data.get("smtp_host", "smtp.gmail.com"),
                    smtp_port=email_data.get("smtp_port", 587),
                    use_tls=email_data.get("use_tls", True),
                    use_ssl=email_data.get("use_ssl", False),
                    username=email_data.get("username", ""),
                    password=email_data.get("password", ""),
                    from_address=email_data.get("from_address", ""),
                    from_name=email_data.get("from_name", "FrameWright"),
                    to_addresses=email_data.get("to_addresses", []),
                    cc_addresses=email_data.get("cc_addresses", []),
                    bcc_addresses=email_data.get("bcc_addresses", []),
                    events=[WebhookEvent(e) for e in email_data.get("events", ["job.completed", "job.failed"])],
                    enabled=email_data.get("enabled", True),
                    timeout=email_data.get("timeout", 30.0),
                    retry_count=email_data.get("retry_count", 3),
                    include_metrics=email_data.get("include_metrics", True),
                    include_timestamps=email_data.get("include_timestamps", True),
                )
                self.add_email(email_data.get("name", "email"), config)

            # Load SMS configs
            for sms_data in data.get("sms", []):
                config = SMSConfig(
                    account_sid=sms_data.get("account_sid", ""),
                    auth_token=sms_data.get("auth_token", ""),
                    from_number=sms_data.get("from_number", ""),
                    to_numbers=sms_data.get("to_numbers", []),
                    events=[WebhookEvent(e) for e in sms_data.get("events", ["job.completed", "job.failed"])],
                    enabled=sms_data.get("enabled", True),
                    timeout=sms_data.get("timeout", 30.0),
                    retry_count=sms_data.get("retry_count", 3),
                    rate_limit_per_minute=sms_data.get("rate_limit_per_minute", 10),
                    max_message_length=sms_data.get("max_message_length", 160),
                )
                self.add_sms(sms_data.get("name", "sms"), config)

        except Exception as e:
            logger.error(f"Failed to load notification config: {e}")

    def save_to_config(self, config_path: Path) -> None:
        """Save notification configurations to file.

        Args:
            config_path: Path to save JSON configuration
        """
        data = {
            "email": [],
            "sms": [],
        }

        for name, sender in self.email_senders.items():
            config = sender.config
            data["email"].append({
                "name": name,
                "smtp_host": config.smtp_host,
                "smtp_port": config.smtp_port,
                "use_tls": config.use_tls,
                "use_ssl": config.use_ssl,
                "username": config.username,
                # Note: password intentionally not saved for security
                "from_address": config.from_address,
                "from_name": config.from_name,
                "to_addresses": config.to_addresses,
                "cc_addresses": config.cc_addresses,
                "bcc_addresses": config.bcc_addresses,
                "events": [e.value for e in config.events],
                "enabled": config.enabled,
                "timeout": config.timeout,
                "retry_count": config.retry_count,
                "include_metrics": config.include_metrics,
                "include_timestamps": config.include_timestamps,
            })

        for name, sender in self.sms_senders.items():
            config = sender.config
            data["sms"].append({
                "name": name,
                "account_sid": config.account_sid,
                # Note: auth_token intentionally not saved for security
                "from_number": config.from_number,
                "to_numbers": config.to_numbers,
                "events": [e.value for e in config.events],
                "enabled": config.enabled,
                "timeout": config.timeout,
                "retry_count": config.retry_count,
                "rate_limit_per_minute": config.rate_limit_per_minute,
                "max_message_length": config.max_message_length,
            })

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved notification config to {config_path}")


# Factory functions

def create_email_notifier(
    smtp_host: str,
    smtp_port: int,
    username: str,
    password: str,
    from_address: str,
    to_addresses: List[str],
    name: str = "email",
    use_tls: bool = True,
    events: Optional[List[WebhookEvent]] = None,
) -> EmailSender:
    """Create an email notification sender.

    Args:
        smtp_host: SMTP server hostname
        smtp_port: SMTP server port
        username: SMTP authentication username
        password: SMTP authentication password
        from_address: Sender email address
        to_addresses: List of recipient email addresses
        name: Name for this notifier
        use_tls: Whether to use STARTTLS
        events: Events to trigger on (default: job completed/failed)

    Returns:
        Configured EmailSender instance
    """
    config = EmailConfig(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        username=username,
        password=password,
        from_address=from_address,
        to_addresses=to_addresses,
        use_tls=use_tls,
        events=events or [WebhookEvent.JOB_COMPLETED, WebhookEvent.JOB_FAILED],
    )
    return EmailSender(config)


def create_gmail_notifier(
    username: str,
    app_password: str,
    to_addresses: List[str],
    name: str = "gmail",
    events: Optional[List[WebhookEvent]] = None,
) -> EmailSender:
    """Create a Gmail email notification sender.

    Note: Requires Gmail app password (not regular password).
    Generate at: https://myaccount.google.com/apppasswords

    Args:
        username: Gmail address (e.g., user@gmail.com)
        app_password: Gmail app password (16 characters)
        to_addresses: List of recipient email addresses
        name: Name for this notifier
        events: Events to trigger on (default: job completed/failed)

    Returns:
        Configured EmailSender instance for Gmail
    """
    config = EmailConfig(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        use_tls=True,
        username=username,
        password=app_password,
        from_address=username,
        from_name="FrameWright",
        to_addresses=to_addresses,
        events=events or [WebhookEvent.JOB_COMPLETED, WebhookEvent.JOB_FAILED],
    )
    return EmailSender(config)


def create_sms_notifier(
    account_sid: str,
    auth_token: str,
    from_number: str,
    to_numbers: List[str],
    name: str = "sms",
    events: Optional[List[WebhookEvent]] = None,
) -> SMSSender:
    """Create a Twilio SMS notification sender.

    Args:
        account_sid: Twilio account SID
        auth_token: Twilio auth token
        from_number: Twilio phone number (e.g., +15551234567)
        to_numbers: List of recipient phone numbers
        name: Name for this notifier
        events: Events to trigger on (default: job completed/failed)

    Returns:
        Configured SMSSender instance
    """
    config = SMSConfig(
        account_sid=account_sid,
        auth_token=auth_token,
        from_number=from_number,
        to_numbers=to_numbers,
        events=events or [WebhookEvent.JOB_COMPLETED, WebhookEvent.JOB_FAILED],
    )
    return SMSSender(config)


def create_notification_manager(
    email_configs: Optional[List[Dict[str, Any]]] = None,
    sms_configs: Optional[List[Dict[str, Any]]] = None,
) -> NotificationManager:
    """Create a notification manager with multiple channels.

    Args:
        email_configs: List of email configuration dictionaries
        sms_configs: List of SMS configuration dictionaries

    Returns:
        Configured NotificationManager instance

    Example:
        manager = create_notification_manager(
            email_configs=[{
                "name": "admin",
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "admin@example.com",
                "password": "app-password",
                "from_address": "admin@example.com",
                "to_addresses": ["team@example.com"],
            }],
            sms_configs=[{
                "name": "alerts",
                "account_sid": "AC...",
                "auth_token": "...",
                "from_number": "+15551234567",
                "to_numbers": ["+15559876543"],
            }],
        )
    """
    manager = NotificationManager()

    if email_configs:
        for email_data in email_configs:
            name = email_data.pop("name", f"email_{len(manager.email_senders)}")
            events_raw = email_data.pop("events", None)
            events = None
            if events_raw:
                events = [
                    WebhookEvent(e) if isinstance(e, str) else e
                    for e in events_raw
                ]

            config = EmailConfig(**email_data)
            if events:
                config.events = events
            manager.add_email(name, config)

    if sms_configs:
        for sms_data in sms_configs:
            name = sms_data.pop("name", f"sms_{len(manager.sms_senders)}")
            events_raw = sms_data.pop("events", None)
            events = None
            if events_raw:
                events = [
                    WebhookEvent(e) if isinstance(e, str) else e
                    for e in events_raw
                ]

            config = SMSConfig(**sms_data)
            if events:
                config.events = events
            manager.add_sms(name, config)

    return manager
