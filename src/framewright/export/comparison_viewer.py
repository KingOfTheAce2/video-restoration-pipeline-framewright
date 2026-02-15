"""Before/After Comparison Viewer Generator.

Generates interactive HTML comparison viewers with slider functionality
for comparing original and restored video frames.

Features:
- Interactive slider comparison
- Side-by-side view
- Zoom functionality
- Multiple frame comparison
- Quality metrics display
- Shareable HTML output

Example:
    >>> viewer = ComparisonViewer()
    >>> viewer.add_frame(original_frame, restored_frame, "Scene 1")
    >>> viewer.generate("comparison.html")
"""

import base64
import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class ComparisonFrame:
    """A single comparison frame pair."""
    original: bytes  # PNG bytes
    restored: bytes  # PNG bytes
    label: str = ""
    timestamp: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


# HTML template with embedded CSS and JavaScript for slider comparison
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FrameWright - Before/After Comparison</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            text-align: center;
        }

        .header h1 {
            font-size: 24px;
            font-weight: 600;
        }

        .header p {
            opacity: 0.9;
            margin-top: 5px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .frame-nav {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }

        .frame-btn {
            background: #2d2d44;
            border: none;
            color: #fff;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .frame-btn:hover {
            background: #3d3d54;
        }

        .frame-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .comparison-container {
            position: relative;
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            overflow: hidden;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }

        .comparison-wrapper {
            position: relative;
            width: 100%;
            cursor: ew-resize;
        }

        .comparison-image {
            display: block;
            width: 100%;
            height: auto;
        }

        .comparison-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 50%;
            height: 100%;
            overflow: hidden;
        }

        .comparison-overlay img {
            display: block;
            width: 200%;
            max-width: none;
            height: auto;
        }

        .comparison-slider {
            position: absolute;
            top: 0;
            left: 50%;
            width: 4px;
            height: 100%;
            background: #fff;
            cursor: ew-resize;
            transform: translateX(-50%);
            z-index: 10;
        }

        .comparison-slider::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 40px;
            height: 40px;
            background: #fff;
            border-radius: 50%;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .comparison-slider::after {
            content: '↔';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 20px;
            color: #333;
        }

        .labels {
            display: flex;
            justify-content: space-between;
            padding: 15px 20px;
            background: #2d2d44;
            border-radius: 0 0 12px 12px;
        }

        .label {
            font-weight: 500;
            opacity: 0.9;
        }

        .label.original { color: #ff6b6b; }
        .label.restored { color: #51cf66; }

        .metrics-panel {
            margin-top: 20px;
            padding: 20px;
            background: #2d2d44;
            border-radius: 12px;
        }

        .metrics-panel h3 {
            margin-bottom: 15px;
            font-size: 16px;
            opacity: 0.8;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }

        .metric {
            background: #1a1a2e;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }

        .metric-value {
            font-size: 24px;
            font-weight: 600;
            color: #667eea;
        }

        .metric-label {
            font-size: 12px;
            opacity: 0.7;
            margin-top: 5px;
        }

        .view-toggle {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 20px;
        }

        .toggle-btn {
            background: #2d2d44;
            border: none;
            color: #fff;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }

        .toggle-btn.active {
            background: #667eea;
        }

        .side-by-side {
            display: none;
            gap: 20px;
        }

        .side-by-side.visible {
            display: grid;
            grid-template-columns: 1fr 1fr;
        }

        .side-by-side img {
            width: 100%;
            border-radius: 8px;
        }

        .side-by-side .image-label {
            text-align: center;
            padding: 10px;
            font-weight: 500;
        }

        .footer {
            text-align: center;
            padding: 30px;
            opacity: 0.6;
            font-size: 14px;
        }

        .footer a {
            color: #667eea;
            text-decoration: none;
        }

        @media (max-width: 768px) {
            .side-by-side.visible {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 FrameWright Comparison</h1>
        <p>{{VIDEO_NAME}} - {{FRAME_COUNT}} frame(s)</p>
    </div>

    <div class="container">
        <div class="view-toggle">
            <button class="toggle-btn active" onclick="setView('slider')">Slider View</button>
            <button class="toggle-btn" onclick="setView('sidebyside')">Side by Side</button>
        </div>

        <div class="frame-nav" id="frameNav">
            {{FRAME_BUTTONS}}
        </div>

        <div id="sliderView">
            <div class="comparison-container">
                <div class="comparison-wrapper" id="comparisonWrapper">
                    <img src="{{FIRST_RESTORED}}" alt="Restored" class="comparison-image" id="restoredImage">
                    <div class="comparison-overlay" id="comparisonOverlay">
                        <img src="{{FIRST_ORIGINAL}}" alt="Original" id="originalImage">
                    </div>
                    <div class="comparison-slider" id="comparisonSlider"></div>
                </div>
                <div class="labels">
                    <span class="label original">◀ Original</span>
                    <span class="label restored">Restored ▶</span>
                </div>
            </div>
        </div>

        <div class="side-by-side" id="sideBySideView">
            <div>
                <img src="{{FIRST_ORIGINAL}}" alt="Original" id="sideOriginal">
                <div class="image-label original">Original</div>
            </div>
            <div>
                <img src="{{FIRST_RESTORED}}" alt="Restored" id="sideRestored">
                <div class="image-label restored">Restored</div>
            </div>
        </div>

        <div class="metrics-panel" id="metricsPanel">
            <h3>Quality Metrics</h3>
            <div class="metrics-grid" id="metricsGrid">
                {{METRICS_HTML}}
            </div>
        </div>
    </div>

    <div class="footer">
        Generated by <a href="https://github.com/your-repo/framewright">FrameWright</a>
    </div>

    <script>
        const frames = {{FRAMES_JSON}};
        let currentFrame = 0;
        let currentView = 'slider';

        // Slider functionality
        const wrapper = document.getElementById('comparisonWrapper');
        const overlay = document.getElementById('comparisonOverlay');
        const slider = document.getElementById('comparisonSlider');
        let isDragging = false;

        function updateSlider(x) {
            const rect = wrapper.getBoundingClientRect();
            let position = (x - rect.left) / rect.width;
            position = Math.max(0, Math.min(1, position));

            overlay.style.width = (position * 100) + '%';
            slider.style.left = (position * 100) + '%';
        }

        wrapper.addEventListener('mousedown', (e) => {
            isDragging = true;
            updateSlider(e.clientX);
        });

        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                updateSlider(e.clientX);
            }
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
        });

        // Touch support
        wrapper.addEventListener('touchstart', (e) => {
            isDragging = true;
            updateSlider(e.touches[0].clientX);
        });

        document.addEventListener('touchmove', (e) => {
            if (isDragging) {
                updateSlider(e.touches[0].clientX);
            }
        });

        document.addEventListener('touchend', () => {
            isDragging = false;
        });

        // Frame navigation
        function selectFrame(index) {
            currentFrame = index;
            const frame = frames[index];

            document.getElementById('originalImage').src = frame.original;
            document.getElementById('restoredImage').src = frame.restored;
            document.getElementById('sideOriginal').src = frame.original;
            document.getElementById('sideRestored').src = frame.restored;

            // Update active button
            document.querySelectorAll('.frame-btn').forEach((btn, i) => {
                btn.classList.toggle('active', i === index);
            });

            // Update metrics
            updateMetrics(frame.metrics);
        }

        function updateMetrics(metrics) {
            const grid = document.getElementById('metricsGrid');
            grid.innerHTML = '';

            for (const [key, value] of Object.entries(metrics)) {
                const div = document.createElement('div');
                div.className = 'metric';
                div.innerHTML = `
                    <div class="metric-value">${typeof value === 'number' ? value.toFixed(2) : value}</div>
                    <div class="metric-label">${key}</div>
                `;
                grid.appendChild(div);
            }
        }

        function setView(view) {
            currentView = view;
            document.querySelectorAll('.toggle-btn').forEach(btn => {
                btn.classList.toggle('active', btn.textContent.toLowerCase().includes(view === 'slider' ? 'slider' : 'side'));
            });

            document.getElementById('sliderView').style.display = view === 'slider' ? 'block' : 'none';
            document.getElementById('sideBySideView').classList.toggle('visible', view === 'sidebyside');
        }

        // Initialize
        if (frames.length > 0) {
            selectFrame(0);
            updateMetrics(frames[0].metrics || {});
        }
    </script>
</body>
</html>
"""


class ComparisonViewer:
    """Generates interactive HTML comparison viewers."""

    def __init__(self, title: str = "FrameWright Comparison"):
        """Initialize viewer.

        Args:
            title: Title for the comparison page
        """
        self.title = title
        self.frames: List[ComparisonFrame] = []
        self.video_name = "Video"

    def set_video_name(self, name: str) -> None:
        """Set video name for display.

        Args:
            name: Video name
        """
        self.video_name = name

    def add_frame(
        self,
        original: "np.ndarray",
        restored: "np.ndarray",
        label: str = "",
        timestamp: float = 0.0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add a comparison frame pair.

        Args:
            original: Original frame (BGR numpy array)
            restored: Restored frame (BGR numpy array)
            label: Frame label
            timestamp: Timestamp in seconds
            metrics: Quality metrics
        """
        if not HAS_OPENCV:
            logger.error("OpenCV required for adding frames")
            return

        # Convert to PNG bytes
        original_png = self._frame_to_png(original)
        restored_png = self._frame_to_png(restored)

        self.frames.append(ComparisonFrame(
            original=original_png,
            restored=restored_png,
            label=label or f"Frame {len(self.frames) + 1}",
            timestamp=timestamp,
            metrics=metrics or {},
        ))

    def add_frame_from_paths(
        self,
        original_path: Path,
        restored_path: Path,
        label: str = "",
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add comparison from file paths.

        Args:
            original_path: Path to original frame
            restored_path: Path to restored frame
            label: Frame label
            metrics: Quality metrics
        """
        if not HAS_OPENCV:
            logger.error("OpenCV required")
            return

        original = cv2.imread(str(original_path))
        restored = cv2.imread(str(restored_path))

        if original is None or restored is None:
            logger.error("Failed to load images")
            return

        self.add_frame(original, restored, label, metrics=metrics)

    def _frame_to_png(self, frame: "np.ndarray") -> bytes:
        """Convert frame to PNG bytes."""
        _, buffer = cv2.imencode('.png', frame)
        return buffer.tobytes()

    def _frame_to_data_uri(self, png_bytes: bytes) -> str:
        """Convert PNG bytes to data URI."""
        b64 = base64.b64encode(png_bytes).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    def generate(
        self,
        output_path: Path,
        include_metrics: bool = True,
    ) -> bool:
        """Generate HTML comparison viewer.

        Args:
            output_path: Output HTML file path
            include_metrics: Include quality metrics panel

        Returns:
            True if generated successfully
        """
        if not self.frames:
            logger.error("No frames to compare")
            return False

        try:
            # Build frame buttons HTML
            frame_buttons = []
            for i, frame in enumerate(self.frames):
                active = "active" if i == 0 else ""
                frame_buttons.append(
                    f'<button class="frame-btn {active}" onclick="selectFrame({i})">'
                    f'{frame.label}</button>'
                )

            # Build metrics HTML for first frame
            metrics_html = ""
            if include_metrics and self.frames[0].metrics:
                for key, value in self.frames[0].metrics.items():
                    formatted = f"{value:.2f}" if isinstance(value, float) else str(value)
                    metrics_html += f'''
                        <div class="metric">
                            <div class="metric-value">{formatted}</div>
                            <div class="metric-label">{key}</div>
                        </div>
                    '''

            # Build frames JSON
            frames_json = []
            for frame in self.frames:
                frames_json.append({
                    "original": self._frame_to_data_uri(frame.original),
                    "restored": self._frame_to_data_uri(frame.restored),
                    "label": frame.label,
                    "timestamp": frame.timestamp,
                    "metrics": frame.metrics,
                })

            # Generate HTML
            import json
            html = HTML_TEMPLATE
            html = html.replace("{{VIDEO_NAME}}", self.video_name)
            html = html.replace("{{FRAME_COUNT}}", str(len(self.frames)))
            html = html.replace("{{FRAME_BUTTONS}}", "\n".join(frame_buttons))
            html = html.replace("{{FIRST_ORIGINAL}}", self._frame_to_data_uri(self.frames[0].original))
            html = html.replace("{{FIRST_RESTORED}}", self._frame_to_data_uri(self.frames[0].restored))
            html = html.replace("{{METRICS_HTML}}", metrics_html)
            html = html.replace("{{FRAMES_JSON}}", json.dumps(frames_json))

            # Write file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)

            logger.info(f"Generated comparison viewer: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate viewer: {e}")
            return False


def create_comparison_from_video(
    original_video: Path,
    restored_video: Path,
    output_html: Path,
    num_samples: int = 5,
    include_metrics: bool = True,
) -> bool:
    """Create comparison viewer from two video files.

    Args:
        original_video: Path to original video
        restored_video: Path to restored video
        output_html: Output HTML path
        num_samples: Number of frames to sample
        include_metrics: Include quality metrics

    Returns:
        True if successful
    """
    if not HAS_OPENCV:
        logger.error("OpenCV required")
        return False

    viewer = ComparisonViewer()
    viewer.set_video_name(original_video.stem)

    # Open videos
    cap_orig = cv2.VideoCapture(str(original_video))
    cap_rest = cv2.VideoCapture(str(restored_video))

    if not cap_orig.isOpened() or not cap_rest.isOpened():
        logger.error("Failed to open videos")
        return False

    total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_orig.get(cv2.CAP_PROP_FPS)

    # Sample frames
    import numpy as np
    sample_indices = np.linspace(
        total_frames * 0.1,
        total_frames * 0.9,
        num_samples,
        dtype=int
    )

    for i, frame_idx in enumerate(sample_indices):
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap_rest.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret_orig, orig_frame = cap_orig.read()
        ret_rest, rest_frame = cap_rest.read()

        if not ret_orig or not ret_rest:
            continue

        # Calculate metrics if requested
        metrics = {}
        if include_metrics:
            # Resize for comparison if needed
            if orig_frame.shape != rest_frame.shape:
                rest_frame_resized = cv2.resize(
                    rest_frame,
                    (orig_frame.shape[1], orig_frame.shape[0])
                )
            else:
                rest_frame_resized = rest_frame

            # PSNR
            mse = np.mean((orig_frame.astype(float) - rest_frame_resized.astype(float)) ** 2)
            if mse > 0:
                metrics["PSNR"] = 20 * np.log10(255.0 / np.sqrt(mse))

            # Sharpness (Laplacian variance)
            gray = cv2.cvtColor(rest_frame, cv2.COLOR_BGR2GRAY)
            metrics["Sharpness"] = cv2.Laplacian(gray, cv2.CV_64F).var()

        timestamp = frame_idx / fps
        label = f"Scene {i + 1} ({timestamp:.1f}s)"

        viewer.add_frame(orig_frame, rest_frame, label, timestamp, metrics)

    cap_orig.release()
    cap_rest.release()

    return viewer.generate(output_html, include_metrics)
