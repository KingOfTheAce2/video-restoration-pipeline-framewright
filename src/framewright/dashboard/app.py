"""Flask application for the web dashboard."""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for the dashboard."""
    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = False
    secret_key: str = ""
    enable_api: bool = True
    enable_websocket: bool = True
    static_folder: Optional[Path] = None
    template_folder: Optional[Path] = None

    # Authentication
    require_auth: bool = False
    username: str = "admin"
    password_hash: str = ""

    # Job sources
    job_store_path: Optional[Path] = None

    # Refresh rates
    auto_refresh_seconds: int = 5
    websocket_ping_seconds: int = 30


def create_app(
    config: Optional[DashboardConfig] = None,
    job_store: Optional[Any] = None,
    progress_tracker: Optional[Any] = None,
) -> Any:
    """Create Flask application."""
    try:
        from flask import Flask, render_template_string, jsonify, request, Response
    except ImportError:
        logger.error("Flask is required for dashboard. Install with: pip install flask")
        return None

    config = config or DashboardConfig()

    # Create Flask app
    app = Flask(
        __name__,
        static_folder=str(config.static_folder) if config.static_folder else None,
        template_folder=str(config.template_folder) if config.template_folder else None,
    )

    app.config["SECRET_KEY"] = config.secret_key or os.urandom(24).hex()

    # Store references
    app.job_store = job_store
    app.progress_tracker = progress_tracker
    app.dashboard_config = config

    # Register routes
    @app.route("/")
    def index():
        """Main dashboard page."""
        return render_template_string(DASHBOARD_TEMPLATE, config=config)

    @app.route("/api/jobs")
    def api_jobs():
        """List all jobs."""
        if not app.job_store:
            return jsonify({"jobs": [], "error": "No job store configured"})

        try:
            jobs = app.job_store.list_jobs()
            return jsonify({
                "jobs": [_job_to_dict(j) for j in jobs],
                "count": len(jobs),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/jobs/<job_id>")
    def api_job_detail(job_id: str):
        """Get job details."""
        if not app.job_store:
            return jsonify({"error": "No job store configured"}), 500

        try:
            job = app.job_store.get_job(job_id)
            if not job:
                return jsonify({"error": "Job not found"}), 404

            result = _job_to_dict(job)

            # Add frame details
            frames = app.job_store.get_frames(job_id)
            result["frames"] = [_frame_to_dict(f) for f in frames[:100]]  # Limit to 100
            result["frame_count"] = len(frames)

            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/jobs/<job_id>/progress")
    def api_job_progress(job_id: str):
        """Get job progress."""
        if not app.progress_tracker:
            return jsonify({"error": "No progress tracker configured"}), 500

        try:
            snapshot = app.progress_tracker.get_snapshot(job_id)
            if not snapshot:
                return jsonify({"error": "No progress data"}), 404

            return jsonify(snapshot.to_dict())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/system")
    def api_system():
        """Get system status."""
        status = _get_system_status()
        return jsonify(status)

    @app.route("/api/models")
    def api_models():
        """List all available models."""
        try:
            from framewright.infrastructure.models import get_model_manager, get_default_registry

            manager = get_model_manager()
            registry = get_default_registry()

            models = []
            for model_info in registry.list_models():
                status = manager.get_model_status(model_info.model_id)
                models.append({
                    "id": model_info.model_id,
                    "name": model_info.name,
                    "category": model_info.category.value,
                    "size_mb": model_info.size_mb,
                    "backend": model_info.backend.value,
                    "description": model_info.description or "",
                    "installed": status.installed,
                    "download_progress": status.download_progress,
                    "path": str(status.path) if status.path else None,
                })

            return jsonify({"models": models, "count": len(models)})
        except Exception as e:
            logger.exception("Failed to list models")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/models/<model_id>/download", methods=["POST"])
    def api_model_download(model_id: str):
        """Download a model."""
        try:
            from framewright.infrastructure.models import get_model_manager

            manager = get_model_manager()
            path = manager.get_model(model_id, download=True)

            return jsonify({
                "success": True,
                "model_id": model_id,
                "path": str(path),
            })
        except Exception as e:
            logger.exception(f"Failed to download model {model_id}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/models/<model_id>", methods=["DELETE"])
    def api_model_delete(model_id: str):
        """Delete a model."""
        try:
            from framewright.infrastructure.models import get_model_manager

            manager = get_model_manager()
            success = manager.delete_model(model_id)

            return jsonify({
                "success": success,
                "model_id": model_id,
            })
        except Exception as e:
            logger.exception(f"Failed to delete model {model_id}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/events")
    def api_events():
        """Server-sent events for real-time updates."""
        def generate():
            while True:
                import time
                time.sleep(config.auto_refresh_seconds)

                data = {"timestamp": datetime.now().isoformat()}

                # Add job summaries
                if app.job_store:
                    try:
                        jobs = app.job_store.list_jobs()
                        data["jobs"] = [
                            {
                                "job_id": j.job_id,
                                "state": j.state.value,
                                "progress": (j.frames_processed / j.total_frames * 100)
                                if j.total_frames > 0 else 0,
                            }
                            for j in jobs
                        ]
                    except:
                        pass

                # Add system status
                data["system"] = _get_system_status()

                yield f"data: {json.dumps(data)}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    return app


def _job_to_dict(job: Any) -> Dict[str, Any]:
    """Convert job record to dictionary."""
    return {
        "job_id": job.job_id,
        "input_path": job.input_path,
        "output_path": job.output_path,
        "state": job.state.value,
        "total_frames": job.total_frames,
        "frames_processed": job.frames_processed,
        "frames_failed": job.frames_failed,
        "progress_percent": (job.frames_processed / job.total_frames * 100)
        if job.total_frames > 0 else 0,
        "current_frame": job.current_frame,
        "avg_frame_time_ms": job.avg_frame_time_ms,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "config_json": job.config_json,
        "error_message": job.error_message,
    }


def _frame_to_dict(frame: Any) -> Dict[str, Any]:
    """Convert frame record to dictionary."""
    return {
        "frame_number": frame.frame_number,
        "state": frame.state.value,
        "input_path": frame.input_path,
        "output_path": frame.output_path,
        "error_message": frame.error_message,
    }


def _get_system_status() -> Dict[str, Any]:
    """Get system resource status."""
    status = {
        "cpu_percent": 0,
        "ram_used_gb": 0,
        "ram_total_gb": 0,
        "vram_used_gb": 0,
        "vram_total_gb": 0,
        "gpu_name": "N/A",
        "gpu_temp": 0,
    }

    try:
        import psutil
        status["cpu_percent"] = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        status["ram_used_gb"] = mem.used / (1024**3)
        status["ram_total_gb"] = mem.total / (1024**3)
    except ImportError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["vram_used_gb"] = torch.cuda.memory_allocated() / (1024**3)
            status["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except:
        pass

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        status["gpu_temp"] = temp
        pynvml.nvmlShutdown()
    except:
        pass

    return status


# Dashboard HTML template
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FrameWright Dashboard</title>
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --accent: #e94560;
            --success: #4ecca3;
            --warning: #ffd93d;
            --error: #ff6b6b;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }
        header {
            background: var(--bg-secondary);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--accent);
        }
        header h1 { font-size: 1.5rem; }
        header h1 span { color: var(--accent); }
        .status-bar {
            display: flex;
            gap: 2rem;
            font-size: 0.9rem;
        }
        .status-item { display: flex; align-items: center; gap: 0.5rem; }
        .status-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--success);
        }
        main { padding: 2rem; display: grid; gap: 2rem; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
        .card {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 1.5rem;
        }
        .card h2 {
            font-size: 1rem;
            color: var(--text-secondary);
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .metric:last-child { border-bottom: none; }
        .metric-value { font-weight: bold; font-size: 1.1rem; }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: var(--bg-primary);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--success));
            transition: width 0.3s ease;
        }
        .job-list { display: flex; flex-direction: column; gap: 1rem; }
        .job-item {
            background: var(--bg-secondary);
            border-radius: 6px;
            padding: 1rem;
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 1rem;
            align-items: center;
        }
        .job-info h3 { font-size: 0.95rem; margin-bottom: 0.25rem; }
        .job-info p { font-size: 0.8rem; color: var(--text-secondary); }
        .job-status {
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.75rem;
            text-transform: uppercase;
        }
        .job-status.processing { background: var(--accent); }
        .job-status.completed { background: var(--success); color: #000; }
        .job-status.failed { background: var(--error); }
        .job-status.pending { background: var(--warning); color: #000; }
        .empty-state {
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
        }
        .model-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1rem;
        }
        .model-card {
            background: var(--bg-secondary);
            border-radius: 6px;
            padding: 1rem;
            border: 2px solid transparent;
            transition: border-color 0.3s;
        }
        .model-card.installed { border-color: var(--success); }
        .model-card.downloading { border-color: var(--warning); }
        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 0.5rem;
        }
        .model-name {
            font-weight: bold;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }
        .model-category {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 3px;
            font-size: 0.7rem;
            text-transform: uppercase;
            background: var(--bg-card);
        }
        .model-info {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin: 0.5rem 0;
        }
        .model-actions {
            margin-top: 0.75rem;
            display: flex;
            gap: 0.5rem;
        }
        .btn {
            padding: 0.4rem 0.8rem;
            border: none;
            border-radius: 4px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        .btn:hover { opacity: 0.8; }
        .btn-primary { background: var(--accent); color: white; }
        .btn-success { background: var(--success); color: #000; }
        .btn-danger { background: var(--error); color: white; }
        .btn-secondary { background: var(--bg-primary); color: var(--text-primary); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        @media (max-width: 768px) {
            .grid-2 { grid-template-columns: 1fr; }
            .model-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <header>
        <h1>Frame<span>Wright</span> Dashboard</h1>
        <div class="status-bar">
            <div class="status-item">
                <div class="status-dot" id="connection-dot"></div>
                <span>Connected</span>
            </div>
            <div class="status-item">
                <span id="last-update">Last update: --</span>
            </div>
        </div>
    </header>
    <main>
        <div class="grid-2">
            <div class="card">
                <h2>System Resources</h2>
                <div class="metric">
                    <span>CPU</span>
                    <span class="metric-value" id="cpu-usage">--%</span>
                </div>
                <div class="progress-bar"><div class="progress-fill" id="cpu-bar" style="width: 0%"></div></div>
                <div class="metric">
                    <span>RAM</span>
                    <span class="metric-value" id="ram-usage">-- / -- GB</span>
                </div>
                <div class="progress-bar"><div class="progress-fill" id="ram-bar" style="width: 0%"></div></div>
                <div class="metric">
                    <span>VRAM</span>
                    <span class="metric-value" id="vram-usage">-- / -- GB</span>
                </div>
                <div class="progress-bar"><div class="progress-fill" id="vram-bar" style="width: 0%"></div></div>
                <div class="metric">
                    <span>GPU</span>
                    <span class="metric-value" id="gpu-name">--</span>
                </div>
                <div class="metric">
                    <span>GPU Temperature</span>
                    <span class="metric-value" id="gpu-temp">--C</span>
                </div>
            </div>
            <div class="card">
                <h2>Quick Stats</h2>
                <div class="metric">
                    <span>Active Jobs</span>
                    <span class="metric-value" id="active-jobs">0</span>
                </div>
                <div class="metric">
                    <span>Completed Today</span>
                    <span class="metric-value" id="completed-today">0</span>
                </div>
                <div class="metric">
                    <span>Total Frames Processed</span>
                    <span class="metric-value" id="total-frames">0</span>
                </div>
                <div class="metric">
                    <span>Avg Frame Time</span>
                    <span class="metric-value" id="avg-frame-time">-- ms</span>
                </div>
            </div>
        </div>
        <div class="card">
            <h2>Active Jobs</h2>
            <div class="job-list" id="job-list">
                <div class="empty-state">No active jobs</div>
            </div>
        </div>
        <div class="card">
            <h2>AI Models</h2>
            <div id="model-list">
                <div class="empty-state">Loading models...</div>
            </div>
        </div>
    </main>
    <script>
        function updateDashboard(data) {
            // Update system metrics
            if (data.system) {
                const s = data.system;
                document.getElementById('cpu-usage').textContent = s.cpu_percent.toFixed(1) + '%';
                document.getElementById('cpu-bar').style.width = s.cpu_percent + '%';
                document.getElementById('ram-usage').textContent = s.ram_used_gb.toFixed(1) + ' / ' + s.ram_total_gb.toFixed(1) + ' GB';
                document.getElementById('ram-bar').style.width = (s.ram_used_gb / s.ram_total_gb * 100) + '%';
                document.getElementById('vram-usage').textContent = s.vram_used_gb.toFixed(1) + ' / ' + s.vram_total_gb.toFixed(1) + ' GB';
                document.getElementById('vram-bar').style.width = s.vram_total_gb > 0 ? (s.vram_used_gb / s.vram_total_gb * 100) + '%' : '0%';
                document.getElementById('gpu-name').textContent = s.gpu_name || 'N/A';
                document.getElementById('gpu-temp').textContent = s.gpu_temp + 'C';
            }
            // Update jobs
            if (data.jobs) {
                const jobList = document.getElementById('job-list');
                if (data.jobs.length === 0) {
                    jobList.innerHTML = '<div class="empty-state">No active jobs</div>';
                } else {
                    jobList.innerHTML = data.jobs.map(j => `
                        <div class="job-item">
                            <div class="job-info">
                                <h3>${j.job_id}</h3>
                                <p>Progress: ${j.progress.toFixed(1)}%</p>
                                <div class="progress-bar"><div class="progress-fill" style="width: ${j.progress}%"></div></div>
                            </div>
                            <span class="job-status ${j.state}">${j.state}</span>
                        </div>
                    `).join('');
                }
                document.getElementById('active-jobs').textContent = data.jobs.filter(j => j.state === 'processing').length;
            }
            document.getElementById('last-update').textContent = 'Last update: ' + new Date().toLocaleTimeString();
        }
        function loadModels() {
            fetch('/api/models')
                .then(r => r.json())
                .then(data => {
                    if (!data.models || data.models.length === 0) {
                        document.getElementById('model-list').innerHTML = '<div class="empty-state">No models available</div>';
                        return;
                    }
                    const modelGrid = document.createElement('div');
                    modelGrid.className = 'model-grid';
                    modelGrid.innerHTML = data.models.map(m => `
                        <div class="model-card ${m.installed ? 'installed' : ''}">
                            <div class="model-header">
                                <div>
                                    <div class="model-name">${m.name}</div>
                                    <span class="model-category">${m.category}</span>
                                </div>
                            </div>
                            <div class="model-info">
                                ${m.size_mb.toFixed(0)} MB
                                ${m.description ? '<br>' + m.description : ''}
                            </div>
                            <div class="model-actions">
                                ${m.installed
                                    ? `<button class="btn btn-success" disabled>✓ Installed</button>
                                       <button class="btn btn-danger" onclick="deleteModel('${m.id}')">Delete</button>`
                                    : `<button class="btn btn-primary" onclick="downloadModel('${m.id}')">Download</button>`
                                }
                            </div>
                        </div>
                    `).join('');
                    document.getElementById('model-list').innerHTML = '';
                    document.getElementById('model-list').appendChild(modelGrid);
                })
                .catch(err => {
                    console.error('Failed to load models:', err);
                    document.getElementById('model-list').innerHTML = '<div class="empty-state">Failed to load models</div>';
                });
        }
        function downloadModel(modelId) {
            if (!confirm(`Download model ${modelId}?`)) return;
            fetch(`/api/models/${modelId}/download`, { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        alert(`Model ${modelId} downloaded successfully!`);
                        loadModels();
                    } else {
                        alert(`Failed to download model: ${data.error}`);
                    }
                })
                .catch(err => alert(`Download error: ${err.message}`));
        }
        function deleteModel(modelId) {
            if (!confirm(`Delete model ${modelId}?`)) return;
            fetch(`/api/models/${modelId}`, { method: 'DELETE' })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        alert(`Model ${modelId} deleted successfully!`);
                        loadModels();
                    } else {
                        alert(`Failed to delete model: ${data.error}`);
                    }
                })
                .catch(err => alert(`Delete error: ${err.message}`));
        }
        // Initial load
        fetch('/api/system').then(r => r.json()).then(data => updateDashboard({system: data}));
        fetch('/api/jobs').then(r => r.json()).then(data => {
            if (data.jobs) updateDashboard({jobs: data.jobs.map(j => ({job_id: j.job_id, state: j.state, progress: j.progress_percent}))});
        });
        loadModels();
        // SSE for real-time updates
        const eventSource = new EventSource('/api/events');
        eventSource.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);
                updateDashboard(data);
            } catch (err) {}
        };
        eventSource.onerror = () => {
            document.getElementById('connection-dot').style.background = '#ff6b6b';
        };
    </script>
</body>
</html>
"""
