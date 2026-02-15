#!/usr/bin/env python3
"""Full-featured FrameWright Dashboard Server with Apple-style UI."""

import http.server
import socketserver
import json
import os
import platform
import subprocess
import threading
import webbrowser
import time
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PORT = 8080
HOST = "127.0.0.1"

# Default paths
DEFAULT_MODEL_DIR = Path.home() / ".framewright" / "models"
DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Job storage
JOBS = {}
JOBS_LOCK = threading.Lock()

# Preset definitions
PRESETS = {
    "fast": {
        "name": "Fast",
        "description": "Quick preview quality - good for testing",
        "model": "realesr-general-x4v3",
        "denoise": False,
        "face_enhance": False,
        "tile_size": 512,
        "fp16": True,
    },
    "balanced": {
        "name": "Balanced",
        "description": "Good balance of quality and speed",
        "model": "realesr-general-x4v3",
        "denoise": True,
        "face_enhance": False,
        "tile_size": 400,
        "fp16": True,
    },
    "quality": {
        "name": "Quality",
        "description": "High quality output for final renders",
        "model": "RealESRGAN_x4plus",
        "denoise": True,
        "face_enhance": True,
        "tile_size": 256,
        "fp16": False,
    },
    "ultra": {
        "name": "Ultra",
        "description": "Maximum quality - slow but best results",
        "model": "RealESRGAN_x4plus",
        "denoise": True,
        "face_enhance": True,
        "tile_size": 128,
        "fp16": False,
    },
    "anime": {
        "name": "Anime",
        "description": "Optimized for anime/cartoon content",
        "model": "realesr-animevideov3",
        "denoise": True,
        "face_enhance": False,
        "tile_size": 400,
        "fp16": True,
    },
}

# Available models
MODELS = [
    {
        "id": "realesr-general-x4v3",
        "name": "Real-ESRGAN General v3",
        "type": "Real-ESRGAN",
        "scale": 4,
        "size_mb": 64,
        "description": "Best for general video content",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    },
    {
        "id": "realesr-animevideov3",
        "name": "Real-ESRGAN Anime Video v3",
        "type": "Real-ESRGAN",
        "scale": 4,
        "size_mb": 64,
        "description": "Optimized for anime content",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    },
    {
        "id": "RealESRGAN_x4plus",
        "name": "Real-ESRGAN x4 Plus",
        "type": "Real-ESRGAN",
        "scale": 4,
        "size_mb": 64,
        "description": "High quality general purpose model",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
    {
        "id": "RealESRGAN_x4plus_anime_6B",
        "name": "Real-ESRGAN x4 Plus Anime 6B",
        "type": "Real-ESRGAN",
        "scale": 4,
        "size_mb": 17,
        "description": "Lightweight anime model",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    },
]


def get_system_info():
    """Get system resource information."""
    info = {
        "cpu_percent": 0.0,
        "ram_used_gb": 0.0,
        "ram_total_gb": 0.0,
        "vram_used_gb": 0.0,
        "vram_total_gb": 0.0,
        "gpu_name": "N/A",
        "gpu_temp": 0,
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "model_dir": str(DEFAULT_MODEL_DIR),
    }
    try:
        import psutil
        info["cpu_percent"] = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        info["ram_used_gb"] = round(mem.used / (1024**3), 2)
        info["ram_total_gb"] = round(mem.total / (1024**3), 2)
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_used_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 2)
            info["vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
    except ImportError:
        pass
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info["gpu_temp"] = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
        pynvml.nvmlShutdown()
    except:
        pass
    return info


def check_requirements():
    """Check what's installed and what's needed."""
    reqs = {
        "python": {"installed": True, "version": platform.python_version()},
        "torch": {"installed": False, "version": None, "cuda": False},
        "opencv": {"installed": False, "version": None},
        "ffmpeg": {"installed": False, "path": None},
        "yt_dlp": {"installed": False, "version": None},
        "realesrgan": {"installed": False},
    }

    try:
        import torch
        reqs["torch"]["installed"] = True
        reqs["torch"]["version"] = torch.__version__
        reqs["torch"]["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            reqs["torch"]["cuda_version"] = torch.version.cuda
            reqs["torch"]["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    try:
        import cv2
        reqs["opencv"]["installed"] = True
        reqs["opencv"]["version"] = cv2.__version__
    except ImportError:
        pass

    # Check ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        reqs["ffmpeg"]["installed"] = True
        reqs["ffmpeg"]["path"] = ffmpeg_path

    try:
        import yt_dlp
        reqs["yt_dlp"]["installed"] = True
        reqs["yt_dlp"]["version"] = yt_dlp.version.__version__
    except ImportError:
        pass

    try:
        from realesrgan import RealESRGANer
        reqs["realesrgan"]["installed"] = True
    except ImportError:
        pass

    return reqs


def get_models_status():
    """Check which models are downloaded."""
    models_status = []
    for model in MODELS:
        model_path = DEFAULT_MODEL_DIR / f"{model['id']}.pth"
        models_status.append({
            **model,
            "downloaded": model_path.exists(),
            "path": str(model_path) if model_path.exists() else None,
        })
    return models_status


def download_model(model_id):
    """Download a model."""
    model = next((m for m in MODELS if m["id"] == model_id), None)
    if not model:
        return {"error": "Model not found"}

    model_path = DEFAULT_MODEL_DIR / f"{model_id}.pth"
    if model_path.exists():
        return {"success": True, "message": "Model already downloaded"}

    try:
        import urllib.request
        print(f"Downloading {model['name']}...")
        urllib.request.urlretrieve(model["url"], model_path)
        return {"success": True, "message": f"Downloaded {model['name']}"}
    except Exception as e:
        return {"error": str(e)}


def download_youtube(url, output_dir=None):
    """Download YouTube video using yt-dlp."""
    if output_dir is None:
        output_dir = Path.home() / "Downloads"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import yt_dlp

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
            'merge_output_format': 'mp4',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            # Handle merged output
            if not Path(filename).exists():
                filename = filename.rsplit('.', 1)[0] + '.mp4'
            return {
                "success": True,
                "path": filename,
                "title": info.get("title", "Unknown"),
                "duration": info.get("duration", 0),
            }
    except ImportError:
        return {"error": "yt-dlp not installed. Run: pip install yt-dlp"}
    except Exception as e:
        return {"error": str(e)}


def list_directory(path):
    """List directory contents for file browser."""
    try:
        p = Path(path)
        if not p.exists():
            return {"error": "Path does not exist"}

        items = []
        for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            try:
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0,
                    "ext": item.suffix.lower() if item.is_file() else "",
                })
            except PermissionError:
                continue

        return {
            "path": str(p),
            "parent": str(p.parent) if p.parent != p else None,
            "items": items,
        }
    except Exception as e:
        return {"error": str(e)}


def get_drives():
    """Get available drives on Windows."""
    if platform.system() == "Windows":
        import string
        drives = []
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                drives.append({"name": f"{letter}:", "path": drive})
        return drives
    else:
        return [{"name": "/", "path": "/"}]


def submit_job(data):
    """Submit a restoration job."""
    job_id = str(uuid.uuid4())[:8]
    job = {
        "id": job_id,
        "input_path": data.get("input_path", ""),
        "output_path": data.get("output_path", ""),
        "preset": data.get("preset", "balanced"),
        "scale": data.get("scale", 4),
        "status": "pending",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
    }

    with JOBS_LOCK:
        JOBS[job_id] = job

    # Start job in background
    threading.Thread(target=run_job, args=(job_id,), daemon=True).start()

    return job


def run_job(job_id):
    """Run a restoration job."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["started_at"] = datetime.now().isoformat()

    try:
        # Build command
        cmd = [
            "python", "-m", "framewright.cli", "restore",
            job["input_path"],
            "-o", job["output_path"],
            "--preset", job["preset"],
            "--scale", str(job["scale"]),
        ]

        # Run restoration
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )

        for line in process.stdout:
            if "%" in line:
                try:
                    for part in line.split():
                        if "%" in part:
                            pct = float(part.replace("%", ""))
                            with JOBS_LOCK:
                                JOBS[job_id]["progress"] = pct
                            break
                except:
                    pass

        process.wait()

        with JOBS_LOCK:
            if process.returncode == 0:
                JOBS[job_id]["status"] = "completed"
                JOBS[job_id]["progress"] = 100
            else:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = f"Exit code: {process.returncode}"
            JOBS[job_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
            JOBS[job_id]["completed_at"] = datetime.now().isoformat()


DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FrameWright</title>
    <style>
        :root {
            --bg-primary: #000000;
            --bg-secondary: #1c1c1e;
            --bg-tertiary: #2c2c2e;
            --bg-elevated: #3a3a3c;
            --text-primary: #ffffff;
            --text-secondary: #8e8e93;
            --accent: #0a84ff;
            --accent-hover: #409cff;
            --success: #30d158;
            --warning: #ff9f0a;
            --error: #ff453a;
            --border: rgba(255,255,255,0.1);
            --radius: 12px;
            --radius-sm: 8px;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }
        header {
            background: rgba(28,28,30,0.8);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            padding: 16px 24px;
            position: sticky;
            top: 0;
            z-index: 100;
            border-bottom: 1px solid var(--border);
        }
        header h1 { font-size: 20px; font-weight: 600; letter-spacing: -0.5px; }
        header h1 span { color: var(--accent); }
        nav { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
        nav button {
            background: var(--bg-tertiary);
            border: none;
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
        }
        nav button:hover { background: var(--bg-elevated); }
        nav button.active { background: var(--accent); }
        main { max-width: 1200px; margin: 0 auto; padding: 24px; }
        .card {
            background: var(--bg-secondary);
            border-radius: var(--radius);
            padding: 20px;
            margin-bottom: 16px;
        }
        .card h2 {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 16px;
        }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: var(--text-secondary); font-size: 14px; }
        .metric-value { font-size: 15px; font-weight: 500; font-variant-numeric: tabular-nums; }
        .progress-bar {
            width: 100%;
            height: 4px;
            background: var(--bg-tertiary);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 8px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--success));
            transition: width 0.3s ease;
        }
        .form-group { margin-bottom: 16px; }
        .form-group label {
            display: block;
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }
        input[type="text"], input[type="url"], select, textarea {
            width: 100%;
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            color: var(--text-primary);
            font-size: 15px;
            outline: none;
            transition: border-color 0.2s;
        }
        input:focus, select:focus, textarea:focus { border-color: var(--accent); }
        textarea { resize: vertical; min-height: 100px; font-family: monospace; }
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 12px 24px;
            border: none;
            border-radius: var(--radius-sm);
            font-size: 15px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-primary { background: var(--accent); color: white; }
        .btn-primary:hover { background: var(--accent-hover); }
        .btn-secondary { background: var(--bg-tertiary); color: var(--text-primary); }
        .btn-secondary:hover { background: var(--bg-elevated); }
        .btn-success { background: var(--success); color: #000; }
        .btn-sm { padding: 8px 16px; font-size: 13px; }
        .file-browser {
            background: var(--bg-tertiary);
            border-radius: var(--radius-sm);
            max-height: 250px;
            overflow-y: auto;
        }
        .file-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 16px;
            cursor: pointer;
            border-bottom: 1px solid var(--border);
            transition: background 0.15s;
        }
        .file-item:hover { background: var(--bg-elevated); }
        .file-item:last-child { border-bottom: none; }
        .file-icon { font-size: 18px; }
        .file-name { flex: 1; font-size: 14px; }
        .file-size { color: var(--text-secondary); font-size: 12px; }
        .drives { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
        .drive-btn {
            padding: 6px 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 13px;
            cursor: pointer;
        }
        .drive-btn:hover { background: var(--bg-elevated); }
        .job-item {
            background: var(--bg-tertiary);
            border-radius: var(--radius-sm);
            padding: 16px;
            margin-bottom: 12px;
        }
        .job-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .job-id { font-weight: 600; }
        .job-status { padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 500; }
        .job-status.pending { background: var(--bg-elevated); }
        .job-status.running { background: var(--accent); }
        .job-status.completed { background: var(--success); color: #000; }
        .job-status.failed { background: var(--error); }
        .job-path { font-size: 13px; color: var(--text-secondary); margin-bottom: 8px; word-break: break-all; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .preset-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
        .preset-card {
            background: var(--bg-tertiary);
            border: 2px solid transparent;
            border-radius: var(--radius-sm);
            padding: 16px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .preset-card:hover { border-color: var(--border); }
        .preset-card.selected { border-color: var(--accent); background: rgba(10,132,255,0.1); }
        .preset-name { font-weight: 600; margin-bottom: 4px; }
        .preset-desc { font-size: 13px; color: var(--text-secondary); }
        .current-path {
            background: var(--bg-primary);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 12px;
            word-break: break-all;
        }
        .actions { display: flex; gap: 12px; margin-top: 20px; flex-wrap: wrap; }
        .tabs-mini { display: flex; gap: 4px; margin-bottom: 16px; }
        .tabs-mini button {
            padding: 6px 12px;
            background: var(--bg-tertiary);
            border: none;
            color: var(--text-secondary);
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
        }
        .tabs-mini button.active { background: var(--accent); color: white; }
        .model-card {
            background: var(--bg-tertiary);
            border-radius: var(--radius-sm);
            padding: 16px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .model-info h3 { font-size: 15px; margin-bottom: 4px; }
        .model-info p { font-size: 13px; color: var(--text-secondary); }
        .model-status { display: flex; align-items: center; gap: 12px; }
        .badge {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        .badge-success { background: var(--success); color: #000; }
        .badge-warning { background: var(--warning); color: #000; }
        .req-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border-radius: var(--radius-sm);
            margin-bottom: 8px;
        }
        .req-name { font-weight: 500; }
        .req-status { display: flex; align-items: center; gap: 8px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; }
        .dot-green { background: var(--success); }
        .dot-red { background: var(--error); }
        .dot-yellow { background: var(--warning); }
        .hint { font-size: 12px; color: var(--text-secondary); margin-top: 8px; }
        .input-row { display: flex; gap: 12px; }
        .input-row input { flex: 1; }
    </style>
</head>
<body>
    <header>
        <h1>Frame<span>Wright</span></h1>
        <nav>
            <button class="active" onclick="showTab('restore')">New Restoration</button>
            <button onclick="showTab('jobs')">Jobs</button>
            <button onclick="showTab('presets')">Presets</button>
            <button onclick="showTab('models')">Models</button>
            <button onclick="showTab('setup')">Setup</button>
            <button onclick="showTab('system')">System</button>
        </nav>
    </header>

    <main>
        <!-- Restore Tab -->
        <div id="tab-restore" class="tab-content active">
            <div class="card">
                <h2>Video Source</h2>
                <div class="tabs-mini">
                    <button class="active" onclick="setSourceMode('file')">Local File</button>
                    <button onclick="setSourceMode('youtube')">YouTube URL</button>
                </div>

                <div id="source-file">
                    <div class="drives" id="drives"></div>
                    <div class="current-path" id="current-path">Select a file...</div>
                    <div class="file-browser" id="file-browser"></div>
                </div>

                <div id="source-youtube" style="display:none;">
                    <div class="form-group">
                        <label>YouTube URL</label>
                        <div class="input-row">
                            <input type="url" id="youtube-url" placeholder="https://www.youtube.com/watch?v=...">
                            <button class="btn btn-secondary" onclick="downloadYouTube()">Download</button>
                        </div>
                        <p class="hint">Video will be downloaded to your Downloads folder, then you can restore it.</p>
                    </div>
                    <div id="youtube-status"></div>
                </div>

                <input type="hidden" id="input-path">
            </div>

            <div class="card">
                <h2>Output Location</h2>
                <div class="form-group">
                    <input type="text" id="output-path" placeholder="Output file path (auto-generated if empty)">
                </div>
            </div>

            <div class="card">
                <h2>Preset</h2>
                <div class="preset-cards" id="preset-cards"></div>
            </div>

            <div class="card">
                <h2>Scale</h2>
                <div class="form-group">
                    <select id="scale">
                        <option value="2">2x Upscale</option>
                        <option value="4" selected>4x Upscale</option>
                    </select>
                </div>
            </div>

            <div class="actions">
                <button class="btn btn-primary" onclick="submitJob()">Start Restoration</button>
            </div>
        </div>

        <!-- Jobs Tab -->
        <div id="tab-jobs" class="tab-content">
            <div class="card">
                <h2>Restoration Jobs</h2>
                <div id="jobs-list">
                    <p style="color: var(--text-secondary);">No jobs yet.</p>
                </div>
            </div>
        </div>

        <!-- Presets Tab -->
        <div id="tab-presets" class="tab-content">
            <div class="card">
                <h2>Available Presets</h2>
                <div id="presets-detail"></div>
            </div>
            <div class="card">
                <h2>Custom Preset (Advanced)</h2>
                <div class="form-group">
                    <label>Preset JSON Configuration</label>
                    <textarea id="custom-preset" placeholder='{"model": "realesr-general-x4v3", "denoise": true, ...}'></textarea>
                    <p class="hint">Edit preset parameters directly. Changes apply to the next job you submit.</p>
                </div>
            </div>
        </div>

        <!-- Models Tab -->
        <div id="tab-models" class="tab-content">
            <div class="card">
                <h2>Model Storage Location</h2>
                <div class="form-group">
                    <input type="text" id="model-dir" placeholder="Model directory path">
                    <p class="hint">Default: ~/.framewright/models</p>
                </div>
            </div>
            <div class="card">
                <h2>Available Models</h2>
                <div id="models-list"></div>
            </div>
        </div>

        <!-- Setup Tab -->
        <div id="tab-setup" class="tab-content">
            <div class="card">
                <h2>Requirements Status</h2>
                <div id="requirements-list"></div>
            </div>
            <div class="card">
                <h2>Installation Commands</h2>
                <div class="form-group">
                    <label>Install PyTorch with CUDA (NVIDIA GPU)</label>
                    <input type="text" readonly value="pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121" onclick="this.select()">
                </div>
                <div class="form-group">
                    <label>Install Real-ESRGAN</label>
                    <input type="text" readonly value="pip install realesrgan" onclick="this.select()">
                </div>
                <div class="form-group">
                    <label>Install yt-dlp (YouTube downloads)</label>
                    <input type="text" readonly value="pip install yt-dlp" onclick="this.select()">
                </div>
                <div class="form-group">
                    <label>Install FFmpeg (Windows - using Chocolatey)</label>
                    <input type="text" readonly value="choco install ffmpeg" onclick="this.select()">
                </div>
                <p class="hint">Click any command to select it, then copy with Ctrl+C</p>
            </div>
        </div>

        <!-- System Tab -->
        <div id="tab-system" class="tab-content">
            <div class="grid">
                <div class="card">
                    <h2>System</h2>
                    <div class="metric">
                        <span class="metric-label">Platform</span>
                        <span class="metric-value" id="sys-platform">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Python</span>
                        <span class="metric-value" id="sys-python">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">CPU Usage</span>
                        <span class="metric-value" id="sys-cpu">--%</span>
                    </div>
                    <div class="progress-bar"><div class="progress-fill" id="cpu-bar" style="width:0%"></div></div>
                    <div class="metric">
                        <span class="metric-label">RAM</span>
                        <span class="metric-value" id="sys-ram">-- / -- GB</span>
                    </div>
                    <div class="progress-bar"><div class="progress-fill" id="ram-bar" style="width:0%"></div></div>
                </div>
                <div class="card">
                    <h2>GPU</h2>
                    <div class="metric">
                        <span class="metric-label">Name</span>
                        <span class="metric-value" id="sys-gpu">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">VRAM</span>
                        <span class="metric-value" id="sys-vram">-- / -- GB</span>
                    </div>
                    <div class="progress-bar"><div class="progress-fill" id="vram-bar" style="width:0%"></div></div>
                    <div class="metric">
                        <span class="metric-label">Temperature</span>
                        <span class="metric-value" id="sys-temp">--°C</span>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        let currentPath = '';
        let selectedPreset = 'balanced';
        let presets = {};
        let sourceMode = 'file';

        // Tab switching
        function showTab(name) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
            document.getElementById('tab-' + name).classList.add('active');
            event.target.classList.add('active');

            if (name === 'system') updateSystem();
            if (name === 'jobs') updateJobs();
            if (name === 'presets') loadPresets();
            if (name === 'models') loadModels();
            if (name === 'setup') loadRequirements();
        }

        // Source mode toggle
        function setSourceMode(mode) {
            sourceMode = mode;
            document.querySelectorAll('.tabs-mini button').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('source-file').style.display = mode === 'file' ? 'block' : 'none';
            document.getElementById('source-youtube').style.display = mode === 'youtube' ? 'block' : 'none';
        }

        // YouTube download
        async function downloadYouTube() {
            const url = document.getElementById('youtube-url').value;
            if (!url) { alert('Please enter a YouTube URL'); return; }

            const status = document.getElementById('youtube-status');
            status.innerHTML = '<p style="color:var(--accent)">Downloading... this may take a minute.</p>';

            try {
                const r = await fetch('/api/youtube', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url: url}),
                });
                const data = await r.json();

                if (data.error) {
                    status.innerHTML = `<p style="color:var(--error)">Error: ${data.error}</p>`;
                } else {
                    status.innerHTML = `<p style="color:var(--success)">Downloaded: ${data.title}</p>`;
                    document.getElementById('input-path').value = data.path;
                    document.getElementById('current-path').textContent = '✓ ' + data.path;
                    document.getElementById('current-path').style.color = 'var(--success)';

                    // Auto-generate output path
                    const ext = data.path.substring(data.path.lastIndexOf('.'));
                    const base = data.path.substring(0, data.path.lastIndexOf('.'));
                    document.getElementById('output-path').value = base + '_restored' + ext;
                }
            } catch (e) {
                status.innerHTML = `<p style="color:var(--error)">Error: ${e.message}</p>`;
            }
        }

        // File browser
        async function loadDrives() {
            const r = await fetch('/api/drives');
            const data = await r.json();
            const container = document.getElementById('drives');
            container.innerHTML = data.map(d =>
                `<button class="drive-btn" onclick="browsePath('${d.path.replace(/\\\\/g, '\\\\\\\\')}')">${d.name}</button>`
            ).join('');
            if (data.length > 0) browsePath(data[0].path);
        }

        async function browsePath(path) {
            const r = await fetch('/api/browse?path=' + encodeURIComponent(path));
            const data = await r.json();
            if (data.error) { alert(data.error); return; }

            currentPath = data.path;
            document.getElementById('current-path').textContent = data.path;
            document.getElementById('current-path').style.color = 'var(--text-secondary)';

            const browser = document.getElementById('file-browser');
            let html = '';

            if (data.parent) {
                html += `<div class="file-item" onclick="browsePath('${data.parent.replace(/\\\\/g, '\\\\\\\\')}')">
                    <span class="file-icon">📁</span><span class="file-name">..</span>
                </div>`;
            }

            for (const item of data.items) {
                const icon = item.is_dir ? '📁' : (['.mp4','.mkv','.avi','.mov','.webm'].includes(item.ext) ? '🎬' : '📄');
                const size = item.is_dir ? '' : formatSize(item.size);
                const escapedPath = item.path.replace(/\\\\/g, '\\\\\\\\');

                if (item.is_dir) {
                    html += `<div class="file-item" onclick="browsePath('${escapedPath}')">
                        <span class="file-icon">${icon}</span><span class="file-name">${item.name}</span>
                    </div>`;
                } else if (['.mp4','.mkv','.avi','.mov','.webm','.wmv','.flv'].includes(item.ext)) {
                    html += `<div class="file-item" onclick="selectFile('${escapedPath}')">
                        <span class="file-icon">${icon}</span><span class="file-name">${item.name}</span>
                        <span class="file-size">${size}</span>
                    </div>`;
                }
            }
            browser.innerHTML = html || '<div style="padding:16px;color:var(--text-secondary)">No video files</div>';
        }

        function selectFile(path) {
            document.getElementById('input-path').value = path;
            document.getElementById('current-path').textContent = '✓ ' + path;
            document.getElementById('current-path').style.color = 'var(--success)';
            const ext = path.substring(path.lastIndexOf('.'));
            const base = path.substring(0, path.lastIndexOf('.'));
            document.getElementById('output-path').value = base + '_restored' + ext;
        }

        function formatSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + ' KB';
            if (bytes < 1024*1024*1024) return (bytes/1024/1024).toFixed(1) + ' MB';
            return (bytes/1024/1024/1024).toFixed(2) + ' GB';
        }

        // Presets
        async function loadPresets() {
            const r = await fetch('/api/presets');
            presets = await r.json();

            // Render preset cards on restore tab
            const cardsHtml = Object.entries(presets).map(([key, p]) => `
                <div class="preset-card ${key === selectedPreset ? 'selected' : ''}" data-preset="${key}" onclick="selectPreset(this)">
                    <div class="preset-name">${p.name}</div>
                    <div class="preset-desc">${p.description}</div>
                </div>
            `).join('');
            document.getElementById('preset-cards').innerHTML = cardsHtml;

            // Render detail view
            const detailHtml = Object.entries(presets).map(([key, p]) => `
                <div class="model-card">
                    <div class="model-info">
                        <h3>${p.name}</h3>
                        <p>${p.description}</p>
                        <p style="margin-top:8px;font-size:12px;color:var(--text-secondary)">
                            Model: ${p.model} | Denoise: ${p.denoise ? 'Yes' : 'No'} |
                            Face Enhance: ${p.face_enhance ? 'Yes' : 'No'} | Tile: ${p.tile_size}px
                        </p>
                    </div>
                </div>
            `).join('');
            document.getElementById('presets-detail').innerHTML = detailHtml;
        }

        function selectPreset(el) {
            document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('selected'));
            el.classList.add('selected');
            selectedPreset = el.dataset.preset;
        }

        // Models
        async function loadModels() {
            const r = await fetch('/api/models');
            const models = await r.json();

            const sysInfo = await (await fetch('/api/system')).json();
            document.getElementById('model-dir').value = sysInfo.model_dir;

            const html = models.map(m => `
                <div class="model-card">
                    <div class="model-info">
                        <h3>${m.name}</h3>
                        <p>${m.description} (${m.size_mb} MB)</p>
                    </div>
                    <div class="model-status">
                        ${m.downloaded
                            ? '<span class="badge badge-success">Downloaded</span>'
                            : `<button class="btn btn-sm btn-secondary" onclick="downloadModel('${m.id}')">Download</button>`
                        }
                    </div>
                </div>
            `).join('');
            document.getElementById('models-list').innerHTML = html;
        }

        async function downloadModel(modelId) {
            const btn = event.target;
            btn.textContent = 'Downloading...';
            btn.disabled = true;

            const r = await fetch('/api/models/download', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model_id: modelId}),
            });
            const data = await r.json();

            if (data.error) {
                alert('Error: ' + data.error);
                btn.textContent = 'Download';
                btn.disabled = false;
            } else {
                loadModels();
            }
        }

        // Requirements
        async function loadRequirements() {
            const r = await fetch('/api/requirements');
            const reqs = await r.json();

            const html = Object.entries(reqs).map(([key, req]) => {
                let status = '';
                let details = '';

                if (key === 'torch') {
                    status = req.installed ? (req.cuda ? 'dot-green' : 'dot-yellow') : 'dot-red';
                    details = req.installed
                        ? `v${req.version}${req.cuda ? ' + CUDA ' + req.cuda_version : ' (CPU only)'}`
                        : 'Not installed';
                } else if (req.installed) {
                    status = 'dot-green';
                    details = req.version || req.path || 'Installed';
                } else {
                    status = 'dot-red';
                    details = 'Not installed';
                }

                return `<div class="req-item">
                    <span class="req-name">${key}</span>
                    <div class="req-status">
                        <span style="color:var(--text-secondary);font-size:13px">${details}</span>
                        <span class="dot ${status}"></span>
                    </div>
                </div>`;
            }).join('');

            document.getElementById('requirements-list').innerHTML = html;
        }

        // Submit job
        async function submitJob() {
            const inputPath = document.getElementById('input-path').value;
            if (!inputPath) { alert('Please select an input video file'); return; }

            const data = {
                input_path: inputPath,
                output_path: document.getElementById('output-path').value,
                preset: selectedPreset,
                scale: parseInt(document.getElementById('scale').value),
            };

            const r = await fetch('/api/jobs', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data),
            });
            const result = await r.json();

            if (result.error) {
                alert('Error: ' + result.error);
            } else {
                alert('Job submitted! ID: ' + result.id);
                document.querySelectorAll('nav button')[1].click();
            }
        }

        // Jobs list
        async function updateJobs() {
            const r = await fetch('/api/jobs');
            const jobs = await r.json();
            const container = document.getElementById('jobs-list');

            if (jobs.length === 0) {
                container.innerHTML = '<p style="color:var(--text-secondary)">No jobs yet.</p>';
                return;
            }

            container.innerHTML = jobs.map(j => `
                <div class="job-item">
                    <div class="job-header">
                        <span class="job-id">Job ${j.id}</span>
                        <span class="job-status ${j.status}">${j.status}</span>
                    </div>
                    <div class="job-path">${j.input_path}</div>
                    <div class="job-path">→ ${j.output_path}</div>
                    ${j.status === 'running' ? `
                        <div class="progress-bar"><div class="progress-fill" style="width:${j.progress}%"></div></div>
                        <div style="font-size:12px;color:var(--text-secondary);margin-top:4px">${j.progress.toFixed(1)}%</div>
                    ` : ''}
                    ${j.error ? `<div style="color:var(--error);font-size:13px;margin-top:8px">${j.error}</div>` : ''}
                </div>
            `).join('');
        }

        // System info
        async function updateSystem() {
            const r = await fetch('/api/system');
            const d = await r.json();
            document.getElementById('sys-platform').textContent = d.platform;
            document.getElementById('sys-python').textContent = d.python_version;
            document.getElementById('sys-cpu').textContent = d.cpu_percent.toFixed(1) + '%';
            document.getElementById('cpu-bar').style.width = d.cpu_percent + '%';
            document.getElementById('sys-ram').textContent = d.ram_used_gb + ' / ' + d.ram_total_gb + ' GB';
            document.getElementById('ram-bar').style.width = (d.ram_used_gb/d.ram_total_gb*100) + '%';
            document.getElementById('sys-gpu').textContent = d.gpu_name;
            document.getElementById('sys-vram').textContent = d.vram_used_gb + ' / ' + d.vram_total_gb + ' GB';
            document.getElementById('vram-bar').style.width = d.vram_total_gb > 0 ? (d.vram_used_gb/d.vram_total_gb*100) + '%' : '0%';
            document.getElementById('sys-temp').textContent = d.gpu_temp + '°C';
        }

        // Init
        loadDrives();
        loadPresets();
        setInterval(() => {
            if (document.getElementById('tab-jobs').classList.contains('active')) updateJobs();
            if (document.getElementById('tab-system').classList.contains('active')) updateSystem();
        }, 3000);
    </script>
</body>
</html>
'''


class Handler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler."""

    def log_message(self, *args):
        pass

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())

        elif path == "/api/system":
            self.send_json(get_system_info())

        elif path == "/api/drives":
            self.send_json(get_drives())

        elif path == "/api/browse":
            browse_path = query.get("path", [""])[0]
            if not browse_path:
                browse_path = os.path.expanduser("~")
            self.send_json(list_directory(browse_path))

        elif path == "/api/jobs":
            with JOBS_LOCK:
                jobs = list(JOBS.values())
            self.send_json(jobs)

        elif path == "/api/presets":
            self.send_json(PRESETS)

        elif path == "/api/models":
            self.send_json(get_models_status())

        elif path == "/api/requirements":
            self.send_json(check_requirements())

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON"}, 400)
            return

        if path == "/api/jobs":
            if not data.get("input_path"):
                self.send_json({"error": "input_path required"}, 400)
                return
            job = submit_job(data)
            self.send_json(job, 201)

        elif path == "/api/youtube":
            if not data.get("url"):
                self.send_json({"error": "url required"}, 400)
                return
            result = download_youtube(data["url"])
            self.send_json(result)

        elif path == "/api/models/download":
            if not data.get("model_id"):
                self.send_json({"error": "model_id required"}, 400)
                return
            result = download_model(data["model_id"])
            self.send_json(result)

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    print("=" * 60)
    print("FrameWright Dashboard")
    print("=" * 60)
    print(f"URL: http://{HOST}:{PORT}")
    print("Opening browser...")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    threading.Thread(
        target=lambda: (time.sleep(1), webbrowser.open(f"http://{HOST}:{PORT}")),
        daemon=True,
    ).start()

    server = ThreadedServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
