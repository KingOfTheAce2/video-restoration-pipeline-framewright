#!/usr/bin/env python3
"""Comprehensive FrameWright Dashboard with ALL CLI features."""

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
import urllib.request
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

# ALL available models
MODELS = [
    # Real-ESRGAN models
    {"id": "realesr-general-x4v3", "name": "Real-ESRGAN General v3", "category": "upscaling", "size_mb": 64,
     "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"},
    {"id": "realesr-animevideov3", "name": "Real-ESRGAN Anime v3", "category": "upscaling", "size_mb": 64,
     "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"},
    {"id": "RealESRGAN_x4plus", "name": "Real-ESRGAN x4 Plus", "category": "upscaling", "size_mb": 64,
     "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"},
    {"id": "RealESRGAN_x4plus_anime_6B", "name": "Real-ESRGAN Anime 6B", "category": "upscaling", "size_mb": 17,
     "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"},
    {"id": "RealESRGAN_x2plus", "name": "Real-ESRGAN x2 Plus", "category": "upscaling", "size_mb": 64,
     "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"},
    # Face restoration
    {"id": "GFPGANv1.4", "name": "GFPGAN v1.4", "category": "face", "size_mb": 348,
     "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"},
    {"id": "CodeFormer", "name": "CodeFormer", "category": "face", "size_mb": 376,
     "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"},
    {"id": "RestoreFormer", "name": "RestoreFormer", "category": "face", "size_mb": 280,
     "url": "https://github.com/wzhouxiff/RestoreFormer/releases/download/v1.0/RestoreFormer.pth"},
    # Frame interpolation
    {"id": "rife-v4.6", "name": "RIFE v4.6", "category": "interpolation", "size_mb": 42,
     "url": "https://github.com/hzwer/Practical-RIFE/releases/download/v4.6/rife46.pth"},
    {"id": "rife-v4.0", "name": "RIFE v4.0", "category": "interpolation", "size_mb": 42,
     "url": "https://github.com/hzwer/Practical-RIFE/releases/download/v4.0/rife40.pth"},
    # Denoising
    {"id": "restormer", "name": "Restormer (Denoise)", "category": "denoise", "size_mb": 100,
     "url": "https://github.com/swz30/Restormer/releases/download/v1.0/restormer_deraining.pth"},
    {"id": "nafnet", "name": "NAFNet (Denoise)", "category": "denoise", "size_mb": 67,
     "url": "https://github.com/megvii-research/NAFNet/releases/download/v1.0.0/NAFNet-SIDD-width64.pth"},
    # Colorization
    {"id": "ddcolor", "name": "DDColor", "category": "colorization", "size_mb": 230,
     "url": "https://huggingface.co/piddnad/DDColor/resolve/main/ddcolor_modelscope.pth"},
    {"id": "deoldify-stable", "name": "DeOldify Stable", "category": "colorization", "size_mb": 250,
     "url": "https://data.deepai.org/deoldify/ColorizeStable_gen.pth"},
]

# Full preset definitions
PRESETS = {
    "fast": {
        "name": "Fast Preview",
        "description": "Quick preview - minimal processing",
        "scale": 2, "model": "realesr-general-x4v3", "quality": 23,
        "denoise": False, "face_enhance": False, "deduplicate": False,
    },
    "balanced": {
        "name": "Balanced",
        "description": "Good quality/speed balance for most videos",
        "scale": 2, "model": "realesr-general-x4v3", "quality": 18,
        "denoise": True, "face_enhance": True, "deduplicate": False,
        "grain_reduction": 0.3, "scratch_sensitivity": 0.5,
    },
    "quality": {
        "name": "High Quality",
        "description": "High quality for final output",
        "scale": 4, "model": "RealESRGAN_x4plus", "quality": 16,
        "denoise": True, "face_enhance": True, "deduplicate": True,
        "grain_reduction": 0.5, "scratch_sensitivity": 0.6,
        "scene_aware": True, "motion_adaptive": True,
    },
    "archive": {
        "name": "Archive",
        "description": "Maximum quality for archival - preserves details",
        "scale": 4, "model": "RealESRGAN_x4plus", "quality": 14,
        "denoise": True, "face_enhance": True, "deduplicate": True,
        "grain_reduction": 0.2, "scratch_sensitivity": 0.7,
        "tap_denoise": True, "tap_preserve_grain": True,
        "scene_aware": True, "motion_adaptive": True,
    },
    "anime": {
        "name": "Anime/Cartoon",
        "description": "Optimized for animated content",
        "scale": 4, "model": "realesr-animevideov3", "quality": 18,
        "denoise": True, "face_enhance": False, "deduplicate": True,
        "grain_reduction": 0.0,
    },
    "film": {
        "name": "Film Restoration",
        "description": "Old film with scratches, grain, and damage",
        "scale": 4, "model": "RealESRGAN_x4plus", "quality": 16,
        "denoise": True, "face_enhance": True, "deduplicate": True,
        "grain_reduction": 0.4, "scratch_sensitivity": 0.8,
        "tap_denoise": True, "tap_preserve_grain": True,
        "scene_aware": True,
    },
    "vhs": {
        "name": "VHS/Tape",
        "description": "VHS tapes with tracking issues and noise",
        "scale": 2, "model": "realesr-general-x4v3", "quality": 18,
        "denoise": True, "face_enhance": True, "deduplicate": True,
        "tap_denoise": True, "qp_artifact_removal": True,
    },
}

# All processing options organized by category
ALL_OPTIONS = {
    "basic": {
        "title": "Basic Settings",
        "options": [
            {"id": "scale", "name": "Upscale Factor", "type": "select", "options": ["2", "4"], "default": "2"},
            {"id": "quality", "name": "Output Quality (CRF)", "type": "number", "min": 0, "max": 51, "default": 18,
             "hint": "Lower = better quality, larger file (0-51)"},
            {"id": "format", "name": "Output Format", "type": "select",
             "options": ["mkv", "mp4", "webm", "avi", "mov"], "default": "mkv"},
        ]
    },
    "upscaling": {
        "title": "Upscaling",
        "options": [
            {"id": "model", "name": "Upscaling Model", "type": "select",
             "options": ["realesr-general-x4v3", "RealESRGAN_x4plus", "realesr-animevideov3",
                        "RealESRGAN_x4plus_anime_6B", "RealESRGAN_x2plus"],
             "default": "realesr-general-x4v3"},
            {"id": "diffusion_sr", "name": "Diffusion SR (Best Quality)", "type": "checkbox", "default": False,
             "hint": "Highest quality but very slow"},
            {"id": "diffusion_steps", "name": "Diffusion Steps", "type": "number", "min": 10, "max": 100, "default": 20,
             "depends_on": "diffusion_sr"},
            {"id": "diffusion_guidance", "name": "Diffusion Guidance", "type": "number", "min": 1, "max": 20,
             "default": 7.5, "depends_on": "diffusion_sr"},
        ]
    },
    "denoise": {
        "title": "Denoising & Grain",
        "options": [
            {"id": "tap_denoise", "name": "TAP Neural Denoise", "type": "checkbox", "default": False,
             "hint": "Advanced AI denoising (Restormer/NAFNet)"},
            {"id": "tap_model", "name": "TAP Model", "type": "select",
             "options": ["restormer", "nafnet", "tap"], "default": "restormer", "depends_on": "tap_denoise"},
            {"id": "tap_strength", "name": "TAP Strength", "type": "range", "min": 0, "max": 1, "step": 0.1,
             "default": 1.0, "depends_on": "tap_denoise"},
            {"id": "tap_preserve_grain", "name": "Preserve Film Grain", "type": "checkbox", "default": False,
             "depends_on": "tap_denoise"},
            {"id": "grain_reduction", "name": "Grain Reduction", "type": "range", "min": 0, "max": 1, "step": 0.1,
             "default": 0.3},
        ]
    },
    "frames": {
        "title": "Frame Processing",
        "options": [
            {"id": "deduplicate", "name": "Remove Duplicate Frames", "type": "checkbox", "default": False},
            {"id": "dedup_threshold", "name": "Dedup Threshold", "type": "range", "min": 0.9, "max": 1.0,
             "step": 0.01, "default": 0.98, "depends_on": "deduplicate"},
            {"id": "enable_rife", "name": "Frame Interpolation (RIFE)", "type": "checkbox", "default": False,
             "hint": "Increase frame rate for smoother video"},
            {"id": "target_fps", "name": "Target FPS", "type": "number", "min": 24, "max": 120, "default": 60,
             "depends_on": "enable_rife"},
            {"id": "rife_model", "name": "RIFE Model", "type": "select",
             "options": ["rife-v4.6", "rife-v4.0", "rife-v2.3"], "default": "rife-v4.6", "depends_on": "enable_rife"},
            {"id": "generate_frames", "name": "Generate Missing Frames", "type": "checkbox", "default": False,
             "hint": "For damaged film with gaps"},
            {"id": "frame_gen_model", "name": "Frame Gen Model", "type": "select",
             "options": ["interpolate_blend", "optical_flow_warp", "svd"], "default": "interpolate_blend",
             "depends_on": "generate_frames"},
            {"id": "max_gap_frames", "name": "Max Gap Frames", "type": "number", "min": 1, "max": 30, "default": 10,
             "depends_on": "generate_frames"},
        ]
    },
    "face": {
        "title": "Face Enhancement",
        "options": [
            {"id": "auto_enhance", "name": "Auto Face & Defect Repair", "type": "checkbox", "default": True},
            {"id": "face_model", "name": "Face Model", "type": "select",
             "options": ["gfpgan", "codeformer", "aesrgan"], "default": "gfpgan", "depends_on": "auto_enhance"},
            {"id": "aesrgan_strength", "name": "AESRGAN Strength", "type": "range", "min": 0, "max": 1,
             "step": 0.1, "default": 0.8, "depends_on": "auto_enhance"},
            {"id": "no_face_restore", "name": "Disable Face Restore", "type": "checkbox", "default": False},
        ]
    },
    "colorize": {
        "title": "Colorization",
        "options": [
            {"id": "colorize", "name": "Colorize B&W Video", "type": "checkbox", "default": False},
            {"id": "colorize_model", "name": "Colorization Model", "type": "select",
             "options": ["ddcolor", "deoldify"], "default": "ddcolor", "depends_on": "colorize"},
            {"id": "colorize_temporal_fusion", "name": "Temporal Fusion", "type": "checkbox", "default": True,
             "depends_on": "colorize", "hint": "Consistent colors across frames"},
        ]
    },
    "reference_enhance": {
        "title": "Reference Enhancement",
        "options": [
            {"id": "reference_enhance", "name": "Reference-Guided Enhancement", "type": "checkbox", "default": False,
             "hint": "Use reference photos to guide detail generation (IP-Adapter + ControlNet)"},
            {"id": "reference_dir", "name": "Reference Images Dir", "type": "text", "default": "",
             "depends_on": "reference_enhance", "hint": "3-10 high-quality photos of same locations/subjects (quality > quantity)"},
            {"id": "reference_strength", "name": "Reference Strength", "type": "range", "min": 0, "max": 1,
             "step": 0.05, "default": 0.35, "depends_on": "reference_enhance"},
            {"id": "reference_guidance", "name": "Guidance Scale", "type": "number", "min": 1, "max": 20,
             "default": 7.5, "depends_on": "reference_enhance"},
            {"id": "reference_ip_scale", "name": "IP-Adapter Scale", "type": "range", "min": 0, "max": 1,
             "step": 0.05, "default": 0.6, "depends_on": "reference_enhance"},
        ]
    },
    "repair": {
        "title": "Defect Repair",
        "options": [
            {"id": "scratch_sensitivity", "name": "Scratch Detection", "type": "range", "min": 0, "max": 1,
             "step": 0.1, "default": 0.5},
            {"id": "no_defect_repair", "name": "Disable Defect Repair", "type": "checkbox", "default": False},
            {"id": "remove_watermark", "name": "Remove Watermark", "type": "checkbox", "default": False},
            {"id": "watermark_auto_detect", "name": "Auto-Detect Watermark", "type": "checkbox", "default": True,
             "depends_on": "remove_watermark"},
            {"id": "remove_subtitles", "name": "Remove Burnt-in Subtitles", "type": "checkbox", "default": False},
            {"id": "subtitle_region", "name": "Subtitle Region", "type": "select",
             "options": ["bottom_third", "bottom_quarter", "top_quarter", "full_frame"],
             "default": "bottom_third", "depends_on": "remove_subtitles"},
        ]
    },
    "audio": {
        "title": "Audio",
        "options": [
            {"id": "audio_enhance", "name": "Enhance Audio", "type": "checkbox", "default": False},
            {"id": "fix_sync", "name": "Fix A/V Sync", "type": "checkbox", "default": False,
             "hint": "Repair audio-video drift"},
        ]
    },
    "advanced": {
        "title": "Advanced",
        "options": [
            {"id": "scene_aware", "name": "Scene-Aware Processing", "type": "checkbox", "default": False,
             "hint": "Adjust intensity per scene"},
            {"id": "motion_adaptive", "name": "Motion-Adaptive Denoise", "type": "checkbox", "default": False},
            {"id": "temporal_method", "name": "Temporal Consistency", "type": "select",
             "options": ["optical_flow", "cross_attention", "hybrid"], "default": "optical_flow"},
            {"id": "qp_artifact_removal", "name": "Codec Artifact Removal", "type": "checkbox", "default": False},
            {"id": "qp_strength", "name": "Artifact Removal Strength", "type": "range", "min": 0, "max": 2,
             "step": 0.1, "default": 1.0, "depends_on": "qp_artifact_removal"},
            {"id": "expand_hdr", "name": "Expand to HDR", "type": "select",
             "options": ["none", "hdr10", "dolby-vision"], "default": "none"},
            {"id": "fix_aspect", "name": "Fix Aspect Ratio", "type": "select",
             "options": ["none", "auto", "4:3", "16:9"], "default": "none"},
            {"id": "ivtc", "name": "Inverse Telecine", "type": "select",
             "options": ["none", "auto", "3:2", "2:3"], "default": "none",
             "hint": "Recover original film frames from telecined video"},
            {"id": "perceptual", "name": "Perceptual Mode", "type": "select",
             "options": ["faithful", "balanced", "enhanced"], "default": "balanced"},
        ]
    },
}


def get_system_info():
    info = {
        "cpu_percent": 0.0, "ram_used_gb": 0.0, "ram_total_gb": 0.0,
        "vram_used_gb": 0.0, "vram_total_gb": 0.0, "gpu_name": "N/A", "gpu_temp": 0,
        "platform": platform.system(), "python_version": platform.python_version(),
        "model_dir": str(DEFAULT_MODEL_DIR),
    }
    try:
        import psutil
        info["cpu_percent"] = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        info["ram_used_gb"] = round(mem.used / (1024**3), 2)
        info["ram_total_gb"] = round(mem.total / (1024**3), 2)
    except: pass
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_used_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 2)
            info["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    except: pass
    return info


def check_requirements():
    reqs = {
        "python": {"installed": True, "version": platform.python_version()},
        "torch": {"installed": False, "version": None, "cuda": False},
        "opencv": {"installed": False}, "ffmpeg": {"installed": False},
        "yt_dlp": {"installed": False}, "realesrgan": {"installed": False},
    }
    try:
        import torch
        reqs["torch"] = {"installed": True, "version": torch.__version__, "cuda": torch.cuda.is_available()}
        if torch.cuda.is_available():
            reqs["torch"]["cuda_version"] = torch.version.cuda
    except: pass
    try:
        import cv2
        reqs["opencv"] = {"installed": True, "version": cv2.__version__}
    except: pass
    if shutil.which("ffmpeg"):
        reqs["ffmpeg"] = {"installed": True, "path": shutil.which("ffmpeg")}
    try:
        import yt_dlp
        reqs["yt_dlp"] = {"installed": True, "version": yt_dlp.version.__version__}
    except: pass
    try:
        from realesrgan import RealESRGANer
        reqs["realesrgan"] = {"installed": True}
    except: pass
    return reqs


def get_models_status():
    status = []
    for m in MODELS:
        path = DEFAULT_MODEL_DIR / f"{m['id']}.pth"
        status.append({**m, "downloaded": path.exists(), "path": str(path) if path.exists() else None})
    return status


def download_model(model_id):
    model = next((m for m in MODELS if m["id"] == model_id), None)
    if not model:
        return {"error": "Model not found"}
    path = DEFAULT_MODEL_DIR / f"{model_id}.pth"
    if path.exists():
        return {"success": True, "message": "Already downloaded"}
    try:
        print(f"Downloading {model['name']}...")
        urllib.request.urlretrieve(model["url"], path)
        return {"success": True, "message": f"Downloaded {model['name']}"}
    except Exception as e:
        return {"error": str(e)}


def download_all_models():
    results = []
    for m in MODELS:
        result = download_model(m["id"])
        results.append({"id": m["id"], "name": m["name"], **result})
    return results


def download_youtube(url, output_dir=None):
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
            if not Path(filename).exists():
                filename = filename.rsplit('.', 1)[0] + '.mp4'
            return {"success": True, "path": filename, "title": info.get("title", "Unknown")}
    except ImportError:
        return {"error": "yt-dlp not installed"}
    except Exception as e:
        return {"error": str(e)}


def list_directory(path):
    try:
        p = Path(path)
        if not p.exists():
            return {"error": "Path does not exist"}
        items = []
        for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            try:
                items.append({
                    "name": item.name, "path": str(item), "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0,
                    "ext": item.suffix.lower() if item.is_file() else "",
                })
            except: continue
        return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "items": items}
    except Exception as e:
        return {"error": str(e)}


def get_drives():
    if platform.system() == "Windows":
        import string
        return [{"name": f"{l}:", "path": f"{l}:\\"} for l in string.ascii_uppercase if os.path.exists(f"{l}:\\")]
    return [{"name": "/", "path": "/"}]


def submit_job(data):
    job_id = str(uuid.uuid4())[:8]
    job = {
        "id": job_id, "input_path": data.get("input_path", ""),
        "output_path": data.get("output_path", ""), "options": data.get("options", {}),
        "status": "pending", "progress": 0, "log": [],
        "created_at": datetime.now().isoformat(),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    threading.Thread(target=run_job, args=(job_id,), daemon=True).start()
    return job


def run_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["started_at"] = datetime.now().isoformat()

    opts = job.get("options", {})
    cmd = ["python", "-m", "framewright.cli", "restore", "--input", job["input_path"], "--output", job["output_path"]]

    # Map options to CLI args
    opt_map = {
        "scale": "--scale", "quality": "--quality", "format": "--format", "model": "--model",
        "grain_reduction": "--grain-reduction", "scratch_sensitivity": "--scratch-sensitivity",
        "target_fps": "--target-fps", "rife_model": "--rife-model", "tap_model": "--tap-model",
        "tap_strength": "--tap-strength", "face_model": "--face-model", "aesrgan_strength": "--aesrgan-strength",
        "colorize_model": "--colorize-model", "subtitle_region": "--subtitle-region",
        "dedup_threshold": "--dedup-threshold", "diffusion_steps": "--diffusion-steps",
        "diffusion_guidance": "--diffusion-guidance", "temporal_method": "--temporal-method",
        "qp_strength": "--qp-strength", "expand_hdr": "--expand-hdr", "fix_aspect": "--fix-aspect",
        "reference_dir": "--reference-dir", "reference_strength": "--reference-strength",
        "reference_guidance": "--reference-guidance", "reference_ip_scale": "--reference-ip-scale",
        "ivtc": "--ivtc", "perceptual": "--perceptual", "frame_gen_model": "--frame-gen-model",
        "max_gap_frames": "--max-gap-frames",
    }
    flag_map = {
        "tap_denoise": "--tap-denoise", "tap_preserve_grain": "--tap-preserve-grain",
        "deduplicate": "--deduplicate", "enable_rife": "--enable-rife", "colorize": "--colorize",
        "colorize_temporal_fusion": "--colorize-temporal-fusion", "auto_enhance": "--auto-enhance",
        "no_face_restore": "--no-face-restore", "no_defect_repair": "--no-defect-repair",
        "remove_watermark": "--remove-watermark", "watermark_auto_detect": "--watermark-auto-detect",
        "remove_subtitles": "--remove-subtitles", "audio_enhance": "--audio-enhance",
        "fix_sync": "--fix-sync", "scene_aware": "--scene-aware", "motion_adaptive": "--motion-adaptive",
        "qp_artifact_removal": "--qp-artifact-removal", "diffusion_sr": "--diffusion-sr",
        "generate_frames": "--generate-frames",
        "reference_enhance": "--reference-enhance",
    }

    for key, arg in opt_map.items():
        if key in opts and opts[key] and opts[key] != "none":
            cmd.extend([arg, str(opts[key])])

    for key, arg in flag_map.items():
        if opts.get(key):
            cmd.append(arg)

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   cwd=str(Path(__file__).parent.parent))
        for line in process.stdout:
            with JOBS_LOCK:
                JOBS[job_id]["log"].append(line.strip())
            if "%" in line:
                try:
                    for part in line.split():
                        if "%" in part:
                            JOBS[job_id]["progress"] = float(part.replace("%", ""))
                            break
                except: pass
        process.wait()
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "completed" if process.returncode == 0 else "failed"
            JOBS[job_id]["progress"] = 100 if process.returncode == 0 else JOBS[job_id]["progress"]
            JOBS[job_id]["completed_at"] = datetime.now().isoformat()
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)


# HTML Dashboard
DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FrameWright - Full Restoration Suite</title>
<style>
:root{--bg:#000;--bg2:#1c1c1e;--bg3:#2c2c2e;--bg4:#3a3a3c;--text:#fff;--text2:#8e8e93;--accent:#0a84ff;--success:#30d158;--warn:#ff9f0a;--error:#ff453a;--border:rgba(255,255,255,.1);--r:12px}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'SF Pro',sans-serif;background:var(--bg);color:var(--text);line-height:1.5}
header{background:rgba(28,28,30,.9);backdrop-filter:blur(20px);padding:16px 24px;position:sticky;top:0;z-index:100;border-bottom:1px solid var(--border)}
h1{font-size:22px;font-weight:600}h1 span{color:var(--accent)}
nav{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
nav button{background:var(--bg3);border:none;color:var(--text);padding:8px 16px;border-radius:20px;font-size:13px;cursor:pointer;transition:.2s}
nav button:hover{background:var(--bg4)}nav button.active{background:var(--accent)}
main{max-width:1400px;margin:0 auto;padding:24px}
.card{background:var(--bg2);border-radius:var(--r);padding:20px;margin-bottom:16px}
.card h2{font-size:12px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.grid-3{grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}
.form-row{display:flex;gap:12px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
.form-group{margin-bottom:12px;flex:1;min-width:200px}
.form-group label{display:block;font-size:12px;color:var(--text2);margin-bottom:6px}
.form-group.full{flex:100%}
input,select,textarea{width:100%;padding:10px 14px;background:var(--bg3);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:14px;outline:none}
input:focus,select:focus{border-color:var(--accent)}
input[type=checkbox]{width:auto;margin-right:8px}
input[type=range]{-webkit-appearance:none;height:4px;background:var(--bg4);border-radius:2px}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;background:var(--accent);border-radius:50%;cursor:pointer}
.btn{display:inline-flex;align-items:center;gap:8px;padding:10px 20px;border:none;border-radius:8px;font-size:14px;font-weight:500;cursor:pointer;transition:.2s}
.btn-primary{background:var(--accent);color:#fff}.btn-primary:hover{background:#409cff}
.btn-secondary{background:var(--bg3);color:var(--text)}.btn-secondary:hover{background:var(--bg4)}
.btn-success{background:var(--success);color:#000}
.btn-sm{padding:6px 12px;font-size:12px}
.hint{font-size:11px;color:var(--text2);margin-top:4px}
.file-browser{background:var(--bg3);border-radius:8px;max-height:200px;overflow-y:auto}
.file-item{display:flex;align-items:center;gap:10px;padding:8px 12px;cursor:pointer;border-bottom:1px solid var(--border);font-size:13px}
.file-item:hover{background:var(--bg4)}.file-item:last-child{border:none}
.drives{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap}
.drive-btn{padding:4px 10px;background:var(--bg3);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:12px;cursor:pointer}
.path-display{background:var(--bg);padding:8px 12px;border-radius:6px;font-size:12px;color:var(--text2);margin-bottom:10px;word-break:break-all}
.tab{display:none}.tab.active{display:block}
.preset-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.preset-card{background:var(--bg3);border:2px solid transparent;border-radius:8px;padding:12px;cursor:pointer;transition:.2s}
.preset-card:hover{border-color:var(--border)}.preset-card.selected{border-color:var(--accent);background:rgba(10,132,255,.1)}
.preset-name{font-weight:600;font-size:13px}.preset-desc{font-size:11px;color:var(--text2)}
.section{border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:16px}
.section-title{font-size:13px;font-weight:600;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.section-title input{margin:0}
.option-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.job-item{background:var(--bg3);border-radius:8px;padding:14px;margin-bottom:10px}
.job-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.status{padding:3px 8px;border-radius:10px;font-size:11px;font-weight:500}
.status.pending{background:var(--bg4)}.status.running{background:var(--accent)}.status.completed{background:var(--success);color:#000}.status.failed{background:var(--error)}
.progress{width:100%;height:4px;background:var(--bg);border-radius:2px;overflow:hidden;margin-top:8px}
.progress-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--success));transition:.3s}
.model-item{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:var(--bg3);border-radius:8px;margin-bottom:8px}
.model-info h3{font-size:14px;margin-bottom:2px}.model-info p{font-size:12px;color:var(--text2)}
.badge{padding:3px 8px;border-radius:10px;font-size:11px}.badge-success{background:var(--success);color:#000}.badge-warn{background:var(--warn);color:#000}
.req-item{display:flex;justify-content:space-between;padding:10px;background:var(--bg3);border-radius:6px;margin-bottom:6px}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-left:8px}
.dot-green{background:var(--success)}.dot-yellow{background:var(--warn)}.dot-red{background:var(--error)}
.toggle{display:flex;align-items:center;gap:8px;cursor:pointer;font-size:13px}
.metric{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--border)}
.metric:last-child{border:none}
.actions{display:flex;gap:12px;margin-top:20px;flex-wrap:wrap}
</style>
</head>
<body>
<header>
<h1>Frame<span>Wright</span> <span style="font-size:12px;color:var(--text2)">Full Suite</span></h1>
<nav>
<button class="active" onclick="showTab('restore')">Restore</button>
<button onclick="showTab('jobs')">Jobs</button>
<button onclick="showTab('models')">Models</button>
<button onclick="showTab('setup')">Setup</button>
<button onclick="showTab('system')">System</button>
</nav>
</header>
<main>
<!-- RESTORE TAB -->
<div id="tab-restore" class="tab active">
<div class="grid">
<div class="card">
<h2>Input Source</h2>
<div style="display:flex;gap:8px;margin-bottom:12px">
<button class="btn btn-sm btn-secondary active" onclick="setSource('file',this)">Local File</button>
<button class="btn btn-sm btn-secondary" onclick="setSource('youtube',this)">YouTube</button>
</div>
<div id="src-file">
<div class="drives" id="in-drives"></div>
<div class="path-display" id="in-path">Select a video file...</div>
<div class="file-browser" id="in-browser"></div>
</div>
<div id="src-youtube" style="display:none">
<div class="form-group">
<label>YouTube URL</label>
<div style="display:flex;gap:8px"><input type="url" id="yt-url" placeholder="https://youtube.com/watch?v=..."><button class="btn btn-sm btn-secondary" onclick="downloadYT()">Download</button></div>
</div>
<div id="yt-status"></div>
</div>
<input type="hidden" id="input-path">
</div>
<div class="card">
<h2>Output Destination</h2>
<div class="drives" id="out-drives"></div>
<div class="path-display" id="out-path">Select output folder...</div>
<div class="file-browser" id="out-browser"></div>
<div class="form-group" style="margin-top:12px">
<label>Output Filename</label>
<input type="text" id="out-filename" placeholder="restored_video.mkv">
</div>
<input type="hidden" id="output-folder">
</div>
</div>

<div class="card">
<h2>Preset</h2>
<div class="preset-grid" id="presets"></div>
</div>

<div class="grid grid-3" id="options-container"></div>

<div class="actions">
<button class="btn btn-primary" onclick="submitJob()">Start Restoration</button>
<button class="btn btn-secondary" onclick="dryRun()">Dry Run (Preview)</button>
</div>
</div>

<!-- JOBS TAB -->
<div id="tab-jobs" class="tab">
<div class="card"><h2>Jobs</h2><div id="jobs-list"><p style="color:var(--text2)">No jobs yet</p></div></div>
</div>

<!-- MODELS TAB -->
<div id="tab-models" class="tab">
<div class="card">
<h2>Model Directory</h2>
<div class="form-group"><input type="text" id="model-dir" readonly></div>
<button class="btn btn-success" onclick="downloadAllModels()">Download All Models</button>
<p class="hint" style="margin-top:8px">Downloads all required models (~1.5 GB total)</p>
</div>
<div class="card"><h2>Available Models</h2><div id="models-list"></div></div>
</div>

<!-- SETUP TAB -->
<div id="tab-setup" class="tab">
<div class="card"><h2>Requirements</h2><div id="reqs"></div></div>
<div class="card">
<h2>Installation Commands</h2>
<div class="form-group"><label>PyTorch with CUDA</label><input readonly value="pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121" onclick="this.select()"></div>
<div class="form-group"><label>Real-ESRGAN</label><input readonly value="pip install realesrgan basicsr facexlib gfpgan" onclick="this.select()"></div>
<div class="form-group"><label>yt-dlp</label><input readonly value="pip install yt-dlp" onclick="this.select()"></div>
<div class="form-group"><label>FFmpeg (Windows)</label><input readonly value="winget install ffmpeg" onclick="this.select()"></div>
</div>
</div>

<!-- SYSTEM TAB -->
<div id="tab-system" class="tab">
<div class="grid">
<div class="card"><h2>System</h2>
<div class="metric"><span>Platform</span><span id="sys-platform">--</span></div>
<div class="metric"><span>Python</span><span id="sys-python">--</span></div>
<div class="metric"><span>CPU</span><span id="sys-cpu">--%</span></div>
<div class="metric"><span>RAM</span><span id="sys-ram">-- / -- GB</span></div>
</div>
<div class="card"><h2>GPU</h2>
<div class="metric"><span>Name</span><span id="sys-gpu">--</span></div>
<div class="metric"><span>VRAM</span><span id="sys-vram">-- / -- GB</span></div>
<div class="metric"><span>Temp</span><span id="sys-temp">--°C</span></div>
</div>
</div>
</div>
</main>

<script>
let inputPath='',outputFolder='',selectedPreset='balanced',options={},allOptions={},presets={};

function showTab(name){
document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));
document.getElementById('tab-'+name).classList.add('active');
event.target.classList.add('active');
if(name==='jobs')updateJobs();if(name==='models')loadModels();if(name==='setup')loadReqs();if(name==='system')updateSystem();
}

function setSource(type,btn){
document.querySelectorAll('#tab-restore .card:first-child .btn').forEach(b=>b.classList.remove('active'));
btn.classList.add('active');
document.getElementById('src-file').style.display=type==='file'?'block':'none';
document.getElementById('src-youtube').style.display=type==='youtube'?'block':'none';
}

async function loadDrives(){
const r=await fetch('/api/drives');const d=await r.json();
['in','out'].forEach(p=>{
document.getElementById(p+'-drives').innerHTML=d.map(dr=>`<button class="drive-btn" onclick="browse('${p}','${dr.path.replace(/\\\\/g,'\\\\\\\\')}')">${dr.name}</button>`).join('');
});
if(d.length)browse('in',d[0].path);
if(d.length)browse('out',d[0].path);
}

async function browse(target,path){
const r=await fetch('/api/browse?path='+encodeURIComponent(path));const d=await r.json();
if(d.error){alert(d.error);return;}
document.getElementById(target+'-path').textContent=d.path;
const browser=document.getElementById(target+'-browser');
let html='';
if(d.parent)html+=`<div class="file-item" onclick="browse('${target}','${d.parent.replace(/\\\\/g,'\\\\\\\\')}')">📁 ..</div>`;
d.items.forEach(i=>{
const esc=i.path.replace(/\\\\/g,'\\\\\\\\');
if(i.is_dir){
html+=`<div class="file-item" onclick="browse('${target}','${esc}')">📁 ${i.name}</div>`;
}else if(target==='in'&&['.mp4','.mkv','.avi','.mov','.webm','.wmv','.flv'].includes(i.ext)){
html+=`<div class="file-item" onclick="selectInput('${esc}','${i.name}')">🎬 ${i.name} <span style="margin-left:auto;color:var(--text2)">${fmtSize(i.size)}</span></div>`;
}
});
if(target==='out'){
document.getElementById('output-folder').value=d.path;
outputFolder=d.path;
}
browser.innerHTML=html||'<div style="padding:12px;color:var(--text2)">No items</div>';
}

function selectInput(path,name){
inputPath=path;
document.getElementById('input-path').value=path;
document.getElementById('in-path').textContent='✓ '+path;
document.getElementById('in-path').style.color='var(--success)';
const base=name.substring(0,name.lastIndexOf('.'));
document.getElementById('out-filename').value=base+'_restored.mkv';
}

function fmtSize(b){if(b<1024)return b+' B';if(b<1024*1024)return(b/1024).toFixed(1)+' KB';if(b<1024*1024*1024)return(b/1024/1024).toFixed(1)+' MB';return(b/1024/1024/1024).toFixed(2)+' GB';}

async function downloadYT(){
const url=document.getElementById('yt-url').value;if(!url){alert('Enter URL');return;}
document.getElementById('yt-status').innerHTML='<p style="color:var(--accent)">Downloading...</p>';
const r=await fetch('/api/youtube',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})});
const d=await r.json();
if(d.error){document.getElementById('yt-status').innerHTML=`<p style="color:var(--error)">${d.error}</p>`;}
else{
document.getElementById('yt-status').innerHTML=`<p style="color:var(--success)">Downloaded: ${d.title}</p>`;
selectInput(d.path,d.title+'.mp4');
}}

async function loadPresets(){
const r=await fetch('/api/presets');presets=await r.json();
const html=Object.entries(presets).map(([k,p])=>`
<div class="preset-card ${k===selectedPreset?'selected':''}" onclick="selectPreset('${k}',this)">
<div class="preset-name">${p.name}</div><div class="preset-desc">${p.description}</div>
</div>`).join('');
document.getElementById('presets').innerHTML=html;
}

function selectPreset(key,el){
document.querySelectorAll('.preset-card').forEach(c=>c.classList.remove('selected'));
el.classList.add('selected');selectedPreset=key;
const p=presets[key];
Object.keys(p).forEach(k=>{if(options[k]!==undefined)options[k]=p[k];});
renderOptions();
}

async function loadOptions(){
const r=await fetch('/api/options');allOptions=await r.json();
renderOptions();
}

function renderOptions(){
const container=document.getElementById('options-container');
let html='';
Object.entries(allOptions).forEach(([cat,data])=>{
html+=`<div class="card"><h2>${data.title}</h2><div class="option-grid">`;
data.options.forEach(opt=>{
const val=options[opt.id]!==undefined?options[opt.id]:opt.default;
const disabled=opt.depends_on&&!options[opt.depends_on]?'disabled':'';
if(opt.type==='checkbox'){
html+=`<label class="toggle"><input type="checkbox" ${val?'checked':''} ${disabled} onchange="setOpt('${opt.id}',this.checked)">${opt.name}</label>`;
}else if(opt.type==='select'){
html+=`<div class="form-group"><label>${opt.name}</label><select ${disabled} onchange="setOpt('${opt.id}',this.value)">${opt.options.map(o=>`<option value="${o}" ${val===o?'selected':''}>${o}</option>`).join('')}</select></div>`;
}else if(opt.type==='range'){
html+=`<div class="form-group"><label>${opt.name}: <span id="val-${opt.id}">${val}</span></label><input type="range" min="${opt.min}" max="${opt.max}" step="${opt.step||0.1}" value="${val}" ${disabled} oninput="setOpt('${opt.id}',parseFloat(this.value));document.getElementById('val-${opt.id}').textContent=this.value"></div>`;
}else if(opt.type==='number'){
html+=`<div class="form-group"><label>${opt.name}</label><input type="number" min="${opt.min}" max="${opt.max}" value="${val}" ${disabled} onchange="setOpt('${opt.id}',parseFloat(this.value))"></div>`;
}
if(opt.hint)html+=`<p class="hint" style="grid-column:1/-1">${opt.hint}</p>`;
});
html+=`</div></div>`;
});
container.innerHTML=html;
}

function setOpt(key,val){options[key]=val;renderOptions();}

async function submitJob(){
if(!inputPath){alert('Select input video');return;}
const outFile=document.getElementById('out-filename').value||'restored.mkv';
const outPath=outputFolder?outputFolder+'/'+outFile:outFile;
const data={input_path:inputPath,output_path:outPath,options:{...options,preset:selectedPreset}};
const r=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
const d=await r.json();
if(d.error)alert('Error: '+d.error);else{alert('Job started: '+d.id);showTab('jobs');document.querySelectorAll('nav button')[1].click();}
}

function dryRun(){options.dry_run=true;submitJob();}

async function updateJobs(){
const r=await fetch('/api/jobs');const jobs=await r.json();
if(!jobs.length){document.getElementById('jobs-list').innerHTML='<p style="color:var(--text2)">No jobs</p>';return;}
document.getElementById('jobs-list').innerHTML=jobs.map(j=>`
<div class="job-item">
<div class="job-header"><span style="font-weight:600">Job ${j.id}</span><span class="status ${j.status}">${j.status}</span></div>
<div style="font-size:12px;color:var(--text2)">${j.input_path}</div>
<div style="font-size:12px;color:var(--text2)">→ ${j.output_path}</div>
${j.status==='running'?`<div class="progress"><div class="progress-fill" style="width:${j.progress}%"></div></div><div style="font-size:11px;color:var(--text2)">${j.progress.toFixed(1)}%</div>`:''}
${j.error?`<div style="color:var(--error);font-size:12px">${j.error}</div>`:''}
</div>`).join('');
}

async function loadModels(){
const r=await fetch('/api/models');const models=await r.json();
const sys=await(await fetch('/api/system')).json();
document.getElementById('model-dir').value=sys.model_dir;
const cats={upscaling:[],face:[],interpolation:[],denoise:[],colorization:[]};
models.forEach(m=>cats[m.category]?.push(m));
let html='';
Object.entries(cats).forEach(([cat,list])=>{
if(!list.length)return;
html+=`<h3 style="font-size:13px;color:var(--text2);margin:16px 0 8px;text-transform:capitalize">${cat}</h3>`;
list.forEach(m=>{
html+=`<div class="model-item"><div class="model-info"><h3>${m.name}</h3><p>${m.size_mb} MB</p></div>
${m.downloaded?'<span class="badge badge-success">Downloaded</span>':`<button class="btn btn-sm btn-secondary" onclick="downloadModel('${m.id}',this)">Download</button>`}</div>`;
});
});
document.getElementById('models-list').innerHTML=html;
}

async function downloadModel(id,btn){
btn.textContent='Downloading...';btn.disabled=true;
const r=await fetch('/api/models/download',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_id:id})});
await r.json();loadModels();
}

async function downloadAllModels(){
if(!confirm('Download all models (~1.5 GB)?'))return;
const r=await fetch('/api/models/download-all',{method:'POST'});
const d=await r.json();
alert('Downloaded '+d.filter(x=>x.success).length+' models');
loadModels();
}

async function loadReqs(){
const r=await fetch('/api/requirements');const reqs=await r.json();
document.getElementById('reqs').innerHTML=Object.entries(reqs).map(([k,v])=>{
let dot='dot-red',info='Not installed';
if(v.installed){dot='dot-green';info=v.version||v.path||'OK';}
if(k==='torch'&&v.installed&&!v.cuda){dot='dot-yellow';info=v.version+' (CPU only)';}
if(k==='torch'&&v.cuda)info=v.version+' + CUDA '+v.cuda_version;
return `<div class="req-item"><span>${k}</span><span>${info}<span class="dot ${dot}"></span></span></div>`;
}).join('');
}

async function updateSystem(){
const r=await fetch('/api/system');const d=await r.json();
document.getElementById('sys-platform').textContent=d.platform;
document.getElementById('sys-python').textContent=d.python_version;
document.getElementById('sys-cpu').textContent=d.cpu_percent.toFixed(1)+'%';
document.getElementById('sys-ram').textContent=d.ram_used_gb+' / '+d.ram_total_gb+' GB';
document.getElementById('sys-gpu').textContent=d.gpu_name;
document.getElementById('sys-vram').textContent=d.vram_used_gb+' / '+d.vram_total_gb+' GB';
document.getElementById('sys-temp').textContent=d.gpu_temp+'°C';
}

// Init
loadDrives();loadPresets();loadOptions();
setInterval(()=>{if(document.getElementById('tab-jobs').classList.contains('active'))updateJobs();},3000);
</script>
</body>
</html>
'''


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        p = urlparse(self.path)
        q = parse_qs(p.query)
        if p.path in ['/', '/index.html']:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif p.path == '/api/system': self.send_json(get_system_info())
        elif p.path == '/api/drives': self.send_json(get_drives())
        elif p.path == '/api/browse': self.send_json(list_directory(q.get('path', [''])[0] or os.path.expanduser('~')))
        elif p.path == '/api/jobs':
            with JOBS_LOCK: self.send_json(list(JOBS.values()))
        elif p.path == '/api/presets': self.send_json(PRESETS)
        elif p.path == '/api/options': self.send_json(ALL_OPTIONS)
        elif p.path == '/api/models': self.send_json(get_models_status())
        elif p.path == '/api/requirements': self.send_json(check_requirements())
        else: self.send_response(404); self.end_headers()

    def do_POST(self):
        p = urlparse(self.path)
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        try: data = json.loads(body) if body else {}
        except: self.send_json({'error': 'Invalid JSON'}, 400); return

        if p.path == '/api/jobs':
            if not data.get('input_path'): self.send_json({'error': 'input_path required'}, 400); return
            self.send_json(submit_job(data), 201)
        elif p.path == '/api/youtube':
            self.send_json(download_youtube(data.get('url', '')))
        elif p.path == '/api/models/download':
            self.send_json(download_model(data.get('model_id', '')))
        elif p.path == '/api/models/download-all':
            self.send_json(download_all_models())
        else: self.send_response(404); self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    print("=" * 60)
    print("FrameWright Full Dashboard")
    print("=" * 60)
    print(f"URL: http://{HOST}:{PORT}")
    print("Opening browser...")
    print("=" * 60)
    threading.Thread(target=lambda: (time.sleep(1), webbrowser.open(f"http://{HOST}:{PORT}")), daemon=True).start()
    Server((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
