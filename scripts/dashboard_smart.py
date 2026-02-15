#!/usr/bin/env python3
"""FrameWright Smart Dashboard - Apple-inspired intuitive UI."""

import http.server
import socketserver
import json
import os
import platform
import re
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
MODEL_DIR = Path.home() / ".framewright" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

JOBS = {}
JOBS_LOCK = threading.Lock()

# Smart content profiles with explanations
CONTENT_PROFILES = {
    "auto": {
        "name": "Auto-Detect",
        "icon": "🔮",
        "description": "Let FrameWright analyze your video and choose the best settings automatically.",
        "tip": "Best for most users. We'll scan your video and optimize everything.",
    },
    "home_video": {
        "name": "Home Videos",
        "icon": "📹",
        "description": "Family recordings, camcorder footage, phone videos.",
        "tip": "Gentle enhancement that keeps the authentic look while improving clarity.",
        "settings": {"scale": 2, "denoise": True, "face_enhance": True, "grain": 0.2},
    },
    "old_film": {
        "name": "Old Film (Color)",
        "icon": "🎞️",
        "description": "Color film: 8mm, 16mm, vintage movies. Already has color — just needs restoration.",
        "tip": "Removes scratches, fixes fading, and cleans damage. Keeps original colors intact.",
        "settings": {"scale": 4, "denoise": True, "face_enhance": True, "scratch_repair": True, "grain": 0.3},
    },
    "vhs_tape": {
        "name": "VHS / Tape",
        "icon": "📼",
        "description": "VHS, Betamax, Hi8, or other tape recordings.",
        "tip": "Fixes tracking issues, color bleeding, and tape noise.",
        "settings": {"scale": 2, "denoise": True, "artifact_removal": True, "color_fix": True},
    },
    "anime": {
        "name": "Anime / Cartoon",
        "icon": "🎨",
        "description": "Animated content, cartoons, motion graphics.",
        "tip": "Uses special AI trained on animation for crisp lines and flat colors.",
        "settings": {"scale": 4, "model": "anime", "denoise": False, "face_enhance": False},
    },
    "bw_restore": {
        "name": "Black & White (Keep B&W)",
        "icon": "🎞️",
        "description": "Maximum restoration for heavily degraded B&W film. AI upscale, denoise, scratch repair, face restore — all tuned for best results.",
        "tip": "Full pipeline: dedup YouTube padding, denoise, 4x upscale, face restore, reference enhancement. Add reference photos for best detail recovery.",
        "settings": {"scale": 4, "denoise": True, "face_enhance": True, "scratch_repair": True},
    },
    "bw_colorize": {
        "name": "Black & White → Color",
        "icon": "🌈",
        "description": "Add color to B&W footage using AI colorization. Turns grayscale into full color.",
        "tip": "AI predicts and adds natural colors to black & white video. Also restores and upscales.",
        "settings": {"colorize": True, "scale": 2, "denoise": True},
    },
    "low_res": {
        "name": "Low Resolution",
        "icon": "🔍",
        "description": "DVD quality, old downloads, compressed videos.",
        "tip": "Maximum upscaling with artifact removal for the sharpest result.",
        "settings": {"scale": 4, "denoise": True, "artifact_removal": True},
    },
    "professional": {
        "name": "Professional",
        "icon": "🎬",
        "description": "High-quality source that needs subtle enhancement.",
        "tip": "Light touch-ups. Preserves original quality while enhancing details.",
        "settings": {"scale": 2, "denoise": False, "face_enhance": True, "quality": 14},
    },
}

# Quality levels explained simply
QUALITY_LEVELS = {
    "quick": {
        "name": "Quick Preview",
        "time": "~2x faster",
        "description": "See results fast. Good for testing settings.",
        "icon": "⚡",
    },
    "balanced": {
        "name": "Balanced",
        "time": "Normal speed",
        "description": "Great quality for everyday use. Recommended for most videos.",
        "icon": "⭐",
        "recommended": True,
    },
    "maximum": {
        "name": "Maximum Quality",
        "time": "~3x slower",
        "description": "Best possible results. Use for important videos you want to preserve forever.",
        "icon": "💎",
    },
}

# Enhancement options - "group": "simple" or "advanced"
ENHANCEMENTS = {
    "upscale": {
        "name": "Upscale Resolution", "group": "simple",
        "description": "Make your video bigger and sharper using AI super-resolution",
        "options": [
            {"value": "2x", "name": "2x (Double)", "tip": "Good for already decent sources (720p->1440p). Faster, less risk of artifacts"},
            {"value": "4x", "name": "4x (Quadruple)", "tip": "Best for low-res sources (480p->Full HD). Slower, more VRAM. Use for old footage"},
        ],
        "why": "AI upscaling adds real detail, not just stretching pixels. 2x is safer; 4x produces bigger jumps but needs good source material.",
    },
    "denoise": {
        "name": "Remove Noise & Grain", "group": "simple",
        "description": "Clean up grainy or noisy footage. Runs before upscaling for best results",
        "options": [
            {"value": "off", "name": "Keep Original", "tip": "Use for film you want to look authentic, or already-clean digital sources"},
            {"value": "light", "name": "Light", "tip": "Best for film — removes noise but keeps texture. Auto-selected when using heavy reference enhancement"},
            {"value": "medium", "name": "Medium", "tip": "Good default for most old video. Balances cleanup vs detail loss"},
            {"value": "strong", "name": "Strong", "tip": "Use for very noisy VHS/analog sources. May soften fine detail. Auto-reduced to Light if reference enhancement is Heavy"},
        ],
        "why": "Runs before upscaling so noise doesn't get amplified. When using reference photos at Heavy strength, denoise auto-caps at Light — the reference step handles detail.",
    },
    "face": {
        "name": "Enhance Faces", "group": "simple",
        "description": "AI reconstructs facial details lost to low resolution or compression",
        "options": [
            {"value": "off", "name": "Off", "tip": "Use when there are no people, or faces look fine already"},
            {"value": "subtle", "name": "Subtle", "tip": "Slight sharpening, preserves original look. Best for documentaries and close-ups"},
            {"value": "full", "name": "Full Restore", "tip": "Rebuilds face details aggressively. Best for low-res or distant faces. May look 'too perfect'"},
        ],
        "why": "Subtle is safer and more natural. Full Restore works wonders on tiny/blurry faces but can over-smooth skin texture on close-ups.",
    },
    "framerate": {
        "name": "Smooth Motion", "group": "simple",
        "description": "AI generates new in-between frames for smoother playback",
        "options": [
            {"value": "off", "name": "Keep Original", "tip": "Use for film (preserves cinematic feel) or when original framerate is fine"},
            {"value": "30", "name": "30 FPS", "tip": "Good upgrade for old 15-24fps video. Natural-looking motion for most content"},
            {"value": "60", "name": "60 FPS", "tip": "Ultra-smooth, best for sports/action. Can look 'soap opera effect' on film"},
        ],
        "why": "Higher FPS = smoother motion, but film purists prefer original rates. 60fps can make cinematic footage look like a TV show.",
    },
    "color": {
        "name": "Fix Colors", "group": "simple",
        "description": "Correct faded or shifted colors from aging tapes and film",
        "options": [
            {"value": "off", "name": "Keep Original", "tip": "Use when colors look correct, or you want the vintage look"},
            {"value": "auto", "name": "Auto Correct", "tip": "Fixes white balance and faded colors. Best for most old footage"},
            {"value": "vivid", "name": "Vivid", "tip": "Boosts saturation beyond original. Good for washed-out VHS, but can look unnatural"},
        ],
        "why": "Auto is the safe choice for most footage. Vivid makes colors pop but may look oversaturated on already-good sources.",
        "hide_for": ["bw_colorize", "bw_restore"],
    },
    "colorize": {
        "name": "AI Colorization", "group": "simple",
        "description": "Add color to black & white footage using AI",
        "options": [
            {"value": "off", "name": "Keep B&W", "tip": "Preserve the original black & white look"},
            {"value": "auto", "name": "Auto Colorize", "tip": "AI predicts colors from context — works well for landscapes and people, less for unusual subjects"},
            {"value": "guided", "name": "Guided", "tip": "Provide reference images for historically accurate colors. More work, better results"},
        ],
        "why": "Auto colorization is impressive but not always historically accurate. Guided mode lets you control the palette with reference images.",
        "only_for": ["bw_colorize"],
    },
    "dedup": {
        "name": "Remove Duplicate Frames", "group": "simple",
        "description": "Detect and remove repeated frames that cause stuttery playback",
        "options": [
            {"value": "off", "name": "Off", "tip": "Use when playback is already smooth, or framerate is intentionally low"},
            {"value": "exact", "name": "Exact", "tip": "Safest — only removes pixel-identical frames. Use when unsure"},
            {"value": "smart", "name": "Smart (mpdecimate)", "tip": "Uses FFmpeg's mpdecimate filter — catches YouTube/telecine padding and blended duplicates. Best for old film re-encoded to higher FPS"},
        ],
        "why": "Exact is safe but misses YouTube-interpolated frames. Smart uses FFmpeg's mpdecimate which detects blended/padded duplicates — essential when YouTube upscaled e.g. 16fps to 25fps.",
    },
    "dedup_threshold": {
        "name": "Deduplication Sensitivity", "group": "simple",
        "description": "How similar frames must be to count as duplicates (lower = more aggressive)",
        "type": "slider", "min": 90, "max": 100, "default": 98, "step": 1,
        "why": "Lower values (0.94-0.96) remove more near-duplicates but risk removing intentional motion. Higher (0.98+) is safer but may miss subtle duplicates. Start at 0.98, lower if duplicates remain."
    },
    "stabilize": {
        "name": "Stabilize Video", "group": "simple",
        "description": "Reduce camera shake and jitter from handheld or wobbly sources",
        "options": [
            {"value": "off", "name": "Off", "tip": "Use for tripod shots, or when the motion is intentional"},
            {"value": "on", "name": "Stabilize", "tip": "Smooths out shake. Slightly crops edges to compensate for motion. Best for handheld footage"},
        ],
        "why": "Stabilization crops the image slightly (5-10%) to compensate for movement. Only use if the source is actually shaky.",
    },
    "output_format": {
        "name": "Output Format", "group": "simple",
        "description": "Container format for the restored video file",
        "options": [
            {"value": "mkv", "name": "MKV", "tip": "Best for archiving — supports all codecs, subtitles, and chapters. May not play on some smart TVs"},
            {"value": "mp4", "name": "MP4", "tip": "Plays everywhere — phones, TVs, web browsers. Slightly less flexible than MKV"},
            {"value": "mov", "name": "MOV", "tip": "Use for Apple ecosystem or video editing in Final Cut Pro / DaVinci Resolve"},
        ],
        "why": "MKV for keeping everything, MP4 for sharing, MOV for editing. Content is identical — only the container differs.",
    },
    "deinterlace": {
        "name": "Deinterlace", "group": "advanced", "section": "Video Processing",
        "description": "Fix combing/horizontal line artifacts from old TV recordings and VHS captures",
        "options": [
            {"value": "off", "name": "Off", "tip": "Use for progressive sources (most digital video, YouTube, film scans)"},
            {"value": "auto", "name": "Auto Detect", "tip": "Safest choice — analyzes each frame and only deinterlaces when needed"},
            {"value": "yadif", "name": "YADIF", "tip": "Fast CPU filter. Good enough for most VHS/TV captures. Low resource use"},
            {"value": "bwdif", "name": "BWDIF", "tip": "Sharper than YADIF with better motion handling. Slightly slower. Best traditional option"},
            {"value": "nnedi", "name": "NNEDI", "tip": "Neural network reconstructs missing lines. Sharpest, fewest artifacts, but needs GPU and is 3-5x slower"},
        ],
        "why": "If you see horizontal lines or combing on motion, you need this. Auto Detect is safest. NNEDI gives best quality but is much slower.",
    },
    "scratch_repair": {
        "name": "Scratch & Damage Repair", "group": "advanced",
        "description": "AI detects and inpaints scratches, dust spots, and physical film damage",
        "options": [
            {"value": "off", "name": "Off", "tip": "Use for digital sources or already-clean film scans"},
            {"value": "light", "name": "Light", "tip": "Fixes obvious scratches and dust. Preserves fine detail. Best for moderate damage"},
            {"value": "full", "name": "Full Repair", "tip": "Aggressively removes all defects. Best for heavily damaged film. May inpaint real detail on rare occasions"},
        ],
        "why": "Light is safer and preserves texture. Full is for badly damaged film where visible scratches are worse than occasional over-correction.",
        "show_for": ["old_film", "bw_restore", "bw_colorize"],
    },
    "audio_enhance": {
        "name": "Audio Enhancement", "group": "advanced",
        "description": "Clean up hiss, hum, clicks, and background noise from old audio tracks",
        "options": [
            {"value": "off", "name": "Keep Original", "tip": "Use when audio is already clean, or you want to handle audio separately"},
            {"value": "traditional", "name": "Traditional", "tip": "FFmpeg EQ, compression, dehum. Predictable results, no AI model needed. Lightest option"},
            {"value": "deepfilter", "name": "DeepFilter", "tip": "AI noise removal — best at preserving speech clarity while removing background noise"},
            {"value": "full", "name": "Full Restore", "tip": "AI + dereverb + declick + dehum. Best for very degraded audio (old film, damaged tape). Most processing"},
        ],
        "why": "DeepFilter is best for speech. Traditional is lighter and doesn't need AI models. Full Restore is for severely degraded audio.",
    },
    "encoder": {
        "name": "Video Encoder", "group": "advanced", "section": "Output & Encoding",
        "description": "Codec for encoding the output video",
        "options": [
            {"value": "h264", "name": "H.264", "tip": "Plays on everything — old phones, TVs, browsers. Larger files. Use when compatibility matters most"},
            {"value": "h265", "name": "H.265/HEVC", "tip": "~50% smaller files, same quality. Most modern devices support it. Best default for archiving"},
            {"value": "av1", "name": "AV1", "tip": "Best compression (~30% smaller than H.265) but encoding is 5-10x slower. Future-proof but impatient"},
        ],
        "why": "H.265 is the sweet spot for most users. H.264 if you need to play on old devices. AV1 if you have time to wait and want smallest files.",
    },
    "crf": {
        "name": "Encoding Quality (CRF)", "group": "advanced",
        "description": "Lower = better quality, larger file. Higher = smaller file, less quality",
        "type": "slider", "min": 0, "max": 51, "default": 18,
        "labels": {"0": "Lossless", "14": "Near Lossless", "18": "High (recommended)", "23": "Balanced", "28": "Preview", "51": "Worst"},
        "why": "CRF (Constant Rate Factor) controls the quality-vs-size tradeoff. 0 = lossless, 18 = visually lossless (recommended), 23 = good balance, 51 = minimum quality.",
    },
    "letterbox": {
        "name": "Letterbox / Black Bars", "group": "advanced",
        "description": "Detect and crop black bars around the video",
        "options": [
            {"value": "off", "name": "Keep As-Is", "tip": "Use when bars are intentional (widescreen film) or there are no black bars"},
            {"value": "auto", "name": "Auto Crop", "tip": "Detects and removes uneven or unnecessary black bars. Safe — won't crop actual content"},
        ],
        "why": "Old transfers often have uneven black bars from wrong aspect ratios. Auto crop cleans this up without losing any real content.",
    },
    "ivtc": {
        "name": "Inverse Telecine (IVTC)", "group": "advanced",
        "description": "Removes duplicate frames inserted when converting film (24fps) to TV video (30fps)",
        "options": [
            {"value": "off", "name": "Off", "tip": "Use for native digital video, or sources that aren't from film-to-TV transfers"},
            {"value": "auto", "name": "Auto Detect", "tip": "Safest — detects the pulldown pattern automatically. Use this if unsure"},
            {"value": "3:2", "name": "3:2 Pulldown", "tip": "For NTSC (US/Japan) film-to-video transfers. Most common pattern"},
            {"value": "2:2", "name": "2:2 Pulldown", "tip": "For PAL (Europe) film-to-video transfers"},
        ],
        "why": "If your source is film transferred to NTSC video, IVTC removes the jutter from repeated frames and restores smooth 24fps. Use Auto if unsure.",
    },
    "vhs_fixes": {
        "name": "VHS / Analog Fixes", "group": "advanced",
        "description": "Fix VHS-specific artifacts: tracking lines, dropout, color bleeding",
        "options": [
            {"value": "off", "name": "Off", "tip": "Use for non-VHS sources. These fixes can harm digital video"},
            {"value": "auto", "name": "Auto Fix", "tip": "Detects and fixes all VHS artifacts. Best for typical VHS captures"},
            {"value": "tracking", "name": "Tracking Only", "tip": "Only fixes horizontal tracking lines. Use when color is fine but you see line glitches"},
        ],
        "why": "Only use this on actual VHS/analog captures. The fixes target tape-specific problems and can cause issues on digital sources.",
        "show_for": ["vhs_tape"],
    },
    "subtitles": {
        "name": "Subtitle Handling", "group": "advanced", "section": "Cleanup & Extras",
        "description": "Handle burnt-in (hardcoded) subtitles in the video",
        "options": [
            {"value": "keep", "name": "Keep As-Is", "tip": "Use when there are no burnt-in subs, or you want to keep them"},
            {"value": "remove", "name": "Remove", "tip": "AI detects text and inpaints over it. Best for clean output. May leave faint traces on complex backgrounds"},
            {"value": "extract", "name": "Extract to File", "tip": "OCR the text to a .srt file, then remove. Best of both worlds — keep the text, clean the video"},
        ],
        "why": "Only works on hardcoded/burnt-in subtitles. Soft subs (separate tracks) are preserved automatically.",
    },
    "watermark": {
        "name": "Watermark Removal", "group": "advanced",
        "description": "Detect and remove station logos, watermarks, or on-screen bugs",
        "options": [
            {"value": "off", "name": "Off", "tip": "Use when there's no watermark, or you want to keep it"},
            {"value": "auto", "name": "Auto Detect", "tip": "AI finds and inpaints over persistent logos. Works well on static watermarks, less on animated ones"},
        ],
        "why": "Best for TV recordings with channel logos. May struggle with large or semi-transparent watermarks.",
    },
    "preview_mode": {
        "name": "Preview First", "group": "advanced",
        "description": "Process a short clip first to check your settings before committing to the full video",
        "options": [
            {"value": "off", "name": "Full Video", "tip": "Process everything. Only use when you're confident in your settings"},
            {"value": "5", "name": "5 Seconds", "tip": "Quick sanity check — enough to see if upscaling and denoising look right"},
            {"value": "10", "name": "10 Seconds", "tip": "Good for checking faces and motion. Recommended for first-time settings"},
            {"value": "30", "name": "30 Seconds", "tip": "Longer test to check temporal consistency and scene transitions"},
        ],
        "why": "A full video can take hours. Always preview first when trying new settings. 10 seconds is usually enough to catch problems.",
    },
    "sr_model": {
        "name": "Upscale Model", "group": "advanced", "section": "AI Models",
        "description": "Which AI architecture to use for upscaling. Each has different speed/quality tradeoffs",
        "options": [
            {"value": "realesrgan", "name": "Real-ESRGAN", "tip": "Best default — fast, reliable, handles most content well. ~2 sec/frame at 4x"},
            {"value": "hat", "name": "HAT", "tip": "Hybrid Attention Transformer — finer detail, 2-3x slower, ~8 GB VRAM at 720p"},
            {"value": "ensemble", "name": "Ensemble", "tip": "Combines HAT + Real-ESRGAN for best quality. 3-5x slower"},
            {"value": "diffusion", "name": "Diffusion", "tip": "Generates realistic detail via diffusion. 10-100x slower. 2-10 sec/frame"},
        ],
        "why": "Start with Real-ESRGAN. Only switch to HAT or Ensemble if you see artifacts or want more detail and can afford the extra time.",
    },
    "face_model": {
        "name": "Face Model", "group": "advanced",
        "description": "Which AI model to use for enhancing faces. Only matters when Face Enhancement is on",
        "options": [
            {"value": "aesrgan", "name": "AESRGAN", "tip": "Lightest touch, most natural result. Best when faces are not too degraded"},
            {"value": "codeformer", "name": "CodeFormer", "tip": "Best at preserving who the person looks like. Safer for known faces. Good balance"},
            {"value": "gfpgan", "name": "GFPGAN", "tip": "Most aggressive restoration — great on very blurry/tiny faces. Can sometimes alter facial features"},
        ],
        "why": "CodeFormer for recognizable people (preserves identity). GFPGAN for severely degraded/tiny faces. AESRGAN for subtle, natural enhancement.",
    },
    "ref_strength": {
        "name": "Reference Enhancement Strength", "group": "advanced",
        "description": "How aggressively reference photos influence restoration. Higher = more detail from references, less from original",
        "options": [
            {"value": "light", "name": "Light", "tip": "Subtle influence — preserves original structure, adds only slight texture hints from references. Best for decent-quality sources"},
            {"value": "medium", "name": "Medium", "tip": "Balanced — transfers moderate detail from references while keeping the original recognizable. Good default"},
            {"value": "heavy", "name": "Heavy", "tip": "Aggressive — fills in lots of detail from references. Best for heavily degraded film where the original has little usable detail"},
        ],
        "why": "Light for good sources, Medium for moderate damage, Heavy for severely degraded footage (early 1900s film, heavy noise/blur). Only applies when Reference Photos are provided.",
    },
    "temporal_method": {
        "name": "Temporal Consistency", "group": "advanced",
        "description": "Prevents frame-to-frame flickering caused by AI processing each frame independently",
        "options": [
            {"value": "off", "name": "Off", "tip": "Fastest — fine for still shots or if you don't notice flickering. Try this first"},
            {"value": "optical_flow", "name": "Optical Flow", "tip": "Uses motion estimation to keep frames consistent. Good balance of speed and quality"},
            {"value": "cross_attention", "name": "Cross-Attention", "tip": "AI looks at neighboring frames. Better than optical flow for complex scenes. Needs more VRAM"},
            {"value": "hybrid", "name": "Hybrid (RAFT + Attention)", "tip": "Best quality: combines optical flow and AI attention. Recommended for archival/historical film. Slower but superior flicker reduction."},
            {"value": "raft", "name": "RAFT", "tip": "State-of-the-art flow estimation. Best quality but GPU-intensive (~2x slower). Use for important projects"},
        ],
        "why": "Only enable if you see flickering in your output. Start with Off, try Optical Flow if flickering appears, upgrade to RAFT for best results.",
    },
    "hdr": {
        "name": "HDR Expansion", "group": "advanced",
        "description": "Convert standard video to HDR for TVs and monitors that support it",
        "options": [
            {"value": "off", "name": "Off (SDR)", "tip": "Keep standard range. Use when your display doesn't support HDR, or for web sharing"},
            {"value": "hdr10", "name": "HDR10", "tip": "Most widely supported HDR format. Works on most modern TVs and monitors"},
            {"value": "hlg", "name": "HLG", "tip": "Looks acceptable on both HDR and non-HDR displays. Best when you don't know the target screen"},
        ],
        "why": "Only useful if you'll watch on an HDR display. HLG is safest since it degrades gracefully on non-HDR screens.",
    },
    "perceptual": {
        "name": "Perceptual Tuning", "group": "advanced", "section": "Look & Feel",
        "description": "Controls how 'enhanced' vs 'authentic' the output looks",
        "options": [
            {"value": "faithful", "name": "Faithful", "tip": "Stays close to the original look. Best for archival, historical, or documentary content"},
            {"value": "balanced", "name": "Balanced", "tip": "Slight improvement without changing the character. Good default for most content"},
            {"value": "enhanced", "name": "Enhanced", "tip": "Makes everything look sharper and more vivid. Great for casual viewing, but less authentic"},
        ],
        "why": "Faithful for history/archives. Balanced for everyday use. Enhanced if you just want it to look as good as possible and don't care about authenticity.",
    },
    "grain_preserve": {
        "name": "Grain Preservation", "group": "advanced",
        "description": "Controls how much original film grain texture to keep in the output",
        "options": [
            {"value": "off", "name": "Remove All", "tip": "Clean, smooth digital look. Best for VHS/digital sources. Makes film look less 'filmic'"},
            {"value": "low", "name": "Low", "tip": "Keeps a hint of texture. Good compromise between clean and authentic"},
            {"value": "medium", "name": "Medium", "tip": "Preserves the film character while reducing heavy grain. Best for most old film"},
            {"value": "high", "name": "High", "tip": "Keeps grain nearly intact. Use when the grain IS the aesthetic (e.g., classic cinema)"},
        ],
        "why": "Film grain gives footage its character. Removing it makes a clean image but can look 'fake'. Medium is a good starting point for old film.",
        "show_for": ["old_film", "bw_restore"],
    },
}


def get_system_info():
    info = {"platform": platform.system(), "python": platform.python_version(),
            "cpu": 0, "ram_used": 0, "ram_total": 0, "gpu": "Not detected", "vram": 0, "vram_total": 0}
    try:
        import psutil
        info["cpu"] = psutil.cpu_percent()
        m = psutil.virtual_memory()
        info["ram_used"], info["ram_total"] = round(m.used/1024**3, 1), round(m.total/1024**3, 1)
    except: pass
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram"] = round(torch.cuda.memory_allocated()/1024**3, 1)
            info["vram_total"] = round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1)
    except: pass
    return info


def check_setup():
    """Check what's installed and provide friendly status."""
    status = {"ready": True, "issues": [], "suggestions": []}

    # Check torch
    try:
        import torch
        if not torch.cuda.is_available():
            status["issues"].append("GPU acceleration not available - processing will be slow")
            status["suggestions"].append("Install PyTorch with CUDA for 10-50x faster processing")
    except ImportError:
        status["ready"] = False
        status["issues"].append("PyTorch not installed - required for AI processing")
        status["suggestions"].append("Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    # Check ffmpeg
    if not shutil.which("ffmpeg"):
        status["ready"] = False
        status["issues"].append("FFmpeg not found - required for video processing")
        status["suggestions"].append("Install FFmpeg: winget install ffmpeg (Windows) or brew install ffmpeg (Mac)")

    # Check models
    essential_models = ["realesr-general-x4v3.pth", "GFPGANv1.4.pth", "HAT-L_SRx4_ImageNet-pretrain.pth"]
    missing = [m for m in essential_models if not (MODEL_DIR / m).exists()]
    status["missing_models"] = missing

    return status


ESSENTIAL_MODELS = {
    "realesr-general-x4v3.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "size_mb": 64, "description": "General video upscaling (Real-ESRGAN v3)",
    },
    "GFPGANv1.4.pth": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        "size_mb": 348, "description": "Face enhancement (GFPGAN v1.4)",
    },
    "HAT-L_SRx4_ImageNet-pretrain.pth": {
        "url": "https://github.com/XPixelGroup/HAT/releases/download/v0.0.0/HAT-L_SRx4_ImageNet-pretrain.pth",
        "size_mb": 91, "description": "HAT-L 4x upscaling (best quality, Hybrid Attention Transformer)",
    },
}

EXTRA_MODELS = {
    "RealESRGAN_x4plus.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "size_mb": 64, "description": "High-quality general upscaling",
    },
    "realesr-animevideov3.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "size_mb": 64, "description": "Anime/cartoon optimized upscaling",
    },
    "RealESRGAN_x4plus_anime_6B.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "size_mb": 17, "description": "Lightweight anime model",
    },
    "codeformer.pth": {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "size_mb": 376, "description": "Face restoration (CodeFormer)",
    },
    # HAT alternative sizes (HAT-L is in ESSENTIAL_MODELS)
    "HAT_SRx4_ImageNet-pretrain.pth": {
        "url": "https://github.com/XPixelGroup/HAT/releases/download/v0.0.0/HAT_SRx4_ImageNet-pretrain.pth",
        "size_mb": 61, "description": "HAT base 4x upscaling (balanced)",
    },
    "HAT-S_SRx4.pth": {
        "url": "https://github.com/XPixelGroup/HAT/releases/download/v0.0.0/HAT-S_SRx4.pth",
        "size_mb": 31, "description": "HAT-S 4x upscaling (fastest HAT)",
    },
    # Diffusion SR — requires diffusers library + HuggingFace download
    "diffusion_sr_info": {
        "url": "", "size_mb": 0,
        "description": "Diffusion SR (requires: pip install diffusers, models via HuggingFace)",
        "info_only": True,
    },
}


def get_all_models_status():
    """Return status of all models."""
    result = {"essential": [], "extra": [], "model_dir": str(MODEL_DIR)}
    for name, info in ESSENTIAL_MODELS.items():
        # HAT models are in hat/ subdirectory
        if name.startswith("HAT"):
            path = MODEL_DIR / "hat" / name
        else:
            path = MODEL_DIR / name
        result["essential"].append({
            "name": name, "description": info["description"],
            "size_mb": info["size_mb"], "downloaded": path.exists(),
        })
    for name, info in EXTRA_MODELS.items():
        if info.get("info_only"):
            result["extra"].append({
                "name": name, "description": info["description"],
                "size_mb": 0, "downloaded": False, "info_only": True,
            })
            continue
        path = MODEL_DIR / name
        result["extra"].append({
            "name": name, "description": info["description"],
            "size_mb": info["size_mb"], "downloaded": path.exists(),
        })
    return result


_DOWNLOAD_STATUS = {}
_DOWNLOAD_STATUS_LOCK = threading.Lock()


def _bg_download(name, url, path):
    """Background download worker."""
    try:
        urllib.request.urlretrieve(url, path)
        with _DOWNLOAD_STATUS_LOCK:
            _DOWNLOAD_STATUS[name] = "downloaded"
    except Exception as e:
        with _DOWNLOAD_STATUS_LOCK:
            _DOWNLOAD_STATUS[name] = f"failed: {e}"
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def download_model_by_name(name):
    """Download a single model by name (non-blocking)."""
    all_models = {**ESSENTIAL_MODELS, **EXTRA_MODELS}
    if name not in all_models:
        return {"error": f"Unknown model: {name}"}
    info = all_models[name]
    if info.get("info_only"):
        return {"name": name, "status": "info only — manual install required"}
    path = MODEL_DIR / name
    if path.exists():
        return {"name": name, "status": "already downloaded"}

    # Check if already downloading
    with _DOWNLOAD_STATUS_LOCK:
        if name in _DOWNLOAD_STATUS:
            status = _DOWNLOAD_STATUS[name]
            if status == "downloading":
                return {"name": name, "status": "downloading"}
            # Previous attempt finished — return result and clear
            del _DOWNLOAD_STATUS[name]
            return {"name": name, "status": status}
        _DOWNLOAD_STATUS[name] = "downloading"

    print(f"Downloading {name} in background...")
    t = threading.Thread(target=_bg_download, args=(name, info["url"], path), daemon=True)
    t.start()
    return {"name": name, "status": "downloading"}


def download_essential_models():
    """Download all essential models (non-blocking)."""
    results = []
    for name, info in ESSENTIAL_MODELS.items():
        model_path = MODEL_DIR / name
        if model_path.exists():
            results.append({"name": name, "status": "already downloaded"})
            continue
        with _DOWNLOAD_STATUS_LOCK:
            if name in _DOWNLOAD_STATUS and _DOWNLOAD_STATUS[name] == "downloading":
                results.append({"name": name, "status": "downloading"})
                continue
            _DOWNLOAD_STATUS[name] = "downloading"
        t = threading.Thread(target=_bg_download, args=(name, info["url"], model_path), daemon=True)
        t.start()
        results.append({"name": name, "status": "downloading"})
    return {"success": True, "models": results}


def list_directory(path):
    try:
        p = Path(path)
        if not p.exists(): return {"error": "Path not found"}
        items = []
        for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            try:
                ext = item.suffix.lower() if item.is_file() else ""
                is_video = ext in ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.wmv', '.flv', '.m4v']
                items.append({
                    "name": item.name, "path": str(item), "is_dir": item.is_dir(),
                    "is_video": is_video, "size": item.stat().st_size if item.is_file() else 0,
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


def analyze_video(path):
    """Analyze video and return smart recommendations."""
    # This would normally use ffprobe and AI analysis
    # For now, return mock analysis based on filename hints
    analysis = {
        "duration": "Unknown",
        "resolution": "Unknown",
        "fps": "Unknown",
        "detected_type": "home_video",
        "detected_issues": [],
        "recommendations": [],
    }

    name = Path(path).name.lower()

    # Simple heuristic detection
    if any(x in name for x in ['vhs', 'tape', 'vcr']):
        analysis["detected_type"] = "vhs_tape"
        analysis["detected_issues"] = ["Tape artifacts likely", "May have tracking issues"]
    elif any(x in name for x in ['film', '8mm', '16mm', 'reel']):
        analysis["detected_type"] = "old_film"
        analysis["detected_issues"] = ["May have scratches", "Film grain present"]
    elif any(x in name for x in ['anime', 'cartoon', 'animated']):
        analysis["detected_type"] = "anime"
    elif any(x in name for x in ['bw', 'b&w', 'black', 'white', '1950', '1940', '1930']):
        analysis["detected_type"] = "bw_restore"
        analysis["detected_issues"] = ["Black & white footage detected"]

    return analysis


def download_youtube(url, download_dir=None):
    try:
        import yt_dlp
        base_dir = Path(download_dir) if download_dir else Path.home() / "Downloads"

        # First extract info to get the title for project folder
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "video")

        # Sanitize title for folder name
        safe_title = "".join(c if c.isalnum() or c in ' _-' else '_' for c in title).strip()
        if not safe_title:
            safe_title = "video"

        # Create project folder: base_dir/VideoTitle/originals/
        project_dir = base_dir / safe_title
        originals_dir = project_dir / "originals"
        restored_dir = project_dir / "restored"
        originals_dir.mkdir(parents=True, exist_ok=True)
        restored_dir.mkdir(parents=True, exist_ok=True)

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': str(originals_dir / '%(title)s.%(ext)s'),
            'merge_output_format': 'mp4',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            if not Path(filename).exists():
                filename = filename.rsplit('.', 1)[0] + '.mp4'

            # Build output path in restored/ subfolder
            restored_path = str(restored_dir / (Path(filename).stem + '_restored.mkv'))

            return {
                "success": True,
                "path": filename,
                "title": info.get("title"),
                "folder": str(project_dir),
                "restored_path": restored_path,
            }
    except ImportError:
        return {"error": "YouTube download not available. Install with: pip install yt-dlp"}
    except Exception as e:
        return {"error": str(e)}


def submit_job(data):
    job_id = str(uuid.uuid4())[:8]
    job = {
        "id": job_id,
        "input": data.get("input", ""),
        "output": data.get("output", ""),
        "profile": data.get("profile", "auto"),
        "quality": data.get("quality", "balanced"),
        "enhancements": data.get("enhancements", {}),
        "ref_dir": data.get("ref_dir", ""),
        "status": "pending",
        "progress": 0,
        "message": "Waiting to start...",
        "stage": "queued",
        "stage_label": "Queued",
        "stages_done": [],
        "elapsed_sec": 0,
        "eta_sec": None,
        "fps": 0,
        "frame_current": 0,
        "frame_total": 0,
        "log": [],
        "created": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    threading.Thread(target=run_job, args=(job_id,), daemon=True).start()
    return job


# Stage definitions for display
STAGES = [
    ("analyze", "Analyzing video"),
    ("extract", "Extracting frames"),
    ("dedup", "Removing duplicates"),
    ("denoise", "Denoising"),
    ("colorize", "Colorizing"),
    ("upscale", "AI upscaling"),
    ("face", "Reference enhancement"),
    ("interpolate", "Frame interpolation"),
    ("encode", "Encoding output"),
    ("done", "Complete"),
]


def _update_job(job_id, **kwargs):
    """Thread-safe job update helper."""
    with JOBS_LOCK:
        if job_id in JOBS:
            log_line = kwargs.pop("log_line", None)
            JOBS[job_id].update(kwargs)
            if log_line:
                # Add timestamp to log line for progress tracking
                timestamp = datetime.now().strftime("%H:%M:%S")
                timestamped_line = f"[{timestamp}] {log_line}"
                JOBS[job_id]["log"].append(timestamped_line)
                if len(JOBS[job_id]["log"]) > 200:
                    JOBS[job_id]["log"] = JOBS[job_id]["log"][-200:]


def _detect_stage(line):
    """Detect processing stage from CLI output line."""
    lower = line.lower()
    if "[1/3]" in lower or "[1/" in lower or "extracting frames" in lower:
        return "extract", "Extracting frames"
    elif "[2/3]" in lower or "[2/" in lower or "processing" in lower:
        return "upscale", "Processing frames"
    elif "[3/3]" in lower or "[3/" in lower or "reassembl" in lower or "encod" in lower:
        return "encode", "Encoding video"
    elif "analyz" in lower or "probing" in lower or "scanning" in lower:
        return "analyze", "Analyzing video"
    elif "extract" in lower or "decod" in lower or "reading frames" in lower:
        return "extract", "Extracting frames"
    elif "dedup" in lower or "duplicate" in lower:
        return "dedup", "Removing duplicates"
    elif "upscal" in lower or "esrgan" in lower or "super-res" in lower or "tile" in lower:
        return "upscale", "AI upscaling"
    elif "denois" in lower or "noise" in lower:
        return "denoise", "Denoising"
    elif "face" in lower or "gfpgan" in lower:
        return "face", "Enhancing faces"
    elif "coloriz" in lower:
        return "colorize", "Colorizing"
    elif "interpol" in lower or "rife" in lower:
        return "interpolate", "Frame interpolation"
    elif "encod" in lower or "mux" in lower or "ffmpeg" in lower or "writing" in lower:
        return "encode", "Encoding output"
    return None, None


def _read_lines_with_cr(stream):
    """Read from a stream handling both \\n and \\r as line terminators (for tqdm)."""
    buf = ""
    while True:
        ch = stream.read(1)
        if not ch:
            if buf.strip():
                yield buf
            break
        if ch in ("\n", "\r"):
            if buf.strip():
                yield buf
            buf = ""
        else:
            buf += ch


# Track running processes for stop functionality
RUNNING_PROCESSES = {}
PROCESSES_LOCK = threading.Lock()


def _run_step(job_id, cmd, stage, stage_label, start_time, stages_done, progress_base=0, progress_span=100):
    """Run a single CLI step and stream its output to the job log."""
    _update_job(job_id, stage=stage, stage_label=stage_label,
                stages_done=list(stages_done), log_line=f">> {stage_label}...")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=0, cwd=str(Path(__file__).parent.parent),
                               env=env)
    with PROCESSES_LOCK:
        RUNNING_PROCESSES[job_id] = process

    frame_total = 0
    last_update = 0
    for line in _read_lines_with_cr(process.stdout):
        # Check if job was cancelled
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job and job.get("status") == "cancelled":
                process.terminate()
                return -1, frame_total

        clean = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
        if not clean:
            continue
        elapsed = time.time() - start_time

        # Throttle updates to max 2/sec to avoid flooding
        now = time.time()
        if now - last_update < 0.5 and "%" not in clean and "/" not in clean:
            continue
        last_update = now

        # Parse tqdm-style progress: "  6%|##4  | 696/11231 [04:40<1:15:38,  2.32it/s]"
        progress = None
        tqdm_match = re.search(r'(\d+)%\|', clean)
        if tqdm_match:
            val = float(tqdm_match.group(1))
            progress = progress_base + (val / 100) * progress_span
        elif "%" in clean:
            for part in clean.split():
                if "%" in part:
                    try:
                        val = float(part.replace("%", "").replace(",", ""))
                        progress = progress_base + (val / 100) * progress_span
                    except ValueError:
                        pass

        # Parse frame counts from tqdm "696/11231" pattern
        frame_current = 0
        extracted = re.search(r'[Ee]xtracted\s+(\d+)\s+frames', clean)
        if extracted:
            frame_total = int(extracted.group(1))
        frame_match = re.search(r'(\d+)/(\d+)', clean)
        if frame_match:
            fc = int(frame_match.group(1))
            ft = int(frame_match.group(2))
            if ft > 10:  # Ignore small ratios like [1/3]
                frame_current = fc
                frame_total = ft

        # Parse tqdm ETA: "<1:15:38" or "<00:30"
        eta = None
        eta_match = re.search(r'<(\d+):(\d+):(\d+)', clean)
        if eta_match:
            eta = int(eta_match.group(1)) * 3600 + int(eta_match.group(2)) * 60 + int(eta_match.group(3))
        else:
            eta_match2 = re.search(r'<(\d+):(\d+)', clean)
            if eta_match2:
                eta = int(eta_match2.group(1)) * 60 + int(eta_match2.group(2))

        # Parse tqdm speed: "2.32it/s"
        fps = 0
        speed_match = re.search(r'(\d+\.?\d*)\s*it/s', clean)
        if speed_match:
            fps = round(float(speed_match.group(1)), 1)
        elif frame_current > 0 and elapsed > 0:
            fps = round(frame_current / elapsed, 1)

        # Build update — only add to log if it's a meaningful line (not tqdm progress)
        update = {"message": clean[:120], "elapsed_sec": round(elapsed),
                  "stage": stage, "stages_done": list(stages_done)}
        if not tqdm_match:
            update["log_line"] = clean[:200]
        if progress is not None:
            update["progress"] = min(round(progress, 1), 99)
        if eta is not None:
            update["eta_sec"] = round(eta)
        if frame_current:
            update["frame_current"] = frame_current
        if frame_total:
            update["frame_total"] = frame_total
        if fps:
            update["fps"] = fps
        _update_job(job_id, **update)

    process.wait()
    with PROCESSES_LOCK:
        RUNNING_PROCESSES.pop(job_id, None)
    return process.returncode, frame_total


def run_job(job_id):
    start_time = time.time()

    _update_job(job_id, status="running", stage="analyze",
                stage_label="Analyzing video", started_at=datetime.now().isoformat(),
                log_line="Starting restoration...")

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return

    enh = job.get("enhancements", {})
    scale = "4" if enh.get("upscale") == "4x" else "2"
    output_path = Path(job["output"])
    input_path = job["input"]

    # Project folder structure — persistent frames for re-runs
    project_dir = output_path.parent
    frames_dir = project_dir / "frames"
    enhanced_dir = project_dir / "enhanced"
    frames_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    stages_done = []

    try:
        # ---- STEP 0: Analyze ----
        _update_job(job_id, stage="analyze", stage_label="Analyzing video",
                    progress=2, elapsed_sec=0, log_line=f"Input: {input_path}")
        stages_done.append("analyze")

        # ---- STEP 1: Extract frames (skip if already extracted) ----
        existing_frames = list(frames_dir.glob("*.png"))
        if existing_frames:
            frame_count = len(existing_frames)
            _update_job(job_id, stage="extract", stage_label="Frames already extracted",
                        stages_done=list(stages_done), progress=10,
                        elapsed_sec=round(time.time() - start_time),
                        frame_total=frame_count,
                        log_line=f"Found {frame_count} existing frames — skipping extraction")
        else:
            cmd_extract = ["python", "-m", "framewright.cli", "extract-frames",
                           "--input", input_path, "--output", str(frames_dir)]
            rc, frame_count = _run_step(job_id, cmd_extract, "extract", "Extracting frames",
                                        start_time, stages_done, progress_base=2, progress_span=13)
            if rc != 0:
                raise Exception(f"Frame extraction failed (exit code {rc})")
            if frame_count == 0:
                frame_count = len(list(frames_dir.glob("*.png")))
        stages_done.append("extract")

        # ---- STEP 1.5: Deduplication (if enabled) ----
        dedup_mode = enh.get("dedup", "off")
        dedup_marker = project_dir / ".dedup_done"  # Marker file to track completion
        dedup_dir = project_dir / "deduped"

        if dedup_mode != "off":
            # Check if deduplication was already completed (marker file exists)
            if dedup_marker.exists():
                _update_job(job_id, stage="dedup", stage_label="Skipping deduplication (already done)",
                            stages_done=list(stages_done), progress=14,
                            elapsed_sec=round(time.time() - start_time),
                            log_line=f"✓ Deduplication already completed - skipping")
                # frames_dir already has deduplicated frames
                frame_count = len(list(frames_dir.glob("*.png")))
                # Set variables for consistency (no frames removed since already done)
                removed = 0
                before_count = frame_count
                after_count = frame_count
            else:
                _update_job(job_id, stage="dedup", stage_label="Removing duplicate frames",
                            stages_done=list(stages_done), progress=14,
                            elapsed_sec=round(time.time() - start_time),
                            log_line=f"▶ Deduplicating with mode: {dedup_mode}")

                before_count = len(list(frames_dir.glob("*.png")))
                removed = 0

                if dedup_mode == "exact":
                    # Exact: full-file MD5 — only removes byte-identical frames
                    import hashlib
                    frames_list = sorted(frames_dir.glob("*.png"))
                    prev_hash = None
                    for fp in frames_list:
                        with open(fp, "rb") as fh:
                            h = hashlib.md5(fh.read()).hexdigest()
                        if h == prev_hash:
                            fp.unlink()
                            removed += 1
                        prev_hash = h
                else:
                    # Smart: use FFmpeg mpdecimate to auto-detect padded/blended duplicates
                    # This handles YouTube re-encoding (e.g. 16fps->25fps) without needing
                    # to know the original framerate — mpdecimate detects it automatically
                    # Get user's threshold (slider value 90-100, convert to 0.90-1.00)
                    threshold_val = int(enh.get("dedup_threshold", 98))
                    threshold = threshold_val / 100.0

                    # Scale mpdecimate parameters based on threshold
                    # Lower threshold = more aggressive = higher hi/lo values
                    scale = (1.0 - threshold) * 10 + 1.0  # 0.98 -> 1.2, 0.96 -> 1.4, 0.94 -> 1.6
                    hi_val = int(64 * 12 * scale)
                    lo_val = int(64 * 5 * scale)

                    _update_job(job_id, log_line=f"▶ Using FFmpeg mpdecimate (threshold={threshold:.2f}, hi={hi_val}, lo={lo_val})...")
                    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
                    # dedup_dir already defined above
                    dedup_dir.mkdir(parents=True, exist_ok=True)
                cmd_dedup = [ffmpeg_bin, "-i", input_path,
                             "-vf", f"mpdecimate=hi={hi_val}:lo={lo_val}:frac=0.33",
                             "-vsync", "vfr",
                             "-qscale:v", "1", "-qmin", "1", "-qmax", "1",
                             str(dedup_dir / "frame_%08d.png")]
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                _update_job(job_id, log_line=f"▶ Running FFmpeg mpdecimate on {before_count} frames...")
                proc = subprocess.Popen(cmd_dedup, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=0, env=env)
                proc.wait()
                _update_job(job_id, log_line=f"✓ FFmpeg mpdecimate completed (exit code: {proc.returncode})")
                if proc.returncode == 0:
                    dedup_frames = sorted(dedup_dir.glob("*.png"))
                    if dedup_frames:
                        # Replace frames_dir with deduped frames
                        for f in frames_dir.glob("*.png"):
                            f.unlink()
                        for idx, fp in enumerate(dedup_frames):
                            fp.rename(frames_dir / f"frame_{idx + 1:08d}.png")
                        removed = before_count - len(list(frames_dir.glob("*.png")))
                    # Clean up temp dir
                    for f in dedup_dir.glob("*"):
                        f.unlink()
                    dedup_dir.rmdir()
                else:
                    _update_job(job_id, log_line="mpdecimate failed, falling back to pixel comparison...")
                    # Fallback: pixel MSE comparison
                    import cv2
                    import numpy as np
                    frames_list = sorted(frames_dir.glob("*.png"))
                    prev_thumb = None
                    for fp in frames_list:
                        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        thumb = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
                        if prev_thumb is not None:
                            if np.mean((thumb - prev_thumb) ** 2) < 30.0:
                                fp.unlink()
                                removed += 1
                                continue
                        prev_thumb = thumb

            # Renumber remaining frames sequentially (FFmpeg needs no gaps)
            if removed > 0:
                remaining = sorted(frames_dir.glob("*.png"))
                for idx, fp in enumerate(remaining):
                    fp.rename(frames_dir / f"_tmp_{idx:08d}.png")
                for idx, fp in enumerate(sorted(frames_dir.glob("_tmp_*.png"))):
                    fp.rename(frames_dir / f"frame_{idx + 1:08d}.png")

            after_count = before_count - removed
            _update_job(job_id, stage="dedup", stage_label="Deduplication complete",
                        stages_done=list(stages_done), progress=15,
                        elapsed_sec=round(time.time() - start_time),
                        frame_total=after_count,
                        log_line=f"Removed {removed} duplicate frames ({before_count} -> {after_count})")
            frame_count = after_count

            # Create marker file to skip dedup on future runs
            dedup_marker.write_text(f"Deduplication completed at {datetime.now().isoformat()}\nMode: {dedup_mode}\nFrames: {before_count} -> {after_count}")
            _update_job(job_id, log_line=f"✓ Deduplication marker created")

        stages_done.append("dedup")

        # ---- STEP 1.7: Denoise BEFORE upscale (if enabled) ----
        # Denoising works better on original-resolution frames:
        #  - Faster (smaller images)
        #  - Noise is at original pixel scale, easier to detect
        #  - Gives the upscaler cleaner input = better upscale results
        # When reference enhancement is "heavy", auto-reduce denoise to avoid
        # over-smoothing — the reference step will regenerate detail anyway.
        denoise_level = enh.get("denoise", "off")
        ref_dir = job.get("ref_dir", "")
        ref_str = enh.get("ref_strength", "medium")
        has_ref_enhance = bool(ref_dir and Path(ref_dir).is_dir() and
                               any(f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
                                   for f in Path(ref_dir).iterdir())) if ref_dir else False

        if denoise_level != "off":
            # Auto-adjust: when heavy reference enhancement is active, cap denoise at light
            # to avoid smoothing away structure that references need to latch onto
            effective_denoise = denoise_level
            if has_ref_enhance and ref_str == "heavy" and denoise_level in ("medium", "strong"):
                effective_denoise = "light"
                _update_job(job_id, log_line=f"Denoise auto-reduced to 'light' (heavy reference enhancement will handle detail)")

            _update_job(job_id, stage="denoise", stage_label="Denoising frames",
                        stages_done=list(stages_done), progress=16,
                        elapsed_sec=round(time.time() - start_time),
                        log_line=f"Denoising with level: {effective_denoise}")
            import cv2
            strength_map = {"light": 5, "medium": 10, "strong": 20}
            h_val = strength_map.get(effective_denoise, 10)
            dn_frames = sorted(frames_dir.glob("*.png"))
            total_dn = len(dn_frames)
            for i, fp in enumerate(dn_frames):
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                denoised = cv2.fastNlMeansDenoisingColored(img, None, h_val, h_val, 7, 21)
                cv2.imwrite(str(fp), denoised)
                if i % max(1, total_dn // 20) == 0:
                    pct = 16 + (i / total_dn) * 4
                    _update_job(job_id, progress=round(pct, 1),
                                elapsed_sec=round(time.time() - start_time),
                                frame_current=i, frame_total=total_dn,
                                message=f"Denoising frame {i+1}/{total_dn}")
            _update_job(job_id, stage="denoise", stage_label="Denoising complete",
                        progress=20, elapsed_sec=round(time.time() - start_time),
                        log_line=f"Denoised {total_dn} frames (effective: {effective_denoise})")
        stages_done.append("denoise")

        # ---- STEP 1.9: Colorize BEFORE upscale (if enabled) ----
        # Colorization models are trained at native resolution.
        # Colorizing first means the upscaler gets full-color input = better results.
        colorize_mode = enh.get("colorize", "off")
        if colorize_mode != "off":
            _update_job(job_id, stage="colorize", stage_label="Colorizing frames",
                        stages_done=list(stages_done), progress=20,
                        elapsed_sec=round(time.time() - start_time),
                        log_line=f"Colorizing with mode: {colorize_mode}")
            try:
                import cv2
                import numpy as np
                color_frames = sorted(frames_dir.glob("*.png"))
                total_cf = len(color_frames)
                _update_job(job_id, log_line=f"Colorizing {total_cf} frames (AI auto-color)")

                # Try DDColor/DeOldify via framewright processor
                try:
                    from framewright.processors.colorization import Colorizer
                    colorizer = Colorizer(model=colorize_mode if colorize_mode != "auto" else "ddcolor")
                    for i, fp in enumerate(color_frames):
                        colorizer.colorize_frame(str(fp), str(fp))
                        if i % max(1, total_cf // 20) == 0:
                            pct = 20 + (i / total_cf) * 5
                            _update_job(job_id, progress=round(pct, 1),
                                        elapsed_sec=round(time.time() - start_time),
                                        frame_current=i, frame_total=total_cf,
                                        message=f"Colorizing frame {i+1}/{total_cf}")
                except ImportError:
                    _update_job(job_id, log_line="Colorization models not installed, skipping (pip install framewright[colorization])")

                _update_job(job_id, stage="colorize", stage_label="Colorization complete",
                            progress=25, elapsed_sec=round(time.time() - start_time),
                            log_line=f"Colorized {total_cf} frames")
            except Exception as e:
                _update_job(job_id, log_line=f"Colorization failed: {e}, continuing without colorization")
        stages_done.append("colorize")

        # ---- STEP 2: Upscale frames with checkpoint/resume support ----
        # Check for existing enhanced frames (allows resuming after crashes)
        existing_enhanced = list(enhanced_dir.glob("*.png"))
        source_frames = list(frames_dir.glob("*.png"))

        if len(existing_enhanced) >= len(source_frames) and len(existing_enhanced) > 0:
            # All frames already upscaled - skip this step
            _update_job(job_id, stage="upscale", stage_label="AI upscaling already complete",
                        stages_done=list(stages_done), progress=60,
                        elapsed_sec=round(time.time() - start_time),
                        log_line=f"✓ Found {len(existing_enhanced)} enhanced frames - skipping upscale")
            stages_done.append("upscale")
        else:
            # Need to upscale (either from scratch or resume partial progress)
            if len(existing_enhanced) > 0:
                _update_job(job_id, log_line=f"Found {len(existing_enhanced)}/{len(source_frames)} enhanced frames - will resume")

            cmd_enhance = ["python", "-m", "framewright.cli", "enhance-frames",
                           "--input", str(frames_dir), "--output", str(enhanced_dir),
                           "--scale", scale]
            sr_model_map = {
                "realesrgan": "realesrgan-x4plus",
                "hat": "hat",
                "ensemble": "ensemble",
                "diffusion": "diffusion",
            }
            sr_model = enh.get("sr_model", "realesrgan")
            model_name = sr_model_map.get(sr_model, "realesrgan-x4plus")
            cmd_enhance.extend(["--model", model_name])

            # Add per-model arguments
            if sr_model == "hat":
                cmd_enhance.extend(["--hat-size", "large"])
            elif sr_model == "diffusion":
                cmd_enhance.extend(["--diffusion-steps", "20", "--diffusion-model", "upscale_a_video"])
            elif sr_model == "ensemble":
                cmd_enhance.extend(["--ensemble-models", "hat,realesrgan", "--ensemble-method", "weighted"])

            rc, _ = _run_step(job_id, cmd_enhance, "upscale", "AI Upscaling",
                              start_time, stages_done, progress_base=20, progress_span=40)
            if rc != 0:
                raise Exception(f"Enhancement failed (exit code {rc})")
            stages_done.append("upscale")

        # ---- STEP 2.5: Reference-guided enhancement (if references provided) ----
        ref_dir = job.get("ref_dir", "")
        if ref_dir and Path(ref_dir).is_dir():
            ref_images = [f for f in Path(ref_dir).iterdir()
                          if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}]
            if ref_images:
                ref_output_dir = project_dir / "ref_enhanced"
                ref_output_dir.mkdir(parents=True, exist_ok=True)

                # Map ref_strength to IP-Adapter parameters
                ref_str = enh.get("ref_strength", "medium")
                ref_params = {
                    "light":  {"strength": "0.25", "ip_scale": "0.4"},
                    "medium": {"strength": "0.40", "ip_scale": "0.6"},
                    "heavy":  {"strength": "0.55", "ip_scale": "0.8"},
                }
                rp = ref_params.get(ref_str, ref_params["medium"])
                _update_job(job_id, log_line=f"Reference strength: {ref_str} (strength={rp['strength']}, ip_scale={rp['ip_scale']})")

                cmd_ref = ["python", "-m", "framewright.processors.reference_enhance",
                           "--frames", str(enhanced_dir),
                           "--output", str(ref_output_dir),
                           "--references", ref_dir,
                           "--strength", rp["strength"],
                           "--ip-scale", rp["ip_scale"]]

                rc, _ = _run_step(job_id, cmd_ref, "face", "Reference Enhancement",
                                  start_time, stages_done, progress_base=60, progress_span=15)
                if rc != 0:
                    _update_job(job_id, log_line=f"Reference enhancement failed (code {rc}), using standard upscaled frames")
                else:
                    # Use ref-enhanced frames for reassembly instead
                    enhanced_dir = ref_output_dir
                    stages_done.append("face")

        # ---- STEP 2.7: Frame interpolation (if enabled) ----
        target_fps = enh.get("framerate", "off")
        if target_fps != "off":
            import cv2
            target = int(target_fps)
            source_fps = 24  # Default for old film
            multiplier = max(1, round(target / source_fps))
            if multiplier > 1:
                _update_job(job_id, stage="interpolate", stage_label="Frame interpolation",
                            stages_done=list(stages_done), progress=75,
                            elapsed_sec=round(time.time() - start_time),
                            log_line=f"Interpolating to {target_fps} FPS (blend {multiplier}x)...")
                interp_dir = project_dir / "interpolated"
                interp_dir.mkdir(parents=True, exist_ok=True)
                # Clean previous interpolated frames
                for f in interp_dir.glob("*.png"):
                    f.unlink()
                src_frames = sorted(enhanced_dir.glob("*.png"))
                total_interp = len(src_frames)
                _update_job(job_id, log_line=f"Generating {multiplier}x frames ({total_interp} -> ~{total_interp * multiplier})")
                out_idx = 1
                for i in range(len(src_frames)):
                    img = cv2.imread(str(src_frames[i]))
                    cv2.imwrite(str(interp_dir / f"frame_{out_idx:08d}.png"), img)
                    out_idx += 1
                    if i < len(src_frames) - 1:
                        img_next = cv2.imread(str(src_frames[i + 1]))
                        for step in range(1, multiplier):
                            alpha = step / multiplier
                            blended = cv2.addWeighted(img, 1 - alpha, img_next, alpha, 0)
                            cv2.imwrite(str(interp_dir / f"frame_{out_idx:08d}.png"), blended)
                            out_idx += 1
                    if i % max(1, total_interp // 20) == 0:
                        pct = 75 + (i / total_interp) * 10
                        _update_job(job_id, progress=round(pct, 1),
                                    elapsed_sec=round(time.time() - start_time),
                                    frame_current=i, frame_total=total_interp,
                                    message=f"Interpolating frame {i+1}/{total_interp}")
                enhanced_dir = interp_dir
                _update_job(job_id, log_line=f"Interpolation complete: {out_idx - 1} frames at {target_fps} FPS")
            else:
                _update_job(job_id, log_line=f"Source already at or above {target_fps} FPS, skipping interpolation")
        stages_done.append("interpolate")

        # ---- STEP 3: Reassemble video ----
        # Ensure frames are sequentially numbered (FFmpeg needs frame_%08d.png without gaps)
        final_frames = sorted(enhanced_dir.glob("*.png"))
        if final_frames:
            _update_job(job_id, log_line=f"Verifying {len(final_frames)} frames for reassembly...")
            for idx, fp in enumerate(final_frames):
                expected = f"frame_{idx + 1:08d}.png"
                if fp.name != expected:
                    fp.rename(enhanced_dir / expected)

        # Use user's CRF slider value directly (no quality-level override)
        crf = enh.get("crf", "18")

        cmd_reassemble = ["python", "-m", "framewright.cli", "reassemble",
                          "--frames-dir", str(enhanced_dir),
                          "--audio", input_path,
                          "--output", str(output_path),
                          "--quality", str(crf)]
        # Use target fps when interpolation was performed
        if target_fps != "off":
            cmd_reassemble.extend(["--fps", str(target_fps)])
        rc, _ = _run_step(job_id, cmd_reassemble, "encode", "Encoding video",
                          start_time, stages_done, progress_base=85, progress_span=15)
        if rc != 0:
            raise Exception(f"Encoding failed (exit code {rc})")
        stages_done.append("encode")

        elapsed = round(time.time() - start_time)
        _update_job(
            job_id, status="completed", progress=100, stage="done",
            stage_label="Complete", elapsed_sec=elapsed, eta_sec=0,
            message="Done! Your video is ready.",
            completed_at=datetime.now().isoformat(),
            log_line=f"Restoration completed. Output: {output_path}",
        )
    except Exception as e:
        elapsed = round(time.time() - start_time)
        _update_job(
            job_id, status="failed", elapsed_sec=elapsed,
            message=str(e), completed_at=datetime.now().isoformat(),
            log_line=f"Error: {e}",
        )


DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FrameWright</title>
<style>
:root {
  --bg: #000; --surface: #1c1c1e; --surface2: #2c2c2e; --surface3: #3a3a3c;
  --text: #fff; --text2: #8e8e93; --accent: #0a84ff; --accent2: #5ac8fa;
  --success: #30d158; --warning: #ff9f0a; --error: #ff453a;
  --radius: 16px; --radius-sm: 12px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
  background: var(--bg); color: var(--text); min-height: 100vh;
  display: flex; flex-direction: column;
}

/* Header */
header {
  background: linear-gradient(180deg, rgba(28,28,30,0.98) 0%, rgba(28,28,30,0.95) 100%);
  backdrop-filter: blur(20px); padding: 20px 32px;
  border-bottom: 1px solid rgba(255,255,255,0.1);
}
.logo { font-size: 24px; font-weight: 700; letter-spacing: -0.5px; }
.logo span { background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

/* Main */
main { flex: 1; max-width: 900px; width: 100%; margin: 0 auto; padding: 32px; }

/* Steps */
.step { display: none; animation: fadeIn 0.3s ease; }
.step.active { display: block; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

/* Step header */
.step-header { text-align: center; margin-bottom: 32px; }
.step-number { display: inline-block; width: 32px; height: 32px; background: var(--accent); border-radius: 50%;
  font-size: 14px; font-weight: 600; line-height: 32px; margin-bottom: 12px; }
.step-title { font-size: 28px; font-weight: 600; margin-bottom: 8px; }
.step-subtitle { color: var(--text2); font-size: 16px; }

/* Cards */
.card { background: var(--surface); border-radius: var(--radius); padding: 24px; margin-bottom: 16px; }
.card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }

/* Profile cards */
.profile-card {
  background: var(--surface2); border: 2px solid transparent; border-radius: var(--radius-sm);
  padding: 20px; cursor: pointer; transition: all 0.2s; text-align: center;
}
.profile-card:hover { background: var(--surface3); transform: translateY(-2px); }
.profile-card.selected { border-color: var(--accent); background: rgba(10,132,255,0.15); }
.profile-icon { font-size: 36px; margin-bottom: 12px; }
.profile-name { font-size: 15px; font-weight: 600; margin-bottom: 4px; }
.profile-desc { font-size: 12px; color: var(--text2); line-height: 1.4; }

/* Quality cards */
.quality-card {
  background: var(--surface2); border: 2px solid transparent; border-radius: var(--radius-sm);
  padding: 20px; cursor: pointer; transition: all 0.2s; position: relative;
}
.quality-card:hover { background: var(--surface3); }
.quality-card.selected { border-color: var(--accent); background: rgba(10,132,255,0.15); }
.quality-card.recommended::after {
  content: "Recommended"; position: absolute; top: -10px; right: 12px;
  background: var(--success); color: #000; font-size: 10px; font-weight: 600;
  padding: 4px 8px; border-radius: 6px;
}
.quality-icon { font-size: 24px; margin-bottom: 8px; }
.quality-name { font-size: 15px; font-weight: 600; }
.quality-time { font-size: 12px; color: var(--accent); margin: 4px 0; }
.quality-desc { font-size: 12px; color: var(--text2); }

/* Enhancement options */
.enhancement { background: var(--surface2); border-radius: var(--radius-sm); padding: 20px; margin-bottom: 12px; }
.enhancement-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.enhancement-name { font-size: 15px; font-weight: 600; }
.enhancement-desc { font-size: 13px; color: var(--text2); margin-bottom: 16px; }
.enhancement-options { display: flex; gap: 8px; flex-wrap: wrap; }
.option-btn {
  background: var(--surface3); border: 2px solid transparent; border-radius: 8px;
  padding: 10px 16px; font-size: 13px; color: var(--text); cursor: pointer; transition: all 0.15s;
}
.option-btn:hover { background: var(--surface); }
.option-btn.selected { border-color: var(--accent); background: rgba(10,132,255,0.2); }
.option-tip { font-size: 11px; color: var(--text2); margin-top: 4px; }
.why-link { font-size: 12px; color: var(--accent); cursor: pointer; }
.why-text { font-size: 12px; color: var(--text2); margin-top: 8px; padding: 12px; background: var(--surface); border-radius: 8px; display: none; }
.why-text.show { display: block; }

/* File browser */
.browser-container { background: var(--surface2); border-radius: var(--radius-sm); overflow: hidden; }
.browser-path { padding: 12px 16px; background: var(--surface3); font-size: 13px; color: var(--text2); border-bottom: 1px solid rgba(255,255,255,0.1); }
.browser-drives { display: flex; gap: 8px; padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.1); flex-wrap: wrap; }
.drive-btn { background: var(--surface); border: none; color: var(--text); padding: 6px 12px; border-radius: 6px; font-size: 12px; cursor: pointer; }
.drive-btn:hover { background: var(--surface3); }
.browser-list { max-height: 300px; overflow-y: auto; }
.browser-item { display: flex; align-items: center; gap: 12px; padding: 12px 16px; cursor: pointer; border-bottom: 1px solid rgba(255,255,255,0.05); }
.browser-item:hover { background: var(--surface3); }
.browser-item.video { background: rgba(10,132,255,0.1); }
.browser-icon { font-size: 20px; }
.browser-name { flex: 1; font-size: 14px; }
.browser-size { font-size: 12px; color: var(--text2); }

/* Folder modal */
.folder-modal {
  display: none;
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0,0,0,0.8);
  z-index: 10000;
  align-items: center;
  justify-content: center;
}
.folder-modal.show {
  display: flex !important;
}

/* Selected file */
.selected-file { display: flex; align-items: center; gap: 16px; padding: 20px; background: rgba(48,209,88,0.15); border-radius: var(--radius-sm); margin-top: 16px; }
.selected-file-icon { font-size: 32px; }
.selected-file-info { flex: 1; }
.selected-file-name { font-size: 15px; font-weight: 500; }
.selected-file-path { font-size: 12px; color: var(--text2); }

/* Buttons */
.btn { display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  padding: 14px 28px; border: none; border-radius: var(--radius-sm);
  font-size: 16px; font-weight: 500; cursor: pointer; transition: all 0.2s; }
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { background: #409cff; transform: scale(1.02); }
.btn-secondary { background: var(--surface2); color: var(--text); }
.btn-secondary:hover { background: var(--surface3); }
.btn-large { padding: 18px 36px; font-size: 17px; border-radius: 14px; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

/* Navigation */
.nav-buttons { display: flex; justify-content: space-between; margin-top: 32px; }

/* YouTube input */
.youtube-input { display: flex; gap: 12px; margin-top: 16px; }
.youtube-input input { flex: 1; background: var(--surface2); border: 1px solid rgba(255,255,255,0.1);
  border-radius: 8px; padding: 14px 16px; color: var(--text); font-size: 15px; }
.youtube-input input:focus { outline: none; border-color: var(--accent); }

/* Summary */
.summary-item { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); }
.summary-item:last-child { border: none; }
.summary-label { color: var(--text2); white-space: nowrap; flex-shrink: 0; }
.summary-value { font-weight: 500; text-align: right; overflow-wrap: break-word; word-break: break-all; max-width: 70%; }

/* Progress */
.progress-container { text-align: center; padding: 40px; }
.progress-ring { width: 120px; height: 120px; margin: 0 auto 24px; }
.progress-ring circle { fill: none; stroke-width: 8; }
.progress-ring .bg { stroke: var(--surface2); }
.progress-ring .fg { stroke: var(--accent); stroke-linecap: round; transform: rotate(-90deg); transform-origin: center;
  transition: stroke-dashoffset 0.3s; }
.progress-text { font-size: 28px; font-weight: 600; }
.progress-message { color: var(--text2); margin-top: 8px; }

/* Status badge */
.status { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 500; }
.status-pending { background: var(--surface3); }
.status-running { background: var(--accent); }
.status-completed { background: var(--success); color: #000; }
.status-failed { background: var(--error); }
.status-cancelled { background: var(--error); }

/* Alerts */
.alert { padding: 16px 20px; border-radius: var(--radius-sm); margin-bottom: 16px; }
.alert-warning { background: rgba(255,159,10,0.15); border-left: 4px solid var(--warning); }
.alert-success { background: rgba(48,209,88,0.15); border-left: 4px solid var(--success); }
.alert-title { font-weight: 600; margin-bottom: 4px; }
.alert-text { font-size: 14px; color: var(--text2); }

/* Tabs */
.tabs { display: flex; gap: 4px; margin-bottom: 24px; }
.tab-btn { background: none; border: none; color: var(--text2); padding: 12px 20px;
  font-size: 14px; cursor: pointer; border-radius: 8px; transition: all 0.15s; }
.tab-btn:hover { background: var(--surface2); }
.tab-btn.active { background: var(--accent); color: #fff; }
</style>
</head>
<body>

<header style="display:flex;justify-content:space-between;align-items:center">
  <div class="logo">Frame<span>Wright</span></div>
  <button class="btn btn-secondary btn-sm" onclick="openModelsPanel()" style="padding:8px 16px;font-size:13px">Models</button>
</header>

<!-- Models Panel Modal -->
<div id="models-modal" style="display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.85);z-index:1000;overflow-y:auto">
  <div style="max-width:700px;margin:40px auto;padding:24px">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px">
      <h2 style="font-size:22px;font-weight:600">AI Models</h2>
      <button onclick="closeModelsPanel()" style="background:none;border:none;color:var(--text);font-size:28px;cursor:pointer">&times;</button>
    </div>
    <div class="card">
      <h3 style="font-size:13px;color:var(--text2);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:16px">Model Directory</h3>
      <div style="font-size:13px;color:var(--text2);background:var(--surface2);padding:10px 14px;border-radius:8px;word-break:break-all" id="model-dir-display">~/.framewright/models</div>
    </div>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
        <h3 style="font-size:13px;color:var(--text2);text-transform:uppercase;letter-spacing:0.5px;margin:0">Essential Models</h3>
        <button class="btn btn-primary btn-sm" onclick="downloadAllModels()" id="btn-dl-all" style="padding:6px 14px;font-size:12px">Download All</button>
      </div>
      <div id="models-essential-list"><p style="color:var(--text2)">Loading...</p></div>
    </div>
    <div class="card">
      <h3 style="font-size:13px;color:var(--text2);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:16px">Additional Models</h3>
      <div id="models-extra-list"><p style="color:var(--text2)">Loading...</p></div>
    </div>
  </div>
</div>

<main>
  <!-- STEP 1: Select Video -->
  <div class="step active" id="step-1">
    <div class="step-header">
      <div class="step-number">1</div>
      <h1 class="step-title">Choose Your Video</h1>
      <p class="step-subtitle">Select a video file from your computer or paste a YouTube URL</p>
    </div>

    <div class="tabs">
      <button class="tab-btn active" onclick="setInputMode('file')">From Computer</button>
      <button class="tab-btn" onclick="setInputMode('youtube')">YouTube URL</button>
    </div>

    <div id="input-file">
      <div class="browser-container">
        <div class="browser-drives" id="drives"></div>
        <div class="browser-path" id="current-path">Select a folder...</div>
        <div class="browser-list" id="file-list"></div>
      </div>
    </div>

    <div id="input-youtube" style="display:none">
      <div class="card">
        <p style="margin-bottom:16px">Paste a YouTube video URL and we'll download it for you.</p>
        <div class="youtube-input">
          <input type="url" id="youtube-url" placeholder="https://www.youtube.com/watch?v=...">
        </div>
        <div style="margin-top:16px">
          <label style="font-size:13px;color:var(--text2);display:block;margin-bottom:8px">Download to folder:</label>
          <div style="display:flex;gap:8px">
            <input type="text" id="youtube-download-dir" style="flex:1;background:var(--surface2);border:1px solid rgba(255,255,255,0.1);border-radius:8px;padding:12px;color:var(--text);font-size:14px" readonly>
            <button class="btn btn-secondary" onclick="browseYouTubeFolder()">Browse</button>
          </div>
        </div>
        <div id="youtube-status" style="margin-top:12px"></div>
      </div>

      <!-- Folder browser modal for YouTube -->
      <div id="yt-folder-modal" style="display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.8);z-index:1000;align-items:center;justify-content:center">
        <div style="background:var(--surface);border-radius:16px;width:90%;max-width:600px;max-height:80vh;overflow:hidden;display:flex;flex-direction:column">
          <div style="padding:20px;border-bottom:1px solid rgba(255,255,255,0.1);display:flex;justify-content:space-between;align-items:center">
            <h3 style="margin:0">Select Download Folder</h3>
            <button onclick="closeYtFolderModal()" style="background:none;border:none;color:var(--text);font-size:24px;cursor:pointer">&times;</button>
          </div>
          <div class="browser-drives" id="yt-drives"></div>
          <div class="browser-path" id="yt-current-path">Select a folder...</div>
          <div class="browser-list" id="yt-folder-list" style="flex:1;overflow-y:auto;max-height:300px"></div>
          <div style="padding:16px;border-top:1px solid rgba(255,255,255,0.1);display:flex;gap:12px;justify-content:flex-end">
            <button class="btn btn-secondary" onclick="closeYtFolderModal()">Cancel</button>
            <button class="btn btn-primary" onclick="selectYtFolder()">Select This Folder</button>
          </div>
        </div>
      </div>

    </div>

    <div class="selected-file" id="selected-file" style="display:none">
      <div class="selected-file-icon">🎬</div>
      <div class="selected-file-info">
        <div class="selected-file-name" id="selected-name"></div>
        <div class="selected-file-path" id="selected-path"></div>
      </div>
    </div>

    <div class="nav-buttons">
      <div></div>
      <button class="btn btn-primary" id="btn-next-1" onclick="handleStep1Continue()" disabled>Continue</button>
    </div>
  </div>

  <!-- STEP 2: What kind of video? -->
  <div class="step" id="step-2">
    <div class="step-header">
      <div class="step-number">2</div>
      <h1 class="step-title">What Kind of Video Is This?</h1>
      <p class="step-subtitle">This helps us choose the best AI models and settings</p>
    </div>

    <div class="card" style="margin-bottom:24px;border:1px solid rgba(10,132,255,0.3);text-align:center">
      <p style="font-size:14px;color:var(--text2)">💡 <strong>Not sure which to choose?</strong> The detailed Model Selection Guide is in <strong>Step 4: Fine-Tune Settings</strong></p>
    </div>

    <div class="card-grid" id="profiles"></div>

    <div class="nav-buttons">
      <button class="btn btn-secondary" onclick="prevStep(1)">Back</button>
      <button class="btn btn-primary" onclick="nextStep(3)">Continue</button>
    </div>
  </div>

  <!-- STEP 3: Quality level -->
  <div class="step" id="step-3">
    <div class="step-header">
      <div class="step-number">3</div>
      <h1 class="step-title">Choose Quality Level</h1>
      <p class="step-subtitle">Higher quality takes longer but produces better results</p>
    </div>

    <div class="card-grid" id="quality-levels"></div>

    <div class="nav-buttons">
      <button class="btn btn-secondary" onclick="prevStep(2)">Back</button>
      <button class="btn btn-primary" onclick="nextStep(4)">Continue</button>
    </div>
  </div>

  <!-- STEP 4: Fine-tune (optional) -->
  <div class="step" id="step-4">
    <div class="step-header">
      <div class="step-number">4</div>
      <h1 class="step-title">Fine-Tune Your Settings</h1>
      <p class="step-subtitle">We've selected good defaults. Adjust if you want more control.</p>
    </div>

    <!-- Model Selection Guide -->
    <div class="card" style="margin-bottom:24px;border:1px solid rgba(10,132,255,0.3)">
      <div style="display:flex;justify-content:space-between;align-items:center;cursor:pointer" onclick="toggleGuide()">
        <div>
          <h3 style="font-size:16px;font-weight:600;margin-bottom:4px">📖 Model Selection Guide</h3>
          <p style="font-size:13px;color:var(--text2)">Understand which models and settings to use</p>
        </div>
        <div id="guide-toggle" style="font-size:20px;transition:transform 0.2s">▼</div>
      </div>

      <div id="guide-content" style="display:none;margin-top:20px;padding-top:20px;border-top:1px solid rgba(255,255,255,0.1)">

        <!-- Upscaling Models Explained -->
        <div style="margin-bottom:24px;padding:16px;background:rgba(255,159,10,0.1);border-left:3px solid var(--warning);border-radius:8px">
          <h4 style="font-size:15px;font-weight:600;margin-bottom:12px;color:var(--warning)">🔬 Upscale Model: Ensemble vs Diffusion</h4>

          <div style="margin-bottom:16px">
            <div style="font-size:14px;font-weight:600;margin-bottom:8px;color:var(--text)">Real-ESRGAN (Default - Recommended)</div>
            <div style="font-size:12px;color:var(--text2);line-height:1.7">
              • <strong>Speed:</strong> Fast (~0.1-0.3 sec/frame on RTX 5090)<br>
              • <strong>Quality:</strong> Excellent for most content<br>
              • <strong>Best for:</strong> All real footage, photos, standard restoration<br>
              • <strong>Models:</strong> realesrgan-x4plus (real), realesr-animevideov3 (anime)<br>
              • <strong>Use when:</strong> 95% of cases - it's fast, reliable, and high quality
            </div>
          </div>

          <div style="margin-bottom:16px">
            <div style="font-size:14px;font-weight:600;margin-bottom:8px;color:var(--text)">Ensemble (Combine Multiple Models)</div>
            <div style="font-size:12px;color:var(--text2);line-height:1.7">
              • <strong>Speed:</strong> Slower (~0.3-1 sec/frame)<br>
              • <strong>Quality:</strong> Slightly better than single model<br>
              • <strong>Best for:</strong> When you want to hedge - combines Real-ESRGAN + HAT models<br>
              • <strong>How it works:</strong> Runs multiple models, blends results (weights configurable)<br>
              • <strong>Use when:</strong> Archival projects where you want absolute best quality
            </div>
          </div>

          <div style="margin-bottom:16px">
            <div style="font-size:14px;font-weight:600;margin-bottom:8px;color:var(--text)">Diffusion (Stable Diffusion SR)</div>
            <div style="font-size:12px;color:var(--text2);line-height:1.7">
              • <strong>Speed:</strong> Very slow (~5-15 sec/frame on RTX 5090)<br>
              • <strong>Quality:</strong> Can hallucinate details (realistic but not always accurate)<br>
              • <strong>Best for:</strong> Heavily damaged footage where some AI "guessing" is acceptable<br>
              • <strong>Warning:</strong> May invent details that weren't there - not ideal for archival<br>
              • <strong>Use when:</strong> Extreme degradation + you prioritize visual appeal over accuracy
            </div>
          </div>

          <div style="background:var(--surface);padding:12px;border-radius:6px;margin-top:12px">
            <div style="font-size:13px;font-weight:600;margin-bottom:6px;color:var(--accent)">💡 Recommendation for Your 1909 Film:</div>
            <div style="font-size:12px;color:var(--text2)">
              Use <strong style="color:var(--text)">Real-ESRGAN (realesrgan-x4plus)</strong> or <strong style="color:var(--text)">Ensemble</strong> if you have time.<br>
              <strong>Avoid Diffusion</strong> for archival - it may invent details that weren't in the original film.
            </div>
          </div>
        </div>

        <div style="margin-bottom:20px">
          <h4 style="font-size:14px;font-weight:600;margin-bottom:12px;color:var(--accent)">⭐ Complete Guide: 1909 B&W Historical Film</h4>
          <div style="background:rgba(10,132,255,0.1);border-left:3px solid var(--accent);padding:16px;border-radius:8px">
            <p style="font-size:13px;margin-bottom:12px"><strong>For heavily degraded B&W footage that should stay B&W:</strong></p>
            <ul style="font-size:12px;line-height:1.8;padding-left:20px;color:var(--text2)">
              <li>Upscale Model: <strong style="color:var(--text)">Real-ESRGAN x4plus</strong> or <strong>Ensemble</strong></li>
              <li>Scale Factor: <strong style="color:var(--text)">4x</strong> (Old footage needs maximum upscaling)</li>
              <li>Denoise: <strong style="color:var(--text)">Medium</strong> (Restormer or TAP)</li>
              <li>Colorization: <strong style="color:var(--error)">❌ DISABLE</strong> (keep B&W)</li>
              <li>Reference Images: <strong style="color:var(--success)">✓ Upload 3-5 historical photos</strong> from 1900s-1910s</li>
              <li>Grain Preservation: <strong style="color:var(--success)">✓ Enable</strong> (authentic film look)</li>
              <li>Frame Deduplication: <strong style="color:var(--success)">✓ Enable</strong> (0.98 threshold for YouTube padding)</li>
              <li>Quality: <strong style="color:var(--text)">Archival</strong> (CRF 18)</li>
            </ul>
          </div>
        </div>

        <div>
          <h4 style="font-size:14px;font-weight:600;margin-bottom:12px;color:var(--accent)">Understanding Video Issues</h4>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px">
            <div style="background:var(--surface2);padding:14px;border-radius:8px">
              <div style="font-size:13px;font-weight:600;margin-bottom:6px">Interlacing / Deinterlacing</div>
              <div style="font-size:11px;color:var(--text2);line-height:1.6">
                <strong>What:</strong> Old analog video drew frames in two passes (odd lines, then even lines)<br>
                <strong>Problem:</strong> Creates "comb" effect on modern screens<br>
                <strong>When:</strong> VHS tapes, old DVDs, TV broadcasts<br>
                <strong>NOT:</strong> Frame duplication (that's different)
              </div>
            </div>
            <div style="background:var(--surface2);padding:14px;border-radius:8px">
              <div style="font-size:13px;font-weight:600;margin-bottom:6px">Letterbox / Pillarbox</div>
              <div style="font-size:11px;color:var(--text2);line-height:1.6">
                <strong>What:</strong> Black bars on edges (aspect ratio mismatch)<br>
                <strong>Letterbox:</strong> Bars on top/bottom (widescreen on old TVs)<br>
                <strong>Pillarbox:</strong> Bars on left/right (old content on widescreen)<br>
                <strong>NOT:</strong> YouTube padding (that's frame duplication)
              </div>
            </div>
            <div style="background:var(--surface2);padding:14px;border-radius:8px">
              <div style="font-size:13px;font-weight:600;margin-bottom:6px">Frame Duplication (YouTube Padding)</div>
              <div style="font-size:11px;color:var(--text2);line-height:1.6">
                <strong>What:</strong> Same frame repeated to increase FPS (16fps → 24fps)<br>
                <strong>When:</strong> Old silent films uploaded to YouTube<br>
                <strong>Fix:</strong> Frame Deduplication (auto-enabled for silent era)<br>
                <strong>NOT:</strong> Interlacing or black bars
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div style="display:flex;justify-content:center;margin-bottom:24px">
      <div style="display:inline-flex;background:var(--surface2);border-radius:10px;padding:3px">
        <button class="tab-btn active" id="btn-simple" onclick="setAdvancedMode(false)" style="border-radius:8px">Simple</button>
        <button class="tab-btn" id="btn-advanced" onclick="setAdvancedMode(true)" style="border-radius:8px">Advanced</button>
      </div>
    </div>

    <div id="enhancements"></div>

    <!-- Reference Photos -->
    <div class="card" style="margin-top:24px;border:1px dashed var(--border)">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
        <span style="font-size:24px">🖼️</span>
        <div>
          <div style="font-weight:600">Reference Photos (Optional)</div>
          <div style="font-size:13px;color:var(--text2)">Add high-quality photos of the same locations/subjects to guide AI detail generation</div>
        </div>
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        <input type="text" id="ref-photos-dir" placeholder="Folder with reference photos..."
          style="flex:1;padding:10px 14px;border-radius:8px;border:1px solid var(--border);background:var(--surface2);color:var(--text);font-size:14px" />
        <button class="btn btn-secondary" onclick="openRefFolderModal()" style="white-space:nowrap">📁 Browse</button>
        <button class="btn btn-secondary" onclick="checkRefFolder()" style="white-space:nowrap">✓ Check</button>
      </div>
      <div id="ref-preview" style="margin-top:12px;display:none">
        <div style="font-size:12px;color:var(--text2)" id="ref-count"></div>
      </div>
      <div style="font-size:12px;color:var(--text2);margin-top:8px">
        <strong>How many?</strong> 3-10 photos is ideal. Quality > quantity. Pick clear, well-lit photos of the same locations/subjects shown in your video.
        <br><br>
        <strong>What works best:</strong> Modern photos of historical locations, clear portraits of people shown in the film, or high-res reference images from the same era.
        <br><br>
        Uses IP-Adapter + ControlNet to transfer texture and detail while preserving original composition. Adds ~1-3 sec/frame. Leave empty to skip.
      </div>
    </div>

    <!-- Folder browser modal for Reference Photos (Step 4 only) -->
    <div id="ref-folder-modal" class="folder-modal">
      <div style="background:var(--surface);border-radius:16px;width:90%;max-width:600px;max-height:80vh;overflow:hidden;display:flex;flex-direction:column">
        <div style="padding:20px;border-bottom:1px solid rgba(255,255,255,0.1);display:flex;justify-content:space-between;align-items:center">
          <h3 style="margin:0">Select Reference Photos Folder</h3>
          <button onclick="closeRefFolderModal()" style="background:none;border:none;color:var(--text);font-size:24px;cursor:pointer">&times;</button>
        </div>
        <div class="browser-drives" id="ref-drives"></div>
        <div class="browser-path" id="ref-current-path">Select a folder...</div>
        <div class="browser-list" id="ref-folder-list" style="flex:1;overflow-y:auto;max-height:300px"></div>
        <div style="padding:16px;border-top:1px solid rgba(255,255,255,0.1);display:flex;gap:12px;justify-content:flex-end">
          <button class="btn btn-secondary" onclick="closeRefFolderModal()">Cancel</button>
          <button class="btn btn-primary" onclick="selectRefFolder()">Select This Folder</button>
        </div>
      </div>
    </div>

    <div class="nav-buttons">
      <button class="btn btn-secondary" onclick="prevStep(3)">Back</button>
      <button class="btn btn-primary" onclick="nextStep(5)">Review & Start</button>
    </div>
  </div>

  <!-- STEP 5: Review & Start -->
  <div class="step" id="step-5">
    <div class="step-header">
      <div class="step-number">5</div>
      <h1 class="step-title">Ready to Restore</h1>
      <p class="step-subtitle">Review your settings and start the restoration</p>
    </div>

    <div class="card">
      <h3 style="margin-bottom:16px">Summary</h3>
      <div id="summary"></div>
    </div>

    <div class="card">
      <h3 style="margin-bottom:12px">Output Location</h3>
      <p style="color:var(--text2);font-size:14px;margin-bottom:12px">Where should we save the restored video?</p>
      <input type="text" id="output-path" style="width:100%;background:var(--surface2);border:1px solid rgba(255,255,255,0.1);border-radius:8px;padding:14px;color:var(--text);font-size:14px">
    </div>

    <div class="card">
      <h3 style="margin-bottom:12px">System Check</h3>
      <div id="hw-check"><p style="color:var(--text2)">Checking...</p></div>
    </div>

    <div class="nav-buttons">
      <button class="btn btn-secondary" onclick="prevStep(4)">Back</button>
      <button class="btn btn-primary btn-large" onclick="startRestoration()">Start Restoration</button>
    </div>
  </div>

  <!-- STEP 6: Progress -->
  <div class="step" id="step-6">
    <div class="progress-container">
      <svg class="progress-ring" viewBox="0 0 120 120">
        <circle class="bg" cx="60" cy="60" r="52"/>
        <circle class="fg" id="progress-circle" cx="60" cy="60" r="52" stroke-dasharray="327" stroke-dashoffset="327"/>
      </svg>
      <div class="progress-text" id="progress-percent">0%</div>
      <div class="progress-message" id="progress-message">Starting...</div>
    </div>

    <!-- Time & speed stats -->
    <div class="card">
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;text-align:center">
        <div>
          <div style="font-size:12px;color:var(--text2)">Elapsed</div>
          <div style="font-size:20px;font-weight:600;font-variant-numeric:tabular-nums" id="stat-elapsed">0:00</div>
        </div>
        <div>
          <div style="font-size:12px;color:var(--text2)">Remaining</div>
          <div style="font-size:20px;font-weight:600;font-variant-numeric:tabular-nums" id="stat-eta">--:--</div>
        </div>
        <div>
          <div style="font-size:12px;color:var(--text2)">Speed</div>
          <div style="font-size:20px;font-weight:600;font-variant-numeric:tabular-nums" id="stat-fps">-- fps</div>
        </div>
      </div>
    </div>

    <!-- Pipeline stages -->
    <div class="card">
      <h2 style="font-size:13px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:16px">Processing Pipeline</h2>
      <div id="pipeline-stages"></div>
    </div>

    <!-- Job info -->
    <div class="card" id="job-details">
      <div class="summary-item">
        <span class="summary-label">Status</span>
        <span class="status" id="job-status">Starting</span>
      </div>
      <div class="summary-item">
        <span class="summary-label">Input</span>
        <span class="summary-value" id="job-input" style="font-size:13px;word-break:break-all">-</span>
      </div>
      <div class="summary-item">
        <span class="summary-label">Output</span>
        <span class="summary-value" id="job-output" style="font-size:13px;word-break:break-all">-</span>
      </div>
      <div class="summary-item" id="frames-row" style="display:none">
        <span class="summary-label">Frames</span>
        <span class="summary-value" id="stat-frames">-</span>
      </div>
    </div>

    <!-- Live log -->
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <h2 style="font-size:13px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:0.5px;margin:0">Activity Log</h2>
        <div style="display:flex;gap:12px">
          <span class="why-link" onclick="copyActivityLog()" title="Copy log to clipboard">Copy</span>
          <span class="why-link" onclick="document.getElementById('log-output').style.display = document.getElementById('log-output').style.display === 'none' ? 'block' : 'none'">Toggle</span>
        </div>
      </div>
      <div id="log-output" style="background:var(--surface2);border-radius:8px;padding:12px;max-height:200px;overflow-y:auto;font-family:monospace;font-size:12px;line-height:1.6;color:var(--text2)">
        Waiting for output...
      </div>
    </div>

    <!-- Progress bar -->
    <div class="card" id="progress-bar-card">
      <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text2);margin-bottom:6px">
        <span id="progress-bar-label">Processing...</span>
        <span id="progress-bar-pct">0%</span>
      </div>
      <div style="background:var(--surface2);border-radius:4px;height:8px;overflow:hidden">
        <div id="progress-bar-fill" style="background:var(--accent);height:100%;width:0%;transition:width 0.5s;border-radius:4px"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text2);margin-top:4px">
        <span id="progress-bar-frames"></span>
        <span id="progress-bar-speed"></span>
      </div>
    </div>

    <div style="display:flex;gap:12px;justify-content:center;margin-top:24px">
      <button class="btn btn-secondary" id="btn-stop-job" onclick="stopJob()" style="background:var(--error)">Stop</button>
      <button class="btn btn-secondary" onclick="location.reload()">Start Another</button>
    </div>
  </div>
</main>

<script>
// State
let inputPath = '';
let selectedProfile = 'auto';
let selectedQuality = 'balanced';
let enhancements = {
  upscale: '2x', denoise: 'light', face: 'subtle', framerate: 'off',
  color: 'auto', colorize: 'off', dedup: 'off', dedup_threshold: 98, stabilize: 'off', output_format: 'mkv',
  deinterlace: 'off', scratch_repair: 'off', audio_enhance: 'full',
  encoder: 'h265', crf: '18', letterbox: 'off', ivtc: 'off', vhs_fixes: 'off',
  subtitles: 'keep', watermark: 'off', preview_mode: 'off',
  sr_model: 'realesrgan', face_model: 'gfpgan', ref_strength: 'medium', temporal_method: 'off',
  hdr: 'off', perceptual: 'balanced', grain_preserve: 'off',
};
let advancedMode = false;
let currentJobId = null;

const profiles = ''' + json.dumps(CONTENT_PROFILES) + ''';
const qualityLevels = ''' + json.dumps(QUALITY_LEVELS) + ''';
const enhancementOptions = ''' + json.dumps(ENHANCEMENTS) + ''';

// Step navigation
function showStep(n) {
  document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
  document.getElementById('step-' + n).classList.add('active');
  window.scrollTo(0, 0);
}
function nextStep(n) { showStep(n); }
function prevStep(n) { showStep(n); }

// Input mode
let inputMode = 'file';
function setInputMode(mode) {
  inputMode = mode;
  document.querySelectorAll('.tabs .tab-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('input-file').style.display = mode === 'file' ? 'block' : 'none';
  document.getElementById('input-youtube').style.display = mode === 'youtube' ? 'block' : 'none';
  // Update button text
  const btn = document.getElementById('btn-next-1');
  if (mode === 'youtube') {
    btn.disabled = false;
    btn.textContent = 'Download & Continue';
  } else {
    btn.textContent = 'Continue';
    btn.disabled = !inputPath;
  }
}

// Handle step 1 continue - for YouTube, download first then advance
async function handleStep1Continue() {
  if (inputMode === 'youtube') {
    const url = document.getElementById('youtube-url').value;
    if (!url) { alert('Please enter a YouTube URL'); return; }
    const btn = document.getElementById('btn-next-1');
    btn.disabled = true;
    btn.textContent = 'Downloading...';
    await downloadYouTube();
    btn.textContent = 'Download & Continue';
    btn.disabled = false;
    if (inputPath) nextStep(2);
  } else {
    if (inputPath) nextStep(2);
  }
}

// File browser
let drives = [];

async function loadDrives() {
  const r = await fetch('/api/drives');
  drives = await r.json();
  document.getElementById('drives').innerHTML = drives.map(d =>
    `<button class="drive-btn" onclick="browse('${d.path.replace(/\\\\/g, '\\\\\\\\')}')">${d.name}</button>`
  ).join('');

  // Set default YouTube download folder
  fetch('/api/default_download_folder').then(r => r.json()).then(data => {
    document.getElementById('youtube-download-dir').value = data.path;
  }).catch(() => {});
  if (drives.length) browse(drives[0].path);
}

async function browse(path) {
  const r = await fetch('/api/browse?path=' + encodeURIComponent(path));
  const data = await r.json();
  if (data.error) { alert(data.error); return; }

  document.getElementById('current-path').textContent = data.path;

  let html = '';
  if (data.parent) {
    html += `<div class="browser-item" onclick="browse('${data.parent.replace(/\\\\/g, '\\\\\\\\')}')">
      <span class="browser-icon">📁</span><span class="browser-name">..</span></div>`;
  }

  data.items.forEach(item => {
    const escaped = item.path.replace(/\\\\/g, '\\\\\\\\');
    if (item.is_dir) {
      html += `<div class="browser-item" onclick="browse('${escaped}')">
        <span class="browser-icon">📁</span><span class="browser-name">${item.name}</span></div>`;
    } else if (item.is_video) {
      html += `<div class="browser-item video" onclick="selectFile('${escaped}', '${item.name}')">
        <span class="browser-icon">🎬</span><span class="browser-name">${item.name}</span>
        <span class="browser-size">${formatSize(item.size)}</span></div>`;
    }
  });

  document.getElementById('file-list').innerHTML = html || '<div style="padding:20px;color:var(--text2)">No video files here</div>';
}

function selectFile(path, name) {
  inputPath = path;
  document.getElementById('selected-file').style.display = 'flex';
  document.getElementById('selected-name').textContent = name;
  document.getElementById('selected-path').textContent = path;
  document.getElementById('btn-next-1').disabled = false;

  // Auto-generate output path
  const base = path.replace(/\\\\.[^.]+$/, '');
  document.getElementById('output-path').value = base + '_restored.mkv';
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + ' KB';
  if (bytes < 1024*1024*1024) return (bytes/1024/1024).toFixed(0) + ' MB';
  return (bytes/1024/1024/1024).toFixed(2) + ' GB';
}

// YouTube
let ytCurrentFolder = '';

async function downloadYouTube() {
  const url = document.getElementById('youtube-url').value;
  const downloadDir = document.getElementById('youtube-download-dir').value;
  if (!url) { alert('Please enter a URL'); return; }

  document.getElementById('youtube-status').innerHTML = '<p style="color:var(--accent)">⏳ Downloading... this may take a moment</p>';

  const r = await fetch('/api/youtube', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, download_dir: downloadDir || null })
  });
  const data = await r.json();

  if (data.error) {
    document.getElementById('youtube-status').innerHTML = `<p style="color:var(--error)">${data.error}</p>`;
  } else {
    document.getElementById('youtube-status').innerHTML = `
      <div class="alert alert-success">
        <div class="alert-title">Downloaded: ${data.title}</div>
        <div class="alert-text" style="margin-top:8px">
          Project folder: <strong>${data.folder}</strong><br>
          Original: <code>originals/</code><br>
          Restored output will go to: <code>restored/</code>
        </div>
      </div>`;
    selectFile(data.path, data.title);
    if (data.restored_path) {
      document.getElementById('output-path').value = data.restored_path;
    }
  }
}

function browseYouTubeFolder() {
  document.getElementById('yt-folder-modal').style.display = 'flex';
  // Initialize drives
  const drivesDiv = document.getElementById('yt-drives');
  drivesDiv.innerHTML = drives.map(d =>
    `<button class="drive-btn" onclick="browseYtFolder('${d.path.replace(/\\\\/g, '\\\\\\\\')}')">${d.name}</button>`
  ).join('');
  if (drives.length) browseYtFolder(drives[0].path);
}

async function browseYtFolder(path) {
  const r = await fetch('/api/browse?path=' + encodeURIComponent(path) + '&folders_only=1');
  const data = await r.json();
  if (data.error) { alert(data.error); return; }

  ytCurrentFolder = data.path;
  document.getElementById('yt-current-path').textContent = data.path;

  let html = '';
  if (data.parent) {
    html += `<div class="browser-item" onclick="browseYtFolder('${data.parent.replace(/\\\\/g, '\\\\\\\\')}')">
      <span class="browser-icon">📁</span><span class="browser-name">..</span></div>`;
  }
  for (const item of data.items) {
    if (item.is_dir) {
      const escaped = item.path.replace(/\\\\/g, '\\\\\\\\');
      html += `<div class="browser-item" onclick="browseYtFolder('${escaped}')">
        <span class="browser-icon">📁</span><span class="browser-name">${item.name}</span></div>`;
    }
  }
  document.getElementById('yt-folder-list').innerHTML = html || '<p style="padding:16px;color:var(--text2)">No subfolders</p>';
}

function closeYtFolderModal() {
  document.getElementById('yt-folder-modal').style.display = 'none';
}

function selectYtFolder() {
  document.getElementById('youtube-download-dir').value = ytCurrentFolder;
  closeYtFolderModal();
}

// Profiles
function renderProfiles() {
  document.getElementById('profiles').innerHTML = Object.entries(profiles).map(([key, p]) => `
    <div class="profile-card ${key === selectedProfile ? 'selected' : ''}" onclick="selectProfile('${key}')">
      <div class="profile-icon">${p.icon}</div>
      <div class="profile-name">${p.name}</div>
      <div class="profile-desc">${p.description}</div>
    </div>
  `).join('');
}

function selectProfile(key) {
  selectedProfile = key;
  renderProfiles();

  // Apply generic profile defaults (bw_restore sets everything explicitly below)
  if (key !== 'bw_restore' && profiles[key].settings) {
    const s = profiles[key].settings;
    if (s.scale) enhancements.upscale = s.scale + 'x';
    if (s.denoise !== undefined) enhancements.denoise = s.denoise ? 'medium' : 'off';
    if (s.face_enhance !== undefined) enhancements.face = s.face_enhance ? 'subtle' : 'off';
  }

  // Profile-specific defaults
  if (key === 'bw_colorize') {
    enhancements.colorize = 'auto';
    enhancements.color = 'off';
  } else {
    enhancements.colorize = 'off';
  }

  if (key === 'bw_restore') {
    // ---- MAXIMUM RESTORATION for heavily degraded B&W film ----
    // Simple settings
    enhancements.upscale = '4x';           // Max upscale — old film is very low res
    enhancements.denoise = 'medium';        // Will auto-cap to light if ref enhance is heavy
    enhancements.face = 'full';             // Full face restore — faces are severely degraded
    enhancements.dedup = 'smart';           // mpdecimate — removes YouTube frame padding
    enhancements.framerate = '30';          // Smooth motion interpolation for B&W film
    enhancements.color = 'off';             // It's B&W, no color correction
    enhancements.stabilize = 'off';         // Old film tripod shots, usually no shake
    enhancements.output_format = 'mp4';     // MP4 for best compatibility
    // Advanced settings — best models
    enhancements.sr_model = 'ensemble';     // Ensemble SR for maximum quality (HAT + Real-ESRGAN)
    enhancements.face_model = 'gfpgan';     // Most aggressive face restore for tiny/blurry faces
    enhancements.ref_strength = 'heavy';    // Max detail from reference photos
    enhancements.temporal_method = 'hybrid'; // Hybrid (RAFT + Attention) for best temporal consistency
    enhancements.scratch_repair = 'full';   // Full scratch/damage repair
    enhancements.grain_preserve = 'off';    // Remove grain — we want clean detail, not film texture
    enhancements.perceptual = 'balanced';   // Balanced perceptual tuning for best results
    enhancements.crf = '14';                // Near-lossless encoding
    enhancements.encoder = 'h265';          // Best quality/size ratio
    enhancements.letterbox = 'auto';        // Clean up any uneven black bars
    enhancements.deinterlace = 'off';       // Film is progressive
    enhancements.ivtc = 'off';              // Not relevant for pre-TV era film
    enhancements.hdr = 'off';               // B&W doesn't benefit from HDR
    enhancements.audio_enhance = 'full';    // Full audio restoration
    // Auto-select maximum quality
    selectedQuality = 'maximum';
    renderQuality();
  } else if (key === 'old_film' || key === 'vhs_tape' || key === 'bw_colorize') {
    enhancements.dedup = 'smart';
    enhancements.ref_strength = 'heavy';
    enhancements.scratch_repair = (key === 'old_film' || key === 'bw_colorize') ? 'full' : 'off';
    enhancements.deinterlace = (key === 'vhs_tape') ? 'auto' : 'off';
    enhancements.vhs_fixes = (key === 'vhs_tape') ? 'auto' : 'off';
    enhancements.ivtc = (key === 'old_film') ? 'auto' : 'off';
    enhancements.grain_preserve = (key === 'old_film') ? 'medium' : 'off';
    enhancements.perceptual = (key === 'old_film') ? 'faithful' : 'balanced';
    enhancements.temporal_method = 'optical_flow';
    enhancements.audio_enhance = 'full';
  } else {
    enhancements.ref_strength = 'medium';
    enhancements.dedup = 'off';
    enhancements.scratch_repair = 'off';
    enhancements.deinterlace = (key === 'home_video') ? 'auto' : 'off';
    enhancements.vhs_fixes = 'off';
    enhancements.ivtc = 'off';
    enhancements.grain_preserve = 'off';
    enhancements.perceptual = 'balanced';
    enhancements.temporal_method = 'off';
    enhancements.audio_enhance = 'full';
  }
}

// Quality levels
function renderQuality() {
  document.getElementById('quality-levels').innerHTML = Object.entries(qualityLevels).map(([key, q]) => `
    <div class="quality-card ${key === selectedQuality ? 'selected' : ''} ${q.recommended ? 'recommended' : ''}" onclick="selectQuality('${key}')">
      <div class="quality-icon">${q.icon}</div>
      <div class="quality-name">${q.name}</div>
      <div class="quality-time">${q.time}</div>
      <div class="quality-desc">${q.description}</div>
    </div>
  `).join('');
}

function selectQuality(key) {
  selectedQuality = key;

  // Auto-apply best settings for high quality presets
  if (key === 'maximum' || key === 'high') {
    enhancements.sr_model = 'ensemble';         // Best upscaling quality
    enhancements.temporal_method = 'hybrid';     // Best temporal consistency
    enhancements.perceptual = 'balanced';        // Best perceptual tuning
  }

  renderQuality();
  renderEnhancements();  // Re-render to show updated settings
}

// Advanced mode toggle
function setAdvancedMode(adv) {
  advancedMode = adv;
  document.getElementById('btn-simple').className = 'tab-btn' + (adv ? '' : ' active');
  document.getElementById('btn-advanced').className = 'tab-btn' + (adv ? ' active' : '');
  renderEnhancements();
}

// Enhancements
function renderEnhancements() {
  let lastSection = '';
  document.getElementById('enhancements').innerHTML = Object.entries(enhancementOptions)
    .filter(([key, e]) => {
      // Hide options not relevant to selected profile
      if (e.hide_for && e.hide_for.includes(selectedProfile)) return false;
      if (e.only_for && !e.only_for.includes(selectedProfile)) return false;
      // show_for: only show if profile matches (when set)
      if (e.show_for && !e.show_for.includes(selectedProfile)) return false;
      // Group filter
      if (!advancedMode && e.group === 'advanced') return false;
      return true;
    })
    .map(([key, e]) => {
      let sectionHeader = '';
      if (e.section && e.section !== lastSection) {
        lastSection = e.section;
        sectionHeader = `<div style="grid-column:1/-1;margin:16px 0 4px;padding:8px 0 4px;border-bottom:1px solid var(--border);font-size:13px;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:1px">${e.section}</div>`;
      }
      let controlHtml;
      if (e.type === 'slider') {
        const parsed = parseInt(enhancements[key]);
        const val = (isNaN(parsed)) ? e.default : parsed;
        const labelEntries = e.labels ? Object.entries(e.labels).sort((a,b) => parseInt(a[0]) - parseInt(b[0])) : [];
        const closestLabel = labelEntries.reduce((best, [lv, lt]) => Math.abs(parseInt(lv) - val) < Math.abs(parseInt(best[0]) - val) ? [lv, lt] : best, labelEntries[0] || ['0','']);

        // For dedup_threshold, display as decimal (0.90 format)
        const isDedup = key === 'dedup_threshold';
        const displayVal = isDedup ? (val / 100).toFixed(2) : val;
        const step = e.step || 1;

        const labelsJson = e.labels ? JSON.stringify(e.labels) : '{}';
        const labelHtml = e.labels ? `<span style="font-size:11px;color:var(--accent);min-width:110px">${closestLabel[1]}</span>` : '';

        // Build oninput handler based on whether this is dedup slider
        const valueDisplayCode = isDedup ? '(this.value / 100).toFixed(2)' : 'this.value';
        const labelUpdateCode = e.labels ? `this.nextElementSibling.nextElementSibling.textContent = (${labelsJson})[this.value] || '';` : '';

        controlHtml = `
          <div style="display:flex;align-items:center;gap:12px;margin-top:8px">
            <span style="font-size:12px;color:var(--text2);min-width:55px">Quality</span>
            <input type="range" min="${e.min}" max="${e.max}" step="${step}" value="${val}"
              style="flex:1;accent-color:var(--accent);height:6px;cursor:pointer"
              oninput="setEnhancement('${key}', this.value); this.nextElementSibling.textContent = ${valueDisplayCode}; ${labelUpdateCode}"
            />
            <span style="font-size:18px;font-weight:bold;color:var(--text);min-width:50px;text-align:center">${displayVal}</span>
            ${labelHtml}
          </div>
          ${labelEntries.length > 0 ? `<div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text2);margin-top:2px;padding:0 67px 0 67px">
            ${labelEntries.map(([lv, lt]) => `<span>${lv}</span>`).join('')}
          </div>` : ''}`;
      } else {
        controlHtml = `
          <div class="enhancement-options">
            ${e.options.map(o => `
              <button class="option-btn ${enhancements[key] === o.value ? 'selected' : ''}" onclick="setEnhancement('${key}', '${o.value}')">
                ${o.name}
                ${o.tip ? `<div class="option-tip">${o.tip}</div>` : ''}
              </button>
            `).join('')}
          </div>`;
      }
      return sectionHeader + `
      <div class="enhancement">
        <div class="enhancement-header">
          <div class="enhancement-name">${e.name}${e.group === 'advanced' ? ' <span style="font-size:10px;color:var(--accent);background:rgba(10,132,255,0.2);padding:2px 6px;border-radius:4px;margin-left:8px">ADV</span>' : ''}</div>
          <span class="why-link" onclick="toggleWhy('${key}')">Why?</span>
        </div>
        <div class="enhancement-desc">${e.description}</div>
        ${controlHtml}
        <div class="why-text" id="why-${key}">${e.why}</div>
      </div>`;
    }).join('');
}

function setEnhancement(key, value) {
  enhancements[key] = value;
  // Don't re-render for slider changes (slider updates its own display inline)
  const eOpt = enhancementOptions[key];
  if (eOpt && eOpt.type === 'slider') return;
  renderEnhancements();
}

function toggleWhy(key) {
  document.getElementById('why-' + key).classList.toggle('show');
}

// Toggle model selection guide
function toggleGuide() {
  const content = document.getElementById('guide-content');
  const toggle = document.getElementById('guide-toggle');
  if (content.style.display === 'none') {
    content.style.display = 'block';
    toggle.style.transform = 'rotate(180deg)';
  } else {
    content.style.display = 'none';
    toggle.style.transform = 'rotate(0deg)';
  }
}

// Models panel
async function openModelsPanel() {
  document.getElementById('models-modal').style.display = 'block';
  await refreshModelsPanel();
}

function closeModelsPanel() {
  document.getElementById('models-modal').style.display = 'none';
}

async function refreshModelsPanel() {
  try {
    const r = await fetch('/api/models');
    const data = await r.json();

    document.getElementById('model-dir-display').textContent = data.model_dir;

    // Essential models
    document.getElementById('models-essential-list').innerHTML = data.essential.map(m => `
      <div style="display:flex;justify-content:space-between;align-items:center;padding:12px 0;border-bottom:1px solid rgba(255,255,255,0.05)">
        <div>
          <div style="font-size:14px;font-weight:500">${m.name}</div>
          <div style="font-size:12px;color:var(--text2)">${m.description} (${m.size_mb} MB)</div>
        </div>
        ${m.downloaded
          ? '<span style="color:var(--success);font-size:13px;font-weight:500">Downloaded</span>'
          : `<button class="btn btn-primary btn-sm" style="padding:6px 12px;font-size:12px" onclick="downloadSingleModel('${m.name}', this)">Download</button>`
        }
      </div>
    `).join('');

    // Extra models
    document.getElementById('models-extra-list').innerHTML = data.extra.map(m => `
      <div style="display:flex;justify-content:space-between;align-items:center;padding:12px 0;border-bottom:1px solid rgba(255,255,255,0.05)">
        <div>
          <div style="font-size:14px;font-weight:500">${m.name}</div>
          <div style="font-size:12px;color:var(--text2)">${m.description}${m.info_only ? '' : ` (${m.size_mb} MB)`}</div>
        </div>
        ${m.info_only
          ? '<span style="color:var(--text2);font-size:12px;font-style:italic">Manual install</span>'
          : m.downloaded
            ? '<span style="color:var(--success);font-size:13px;font-weight:500">Downloaded</span>'
            : `<button class="btn btn-secondary btn-sm" style="padding:6px 12px;font-size:12px" onclick="downloadSingleModel('${m.name}', this)">Download</button>`
        }
      </div>
    `).join('');

    // Update Download All button
    const allDownloaded = data.essential.every(m => m.downloaded);
    const dlBtn = document.getElementById('btn-dl-all');
    if (allDownloaded) { dlBtn.textContent = 'All Downloaded'; dlBtn.disabled = true; }
  } catch (e) {
    document.getElementById('models-essential-list').innerHTML = '<p style="color:var(--error)">Failed to load models.</p>';
  }
}

async function downloadSingleModel(name, btn) {
  if (btn) { btn.disabled = true; btn.textContent = 'Downloading...'; }
  try {
    const r = await fetch('/api/models/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    const result = await r.json();
    if (result.status === 'downloading') {
      // Poll until done
      const poll = async () => {
        const pr = await fetch('/api/models/download', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name })
        });
        const ps = await pr.json();
        if (ps.status === 'downloading') {
          if (btn) btn.textContent = 'Downloading...';
          setTimeout(async () => { await poll(); }, 2000);
        } else {
          await refreshModelsPanel();
        }
      };
      setTimeout(async () => { await poll(); }, 2000);
    } else {
      await refreshModelsPanel();
    }
  } catch (e) {
    alert('Download failed: ' + e.message);
    if (btn) { btn.disabled = false; btn.textContent = 'Download'; }
  }
}

async function downloadAllModels() {
  const btn = document.getElementById('btn-dl-all');
  btn.disabled = true; btn.textContent = 'Downloading...';
  try {
    const r = await fetch('/api/models/download_essential', { method: 'POST' });
    const result = await r.json();
    const anyDownloading = result.models && result.models.some(m => m.status === 'downloading');
    if (anyDownloading) {
      const pollAll = async () => {
        const mr = await fetch('/api/models');
        const data = await mr.json();
        const stillMissing = data.essential.some(m => !m.downloaded);
        if (stillMissing) {
          btn.textContent = 'Downloading...';
          setTimeout(async () => { await pollAll(); }, 3000);
        } else {
          await refreshModelsPanel();
        }
      };
      setTimeout(async () => { await pollAll(); }, 3000);
    } else {
      await refreshModelsPanel();
    }
  } catch (e) {
    alert('Download failed: ' + e.message);
    btn.disabled = false; btn.textContent = 'Download All';
  }
}

// Summary
function renderSummary() {
  const profile = profiles[selectedProfile];
  const quality = qualityLevels[selectedQuality];

  let summaryHtml = `
    <div class="summary-item"><span class="summary-label">Video</span><span class="summary-value">${inputPath.split(/[\\\\/]/).pop()}</span></div>
    <div class="summary-item"><span class="summary-label">Content Type</span><span class="summary-value">${profile.icon} ${profile.name}</span></div>
    <div class="summary-item"><span class="summary-label">Quality</span><span class="summary-value">${quality.icon} ${quality.name}</span></div>
    <div class="summary-item"><span class="summary-label">Upscale</span><span class="summary-value">${enhancements.upscale}</span></div>
    <div class="summary-item"><span class="summary-label">Denoise</span><span class="summary-value">${enhancements.denoise}</span></div>
    <div class="summary-item"><span class="summary-label">Face Enhancement</span><span class="summary-value">${enhancements.face}</span></div>
    <div class="summary-item"><span class="summary-label">Frame Rate</span><span class="summary-value">${enhancements.framerate === 'off' ? 'Keep Original' : enhancements.framerate + ' FPS'}</span></div>
    <div class="summary-item"><span class="summary-label">Deduplication</span><span class="summary-value">${enhancements.dedup}</span></div>`;

  const refDir = document.getElementById('ref-photos-dir').value;
  if (refDir) {
    summaryHtml += `<div class="summary-item"><span class="summary-label">Reference Photos</span><span class="summary-value">${enhancements.ref_strength} enhancement</span></div>`;
  }

  if (selectedProfile === 'bw_colorize') {
    summaryHtml += `<div class="summary-item"><span class="summary-label">Colorization</span><span class="summary-value">${enhancements.colorize}</span></div>`;
  } else {
    summaryHtml += `<div class="summary-item"><span class="summary-label">Color Fix</span><span class="summary-value">${enhancements.color}</span></div>`;
  }

  summaryHtml += `
    <div class="summary-item"><span class="summary-label">Output Format</span><span class="summary-value">${enhancements.output_format.toUpperCase()}</span></div>`;

  document.getElementById('summary').innerHTML = summaryHtml;

  // Update output path extension to match selected format
  const outPath = document.getElementById('output-path').value;
  if (outPath) {
    const base = outPath.replace(/\\\\.[^.]+$/, '');
    document.getElementById('output-path').value = base + '.' + enhancements.output_format;
  }

  // Run hardware check
  checkHardware();
}

// Hardware / dependency check
async function checkHardware() {
  const container = document.getElementById('hw-check');
  if (!container) return;
  container.innerHTML = '<p style="color:var(--text2)">Checking system...</p>';

  try {
    const [sysR, setupR] = await Promise.all([fetch('/api/system'), fetch('/api/setup')]);
    const sys = await sysR.json();
    const setup = await setupR.json();

    let html = '';
    // GPU status
    if (sys.gpu && sys.gpu !== 'Not detected') {
      html += `<div class="alert alert-success"><div class="alert-title">GPU Ready</div>
        <div class="alert-text">${sys.gpu} (${sys.vram_total} GB VRAM)</div></div>`;
    } else {
      html += `<div class="alert alert-warning"><div class="alert-title">No GPU Detected</div>
        <div class="alert-text">Processing will be very slow on CPU only. Consider installing PyTorch with CUDA.</div></div>`;
    }

    // Issues
    if (setup.issues && setup.issues.length > 0) {
      setup.issues.forEach(issue => {
        html += `<div class="alert alert-warning"><div class="alert-text">${issue}</div></div>`;
      });
    }

    // Models check
    if (setup.missing_models && setup.missing_models.length > 0) {
      html += `<div class="alert alert-warning">
        <div class="alert-title">AI Models Not Downloaded</div>
        <div class="alert-text">Missing: ${setup.missing_models.join(', ')}<br>
          Models are required for restoration. Download them now (~130 MB total).</div>
        <button class="btn btn-primary btn-sm" style="margin-top:12px" id="btn-download-models" onclick="downloadModels()">Download Models</button>
      </div>`;
    } else {
      html += `<div class="alert alert-success"><div class="alert-title">AI Models Ready</div>
        <div class="alert-text">All essential models are downloaded.</div></div>`;
    }

    // Other suggestions
    if (setup.suggestions && setup.suggestions.length > 0) {
      setup.suggestions.forEach(sug => {
        html += `<div class="alert alert-warning"><div class="alert-text">${sug}</div></div>`;
      });
    }

    container.innerHTML = html;
  } catch (e) {
    container.innerHTML = '<p style="color:var(--text2)">Could not check system status.</p>';
  }
}

// Download essential models
async function downloadModels() {
  const btn = document.getElementById('btn-download-models');
  if (btn) { btn.disabled = true; btn.textContent = 'Downloading...'; }

  try {
    const r = await fetch('/api/models/download_essential', { method: 'POST' });
    const data = await r.json();
    if (data.error) {
      alert('Download error: ' + data.error);
      if (btn) { btn.disabled = false; btn.textContent = 'Download Models'; }
    } else {
      // Re-run the check to update the display
      checkHardware();
    }
  } catch (e) {
    alert('Download failed: ' + e.message);
    if (btn) { btn.disabled = false; btn.textContent = 'Download Models'; }
  }
}

// Pipeline stage config
const pipelineStages = [
  {id: 'analyze', label: 'Analyze', icon: '🔍'},
  {id: 'extract', label: 'Extract Frames', icon: '🎞️'},
  {id: 'dedup', label: 'Deduplicate', icon: '🔄'},
  {id: 'denoise', label: 'Denoise', icon: '✨'},
  {id: 'colorize', label: 'Colorize', icon: '🌈'},
  {id: 'upscale', label: 'AI Upscale', icon: '🔬'},
  {id: 'face', label: 'Reference Enhance', icon: '🖼️'},
  {id: 'interpolate', label: 'Interpolate', icon: '🎬'},
  {id: 'color_grade', label: 'Color Grade', icon: '🎨'},
  {id: 'encode', label: 'Encode', icon: '💾'},
  {id: 'upload', label: 'YouTube Upload', icon: '📤'},
];

function formatTime(seconds) {
  if (seconds == null || seconds < 0) return '--:--';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return h + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
  return m + ':' + String(s).padStart(2, '0');
}

function renderPipeline(currentStage, stagesDone) {
  const container = document.getElementById('pipeline-stages');
  if (!container) return;
  container.innerHTML = pipelineStages.map(s => {
    let state = 'pending';
    if (stagesDone && stagesDone.includes(s.id)) state = 'done';
    if (s.id === currentStage) state = 'active';

    const colors = {
      pending: 'color:var(--text2);opacity:0.4',
      active: 'color:var(--accent)',
      done: 'color:var(--success)',
    };
    const indicators = { pending: '○', active: '◉', done: '✓' };

    return `<div style="display:flex;align-items:center;gap:12px;padding:8px 0;${colors[state]}">
      <span style="font-size:14px;width:20px;text-align:center">${indicators[state]}</span>
      <span style="font-size:14px">${s.icon}</span>
      <span style="font-size:14px;flex:1">${s.label}</span>
      ${state === 'active' ? '<span style="font-size:11px;background:var(--accent);color:#fff;padding:2px 8px;border-radius:10px">Processing</span>' : ''}
    </div>`;
  }).join('');
}

// Start restoration
async function startRestoration() {
  const outputPath = document.getElementById('output-path').value;
  if (!outputPath) { alert('Please specify output path'); return; }

  showStep(6);
  renderPipeline('analyze', []);

  const r = await fetch('/api/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      input: inputPath,
      output: outputPath,
      profile: selectedProfile,
      quality: selectedQuality,
      enhancements: enhancements,
      ref_dir: document.getElementById('ref-photos-dir').value || ''
    })
  });

  const job = await r.json();
  currentJobId = job.id;

  document.getElementById('job-input').textContent = inputPath;
  document.getElementById('job-output').textContent = outputPath;

  pollProgress();
}

async function pollProgress() {
  if (!currentJobId) return;

  try {
    const r = await fetch('/api/jobs');
    const jobs = await r.json();
    const job = jobs.find(j => j.id === currentJobId);

    if (job) {
      // Progress ring
      const pct = job.progress || 0;
      const circumference = 327;
      const offset = circumference - (pct / 100) * circumference;
      document.getElementById('progress-circle').style.strokeDashoffset = offset;
      document.getElementById('progress-percent').textContent = Math.round(pct) + '%';
      document.getElementById('progress-message').textContent = job.stage_label || job.message;

      // Time stats — compute elapsed client-side from started_at for live updates
      let elapsed = job.elapsed_sec || 0;
      if (job.started_at && (job.status === 'running' || job.status === 'pending')) {
        elapsed = Math.round((Date.now() - new Date(job.started_at).getTime()) / 1000);
      }
      document.getElementById('stat-elapsed').textContent = formatTime(elapsed);
      document.getElementById('stat-eta').textContent = job.eta_sec != null ? formatTime(job.eta_sec) : '--:--';
      document.getElementById('stat-fps').textContent = job.fps ? job.fps + ' fps' : '-- fps';

      // Frame count
      if (job.frame_total > 0) {
        document.getElementById('frames-row').style.display = 'flex';
        document.getElementById('stat-frames').textContent = (job.frame_current || 0) + ' / ' + job.frame_total;
      }

      // Progress bar
      const barFill = document.getElementById('progress-bar-fill');
      const barPct = document.getElementById('progress-bar-pct');
      const barLabel = document.getElementById('progress-bar-label');
      const barFrames = document.getElementById('progress-bar-frames');
      const barSpeed = document.getElementById('progress-bar-speed');
      barFill.style.width = pct + '%';
      barPct.textContent = Math.round(pct) + '%';
      barLabel.textContent = job.stage_label || 'Processing...';
      if (job.frame_total > 0 && job.frame_current > 0) {
        barFrames.textContent = job.frame_current + ' / ' + job.frame_total + ' frames';
      }
      if (job.fps) {
        barSpeed.textContent = job.fps + ' frames/sec';
      }

      // Pipeline stages
      renderPipeline(job.stage, job.stages_done);

      // Status badge
      const statusEl = document.getElementById('job-status');
      const statusLabels = { pending: 'Queued', running: 'Running', completed: 'Completed', failed: 'Failed', cancelled: 'Cancelled' };
      statusEl.textContent = statusLabels[job.status] || job.status;
      statusEl.className = 'status status-' + job.status;

      // Live log
      if (job.log && job.log.length > 0) {
        const logEl = document.getElementById('log-output');
        logEl.innerHTML = job.log.map(l => `<div>${l}</div>`).join('');
        logEl.scrollTop = logEl.scrollHeight;
      }

      // Change ring color and hide stop button on completion/failure/cancel
      const ring = document.getElementById('progress-circle');
      const stopBtn = document.getElementById('btn-stop-job');
      if (job.status === 'completed') {
        ring.style.stroke = 'var(--success)';
        barFill.style.background = 'var(--success)';
        document.getElementById('progress-message').textContent = 'Your video is ready!';
        stopBtn.style.display = 'none';
      } else if (job.status === 'failed') {
        ring.style.stroke = 'var(--error)';
        barFill.style.background = 'var(--error)';
        stopBtn.style.display = 'none';
      } else if (job.status === 'cancelled') {
        ring.style.stroke = 'var(--error)';
        barFill.style.background = 'var(--error)';
        document.getElementById('progress-message').textContent = 'Cancelled';
        stopBtn.style.display = 'none';
      }

      if (job.status === 'running' || job.status === 'pending') {
        setTimeout(pollProgress, 1000);
      }
    }
  } catch (e) {
    setTimeout(pollProgress, 2000);
  }
}

// Check button: validate typed path and show image count
async function checkRefFolder() {
  const current = document.getElementById('ref-photos-dir').value;
  if (!current) {
    alert('Please enter a folder path first');
    return;
  }
  try {
    const r = await fetch('/api/browse?path=' + encodeURIComponent(current));
    const data = await r.json();
    if (data.error) {
      alert('Path not found: ' + current);
      return;
    }
    if (data.path) {
      document.getElementById('ref-photos-dir').value = data.path;
      const imgs = (data.items || []).filter(i => !i.is_dir && /\\.(jpg|jpeg|png|bmp|tiff|webp)$/i.test(i.name));
      const preview = document.getElementById('ref-preview');
      const count = document.getElementById('ref-count');
      if (imgs.length > 0) {
        preview.style.display = 'block';
        count.textContent = imgs.length + ' reference image(s) found ✓';
        count.style.color = 'var(--success)';
      } else {
        preview.style.display = 'block';
        count.textContent = 'No images found in this folder';
        count.style.color = 'var(--warning)';
      }
    }
  } catch (e) {
    alert('Error checking folder: ' + e.message);
  }
}

// Browse button: open folder picker modal (STEP 4 ONLY)
async function openRefFolderModal() {
  if (!drives || drives.length === 0) {
    const r = await fetch('/api/drives');
    drives = await r.json();
  }
  const modal = document.getElementById('ref-folder-modal');
  modal.style.display = 'flex';
  modal.style.position = 'fixed';
  modal.style.top = '0';
  modal.style.left = '0';
  modal.style.right = '0';
  modal.style.bottom = '0';
  modal.style.background = 'rgba(0,0,0,0.9)';
  modal.style.zIndex = '99999';
  modal.style.alignItems = 'center';
  modal.style.justifyContent = 'center';

  const drivesDiv = document.getElementById('ref-drives');
  drivesDiv.innerHTML = drives.map(d =>
    `<button class="drive-btn" onclick="browseRefFolderPath('${d.path.replace(/\\\\/g, '\\\\\\\\')}')">${d.name}</button>`
  ).join('');
  if (drives.length) browseRefFolderPath(drives[0].path);
}

let refCurrentFolder = '';

async function browseRefFolderPath(path) {
  const r = await fetch('/api/browse?path=' + encodeURIComponent(path));
  const data = await r.json();
  if (data.error) { alert(data.error); return; }

  refCurrentFolder = data.path;
  document.getElementById('ref-current-path').textContent = data.path;

  let html = '';
  if (data.parent) {
    html += `<div class="browser-item" onclick="browseRefFolderPath('${data.parent.replace(/\\\\/g, '\\\\\\\\')}')">
      <span class="browser-icon">📁</span><span class="browser-name">..</span></div>`;
  }
  for (const item of data.items) {
    if (item.is_dir) {
      const escaped = item.path.replace(/\\\\/g, '\\\\\\\\');
      html += `<div class="browser-item" onclick="browseRefFolderPath('${escaped}')">
        <span class="browser-icon">📁</span><span class="browser-name">${item.name}</span></div>`;
    }
  }
  document.getElementById('ref-folder-list').innerHTML = html || '<p style="padding:16px;color:var(--text2)">No subfolders</p>';

  // Count image files for preview
  const imgs = (data.items || []).filter(i => !i.is_dir && /\\.(jpg|jpeg|png|bmp|tiff|webp)$/i.test(i.name));
  const preview = document.getElementById('ref-preview');
  const count = document.getElementById('ref-count');
  if (imgs.length > 0) {
    preview.style.display = 'block';
    count.textContent = imgs.length + ' reference image(s) found in this folder';
  } else {
    preview.style.display = 'none';
  }
}

function closeRefFolderModal() {
  document.getElementById('ref-folder-modal').style.display = 'none';
}

function selectRefFolder() {
  document.getElementById('ref-photos-dir').value = refCurrentFolder;

  // Trigger final count update
  fetch('/api/browse?path=' + encodeURIComponent(refCurrentFolder))
    .then(r => r.json())
    .then(data => {
      const imgs = (data.items || []).filter(i => !i.is_dir && /\\.(jpg|jpeg|png|bmp|tiff|webp)$/i.test(i.name));
      const preview = document.getElementById('ref-preview');
      const count = document.getElementById('ref-count');
      if (imgs.length > 0) {
        preview.style.display = 'block';
        count.textContent = imgs.length + ' reference image(s) found';
      } else {
        preview.style.display = 'block';
        count.textContent = 'No images found in this folder — add .jpg/.png reference photos';
      }
    });

  closeRefFolderModal();
}

function copyActivityLog() {
  const logOutput = document.getElementById('log-output');
  const logText = logOutput.innerText || logOutput.textContent;

  navigator.clipboard.writeText(logText).then(() => {
    // Show temporary feedback
    const originalText = 'Copy';
    const btn = event.target;
    btn.textContent = 'Copied!';
    btn.style.color = 'var(--accent)';
    setTimeout(() => {
      btn.textContent = originalText;
      btn.style.color = '';
    }, 2000);
  }).catch(err => {
    // Fallback: create a text area and copy
    const textarea = document.createElement('textarea');
    textarea.value = logText;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
    alert('Log copied to clipboard!');
  });
}

async function stopJob() {
  if (!currentJobId) return;
  if (!confirm('Stop the current restoration? Extracted frames will be kept for re-use.')) return;
  try {
    await fetch('/api/jobs/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: currentJobId })
    });
  } catch (e) {
    console.error('Stop failed:', e);
  }
}

// Step 4 and 5 setup
document.querySelector('[onclick="nextStep(4)"]').addEventListener('click', renderEnhancements);
document.querySelector('[onclick="nextStep(5)"]').addEventListener('click', renderSummary);

// Init
loadDrives();
renderProfiles();
renderQuality();
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
        elif p.path == '/api/system':
            self.send_json(get_system_info())
        elif p.path == '/api/setup':
            self.send_json(check_setup())
        elif p.path == '/api/drives':
            self.send_json(get_drives())
        elif p.path == '/api/default_download_folder':
            self.send_json({"path": str(Path.home() / "Downloads")})
        elif p.path == '/api/browse':
            self.send_json(list_directory(q.get('path', [''])[0] or os.path.expanduser('~')))
        elif p.path == '/api/jobs':
            with JOBS_LOCK:
                self.send_json(list(JOBS.values()))
        elif p.path == '/api/models':
            self.send_json(get_all_models_status())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        p = urlparse(self.path)
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        try:
            data = json.loads(body) if body else {}
        except:
            self.send_json({'error': 'Invalid JSON'}, 400)
            return

        if p.path == '/api/jobs':
            self.send_json(submit_job(data), 201)
        elif p.path == '/api/youtube':
            self.send_json(download_youtube(data.get('url', ''), data.get('download_dir')))
        elif p.path == '/api/analyze':
            self.send_json(analyze_video(data.get('path', '')))
        elif p.path == '/api/models/download_essential':
            self.send_json(download_essential_models())
        elif p.path == '/api/models/download':
            self.send_json(download_model_by_name(data.get('name', '')))
        elif p.path == '/api/jobs/stop':
            job_id = data.get('id', '')
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["status"] = "cancelled"
            with PROCESSES_LOCK:
                proc = RUNNING_PROCESSES.get(job_id)
                if proc:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
            self.send_json({"success": True, "message": "Job cancelled"})
        else:
            self.send_response(404)
            self.end_headers()

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
    print("=" * 50)
    print("FrameWright Smart Dashboard")
    print("=" * 50)
    print(f"Open: http://{HOST}:{PORT}")
    print("=" * 50)
    threading.Thread(target=lambda: (time.sleep(1), webbrowser.open(f"http://{HOST}:{PORT}")), daemon=True).start()
    Server((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
