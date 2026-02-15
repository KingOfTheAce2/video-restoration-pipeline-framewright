"""Configuration module for FrameWright video restoration pipeline."""
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional, Dict, Any, ClassVar, List


# Configuration presets for common use cases
PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "scale_factor": 2,
        "model_name": "realesrgan-x2plus",
        "crf": 23,
        "preset": "fast",
        "parallel_frames": 4,
        "enable_checkpointing": False,
        "enable_validation": False,
    },
    "quality": {
        "scale_factor": 4,
        "model_name": "realesrgan-x4plus",
        "crf": 18,
        "preset": "slow",
        "parallel_frames": 2,
        "enable_checkpointing": True,
        "enable_validation": True,
    },
    "archive": {
        "scale_factor": 4,
        "model_name": "realesrgan-x4plus",
        "crf": 15,
        "preset": "veryslow",
        "parallel_frames": 1,
        "enable_checkpointing": True,
        "enable_validation": True,
        "min_ssim_threshold": 0.9,
        "min_psnr_threshold": 30.0,
    },
    "anime": {
        "scale_factor": 4,
        "model_name": "realesr-animevideov3",
        "crf": 18,
        "preset": "medium",
        "parallel_frames": 2,
        "enable_checkpointing": True,
    },
    "film_restoration": {
        "scale_factor": 4,
        "model_name": "realesrgan-x4plus",
        "crf": 16,
        "preset": "slow",
        "parallel_frames": 2,
        "enable_checkpointing": True,
        "enable_validation": True,
        "enable_auto_enhance": True,
        "auto_defect_repair": True,
        "auto_face_restore": True,
        "scratch_sensitivity": 0.7,
        "grain_reduction": 0.4,
    },
    "ultimate": {
        # Maximum quality preset for powerful hardware (RTX 5090 / 32GB+ VRAM)
        "scale_factor": 4,
        "model_name": "realesrgan-x4plus",
        "crf": 14,  # Near-lossless
        "preset": "veryslow",
        "parallel_frames": 1,  # Sequential for quality
        "enable_checkpointing": True,
        "enable_validation": True,
        "min_ssim_threshold": 0.92,
        "min_psnr_threshold": 35.0,
        # Enable all cutting-edge features
        "enable_tap_denoise": True,
        "tap_model": "restormer",
        "tap_strength": 0.8,
        "enable_qp_artifact_removal": True,
        "qp_strength": 1.0,
        "sr_model": "realesrgan",  # Or "diffusion" for maximum quality
        "face_model": "aesrgan",
        "auto_face_restore": True,
        "temporal_method": "hybrid",
        "cross_attention_window": 7,
        "enable_auto_enhance": True,
        "auto_defect_repair": True,
        "scratch_sensitivity": 0.8,
        "grain_reduction": 0.3,
        # v2.0 features
        "enable_scene_intelligence": True,
        "enable_vmaf_analysis": True,
    },
    "authentic": {
        # Authentic restoration preset - preserves period character
        # For historic footage that shouldn't look "too modern"
        "scale_factor": 2,  # Conservative upscale
        "model_name": "realesrgan-x2plus",
        "crf": 16,
        "preset": "slow",
        "parallel_frames": 2,
        "enable_checkpointing": True,
        "enable_validation": True,
        # Authenticity preservation
        "enable_authenticity_guard": True,
        "preserve_era_character": True,
        "max_enhancement_strength": 0.5,
        "preserve_grain": True,
        "grain_preservation_level": 0.6,
        # Conservative processing
        "enable_tap_denoise": True,
        "tap_model": "nafnet",
        "tap_strength": 0.4,
        "tap_preserve_grain": True,
        "face_model": "codeformer",
        "auto_face_restore": True,
        "grain_reduction": 0.2,  # Minimal grain reduction
        "temporal_method": "optical_flow",
        "enable_scene_intelligence": True,
    },
    "vhs": {
        # VHS/analog tape restoration preset
        "scale_factor": 2,
        "model_name": "realesrgan-x2plus",
        "crf": 18,
        "preset": "medium",
        "parallel_frames": 2,
        "enable_checkpointing": True,
        # VHS-specific restoration
        "enable_vhs_restoration": True,
        "vhs_remove_tracking": True,
        "vhs_remove_dropout": True,
        "vhs_fix_chroma": True,
        "vhs_tbc_simulation": True,
        "vhs_remove_dot_crawl": True,
        # Additional processing
        "enable_tap_denoise": True,
        "tap_model": "nafnet",
        "tap_strength": 0.6,
        "temporal_method": "optical_flow",
        "enable_scene_intelligence": True,
    },
    "archive": {
        # Optimized for archive/historical footage restoration
        # Includes missing frame generation and old film specific features
        "scale_factor": 4,
        "model_name": "realesrgan-x4plus",
        "crf": 14,
        "preset": "veryslow",
        "parallel_frames": 1,
        "enable_checkpointing": True,
        "enable_validation": True,
        "min_ssim_threshold": 0.90,
        "min_psnr_threshold": 32.0,
        # Archive-specific features
        "enable_deduplication": True,  # Remove duplicate frames from padded FPS
        "deduplication_threshold": 0.98,
        "enable_frame_generation": True,  # Generate missing/damaged frames
        "frame_gen_model": "optical_flow_warp",  # SVD for best quality (requires more VRAM)
        "max_gap_frames": 10,  # Maximum frames to generate in a gap
        # Full restoration pipeline
        "enable_tap_denoise": True,
        "tap_model": "restormer",
        "tap_strength": 0.9,  # Stronger for archive footage
        "tap_preserve_grain": True,  # Preserve film grain character
        "enable_qp_artifact_removal": True,
        "qp_strength": 0.8,
        "sr_model": "realesrgan",
        "face_model": "aesrgan",
        "auto_face_restore": True,
        "temporal_method": "hybrid",
        "cross_attention_window": 9,  # Wider window for better consistency
        "temporal_blend_strength": 0.6,
        # Defect repair for scratches/dust
        "enable_auto_enhance": True,
        "auto_defect_repair": True,
        "scratch_sensitivity": 0.85,
        "dust_sensitivity": 0.75,
        "grain_reduction": 0.2,  # Preserve some grain for authenticity
        # Frame interpolation for smoother playback
        "enable_interpolation": True,
        "target_fps": 24,  # Standard film rate
    },
    "rtx5090": {
        # RTX 5090 / High-End GPU preset (32GB+ VRAM, 64GB+ RAM)
        # Maximum quality with no compromises - full resolution, large temporal windows
        "scale_factor": 4,
        "model_name": "realesrgan-x4plus",
        "crf": 12,  # Near-lossless quality
        "preset": "veryslow",
        "parallel_frames": 1,  # Sequential for maximum quality
        "enable_checkpointing": True,
        "enable_validation": True,
        "min_ssim_threshold": 0.95,
        "min_psnr_threshold": 38.0,
        # No tiling - process full resolution with 32GB VRAM
        "tile_size": None,  # Disable tiling entirely
        # Super-Resolution: Use HAT or VRT for maximum quality
        "sr_model": "hat",  # HAT > VRT > Real-ESRGAN for quality
        "enable_ensemble_sr": True,  # Run multiple models and vote
        "ensemble_models": ["hat", "vrt", "realesrgan"],
        "ensemble_voting": "weighted",  # weighted, max_quality, per_region
        # Extended temporal windows (2-3x normal for better consistency)
        "temporal_window": 16,  # Standard is 7, RTX 5090 can handle 16+
        "cross_attention_window": 16,  # Match temporal window
        "temporal_method": "hybrid",
        "temporal_blend_strength": 0.7,
        # RAFT optical flow (better than Farneback)
        "optical_flow_method": "raft",
        "enable_bidirectional_flow": True,
        # TAP Neural Denoising with maximum quality
        "enable_tap_denoise": True,
        "tap_model": "restormer",
        "tap_strength": 0.85,
        "tap_temporal_window": 9,  # Extended temporal context
        # QP Artifact Removal for YouTube/compressed sources
        "enable_qp_artifact_removal": True,
        "qp_strength": 1.0,
        "qp_adaptive": True,
        # Face Enhancement with AESRGAN
        "face_model": "aesrgan",
        "auto_face_restore": True,
        "aesrgan_strength": 0.9,
        # Colorization with temporal consistency
        "colorization_model": "ddcolor",
        "enable_temporal_colorization": True,  # Bidirectional color fusion
        "colorization_temporal_window": 7,
        "colorization_propagation": "bidirectional",
        # Diffusion-based enhancement for maximum quality
        "enable_diffusion_sr": True,
        "diffusion_steps": 30,
        "diffusion_guidance": 7.5,
        # Full defect repair
        "enable_auto_enhance": True,
        "auto_defect_repair": True,
        "scratch_sensitivity": 0.9,
        "dust_sensitivity": 0.85,
        "grain_reduction": 0.25,
        # Scene intelligence for adaptive processing
        "enable_scene_intelligence": True,
        "enable_vmaf_analysis": True,
        # Frame interpolation with RIFE v4.6
        "enable_interpolation": True,
        "target_fps": 60,  # High FPS output
        "rife_model": "rife-v4.6",
        # Archive features
        "enable_deduplication": True,
        "deduplication_threshold": 0.98,
        "enable_frame_generation": True,
        "frame_gen_model": "svd",  # Use Stable Video Diffusion for missing frames
        "max_gap_frames": 15,
    },
}


@dataclass
class Config:
    """Configuration for video restoration pipeline.

    Attributes:
        project_dir: Root directory for all processing files
        scale_factor: Upscaling factor (2x or 4x)
        model_name: Real-ESRGAN model to use
        crf: Constant Rate Factor for x265 encoding (0-51, lower is better quality)
        preset: x265 encoding preset (ultrafast to veryslow)
        output_format: Output video container format
        temp_dir: Temporary directory for intermediate files (auto-created)
        frames_dir: Directory for extracted frames (auto-created)
        enhanced_dir: Directory for enhanced frames (auto-created)

        # Robustness options
        enable_checkpointing: Enable checkpoint/resume functionality
        checkpoint_interval: Save checkpoint every N frames
        enable_validation: Enable quality validation
        min_ssim_threshold: Minimum SSIM for quality validation
        min_psnr_threshold: Minimum PSNR for quality validation
        enable_disk_validation: Pre-check disk space
        disk_safety_margin: Extra disk space buffer (1.2 = 20% extra)
        enable_vram_monitoring: Monitor GPU VRAM usage
        tile_size: Tile size for Real-ESRGAN (0 = auto, None = no tiling)
        max_retries: Maximum retry attempts for transient errors
        retry_delay: Initial delay between retries in seconds
        parallel_frames: Number of frames to process in parallel (1 = sequential)
        continue_on_error: Continue processing even if some frames fail

        # Multi-GPU distribution options
        enable_multi_gpu: Enable multi-GPU frame distribution for faster processing
        gpu_ids: Specific GPU IDs to use (None = auto-detect all available GPUs)
        gpu_load_balance_strategy: Load balancing strategy (round_robin, least_loaded, vram_aware, weighted)
        workers_per_gpu: Number of worker threads per GPU for parallel processing
        enable_work_stealing: Allow idle workers to steal work from busy workers

        # Directory configuration
        model_dir: Directory for storing model files (default: ~/.framewright/models)
        _output_dir_override: Override for output directory (if None, uses project_dir/output/)
        _frames_dir_override: Override for frames directory (if None, uses project_dir/frames/)
        _enhanced_dir_override: Override for enhanced directory (if None, uses project_dir/enhanced/)

        # Colorization options
        enable_colorization: Enable automatic colorization of black-and-white footage
        colorization_model: Colorization model to use ('ddcolor' or 'deoldify')
        colorization_strength: Strength of colorization effect (0.0-1.0)

        # Watermark removal options
        enable_watermark_removal: Enable watermark removal processing
        watermark_mask_path: Path to a mask image defining watermark region
        watermark_auto_detect: Automatically detect watermark location

        # Burnt-in subtitle removal options
        enable_subtitle_removal: Enable burnt-in subtitle detection and removal
        subtitle_region: Region to scan for subtitles (bottom_third, bottom_quarter, top_quarter, full_frame)
        subtitle_ocr_engine: OCR engine for detection (auto, easyocr, tesseract, paddleocr)
        subtitle_languages: List of language codes for OCR detection
    """

    project_dir: Path
    output_dir: Optional[Path] = None  # Explicit output directory (defaults to project_dir)
    scale_factor: Literal[2, 4] = 4
    model_name: str = "realesrgan-x4plus"
    crf: int = 18
    preset: str = "medium"
    output_format: str = "mkv"

    # Robustness options
    enable_checkpointing: bool = True
    checkpoint_interval: int = 100
    enable_validation: bool = True
    min_ssim_threshold: float = 0.85
    min_psnr_threshold: float = 25.0
    enable_disk_validation: bool = True
    disk_safety_margin: float = 1.2
    enable_vram_monitoring: bool = True
    tile_size: Optional[int] = 0  # 0 = auto, None = no tiling
    max_retries: int = 3
    retry_delay: float = 1.0
    parallel_frames: int = 1
    continue_on_error: bool = True  # Continue processing even if some frames fail

    # Frame caching options (performance optimization)
    enable_frame_caching: bool = True  # Cache intermediate frames to avoid reprocessing
    frame_cache_max_mb: int = 2048  # Maximum memory for frame cache (in MB)
    frame_cache_eviction: str = "lru"  # Eviction policy: lru, lfu, fifo, size

    # Async I/O options (performance optimization)
    enable_async_io: bool = True  # Use async I/O for FFmpeg calls (15-20% faster)

    # GPU selection and multi-GPU distribution options
    require_gpu: bool = True  # Require GPU for processing - blocks CPU fallback to prevent runaway CPU usage
    gpu_id: Optional[int] = None  # Select specific GPU by index (--gpu N), None = auto-select
    enable_multi_gpu: bool = False  # Enable multi-GPU frame distribution (--multi-gpu)
    gpu_ids: Optional[List[int]] = None  # Specific GPU IDs to use (None = auto-detect all)
    gpu_load_balance_strategy: str = "vram_aware"  # round_robin, least_loaded, vram_aware, weighted
    workers_per_gpu: int = 2  # Worker threads per GPU for parallel processing
    enable_work_stealing: bool = True  # Allow idle workers to steal work from busy ones

    # RIFE frame interpolation options (optional)
    enable_interpolation: bool = False  # Must explicitly enable RIFE
    target_fps: Optional[float] = None  # Target frame rate (None = auto from source)
    rife_model: str = "rife-v4.6"  # RIFE model version
    rife_gpu_id: int = 0  # GPU for RIFE processing

    # Frame deduplication options (for old films with artificial FPS padding)
    enable_deduplication: bool = False  # Detect and remove duplicate frames
    deduplication_threshold: float = 0.98  # Similarity threshold (0.98 = very similar)
    expected_source_fps: Optional[float] = None  # Hint for original FPS (e.g., 18 for 1909 film)

    # Auto-enhancement options (fully automated processing)
    enable_auto_enhance: bool = False  # Enable automatic enhancement pipeline
    auto_detect_content: bool = True  # Auto-detect content type (faces, animation, etc.)
    auto_defect_repair: bool = True  # Auto-detect and repair defects
    auto_face_restore: bool = True  # Auto face restoration when faces detected
    scratch_sensitivity: float = 0.5  # Sensitivity for scratch detection (0-1)
    dust_sensitivity: float = 0.5  # Sensitivity for dust detection (0-1)
    grain_reduction: float = 0.3  # Film grain reduction strength (0-1)

    # Model download location
    model_download_dir: Optional[Path] = None  # Custom model download location

    # Directory configuration (new flexible paths)
    model_dir: Path = field(default_factory=lambda: Path.home() / ".framewright" / "models")
    _output_dir_override: Optional[Path] = None  # Internal: explicit output directory
    _frames_dir_override: Optional[Path] = None  # Internal: explicit frames directory
    _enhanced_dir_override: Optional[Path] = None  # Internal: explicit enhanced directory

    # Colorization options
    enable_colorization: bool = False
    colorization_model: str = "ddcolor"  # "ddcolor" or "deoldify"
    colorization_strength: float = 1.0  # Strength of colorization (0.0-1.0)

    # Watermark removal options
    enable_watermark_removal: bool = False
    watermark_mask_path: Optional[Path] = None  # Path to watermark mask image
    watermark_auto_detect: bool = True  # Auto-detect watermark location

    # Burnt-in subtitle removal options
    enable_subtitle_removal: bool = False
    subtitle_region: str = "bottom_third"  # bottom_third, bottom_quarter, top_quarter, full_frame
    subtitle_ocr_engine: str = "auto"  # auto, easyocr, tesseract, paddleocr
    subtitle_languages: List[str] = field(default_factory=lambda: ["en"])

    # TAP Neural Denoising (advanced - Restormer/NAFNet with temporal attention)
    enable_tap_denoise: bool = False
    tap_model: str = "restormer"  # restormer, nafnet, tap
    tap_strength: float = 1.0  # Denoising strength (0-1)
    tap_preserve_grain: bool = False  # Preserve film grain character

    # Super-Resolution Model Selection
    sr_model: str = "realesrgan"  # realesrgan, diffusion, basicvsr
    diffusion_steps: int = 20  # Number of diffusion steps (for diffusion SR)
    diffusion_guidance: float = 7.5  # Classifier-free guidance scale

    # Face Enhancement Model Selection
    face_model: str = "gfpgan"  # gfpgan, codeformer, aesrgan
    aesrgan_strength: float = 0.8  # AESRGAN enhancement strength

    # QP-Aware Codec Artifact Removal
    enable_qp_artifact_removal: bool = False
    qp_auto_detect: bool = True  # Auto-detect QP from video metadata
    qp_strength: float = 1.0  # Artifact removal strength multiplier

    # Exemplar-Based Colorization (SwinTExCo)
    colorization_reference_images: List[Path] = field(default_factory=list)
    colorization_temporal_fusion: bool = True  # Enable temporal consistency

    # Reference-Guided Enhancement (IP-Adapter + ControlNet)
    enable_reference_enhance: bool = False
    reference_images_dir: Optional[Path] = None
    reference_strength: float = 0.35
    reference_guidance_scale: float = 7.5
    reference_ip_adapter_scale: float = 0.6

    # Missing Frame Generation
    enable_frame_generation: bool = False
    frame_gen_model: str = "interpolate_blend"  # svd, optical_flow_warp, interpolate_blend
    max_gap_frames: int = 10  # Maximum frames to generate in a gap

    # Enhanced Temporal Consistency (Cross-Frame Attention)
    temporal_method: str = "optical_flow"  # optical_flow, cross_attention, hybrid, raft
    cross_attention_window: int = 7  # Frames for attention window
    temporal_blend_strength: float = 0.8  # Temporal blending strength
    temporal_window: int = 7  # General temporal window size

    # RAFT Optical Flow (better than Farneback)
    optical_flow_method: str = "farneback"  # farneback, dis, raft
    enable_bidirectional_flow: bool = False  # Use bidirectional flow estimation

    # Temporal Colorization Consistency
    enable_temporal_colorization: bool = False  # Apply temporal consistency to colorization
    colorization_temporal_window: int = 7  # Frames for color propagation
    colorization_propagation: str = "bidirectional"  # forward, backward, bidirectional

    # HAT Upscaler (Hybrid Attention Transformer)
    enable_hat: bool = False  # Use HAT instead of Real-ESRGAN
    hat_model_size: str = "large"  # small, base, large

    # Ensemble Super-Resolution
    enable_ensemble_sr: bool = False  # Run multiple SR models and combine
    ensemble_models: List[str] = field(default_factory=lambda: ["hat", "realesrgan"])
    ensemble_voting: str = "weighted"  # weighted, max_quality, per_region, median

    # ==================== v2.0 ADVANCED FEATURES ====================

    # Authenticity Preservation (prevents over-processing of historic footage)
    enable_authenticity_guard: bool = False  # Enable era-aware processing limits
    preserve_era_character: bool = True  # Maintain period-appropriate aesthetics
    auto_detect_era: bool = True  # Automatically detect footage era
    source_era: Optional[str] = None  # Manual: silent_film, early_talkies, golden_age, etc.
    max_enhancement_strength: float = 0.7  # Global limit on enhancement intensity
    preserve_grain: bool = False  # Preserve film grain character
    grain_preservation_level: float = 0.5  # How much grain to keep (0=none, 1=all)

    # AI Scene Intelligence (content-aware adaptive processing)
    enable_scene_intelligence: bool = False  # Enable AI content analysis
    scene_detect_faces: bool = True  # Detect and enhance faces
    scene_detect_text: bool = True  # Detect and preserve text/titles
    scene_adaptive_settings: bool = True  # Adjust settings per scene type

    # VHS/Analog Restoration
    enable_vhs_restoration: bool = False  # Enable VHS-specific processing
    vhs_auto_detect_format: bool = True  # Auto-detect VHS, Betamax, Hi8, etc.
    vhs_source_format: str = "vhs"  # vhs, betamax, hi8, video8, umatic
    vhs_remove_tracking: bool = True  # Fix tracking line artifacts
    vhs_remove_dropout: bool = True  # Repair dropout regions
    vhs_fix_chroma: bool = True  # Fix color bleeding/delay
    vhs_tbc_simulation: bool = True  # Time base corrector simulation
    vhs_remove_dot_crawl: bool = True  # Remove composite video artifacts
    vhs_preserve_character: bool = True  # Don't over-process analog character

    # Quality Analysis (VMAF, heatmaps)
    enable_vmaf_analysis: bool = False  # Calculate VMAF scores
    vmaf_model: str = "vmaf_v0.6.1"  # VMAF model version
    enable_quality_heatmaps: bool = False  # Generate quality heatmaps
    quality_report_format: str = "html"  # html, json, both

    # Webhook Notifications
    enable_webhooks: bool = False  # Enable webhook notifications
    webhook_config_path: Optional[Path] = None  # Path to webhook config JSON

    # Distributed Processing
    enable_distributed: bool = False  # Enable distributed render farm
    coordinator_address: Optional[str] = None  # Coordinator server address
    worker_mode: bool = False  # Run as worker node
    chunk_size: int = 100  # Frames per work chunk

    # LUT Integration
    input_lut_path: Optional[Path] = None  # LUT to apply before processing
    output_lut_path: Optional[Path] = None  # LUT to apply after processing

    # Seasonal Color Grading (applied after colorization, before encoding)
    seasonal_color_grade: Optional[str] = None  # winter, spring, summer, autumn (None = disabled)
    color_grade_strength: float = 0.7  # How strongly to apply the grade (0.0-1.0)

    # YouTube Upload (optional final step after encoding)
    enable_youtube_upload: bool = False  # Upload finished video to YouTube
    youtube_client_secrets: Optional[Path] = None  # OAuth client_secrets.json path
    youtube_privacy: str = "private"  # public, unlisted, private
    youtube_title: Optional[str] = None  # Video title (None = auto from filename)
    youtube_description: str = ""  # Video description
    youtube_tags: List[str] = field(default_factory=lambda: ["restoration", "framewright"])
    youtube_playlist_id: Optional[str] = None  # Optional playlist to add to

    # EDL Support
    edl_input_path: Optional[Path] = None  # Import restoration settings from EDL
    edl_output_path: Optional[Path] = None  # Export EDL with restoration metadata

    # Natural Language Interface
    natural_language_mode: bool = False  # Enable NLP command parsing
    nlp_preserve_authenticity: bool = True  # Default authenticity for NLP commands

    # ==================== v2.1 MODULAR FEATURES ====================

    # Scene-Aware Processing (uses SceneIntelligence for adaptive processing)
    enable_scene_aware: bool = False  # Per-scene intensity adjustment
    scene_aware_intensity_scale: float = 1.0  # Global intensity multiplier
    scene_aware_preserve_faces: bool = True  # Lighter processing for faces
    scene_aware_preserve_text: bool = True  # Preserve text sharpness

    # Motion-Adaptive Denoising
    enable_motion_adaptive: bool = False  # Modulate denoise by motion level
    motion_adaptive_sensitivity: float = 0.5  # Motion detection sensitivity

    # Audio-Visual Sync Repair
    enable_av_sync_repair: bool = False  # Detect and fix A/V drift
    av_sync_max_drift_ms: float = 500.0  # Maximum drift to correct

    # HDR Expansion
    enable_hdr_expansion: bool = False  # SDR to HDR conversion
    hdr_target_format: str = "hdr10"  # hdr10, hdr10plus, dolby_vision, hlg
    hdr_peak_brightness: int = 1000  # Target peak brightness (nits)

    # Aspect Ratio Correction
    enable_aspect_correction: bool = False  # Fix stretched/squeezed video
    aspect_target_ratio: Optional[str] = None  # Target ratio (e.g., "16:9", "4:3", "auto")
    aspect_crop_letterbox: bool = True  # Auto-detect and crop letterboxing

    # IVTC (Inverse Telecine)
    enable_ivtc: bool = False  # Remove telecine pulldown
    ivtc_pattern: str = "auto"  # auto, 3:2, 2:3, 2:2

    # Perceptual Tuning
    enable_perceptual_tuning: bool = False  # Balance faithful vs enhanced
    perceptual_mode: str = "balanced"  # faithful, balanced, enhanced
    perceptual_balance: float = 0.5  # 0.0 = faithful, 1.0 = enhanced

    # Sidecar Metadata Export
    enable_sidecar: bool = False  # Export JSON sidecar with settings/metrics

    # Notifications (Email/SMS)
    enable_notifications: bool = False  # Enable email/SMS notifications
    notification_config_path: Optional[Path] = None  # Path to notification config

    # Daemon Mode
    enable_daemon: bool = False  # Run as background daemon
    daemon_auto_resume: bool = True  # Resume crashed jobs

    # Scheduled Processing
    enable_scheduling: bool = False  # Enable scheduled job processing
    scheduler_config_path: Optional[Path] = None  # Path to scheduler config

    # Media Library Integration
    enable_library_integration: bool = False  # Auto-add to Plex/Jellyfin
    library_server_type: str = "plex"  # plex, jellyfin, emby
    library_server_url: Optional[str] = None
    library_api_token: Optional[str] = None

    # Proxy Workflow
    enable_proxy_workflow: bool = False  # Use proxy for tuning

    # Quality Tracking
    enable_quality_tracking: bool = False  # Track PSNR/SSIM trends

    # ==================== PREPROCESSING FIXES ====================

    # Interlace Detection and Deinterlacing
    enable_interlace_fix: bool = False  # Detect and fix interlaced content
    interlace_method: str = "auto"  # auto, yadif, bwdif, nnedi

    # Letterbox/Black Bar Cropping
    enable_letterbox_crop: bool = False  # Detect and crop black bars

    # Film Color Fading Correction
    enable_film_color_correction: bool = False  # Detect and correct color fading
    film_stock_override: Optional[str] = None  # Manual: kodachrome, ektachrome, etc.

    # Audio Sync Drift Fixing
    enable_audio_sync_fix: bool = False  # Detect and fix A/V sync drift
    audio_sync_method: str = "resample"  # resample, pts_adjust

    # Auto-generated paths
    temp_dir: Path = field(init=False)
    frames_dir: Path = field(init=False)
    unique_frames_dir: Path = field(init=False)  # For deduplicated frames
    enhanced_dir: Path = field(init=False)
    interpolated_dir: Path = field(init=False)  # For RIFE output
    checkpoint_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Initialize derived paths and validate configuration."""
        # Ensure project_dir is a Path object
        if not isinstance(self.project_dir, Path):
            self.project_dir = Path(self.project_dir)

        # Convert model_dir to Path if needed
        if not isinstance(self.model_dir, Path):
            self.model_dir = Path(self.model_dir)

        # Convert override paths to Path if provided as strings
        if self._output_dir_override is not None and not isinstance(self._output_dir_override, Path):
            self._output_dir_override = Path(self._output_dir_override)
        if self._frames_dir_override is not None and not isinstance(self._frames_dir_override, Path):
            self._frames_dir_override = Path(self._frames_dir_override)
        if self._enhanced_dir_override is not None and not isinstance(self._enhanced_dir_override, Path):
            self._enhanced_dir_override = Path(self._enhanced_dir_override)

        # Convert watermark_mask_path to Path if provided as string
        if self.watermark_mask_path is not None and not isinstance(self.watermark_mask_path, Path):
            self.watermark_mask_path = Path(self.watermark_mask_path)

        # Create derived directories (using overrides if provided)
        self.temp_dir = self.project_dir / "temp"
        self.frames_dir = self._frames_dir_override if self._frames_dir_override is not None else self.temp_dir / "frames"
        self.unique_frames_dir = self.temp_dir / "unique_frames"  # Deduplicated frames
        self.enhanced_dir = self._enhanced_dir_override if self._enhanced_dir_override is not None else self.temp_dir / "enhanced"
        self.interpolated_dir = self.temp_dir / "interpolated"
        self.checkpoint_dir = self.project_dir / ".framewright"

        # Validate scale factor
        if self.scale_factor not in (2, 4):
            raise ValueError("scale_factor must be 2 or 4")

        # Validate CRF range
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be between 0 and 51")

        # Validate model name based on scale factor
        valid_models = {
            2: ["realesrgan-x2plus"],
            4: ["realesrgan-x4plus", "realesrgan-x4plus-anime", "realesr-animevideov3"]
        }

        if self.model_name not in valid_models.get(self.scale_factor, []):
            raise ValueError(
                f"Invalid model '{self.model_name}' for scale factor {self.scale_factor}. "
                f"Valid models: {valid_models.get(self.scale_factor, [])}"
            )

        # Validate threshold ranges
        if not 0.0 <= self.min_ssim_threshold <= 1.0:
            raise ValueError("min_ssim_threshold must be between 0.0 and 1.0")

        if self.min_psnr_threshold < 0:
            raise ValueError("min_psnr_threshold must be non-negative")

        # Validate retry settings
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")

        # Validate parallel frames
        if self.parallel_frames < 1:
            raise ValueError("parallel_frames must be at least 1")

        # Validate tile size
        if self.tile_size is not None and self.tile_size < 0:
            raise ValueError("tile_size must be non-negative or None")

        # Validate RIFE options
        if self.target_fps is not None and self.target_fps <= 0:
            raise ValueError("target_fps must be positive")

        valid_rife_models = ['rife-v2.3', 'rife-v4.0', 'rife-v4.6']
        if self.rife_model not in valid_rife_models:
            raise ValueError(
                f"Invalid RIFE model '{self.rife_model}'. "
                f"Valid models: {valid_rife_models}"
            )

        # Validate deduplication options
        if not 0.0 <= self.deduplication_threshold <= 1.0:
            raise ValueError("deduplication_threshold must be between 0.0 and 1.0")
        if self.expected_source_fps is not None and self.expected_source_fps <= 0:
            raise ValueError("expected_source_fps must be positive")

        # Validate GPU options
        if self.gpu_id is not None:
            if not isinstance(self.gpu_id, int) or self.gpu_id < 0:
                raise ValueError(f"Invalid gpu_id: {self.gpu_id}. Must be non-negative integer or None.")

        # Validate multi-GPU options
        valid_strategies = ["round_robin", "least_loaded", "vram_aware", "weighted"]
        if self.gpu_load_balance_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid GPU load balance strategy '{self.gpu_load_balance_strategy}'. "
                f"Valid strategies: {valid_strategies}"
            )

        if self.workers_per_gpu < 1:
            raise ValueError("workers_per_gpu must be at least 1")

        if self.gpu_ids is not None:
            if not isinstance(self.gpu_ids, list):
                raise ValueError("gpu_ids must be a list of integers or None")
            for gid in self.gpu_ids:
                if not isinstance(gid, int) or gid < 0:
                    raise ValueError(f"Invalid GPU ID: {gid}. Must be non-negative integer.")

        # Validate auto-enhancement options
        if not 0.0 <= self.scratch_sensitivity <= 1.0:
            raise ValueError("scratch_sensitivity must be between 0.0 and 1.0")
        if not 0.0 <= self.dust_sensitivity <= 1.0:
            raise ValueError("dust_sensitivity must be between 0.0 and 1.0")
        if not 0.0 <= self.grain_reduction <= 1.0:
            raise ValueError("grain_reduction must be between 0.0 and 1.0")

        # Validate colorization options
        valid_colorization_models = ["ddcolor", "deoldify"]
        if self.colorization_model not in valid_colorization_models:
            raise ValueError(
                f"Invalid colorization model '{self.colorization_model}'. "
                f"Valid models: {valid_colorization_models}"
            )
        if not 0.0 <= self.colorization_strength <= 1.0:
            raise ValueError("colorization_strength must be between 0.0 and 1.0")

        # Validate reference enhancement options
        if self.reference_images_dir is not None and not isinstance(self.reference_images_dir, Path):
            self.reference_images_dir = Path(self.reference_images_dir)
        if not 0.0 <= self.reference_strength <= 1.0:
            raise ValueError("reference_strength must be between 0.0 and 1.0")
        if self.reference_guidance_scale < 0:
            raise ValueError("reference_guidance_scale must be non-negative")
        if not 0.0 <= self.reference_ip_adapter_scale <= 1.0:
            raise ValueError("reference_ip_adapter_scale must be between 0.0 and 1.0")

        # Validate watermark options
        if self.watermark_mask_path is not None:
            if not self.watermark_mask_path.exists():
                raise ValueError(f"Watermark mask file not found: {self.watermark_mask_path}")
            # Validate file extension for common image formats
            valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
            if self.watermark_mask_path.suffix.lower() not in valid_extensions:
                raise ValueError(
                    f"Invalid watermark mask format '{self.watermark_mask_path.suffix}'. "
                    f"Supported formats: {valid_extensions}"
                )

        # Validate subtitle removal options
        valid_subtitle_regions = ["bottom_third", "bottom_quarter", "top_quarter", "full_frame"]
        if self.subtitle_region not in valid_subtitle_regions:
            raise ValueError(
                f"Invalid subtitle region '{self.subtitle_region}'. "
                f"Valid regions: {valid_subtitle_regions}"
            )

        valid_ocr_engines = ["auto", "easyocr", "tesseract", "paddleocr"]
        if self.subtitle_ocr_engine not in valid_ocr_engines:
            raise ValueError(
                f"Invalid OCR engine '{self.subtitle_ocr_engine}'. "
                f"Valid engines: {valid_ocr_engines}"
            )

        # Validate TAP denoising options
        valid_tap_models = ["restormer", "nafnet", "tap"]
        if self.tap_model not in valid_tap_models:
            raise ValueError(
                f"Invalid TAP model '{self.tap_model}'. "
                f"Valid models: {valid_tap_models}"
            )
        if not 0.0 <= self.tap_strength <= 1.0:
            raise ValueError("tap_strength must be between 0.0 and 1.0")

        # Validate SR model options
        valid_sr_models = ["realesrgan", "diffusion", "basicvsr"]
        if self.sr_model not in valid_sr_models:
            raise ValueError(
                f"Invalid SR model '{self.sr_model}'. "
                f"Valid models: {valid_sr_models}"
            )
        if self.diffusion_steps < 1:
            raise ValueError("diffusion_steps must be at least 1")
        if self.diffusion_guidance < 0:
            raise ValueError("diffusion_guidance must be non-negative")

        # Validate face model options
        valid_face_models = ["gfpgan", "codeformer", "aesrgan"]
        if self.face_model not in valid_face_models:
            raise ValueError(
                f"Invalid face model '{self.face_model}'. "
                f"Valid models: {valid_face_models}"
            )
        if not 0.0 <= self.aesrgan_strength <= 1.0:
            raise ValueError("aesrgan_strength must be between 0.0 and 1.0")

        # Validate QP artifact removal options
        if not 0.1 <= self.qp_strength <= 3.0:
            raise ValueError("qp_strength must be between 0.1 and 3.0")

        # Validate frame generation options
        valid_frame_gen_models = ["svd", "optical_flow_warp", "interpolate_blend"]
        if self.frame_gen_model not in valid_frame_gen_models:
            raise ValueError(
                f"Invalid frame generation model '{self.frame_gen_model}'. "
                f"Valid models: {valid_frame_gen_models}"
            )
        if self.max_gap_frames < 1:
            raise ValueError("max_gap_frames must be at least 1")

        # Validate temporal consistency options
        valid_temporal_methods = ["optical_flow", "cross_attention", "hybrid"]
        if self.temporal_method not in valid_temporal_methods:
            raise ValueError(
                f"Invalid temporal method '{self.temporal_method}'. "
                f"Valid methods: {valid_temporal_methods}"
            )
        if self.cross_attention_window < 1:
            raise ValueError("cross_attention_window must be at least 1")
        if not 0.0 <= self.temporal_blend_strength <= 1.0:
            raise ValueError("temporal_blend_strength must be between 0.0 and 1.0")

        # Convert colorization reference images to Path objects
        if self.colorization_reference_images:
            self.colorization_reference_images = [
                Path(p) if not isinstance(p, Path) else p
                for p in self.colorization_reference_images
            ]

        # Convert output_dir to Path if provided
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

        # Convert model_download_dir to Path if provided
        if self.model_download_dir is not None and not isinstance(self.model_download_dir, Path):
            self.model_download_dir = Path(self.model_download_dir)

        # Validate v2.1 modular feature options
        if not 0.0 <= self.scene_aware_intensity_scale <= 2.0:
            raise ValueError("scene_aware_intensity_scale must be between 0.0 and 2.0")

        if not 0.0 <= self.motion_adaptive_sensitivity <= 1.0:
            raise ValueError("motion_adaptive_sensitivity must be between 0.0 and 1.0")

        if self.av_sync_max_drift_ms < 0:
            raise ValueError("av_sync_max_drift_ms must be non-negative")

        valid_hdr_formats = ["hdr10", "hdr10plus", "dolby_vision", "hlg"]
        if self.hdr_target_format not in valid_hdr_formats:
            raise ValueError(
                f"Invalid HDR format '{self.hdr_target_format}'. "
                f"Valid formats: {valid_hdr_formats}"
            )

        if self.hdr_peak_brightness < 100 or self.hdr_peak_brightness > 10000:
            raise ValueError("hdr_peak_brightness must be between 100 and 10000 nits")

        valid_ivtc_patterns = ["auto", "3:2", "2:3", "2:2", "mixed"]
        if self.ivtc_pattern not in valid_ivtc_patterns:
            raise ValueError(
                f"Invalid IVTC pattern '{self.ivtc_pattern}'. "
                f"Valid patterns: {valid_ivtc_patterns}"
            )

        valid_perceptual_modes = ["faithful", "balanced", "enhanced"]
        if self.perceptual_mode not in valid_perceptual_modes:
            raise ValueError(
                f"Invalid perceptual mode '{self.perceptual_mode}'. "
                f"Valid modes: {valid_perceptual_modes}"
            )

        if not 0.0 <= self.perceptual_balance <= 1.0:
            raise ValueError("perceptual_balance must be between 0.0 and 1.0")

        valid_library_types = ["plex", "jellyfin", "emby"]
        if self.library_server_type not in valid_library_types:
            raise ValueError(
                f"Invalid library server type '{self.library_server_type}'. "
                f"Valid types: {valid_library_types}"
            )

        # Validate seasonal color grading
        valid_seasons = [None, "winter", "spring", "summer", "autumn"]
        if self.seasonal_color_grade not in valid_seasons:
            raise ValueError(
                f"Invalid seasonal_color_grade '{self.seasonal_color_grade}'. "
                f"Valid options: {[s for s in valid_seasons if s]}"
            )
        if not 0.0 <= self.color_grade_strength <= 1.0:
            raise ValueError("color_grade_strength must be between 0.0 and 1.0")

        # Validate YouTube upload options
        valid_yt_privacy = ["public", "unlisted", "private"]
        if self.youtube_privacy not in valid_yt_privacy:
            raise ValueError(
                f"Invalid youtube_privacy '{self.youtube_privacy}'. "
                f"Valid options: {valid_yt_privacy}"
            )
        if self.youtube_client_secrets is not None and not isinstance(self.youtube_client_secrets, Path):
            self.youtube_client_secrets = Path(self.youtube_client_secrets)

        # Validate preprocessing fix options
        valid_interlace_methods = ["auto", "yadif", "bwdif", "nnedi"]
        if self.interlace_method not in valid_interlace_methods:
            raise ValueError(
                f"Invalid interlace method '{self.interlace_method}'. "
                f"Valid methods: {valid_interlace_methods}"
            )

        valid_film_stocks = [
            None, "kodachrome", "ektachrome", "kodacolor", "vision",
            "technicolor_2", "technicolor_3", "technicolor_ib",
            "agfacolor", "fujifilm", "eastmancolor",
            "early_color", "classic_color", "modern_color"
        ]
        if self.film_stock_override not in valid_film_stocks:
            raise ValueError(
                f"Invalid film stock override '{self.film_stock_override}'. "
                f"Valid stocks: {[s for s in valid_film_stocks if s]}"
            )

        valid_audio_sync_methods = ["resample", "pts_adjust"]
        if self.audio_sync_method not in valid_audio_sync_methods:
            raise ValueError(
                f"Invalid audio sync method '{self.audio_sync_method}'. "
                f"Valid methods: {valid_audio_sync_methods}"
            )

        # Convert notification_config_path to Path if provided
        if self.notification_config_path is not None and not isinstance(self.notification_config_path, Path):
            self.notification_config_path = Path(self.notification_config_path)

        # Convert scheduler_config_path to Path if provided
        if self.scheduler_config_path is not None and not isinstance(self.scheduler_config_path, Path):
            self.scheduler_config_path = Path(self.scheduler_config_path)

    def get_output_dir(self) -> Path:
        """Get the effective output directory.

        Returns:
            output_dir if set, otherwise _output_dir_override if set,
            otherwise project_dir/output/
        """
        if self.output_dir is not None:
            return self.output_dir
        if self._output_dir_override is not None:
            return self._output_dir_override
        return self.project_dir / "output"

    def get_frames_dir(self) -> Path:
        """Get the effective frames directory.

        Returns:
            _frames_dir_override if set, otherwise the computed frames_dir
        """
        if self._frames_dir_override is not None:
            return self._frames_dir_override
        return self.frames_dir

    def get_enhanced_dir(self) -> Path:
        """Get the effective enhanced frames directory.

        Returns:
            _enhanced_dir_override if set, otherwise the computed enhanced_dir
        """
        if self._enhanced_dir_override is not None:
            return self._enhanced_dir_override
        return self.enhanced_dir

    def get_model_path(self, model_name: str) -> Path:
        """Get the full path for a model file.

        Args:
            model_name: Name of the model (with or without extension)

        Returns:
            Full path to the model file in the model directory
        """
        return self.model_dir / model_name

    def ensure_directories(self) -> None:
        """Create all required directories for the pipeline.

        Creates:
            - project_dir
            - model_dir
            - output_dir (from get_output_dir())
            - frames_dir (from get_frames_dir())
            - enhanced_dir (from get_enhanced_dir())
            - temp_dir
            - checkpoint_dir (if checkpointing enabled)
            - interpolated_dir (if interpolation enabled)
        """
        directories: List[Path] = [
            self.project_dir,
            self.model_dir,
            self.get_output_dir(),
            self.get_frames_dir(),
            self.get_enhanced_dir(),
            self.temp_dir,
        ]

        if self.enable_checkpointing:
            directories.append(self.checkpoint_dir)

        if self.enable_interpolation:
            directories.append(self.interpolated_dir)

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def validate_paths(self) -> List[str]:
        """Validate all configured paths.

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors: List[str] = []

        # Check if project_dir parent exists or can be created
        if not self.project_dir.parent.exists():
            try:
                self.project_dir.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                errors.append(f"Cannot create project directory parent: {e}")

        # Check model_dir is writable (or its parent)
        model_dir_to_check = self.model_dir if self.model_dir.exists() else self.model_dir.parent
        if model_dir_to_check.exists() and not os.access(model_dir_to_check, os.W_OK):
            errors.append(f"Model directory is not writable: {self.model_dir}")

        # Check watermark mask if specified
        if self.watermark_mask_path is not None and not self.watermark_mask_path.exists():
            errors.append(f"Watermark mask file not found: {self.watermark_mask_path}")

        # Validate override directories if specified
        for name, path in [
            ("output_dir_override", self._output_dir_override),
            ("frames_dir_override", self._frames_dir_override),
            ("enhanced_dir_override", self._enhanced_dir_override),
        ]:
            if path is not None:
                parent = path.parent
                if parent.exists() and not os.access(parent, os.W_OK):
                    errors.append(f"{name} parent directory is not writable: {parent}")

        return errors

    def get_output_path(self, filename: Optional[str] = None) -> Path:
        """Get the full output file path.

        Args:
            filename: Optional filename (defaults to 'restored_video.{format}')

        Returns:
            Full path to output file
        """
        if filename is None:
            filename = f"restored_video.{self.output_format}"
        return self.get_output_dir() / filename

    def create_directories(self) -> None:
        """Create all required directories for the pipeline."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.enhanced_dir.mkdir(parents=True, exist_ok=True)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_deduplication:
            self.unique_frames_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_interpolation:
            self.interpolated_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_checkpointing:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def cleanup_temp(self) -> None:
        """Remove temporary directories and their contents."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Useful for serialization and checkpointing.
        """
        data = {}
        for key in [
            'project_dir', 'output_dir', 'scale_factor', 'model_name', 'crf', 'preset',
            'output_format', 'enable_checkpointing', 'checkpoint_interval',
            'enable_validation', 'min_ssim_threshold', 'min_psnr_threshold',
            'enable_disk_validation', 'disk_safety_margin', 'enable_vram_monitoring',
            'tile_size', 'max_retries', 'retry_delay', 'parallel_frames',
            'continue_on_error', 'require_gpu', 'gpu_id', 'enable_multi_gpu', 'gpu_ids', 'gpu_load_balance_strategy',
            'workers_per_gpu', 'enable_work_stealing', 'enable_interpolation', 'target_fps',
            'rife_model', 'rife_gpu_id', 'enable_deduplication', 'deduplication_threshold',
            'expected_source_fps', 'enable_auto_enhance', 'auto_detect_content',
            'auto_defect_repair', 'auto_face_restore', 'scratch_sensitivity',
            'dust_sensitivity', 'grain_reduction', 'model_download_dir', 'model_dir',
            'enable_colorization', 'colorization_model', 'colorization_strength',
            'enable_reference_enhance', 'reference_images_dir', 'reference_strength',
            'reference_guidance_scale', 'reference_ip_adapter_scale',
            'enable_watermark_removal', 'watermark_mask_path', 'watermark_auto_detect',
            'enable_subtitle_removal', 'subtitle_region', 'subtitle_ocr_engine', 'subtitle_languages',
            # v2.1 modular features
            'enable_scene_aware', 'scene_aware_intensity_scale', 'scene_aware_preserve_faces',
            'scene_aware_preserve_text', 'enable_motion_adaptive', 'motion_adaptive_sensitivity',
            'enable_av_sync_repair', 'av_sync_max_drift_ms', 'enable_hdr_expansion',
            'hdr_target_format', 'hdr_peak_brightness', 'enable_aspect_correction',
            'aspect_target_ratio', 'aspect_crop_letterbox', 'enable_ivtc', 'ivtc_pattern',
            'enable_perceptual_tuning', 'perceptual_mode', 'perceptual_balance',
            'enable_sidecar', 'enable_notifications', 'notification_config_path',
            'enable_daemon', 'daemon_auto_resume', 'enable_scheduling', 'scheduler_config_path',
            'enable_library_integration', 'library_server_type', 'library_server_url',
            'library_api_token', 'enable_proxy_workflow', 'enable_quality_tracking',
            # Preprocessing fixes
            'enable_interlace_fix', 'interlace_method', 'enable_letterbox_crop',
            'enable_film_color_correction', 'film_stock_override',
            'enable_audio_sync_fix', 'audio_sync_method'
        ]:
            val = getattr(self, key)
            if isinstance(val, Path):
                data[key] = str(val)
            elif val is None:
                data[key] = None
            else:
                data[key] = val
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            Config instance
        """
        # Convert project_dir to Path
        if 'project_dir' in data and isinstance(data['project_dir'], str):
            data['project_dir'] = Path(data['project_dir'])

        # Convert output_dir to Path if provided
        if 'output_dir' in data and isinstance(data['output_dir'], str):
            data['output_dir'] = Path(data['output_dir'])

        # Convert model_download_dir to Path if provided
        if 'model_download_dir' in data and isinstance(data['model_download_dir'], str):
            data['model_download_dir'] = Path(data['model_download_dir'])

        # Convert model_dir to Path if provided
        if 'model_dir' in data and isinstance(data['model_dir'], str):
            data['model_dir'] = Path(data['model_dir'])

        # Convert reference_images_dir to Path if provided
        if 'reference_images_dir' in data and isinstance(data['reference_images_dir'], str):
            data['reference_images_dir'] = Path(data['reference_images_dir'])

        # Convert watermark_mask_path to Path if provided
        if 'watermark_mask_path' in data and isinstance(data['watermark_mask_path'], str):
            data['watermark_mask_path'] = Path(data['watermark_mask_path'])

        # Convert notification_config_path to Path if provided
        if 'notification_config_path' in data and isinstance(data['notification_config_path'], str):
            data['notification_config_path'] = Path(data['notification_config_path'])

        # Convert scheduler_config_path to Path if provided
        if 'scheduler_config_path' in data and isinstance(data['scheduler_config_path'], str):
            data['scheduler_config_path'] = Path(data['scheduler_config_path'])

        # Filter to only valid init parameters
        valid_keys = {
            'project_dir', 'output_dir', 'scale_factor', 'model_name', 'crf', 'preset',
            'output_format', 'enable_checkpointing', 'checkpoint_interval',
            'enable_validation', 'min_ssim_threshold', 'min_psnr_threshold',
            'enable_disk_validation', 'disk_safety_margin', 'enable_vram_monitoring',
            'tile_size', 'max_retries', 'retry_delay', 'parallel_frames',
            'continue_on_error', 'require_gpu', 'gpu_id', 'enable_multi_gpu', 'gpu_ids', 'gpu_load_balance_strategy',
            'workers_per_gpu', 'enable_work_stealing', 'enable_interpolation', 'target_fps',
            'rife_model', 'rife_gpu_id', 'enable_deduplication', 'deduplication_threshold',
            'expected_source_fps', 'enable_auto_enhance', 'auto_detect_content',
            'auto_defect_repair', 'auto_face_restore', 'scratch_sensitivity',
            'dust_sensitivity', 'grain_reduction', 'model_download_dir', 'model_dir',
            'enable_colorization', 'colorization_model', 'colorization_strength',
            'enable_reference_enhance', 'reference_images_dir', 'reference_strength',
            'reference_guidance_scale', 'reference_ip_adapter_scale',
            'enable_watermark_removal', 'watermark_mask_path', 'watermark_auto_detect',
            'enable_subtitle_removal', 'subtitle_region', 'subtitle_ocr_engine', 'subtitle_languages',
            '_output_dir_override', '_frames_dir_override', '_enhanced_dir_override',
            # v2.1 modular features
            'enable_scene_aware', 'scene_aware_intensity_scale', 'scene_aware_preserve_faces',
            'scene_aware_preserve_text', 'enable_motion_adaptive', 'motion_adaptive_sensitivity',
            'enable_av_sync_repair', 'av_sync_max_drift_ms', 'enable_hdr_expansion',
            'hdr_target_format', 'hdr_peak_brightness', 'enable_aspect_correction',
            'aspect_target_ratio', 'aspect_crop_letterbox', 'enable_ivtc', 'ivtc_pattern',
            'enable_perceptual_tuning', 'perceptual_mode', 'perceptual_balance',
            'enable_sidecar', 'enable_notifications', 'notification_config_path',
            'enable_daemon', 'daemon_auto_resume', 'enable_scheduling', 'scheduler_config_path',
            'enable_library_integration', 'library_server_type', 'library_server_url',
            'library_api_token', 'enable_proxy_workflow', 'enable_quality_tracking',
            # Preprocessing fixes
            'enable_interlace_fix', 'interlace_method', 'enable_letterbox_crop',
            'enable_film_color_correction', 'film_stock_override',
            'enable_audio_sync_fix', 'audio_sync_method'
        }

        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def get_hash(self) -> str:
        """Generate hash of configuration for change detection.

        Returns:
            SHA256 hash (first 16 characters) of configuration
        """
        # Only hash settings that affect output
        hash_data = {
            'scale_factor': self.scale_factor,
            'model_name': self.model_name,
            'crf': self.crf,
            'preset': self.preset,
            'tile_size': self.tile_size,
        }
        config_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    @classmethod
    def from_preset(cls, preset_name: str, project_dir: Path, **overrides) -> "Config":
        """Create a configuration from a preset name.

        Args:
            preset_name: Name of the preset ('fast', 'quality', 'archive', 'anime', 'film_restoration')
            project_dir: Root directory for processing files
            **overrides: Additional parameters to override preset defaults

        Returns:
            Config instance with preset settings

        Raises:
            ValueError: If preset_name is not recognized
        """
        if preset_name not in PRESETS:
            available = ', '.join(PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")

        preset_config = PRESETS[preset_name].copy()
        preset_config['project_dir'] = project_dir
        preset_config.update(overrides)

        return cls(**preset_config)

    @classmethod
    def list_presets(cls) -> Dict[str, str]:
        """Get list of available presets with descriptions.

        Returns:
            Dictionary mapping preset names to descriptions
        """
        descriptions = {
            "fast": "Quick processing with 2x upscale, minimal quality checks",
            "quality": "High quality 4x upscale with validation enabled",
            "archive": "Archival quality with frame generation and old film features",
            "anime": "Optimized for anime/animation content",
            "film_restoration": "Full restoration for old films with defect repair",
            "ultimate": "Maximum quality for RTX 5090/high-end hardware (32GB+ VRAM)",
            "authentic": "Preserve period character - ideal for historic footage",
            "vhs": "VHS/analog tape restoration with tracking and dropout repair",
        }
        return descriptions

    def save_preset(self, filepath: Path, name: str = "custom") -> None:
        """Save current configuration as a preset file.

        Args:
            filepath: Path to save the preset JSON file
            name: Name to identify this preset
        """
        preset_data = {
            "name": name,
            "version": "1.3.1",
            "config": self.to_dict(),
        }
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(preset_data, f, indent=2)

    @classmethod
    def load_preset_file(cls, filepath: Path) -> "Config":
        """Load a configuration from a preset file.

        Args:
            filepath: Path to the preset JSON file

        Returns:
            Config instance from the preset file

        Raises:
            FileNotFoundError: If preset file doesn't exist
            ValueError: If preset file is invalid
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Preset file not found: {filepath}")

        with open(filepath, 'r') as f:
            preset_data = json.load(f)

        if 'config' not in preset_data:
            raise ValueError("Invalid preset file: missing 'config' key")

        return cls.from_dict(preset_data['config'])

    def get_tile_size_for_resolution(
        self,
        width: int,
        height: int,
        available_vram_mb: Optional[int] = None
    ) -> int:
        """Calculate appropriate tile size for given resolution.

        Args:
            width: Frame width
            height: Frame height
            available_vram_mb: Available VRAM (auto-detected if None)

        Returns:
            Tile size (0 if no tiling needed)
        """
        if self.tile_size is None:
            return 0  # No tiling

        if self.tile_size > 0:
            return self.tile_size  # Use configured size

        # Auto-calculate tile size
        from .utils.gpu import calculate_optimal_tile_size

        return calculate_optimal_tile_size(
            frame_resolution=(width, height),
            scale_factor=self.scale_factor,
            available_vram_mb=available_vram_mb,
            model_name=self.model_name,
        )


@dataclass
class RestoreOptions:
    """Additional options for video restoration.

    Separate from Config to allow per-restoration customization.
    """
    source: str  # URL or file path
    output_path: Optional[Path] = None
    cleanup: bool = True
    resume: bool = True  # Resume from checkpoint if available
    validate_output: bool = True
    skip_audio: bool = False
    dry_run: bool = False  # Validate only, don't process

    # Preview/approval options
    preview_before_reassembly: bool = False  # Pause for user to inspect frames
    preview_frame_count: int = 5  # Number of sample frames to show

    # RIFE interpolation overrides (can override Config settings per-run)
    enable_rife: Optional[bool] = None  # None = use Config setting
    target_fps: Optional[float] = None  # None = use Config setting or auto-detect

    def __post_init__(self) -> None:
        """Validate options."""
        if self.output_path is not None and not isinstance(self.output_path, Path):
            self.output_path = Path(self.output_path)

        if self.target_fps is not None and self.target_fps <= 0:
            raise ValueError("target_fps must be positive")

        if self.preview_frame_count < 1:
            raise ValueError("preview_frame_count must be at least 1")


@dataclass
class CloudConfig:
    """Configuration for cloud processing.

    Attributes:
        provider: Cloud GPU provider ('runpod', 'vastai').
        api_key: Provider API key (can also be set via environment variable).
        gpu_type: Preferred GPU type (e.g., 'RTX_4090', 'A100_80GB').
        storage_backend: Cloud storage backend ('s3', 'gcs', 'azure').
        storage_bucket: Storage bucket/container name.
        storage_credentials: Storage-specific credentials dict.
        max_runtime_minutes: Maximum job runtime before timeout.
        auto_cleanup: Automatically delete remote files after completion.
        use_serverless: Use serverless endpoints if available.
        endpoint_id: Provider-specific endpoint ID for serverless.
    """

    provider: str = "runpod"  # runpod, vastai
    api_key: Optional[str] = None
    gpu_type: str = "RTX_4090"
    storage_backend: str = "s3"  # s3, gcs, azure
    storage_bucket: Optional[str] = None
    storage_credentials: Dict[str, Any] = field(default_factory=dict)
    max_runtime_minutes: int = 120
    auto_cleanup: bool = True
    use_serverless: bool = True
    endpoint_id: Optional[str] = None

    # Valid providers and GPU types
    VALID_PROVIDERS: ClassVar[List[str]] = ["runpod", "vastai"]
    VALID_STORAGE_BACKENDS: ClassVar[List[str]] = ["s3", "gcs", "azure"]
    VALID_GPU_TYPES: ClassVar[List[str]] = [
        "RTX_4090",
        "RTX_3090",
        "RTX_4080",
        "RTX_3080",
        "A100_80GB",
        "A100_40GB",
        "A6000",
        "H100",
        "L40",
    ]

    def __post_init__(self) -> None:
        """Validate cloud configuration."""
        if self.provider not in self.VALID_PROVIDERS:
            raise ValueError(
                f"Invalid provider '{self.provider}'. "
                f"Valid providers: {self.VALID_PROVIDERS}"
            )

        if self.storage_backend not in self.VALID_STORAGE_BACKENDS:
            raise ValueError(
                f"Invalid storage backend '{self.storage_backend}'. "
                f"Valid backends: {self.VALID_STORAGE_BACKENDS}"
            )

        if self.gpu_type not in self.VALID_GPU_TYPES:
            raise ValueError(
                f"Invalid GPU type '{self.gpu_type}'. "
                f"Valid types: {self.VALID_GPU_TYPES}"
            )

        if self.max_runtime_minutes < 1:
            raise ValueError("max_runtime_minutes must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "api_key": self.api_key,
            "gpu_type": self.gpu_type,
            "storage_backend": self.storage_backend,
            "storage_bucket": self.storage_bucket,
            "storage_credentials": self.storage_credentials,
            "max_runtime_minutes": self.max_runtime_minutes,
            "auto_cleanup": self.auto_cleanup,
            "use_serverless": self.use_serverless,
            "endpoint_id": self.endpoint_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CloudConfig":
        """Create from dictionary."""
        valid_keys = {
            "provider",
            "api_key",
            "gpu_type",
            "storage_backend",
            "storage_bucket",
            "storage_credentials",
            "max_runtime_minutes",
            "auto_cleanup",
            "use_serverless",
            "endpoint_id",
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def load_from_file(cls, filepath: Path) -> "CloudConfig":
        """Load cloud configuration from JSON file.

        Args:
            filepath: Path to configuration file.

        Returns:
            CloudConfig instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file contains invalid configuration.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Cloud config file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def save_to_file(self, filepath: Path) -> None:
        """Save cloud configuration to JSON file.

        Args:
            filepath: Path to save configuration.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Don't save API key to file for security
        data = self.to_dict()
        if data.get("api_key"):
            data["api_key"] = "***REDACTED***"
        if data.get("storage_credentials"):
            data["storage_credentials"] = {"***": "REDACTED"}

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get default path for cloud configuration file."""
        return Path.home() / ".framewright" / "cloud_config.json"

    def get_provider_instance(self):
        """Get configured cloud provider instance.

        Returns:
            CloudProvider instance configured with this config.

        Raises:
            ImportError: If cloud dependencies not installed.
        """
        try:
            from .cloud import RunPodProvider, VastAIProvider

            if self.provider == "runpod":
                return RunPodProvider(
                    api_key=self.api_key,
                    endpoint_id=self.endpoint_id,
                    use_serverless=self.use_serverless,
                )
            elif self.provider == "vastai":
                return VastAIProvider(
                    api_key=self.api_key,
                )
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

        except ImportError:
            raise ImportError(
                "Cloud dependencies not installed. "
                "Install with: pip install framewright[cloud]"
            )

    def get_storage_instance(self):
        """Get configured storage provider instance.

        Returns:
            CloudStorageProvider instance configured with this config.

        Raises:
            ImportError: If cloud dependencies not installed.
        """
        try:
            from .cloud.storage import get_storage_provider

            return get_storage_provider(
                self.storage_backend,
                bucket=self.storage_bucket,
                credentials=self.storage_credentials,
            )

        except ImportError:
            raise ImportError(
                "Cloud dependencies not installed. "
                "Install with: pip install framewright[cloud]"
            )
