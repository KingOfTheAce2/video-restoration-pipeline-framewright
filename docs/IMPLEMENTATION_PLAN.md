# FrameWright: Ultimate Video Restoration Pipeline - Implementation Plan

## Vision Statement

Transform FrameWright into the world's most intuitive, powerful video/audio/image restoration pipeline - where complexity is invisible and results are extraordinary. Following Apple's design philosophy: **"It just works"** at the surface, with **"There's more if you want it"** underneath.

---

## Executive Summary

Based on comprehensive research of:
- 187 Python modules in the current codebase
- 53 specialized processors
- 28+ utility modules
- Industry leaders (Topaz, DaVinci Resolve, Adobe)
- Cutting-edge AI research (2025-2026)
- Apple design principles

This plan outlines a strategic transformation in **4 phases over 12 months**.

---

## Part 1: Architecture Streamlining

### CRITICAL PRINCIPLE: Hardware Flexibility Must Be Preserved

**The multiple modules exist for good reason.** Different hardware requires different approaches:

| Module Pair | Why BOTH Must Exist |
|-------------|---------------------|
| `pytorch_realesrgan.py` + `ncnn_vulkan.py` | PyTorch needs CUDA; NCNN/Vulkan works on AMD/Intel/older GPUs |
| `temporal_denoise.py` + `tap_denoise.py` | Traditional is CPU-friendly (2GB); Neural needs 8GB+ VRAM |
| `colorization.py` + `swintexco_colorize.py` | DeOldify works on 2GB; SwinTExCo needs 8GB+ |
| `face_restore.py` + `aesrgan_face.py` | GFPGAN is lightweight; AESRGAN is high-quality but heavy |
| `hat_upscaler.py` + `diffusion_sr.py` | HAT works on 8GB; Diffusion needs 12GB+ |

**"Consolidation" means UNIFIED INTERFACES, not feature removal.**

### What Consolidation Actually Means

```python
# WRONG: Delete modules to reduce code
# del tap_denoise.py  # NO! This removes 8GB+ VRAM capability

# CORRECT: Unified interface with multiple backends preserved
class Denoiser:
    """Unified denoising - ALL backends preserved"""

    BACKENDS = {
        "cpu":        TraditionalDenoiser,    # Works everywhere, no GPU
        "cuda_2gb":   BasicTemporalDenoiser,  # GTX 1050 Ti level
        "cuda_4gb":   TemporalDenoiser,       # GTX 1650/1660 level
        "cuda_8gb":   TAPDenoiser,            # RTX 3060/3070 level
        "cuda_12gb+": AdvancedTAPDenoiser,    # RTX 3080+ level
        "ncnn":       NCNNDenoiser,           # AMD/Intel/Mac GPUs
        "tensorrt":   TensorRTDenoiser,       # NVIDIA optimized
        "coreml":     CoreMLDenoiser,         # Apple Silicon ANE
    }

    def __init__(self, config: Config):
        # Auto-select best backend for user's hardware
        self.backend = self._select_optimal_backend(config.hardware)

    def _select_optimal_backend(self, hw: HardwareInfo) -> BaseDenoiser:
        """Automatic fallback chain based on available hardware"""
        if hw.has_cuda and hw.vram_gb >= 12:
            return self.BACKENDS["cuda_12gb+"]()
        elif hw.has_cuda and hw.vram_gb >= 8:
            return self.BACKENDS["cuda_8gb"]()
        elif hw.has_cuda and hw.vram_gb >= 4:
            return self.BACKENDS["cuda_4gb"]()
        elif hw.has_cuda:
            return self.BACKENDS["cuda_2gb"]()
        elif hw.has_vulkan:
            return self.BACKENDS["ncnn"]()
        elif hw.has_apple_silicon:
            return self.BACKENDS["coreml"]()
        else:
            return self.BACKENDS["cpu"]()  # Always works
```

### Hardware Tiers (All Must Be Supported)

| Tier | Example Hardware | VRAM | What Works | Target Users |
|------|------------------|------|------------|--------------|
| **CPU Only** | Any CPU, integrated graphics | 0 | NCNN CPU, basic denoise, audio | Laptops, old PCs |
| **Entry GPU** | GTX 1050 Ti, RX 570 | 4GB | Real-ESRGAN, GFPGAN, RIFE | Budget builders |
| **Mid GPU** | RTX 3060, RX 6700 XT | 8-12GB | HAT, TAP denoise, CodeFormer | Most users |
| **High GPU** | RTX 4080, RX 7900 XT | 16GB | Diffusion SR, ensemble models | Enthusiasts |
| **Extreme GPU** | RTX 4090, RTX 5090 | 24-32GB | Everything at max quality | Professionals |
| **Apple Silicon** | M1/M2/M3 Pro/Max/Ultra | Unified | CoreML optimized models | Mac users |
| **Cloud** | RunPod A100, Vast.ai | 40-80GB | All features, no local limits | Anyone |

### Current State Issues (Code Organization Only)

| Problem | Impact | Files Affected | Solution |
|---------|--------|----------------|----------|
| Dual error systems | Maintenance confusion | `errors.py`, `exceptions.py` | Merge into one file (keep all exceptions) |
| 3 GPU detection methods | Duplicated detection code | `gpu.py`, `multi_gpu.py`, `gpu_memory_optimizer.py` | Shared detection, separate backends |
| Scattered model management | Same download logic repeated | `model_manager.py`, `model_cache.py`, `profiles.py` | Unified manager, preserve all models |
| 50+ config parameters | Overwhelming for users | `config.py` | Better defaults + progressive disclosure |

### Proposed Consolidated Architecture

```
src/framewright/
├── core/                          # Core abstractions (NEW)
│   ├── __init__.py
│   ├── config.py                  # Simplified config (replaces current)
│   ├── errors.py                  # Unified exception hierarchy
│   ├── events.py                  # Event system for coordination
│   └── types.py                   # Shared type definitions
│
├── infrastructure/                # Infrastructure layer (NEW)
│   ├── gpu/
│   │   ├── __init__.py
│   │   ├── detector.py            # Unified GPU detection
│   │   ├── memory.py              # Memory management
│   │   ├── distributor.py         # Multi-GPU distribution
│   │   └── backends/
│   │       ├── nvidia.py
│   │       ├── amd.py
│   │       └── intel.py
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── frame_cache.py
│   │   ├── model_cache.py
│   │   └── eviction.py
│   └── models/
│       ├── __init__.py
│       ├── manager.py             # Unified model manager
│       ├── registry.py            # Model registry
│       └── downloader.py
│
├── engine/                        # Processing engine (NEW)
│   ├── __init__.py
│   ├── pipeline.py                # Pipeline orchestrator
│   ├── scheduler.py               # Job scheduling
│   └── checkpoint.py              # State persistence
│
├── processors/                    # Consolidated processors
│   ├── analysis/                  # Content analysis (merged)
│   │   ├── content_analyzer.py    # analyzer + scene_intelligence
│   │   ├── quality_scorer.py
│   │   └── degradation_detector.py
│   ├── restoration/               # Core restoration
│   │   ├── faces.py               # face_restore + aesrgan_face
│   │   ├── defects.py
│   │   ├── colorization.py        # All 3 colorization methods
│   │   └── stabilization.py
│   ├── enhancement/               # Enhancement
│   │   ├── super_resolution.py    # Unified SR (ESRGAN, HAT, Diffusion)
│   │   ├── denoising.py           # temporal + tap merged
│   │   └── hdr.py
│   ├── format/                    # Format handlers
│   │   ├── film.py
│   │   ├── vhs.py
│   │   ├── interlace.py
│   │   └── aspect.py
│   └── audio/                     # Audio (merged)
│       ├── enhancer.py            # audio_enhance + audio_restoration
│       └── sync.py
│
├── ui/                            # User interfaces
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py                # Single entry point
│   │   ├── wizard.py              # Interactive wizard
│   │   └── progress.py            # Unified progress display
│   ├── dashboard/                 # Web dashboard
│   └── api/                       # Programmatic API
│
└── presets/                       # Simplified presets
    ├── __init__.py
    ├── registry.py
    └── presets.yaml               # External config
```

### Consolidation Tasks (Unified Interfaces, ALL Backends Preserved)

| Current Modules | Unified Interface | What's PRESERVED | Savings Source |
|-----------------|-------------------|------------------|----------------|
| `temporal_denoise.py` + `tap_denoise.py` | `denoising.py` with `DenoiserBackend` | CPU, CUDA 4GB, CUDA 8GB+, NCNN backends | Shared validation, config parsing |
| `colorization.py` + `swintexco_colorize.py` + `temporal_colorization.py` | `colorization.py` with `ColorizerBackend` | DeOldify, DDColor, SwinTExCo, temporal propagation | Shared frame I/O, color space conversion |
| `audio_enhance.py` + `audio_restoration.py` | `audio/enhancer.py` with `AudioBackend` | Traditional FFmpeg, AI models, DeepFilter | Shared audio loading, format conversion |
| `face_restore.py` + `aesrgan_face.py` | `faces.py` with `FaceBackend` | GFPGAN 1.3/1.4, CodeFormer, RestoreFormer, AESRGAN | Shared face detection, alignment |
| `analyzer.py` + `scene_intelligence.py` + `scene_detection.py` | `content_analyzer.py` | All analysis capabilities | Shared frame sampling, metrics |
| `gpu.py` + `multi_gpu.py` + `gpu_memory_optimizer.py` | `infrastructure/gpu/` | NVIDIA, AMD, Intel, Apple detection | Shared detection logic only |
| `errors.py` + `exceptions.py` | `core/errors.py` | ALL exception types | Remove duplicate definitions |

**Savings come from:**
- Removing duplicate validation code (~40%)
- Shared I/O operations (~30%)
- Unified configuration parsing (~20%)
- Consolidated error handling (~10%)

**Total estimated reduction: ~3,300 lines** while **preserving ALL hardware backends and capabilities**.

---

## Part 2: Apple Design Principles Implementation

### The "Three Command" Philosophy

Users should need at most 3 different command patterns:

```bash
# 1. Just Works (80% of users)
framewright video.mp4

# 2. I Have Preferences (15% of users)
framewright video.mp4 --quality best
framewright video.mp4 --speed 30m
framewright video.mp4 --style film

# 3. I Know What I'm Doing (5% of users)
framewright video.mp4 --preset custom --config myconfig.yaml
```

### Simplified Preset System (Hardware-Aware)

**Current**: 10+ presets with overlapping concepts
**Proposed**: 3 primary + 4 style modifiers + AUTOMATIC HARDWARE ADAPTATION

```yaml
# presets/presets.yaml
primary:
  fast:
    description: "Quick results, good quality"
    scale: 2
    # Backends auto-selected based on hardware:
    processors:
      denoise: {backend: auto}      # CPU if no GPU, CUDA if available
      upscale: {backend: auto}      # NCNN for AMD/Intel, CUDA for NVIDIA

  balanced:
    description: "Great quality in reasonable time"
    scale: 4
    processors:
      analysis: {backend: auto}
      face_restore: {backend: auto}  # GFPGAN for 4GB, CodeFormer for 8GB+
      denoise: {backend: auto}       # Traditional for 4GB, TAP for 8GB+
      upscale: {backend: auto}       # Real-ESRGAN for 4GB, HAT for 12GB+

  best:
    description: "Maximum quality, no compromises"
    scale: 4
    processors:
      analysis: {backend: auto}
      face_restore: {backend: auto}
      denoise: {backend: auto}
      upscale: {backend: auto}       # Diffusion for 16GB+, HAT for 8GB+
      hdr: {backend: auto}

# Hardware tier overrides (automatic)
hardware_tiers:
  cpu_only:
    max_scale: 2
    force_backends:
      upscale: ncnn_cpu
      denoise: traditional
      face_restore: gfpgan_cpu

  vram_4gb:
    max_scale: 4
    force_backends:
      upscale: realesrgan
      denoise: temporal_basic
      face_restore: gfpgan

  vram_8gb:
    max_scale: 4
    force_backends:
      upscale: hat_small
      denoise: tap
      face_restore: codeformer

  vram_12gb:
    max_scale: 4
    force_backends:
      upscale: hat_large
      denoise: tap_extended
      face_restore: codeformer

  vram_16gb_plus:
    max_scale: 4
    force_backends:
      upscale: diffusion_sr
      denoise: tap_full
      face_restore: ensemble

  vram_24gb_plus:
    max_scale: 4
    force_backends:
      upscale: ensemble_sr
      denoise: full_pipeline
      face_restore: ensemble

  apple_silicon:
    max_scale: 4
    force_backends:
      upscale: coreml_esrgan
      denoise: coreml_denoise
      face_restore: coreml_face

styles:
  film:
    inherits: balanced
    processors_add: [grain_preservation, film_color_correction]

  animation:
    inherits: balanced
    model: realesr-animevideov3
    processors_remove: [face_restore]

  home-video:
    inherits: balanced
    processors_add: [vhs_restoration, audio_enhance]

  archive:
    inherits: best
    processors_add: [frame_generation, subtitle_preserve]
```

### Hardware Auto-Detection Flow

```python
# What happens when user runs: framewright video.mp4

def select_optimal_config(video_path: Path) -> Config:
    # 1. Detect hardware
    hardware = HardwareDetector().detect()
    # Returns: HardwareInfo(gpu="RTX 3060", vram_gb=12, has_cuda=True, ...)

    # 2. Analyze content
    content = ContentAnalyzer().quick_analyze(video_path)
    # Returns: ContentInfo(is_film=False, has_faces=True, is_interlaced=False, ...)

    # 3. Start with balanced preset
    config = PresetRegistry.get("balanced")

    # 4. Apply hardware tier overrides
    tier = hardware.get_tier()  # "vram_12gb"
    config = config.with_hardware_overrides(HARDWARE_TIERS[tier])

    # 5. Apply content-specific adjustments
    if content.is_animation:
        config = config.with_style("animation")
    elif content.is_film:
        config = config.with_style("film")

    # 6. Final validation
    config.validate_for_hardware(hardware)

    return config
```

### Human-Readable Progress

**Before:**
```
Frame 1234/5678 │ 45 fps │ ETA 3:45
████████████░░░░░░░░░░░░ 45%
```

**After:**
```
Enhancing your video...

████████████░░░░░░░░░░░░ 45%

Stage: Making faces look natural (3 of 5)
Time: 4h done → ~5h remaining
GPU: Working smoothly at 92%

Next: Improving clarity and sharpness
```

### Error Messages That Teach

**Before:**
```
VRAMError: CUDA out of memory
```

**After:**
```
Not enough GPU memory

Your RTX 3080 (10GB) needs more memory for this video.

Quick fixes:
  framewright video.mp4 --quality fast     (uses half the memory)
  framewright video.mp4 --tile 512         (processes smaller sections)

Need help? framewright help gpu-memory
```

---

## Part 3: New Features Roadmap

### Phase 1: Foundation (Months 1-3)

#### 1.1 Temporal Consistency Metrics (WCS-Style)
**Priority: CRITICAL**

```python
# New: processors/analysis/temporal_consistency.py
class TemporalConsistencyAnalyzer:
    """World Consistency Score implementation"""

    def analyze(self, frames: List[Frame]) -> TemporalReport:
        return TemporalReport(
            object_permanence=self._check_object_permanence(frames),
            relation_stability=self._check_relation_stability(frames),
            flicker_penalty=self._calculate_flicker(frames),
            causal_compliance=self._check_causality(frames),
            overall_score=self._calculate_wcs(frames)
        )
```

#### 1.2 Smart Preset Auto-Selection
**Priority: HIGH**

```python
# New: core/smart_selector.py
class SmartPresetSelector:
    """Analyzes input and selects optimal restoration chain"""

    def select(self, input_path: Path) -> PresetConfig:
        analysis = ContentAnalyzer().analyze(input_path)

        # Content-based selection
        if analysis.is_film:
            base = "film"
        elif analysis.is_animation:
            base = "animation"
        elif analysis.has_vhs_artifacts:
            base = "home-video"
        else:
            base = "balanced"

        # Hardware-based adjustments
        gpu_info = GPUDetector().detect()
        if gpu_info.vram_gb < 8:
            return self._optimize_for_low_vram(base)
        elif gpu_info.vram_gb >= 24:
            return self._optimize_for_high_vram(base)

        return PresetRegistry.get(base)
```

#### 1.3 Grain Preservation Mode
**Priority: HIGH**

```python
# New: processors/restoration/grain_manager.py
class GrainManager:
    """Extract, preserve, and restore film grain"""

    def extract_profile(self, frames: List[Frame]) -> GrainProfile:
        """Analyze and extract grain characteristics"""

    def remove_grain(self, frame: Frame) -> Tuple[Frame, GrainMask]:
        """Remove grain while preserving detail"""

    def restore_grain(self, frame: Frame, profile: GrainProfile,
                      opacity: float = 0.35) -> Frame:
        """Re-apply authentic grain after processing"""
```

#### 1.4 TensorRT Acceleration
**Priority: HIGH**

```python
# New: infrastructure/gpu/tensorrt_backend.py
class TensorRTBackend:
    """TensorRT-optimized inference backend"""

    def optimize_model(self, model_path: Path) -> Path:
        """Convert PyTorch model to TensorRT engine"""

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run TensorRT inference (up to 100% faster)"""
```

### Phase 2: Core Differentiation (Months 4-6)

#### 2.1 Diffusion-Based Super Resolution
**Priority: CRITICAL**

Based on FlashVSR research - one-step diffusion for real-time capable 4K upscaling.

```python
# New: processors/enhancement/diffusion_sr.py
class FlashSRProcessor:
    """One-step diffusion super resolution"""

    def __init__(self, config: FlashSRConfig):
        self.model = FlashVSR(
            sparse_attention=True,  # Locality-constrained
            guidance_scale=1.0,     # No classifier-free guidance needed
        )

    def upscale(self, frames: List[Frame], scale: int = 4) -> List[Frame]:
        """Real-time capable 4K upscaling (~17 FPS on A100)"""
```

#### 2.2 DeepFilterNet Audio Integration
**Priority: HIGH**

State-of-the-art audio restoration with 10-20ms latency.

```python
# New: processors/audio/deepfilter.py
class DeepFilterEnhancer:
    """DeepFilterNet 3 integration for audio restoration"""

    def enhance(self, audio: AudioSegment) -> AudioSegment:
        return self.pipeline(
            denoise=True,           # Adaptive noise removal
            dereverb=True,          # Room reverb reduction
            declip=True,            # Clipping restoration
            normalize=True,         # EBU R128 loudness
        )
```

#### 2.3 Real-Time Preview Server
**Priority: HIGH**

```python
# New: ui/preview/server.py
class PreviewServer:
    """Live preview without full render"""

    def start(self, input_path: Path, config: Config):
        """Start preview server on localhost:8080"""

    def render_segment(self, start: float, duration: float = 5.0):
        """Render short preview segment on demand"""

    def compare_settings(self, configs: List[Config]) -> ComparisonView:
        """Side-by-side comparison of different settings"""
```

#### 2.4 One-Click Restoration Wizard
**Priority: HIGH**

```python
# New: ui/cli/wizard.py
class RestorationWizard:
    """Guided restoration flow"""

    def run(self, input_path: Path) -> Config:
        # Step 1: Analyze
        self.display("Analyzing your video...")
        analysis = self.analyzer.analyze(input_path)

        # Step 2: Recommend
        recommendation = self.selector.select(analysis)
        self.display(f"Recommended: {recommendation.name}")
        self.display(f"  {recommendation.description}")

        # Step 3: Preview (optional)
        if self.ask_preview():
            preview = self.preview_server.render_segment(
                input_path, start=analysis.best_sample_time
            )
            self.display_preview(preview)

        # Step 4: Confirm
        if self.confirm():
            return recommendation.config
        else:
            return self.advanced_options()
```

### Phase 3: Advanced Features (Months 7-9)

#### 3.1 Cross-Attention Temporal Consistency
**Priority: HIGH**

TE-3DVAE approach from DiffVSR research.

```python
# New: processors/enhancement/temporal_vae.py
class TemporalVAE:
    """3D VAE with temporal attention for consistency"""

    def __init__(self):
        self.encoder = TemporalEncoder3D()
        self.attention = CrossFrameAttention(
            num_heads=8,
            window_size=16,  # Extended temporal window
        )
        self.decoder = TemporalDecoder3D()

    def process_batch(self, frames: List[Frame]) -> List[Frame]:
        """Process frames with temporal consistency enforcement"""
```

#### 3.2 HDR Dolby Vision Export
**Priority: MEDIUM**

```python
# New: processors/enhancement/hdr_export.py
class HDRExporter:
    """Dual HDR10 / Dolby Vision export"""

    def export(self, frames: List[Frame],
               format: Literal["hdr10", "dolby_vision", "both"]):
        if format in ["hdr10", "both"]:
            self._export_hdr10(frames)
        if format in ["dolby_vision", "both"]:
            self._export_dolby_vision(frames)
```

#### 3.3 Cloud Rendering Integration
**Priority: MEDIUM**

```python
# New: infrastructure/cloud/coordinator.py
class CloudCoordinator:
    """Coordinate cloud rendering on RunPod/Vast.ai"""

    def offload_job(self, job: Job, provider: str = "runpod") -> CloudJob:
        """Offload heavy processing to cloud"""

    def monitor_progress(self, cloud_job: CloudJob) -> Progress:
        """Track remote job progress"""

    def download_results(self, cloud_job: CloudJob) -> Path:
        """Download completed results"""
```

### Phase 4: Innovation (Months 10-12)

#### 4.1 Text-Guided Upscaling
**Priority: MEDIUM**

```python
# New: processors/enhancement/guided_sr.py
class GuidedSuperResolution:
    """Text-guided texture generation (Upscale-A-Video style)"""

    def upscale(self, frame: Frame,
                guidance: str = "high quality, sharp details, film grain"):
        """Generate textures guided by text description"""
```

#### 4.2 Generative Frame Extension
**Priority: LOW**

```python
# New: processors/restoration/frame_generator.py
class GenerativeFrameExtender:
    """Generate missing/extend frames using diffusion"""

    def extend_clip(self, frames: List[Frame],
                    direction: str, seconds: float) -> List[Frame]:
        """Extend clip forward or backward"""

    def fill_gap(self, before: List[Frame], after: List[Frame],
                 gap_frames: int) -> List[Frame]:
        """Generate frames to fill temporal gap"""
```

#### 4.3 NPU/Neural Engine Support
**Priority: MEDIUM**

```python
# New: infrastructure/gpu/backends/apple_silicon.py
class AppleSiliconBackend:
    """Apple Neural Engine optimization"""

    def convert_to_coreml(self, model: torch.nn.Module) -> Path:
        """Convert to Core ML for 78x speedup on ANE"""
```

---

## Part 4: Long-Form Temporal Consistency (7,000+ Frames)

### The Core Challenge

A 4-minute video at 30fps = 7,200 frames. A feature film = 150,000+ frames.

**The problem:** Most AI models process 4-16 frame windows. Without explicit long-form consistency:
- Colors drift gradually over thousands of frames
- Faces look different in minute 1 vs minute 10
- Grain/noise characteristics change scene to scene
- Enhancement "style" varies unpredictably

### Multi-Scale Temporal Architecture

```
Long-Form Consistency Hierarchy:

┌─────────────────────────────────────────────────────────────────────┐
│ GLOBAL CONSISTENCY (Entire Video)                                    │
│ - Color palette anchors extracted from key frames                   │
│ - Global noise/grain profile                                        │
│ - Face identity embeddings (maintain same face across all frames)   │
│ - Enhancement "style vector" locked for consistency                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│ SCENE 1       │         │ SCENE 2       │         │ SCENE N       │
│ Frames 0-450  │         │ Frames 451-890│         │ Frames X-Y    │
│               │         │               │         │               │
│ Scene-level   │         │ Scene-level   │         │ Scene-level   │
│ color anchor  │         │ color anchor  │         │ color anchor  │
│ (inherits     │         │ (inherits     │         │ (inherits     │
│  global)      │         │  global)      │         │  global)      │
└───────────────┘         └───────────────┘         └───────────────┘
        │                           │                           │
   ┌────┴────┐                 ┌────┴────┐                 ┌────┴────┐
   ▼         ▼                 ▼         ▼                 ▼         ▼
┌─────┐   ┌─────┐           ┌─────┐   ┌─────┐           ┌─────┐   ┌─────┐
│Chunk│   │Chunk│           │Chunk│   │Chunk│           │Chunk│   │Chunk│
│1-50 │   │51-100│          │451- │   │501- │           │...  │   │...  │
│     │   │      │          │500  │   │550  │           │     │   │     │
│16-frame│ │16-frame│       │16-frame│ │16-frame│       │     │   │     │
│window │ │window │         │window │ │window │         │     │   │     │
└─────┘   └─────┘           └─────┘   └─────┘           └─────┘   └─────┘
```

### Implementation: Hierarchical Consistency Manager

```python
# New: engine/temporal_consistency.py

class LongFormConsistencyManager:
    """Maintains consistency across thousands of frames"""

    def __init__(self, config: Config):
        self.global_anchors = GlobalAnchors()
        self.scene_anchors = {}  # scene_id -> SceneAnchors
        self.chunk_overlap = 4   # Frames overlap between chunks

    def initialize_from_video(self, video_path: Path) -> ConsistencyProfile:
        """
        Pre-analyze entire video to establish consistency anchors.
        This runs BEFORE any processing begins.
        """
        # 1. Detect all scene boundaries
        scenes = self.scene_detector.detect_all(video_path)

        # 2. Extract global anchors from representative frames
        key_frames = self._select_key_frames(video_path, count=20)
        self.global_anchors = GlobalAnchors(
            color_palette=self._extract_color_palette(key_frames),
            grain_profile=self._extract_grain_profile(key_frames),
            brightness_curve=self._extract_brightness_curve(key_frames),
            face_embeddings=self._extract_face_identities(key_frames),
            style_vector=self._compute_style_vector(key_frames),
        )

        # 3. Extract per-scene anchors (inherit from global)
        for scene in scenes:
            scene_key_frame = self._get_scene_representative(scene)
            self.scene_anchors[scene.id] = SceneAnchors(
                parent=self.global_anchors,
                local_color_shift=self._compute_color_shift(scene_key_frame),
                local_brightness=self._compute_brightness(scene_key_frame),
                motion_characteristics=self._analyze_motion(scene),
            )

        return ConsistencyProfile(
            global_anchors=self.global_anchors,
            scene_anchors=self.scene_anchors,
            scene_boundaries=scenes,
        )

    def process_chunk_with_consistency(
        self,
        chunk: FrameChunk,
        processor: BaseProcessor,
        profile: ConsistencyProfile,
    ) -> FrameChunk:
        """
        Process a chunk while enforcing consistency constraints.
        """
        scene = profile.get_scene_for_frame(chunk.start_frame)
        anchors = profile.scene_anchors[scene.id]

        # Process frames with local temporal window
        processed = processor.process(chunk.frames)

        # Apply consistency corrections
        corrected = self._apply_consistency_corrections(
            processed,
            anchors,
            profile.global_anchors,
        )

        return corrected

    def _apply_consistency_corrections(
        self,
        frames: List[Frame],
        scene_anchors: SceneAnchors,
        global_anchors: GlobalAnchors,
    ) -> List[Frame]:
        """
        Correct processed frames to match established anchors.
        """
        corrected = []
        for frame in frames:
            # Color consistency
            frame = self._match_color_palette(frame, global_anchors.color_palette)

            # Brightness consistency
            frame = self._match_brightness_curve(frame, global_anchors.brightness_curve)

            # Grain consistency (re-apply consistent grain)
            frame = self._match_grain_profile(frame, global_anchors.grain_profile)

            # Face identity consistency
            frame = self._enforce_face_identity(frame, global_anchors.face_embeddings)

            corrected.append(frame)

        return corrected


class GlobalAnchors:
    """Anchors that apply to the ENTIRE video"""

    color_palette: ColorPalette        # Dominant colors, white balance
    grain_profile: GrainProfile        # Noise characteristics
    brightness_curve: BrightnessCurve  # Overall exposure
    face_embeddings: Dict[str, FaceEmbedding]  # Known faces
    style_vector: StyleVector          # Enhancement "style"


class SceneAnchors:
    """Anchors specific to a scene (inherits from global)"""

    parent: GlobalAnchors
    local_color_shift: ColorShift      # Scene-specific color
    local_brightness: float            # Scene-specific exposure
    motion_characteristics: MotionProfile  # Camera movement, action
```

### Chunk Processing with Overlap

```python
class ChunkedProcessor:
    """Process video in overlapping chunks for seamless transitions"""

    def __init__(self, chunk_size: int = 50, overlap: int = 4):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_video(
        self,
        frames: Iterator[Frame],
        total_frames: int,
        processor: BaseProcessor,
        consistency: ConsistencyProfile,
    ) -> Iterator[Frame]:
        """
        Process entire video maintaining consistency.

        For 7,200 frames with chunk_size=50, overlap=4:
        - Chunk 1: frames 0-49 (output 0-45, blend 46-49)
        - Chunk 2: frames 46-95 (blend 46-49, output 50-91, blend 92-95)
        - Chunk 3: frames 92-141 (blend 92-95, output 96-137, blend 138-141)
        - ... and so on
        """
        buffer = []
        prev_overlap = None
        chunk_idx = 0

        for frame in frames:
            buffer.append(frame)

            if len(buffer) >= self.chunk_size:
                # Process this chunk
                chunk = FrameChunk(
                    frames=buffer,
                    start_frame=chunk_idx * (self.chunk_size - self.overlap),
                    chunk_idx=chunk_idx,
                )

                processed = self.consistency_manager.process_chunk_with_consistency(
                    chunk, processor, consistency
                )

                # Blend overlap region with previous chunk
                if prev_overlap is not None:
                    processed[:self.overlap] = self._blend_overlap(
                        prev_overlap, processed[:self.overlap]
                    )

                # Yield non-overlap frames
                for frame in processed[:-self.overlap]:
                    yield frame

                # Save overlap for next chunk
                prev_overlap = processed[-self.overlap:]

                # Keep overlap frames in buffer for context
                buffer = buffer[-self.overlap:]
                chunk_idx += 1

        # Process final partial chunk
        if buffer:
            # ... handle remaining frames

    def _blend_overlap(
        self,
        prev_frames: List[Frame],
        curr_frames: List[Frame],
    ) -> List[Frame]:
        """
        Smooth transition between chunks using temporal blending.
        """
        blended = []
        for i, (prev, curr) in enumerate(zip(prev_frames, curr_frames)):
            # Linear blend weight (can be improved with optical flow)
            weight = i / len(prev_frames)
            blended_frame = self._temporal_blend(prev, curr, weight)
            blended.append(blended_frame)
        return blended
```

### Face Identity Persistence

```python
class FaceIdentityTracker:
    """
    Ensure the same face looks consistent across 7,000+ frames.
    Critical for face restoration - prevents "face drift" over time.
    """

    def __init__(self):
        self.known_faces: Dict[str, FaceIdentity] = {}
        self.face_history: Dict[str, List[FaceAppearance]] = {}

    def register_faces_from_video(self, video_path: Path) -> None:
        """
        Pre-scan video to identify all unique faces.
        """
        # Sample frames throughout video
        sample_frames = self._sample_frames(video_path, count=100)

        for frame in sample_frames:
            faces = self.face_detector.detect(frame)
            for face in faces:
                embedding = self.face_encoder.encode(face)
                identity = self._match_or_create_identity(embedding)
                self._update_identity_stats(identity, face)

    def restore_face_consistently(
        self,
        face: Face,
        frame_idx: int,
        restorer: FaceRestorer,
    ) -> Face:
        """
        Restore face while maintaining identity consistency.
        """
        # Find which known identity this face belongs to
        identity = self._match_to_known_identity(face)

        if identity is None:
            # Unknown face - restore normally
            return restorer.restore(face)

        # Get identity's "canonical" appearance
        canonical = identity.get_canonical_appearance()

        # Restore with identity guidance
        restored = restorer.restore_with_guidance(
            face,
            identity_embedding=identity.embedding,
            canonical_features=canonical,
        )

        # Verify restored face still matches identity
        restored_embedding = self.face_encoder.encode(restored)
        similarity = self._compute_similarity(restored_embedding, identity.embedding)

        if similarity < 0.85:
            # Restoration drifted too far - blend with canonical
            restored = self._blend_toward_canonical(restored, canonical, strength=0.3)

        return restored
```

### Color Consistency Across Scenes

```python
class ColorConsistencyEnforcer:
    """
    Prevent color drift over thousands of frames.
    """

    def __init__(self, global_palette: ColorPalette):
        self.global_palette = global_palette
        self.correction_history = []

    def enforce_consistency(
        self,
        frame: Frame,
        frame_idx: int,
        scene_anchors: SceneAnchors,
    ) -> Frame:
        """
        Correct frame colors to match established palette.
        """
        # Extract current frame's color statistics
        current_stats = self._extract_color_stats(frame)

        # Compare to global palette
        drift = self._compute_color_drift(current_stats, self.global_palette)

        # If drift exceeds threshold, apply correction
        if drift.magnitude > 0.05:  # 5% drift threshold
            # Compute correction that brings colors back to palette
            correction = self._compute_correction(
                current_stats,
                self.global_palette,
                scene_anchors.local_color_shift,
            )

            # Apply smoothly (avoid sudden color jumps)
            smoothed_correction = self._smooth_correction(
                correction,
                self.correction_history,
                window=10,
            )

            frame = self._apply_color_correction(frame, smoothed_correction)

            self.correction_history.append(smoothed_correction)

        return frame

    def _smooth_correction(
        self,
        correction: ColorCorrection,
        history: List[ColorCorrection],
        window: int,
    ) -> ColorCorrection:
        """
        Smooth corrections over time to avoid jarring changes.
        """
        if len(history) < window:
            return correction

        # Weighted average with recent corrections
        recent = history[-window:]
        weights = [0.5 ** (window - i) for i in range(window)]
        weights.append(1.0)  # Current correction

        all_corrections = recent + [correction]

        return self._weighted_average_correction(all_corrections, weights)
```

### Memory-Efficient Long-Form Processing

```python
class StreamingConsistencyProcessor:
    """
    Process 7,000+ frames without loading all into memory.
    Maintains consistency state efficiently.
    """

    def __init__(self, config: Config):
        self.max_frames_in_memory = config.max_frames_in_memory  # e.g., 100
        self.consistency_state_size = 50  # Anchors + recent history

    def process_video_streaming(
        self,
        input_path: Path,
        output_path: Path,
        processor: BaseProcessor,
    ) -> None:
        """
        Stream process entire video with bounded memory.
        """
        # Phase 1: Pre-analyze (samples only, not full video)
        profile = self.consistency_manager.initialize_from_video(input_path)

        # Phase 2: Stream process
        with VideoReader(input_path) as reader:
            with VideoWriter(output_path, reader.metadata) as writer:

                chunk_buffer = []

                for frame_idx, frame in enumerate(reader):
                    chunk_buffer.append(frame)

                    if len(chunk_buffer) >= self.chunk_size:
                        # Process chunk with consistency
                        processed = self._process_chunk(
                            chunk_buffer, frame_idx, processor, profile
                        )

                        # Write processed frames (except overlap)
                        for processed_frame in processed[:-self.overlap]:
                            writer.write(processed_frame)

                        # Keep only overlap for next chunk
                        chunk_buffer = chunk_buffer[-self.overlap:]

                        # Clear GPU memory periodically
                        if frame_idx % 500 == 0:
                            torch.cuda.empty_cache()

                # Handle final frames
                # ...
```

### Metrics for Long-Form Consistency

```python
class LongFormConsistencyMetrics:
    """
    Measure consistency across entire video, not just local windows.
    """

    def evaluate(self, video_path: Path) -> ConsistencyReport:
        """
        Sample frames throughout video and measure drift.
        """
        # Sample 100 frames evenly distributed
        samples = self._sample_evenly(video_path, count=100)

        return ConsistencyReport(
            # Color consistency
            color_drift_over_time=self._measure_color_drift(samples),
            max_color_jump=self._find_max_color_jump(samples),

            # Brightness consistency
            brightness_variance=self._measure_brightness_variance(samples),
            flicker_score=self._measure_flicker(samples),

            # Face consistency
            face_identity_stability=self._measure_face_stability(samples),
            face_quality_variance=self._measure_face_quality_variance(samples),

            # Grain/noise consistency
            grain_consistency=self._measure_grain_consistency(samples),

            # Overall score
            long_form_wcs=self._compute_long_form_wcs(samples),
        )

    def _measure_color_drift(self, samples: List[Frame]) -> float:
        """
        Measure how much colors drift from start to end.
        Score of 0 = no drift, 1 = complete drift.
        """
        first_palette = self._extract_palette(samples[0])
        last_palette = self._extract_palette(samples[-1])

        return self._palette_distance(first_palette, last_palette)
```

### Configuration for Long-Form Processing

```yaml
# presets/presets.yaml - long-form settings
long_form:
  # Consistency enforcement level
  consistency_strength: 0.8  # 0=none, 1=strict

  # Chunk processing
  chunk_size: 50
  chunk_overlap: 4

  # Anchor extraction
  key_frame_count: 20
  scene_detection_threshold: 0.3

  # Face consistency
  face_identity_threshold: 0.85
  face_correction_strength: 0.3

  # Color consistency
  color_drift_threshold: 0.05
  color_correction_smoothing: 10

  # Memory management
  max_frames_in_memory: 100
  checkpoint_interval: 500

  # Quality targets
  target_long_form_wcs: 95
```

---

## Part 5: Quality Metrics & Testing

### Short-Form Temporal Consistency (Local Windows)

```python
class WorldConsistencyScore:
    """Unified video quality metric"""

    METRICS = {
        "object_permanence": 0.25,   # Objects don't disappear
        "relation_stability": 0.25,  # Spatial relationships stable
        "flicker_penalty": 0.30,     # Frame-to-frame consistency
        "causal_compliance": 0.20,   # Actions have consequences
    }

    def calculate(self, original: Video, restored: Video) -> float:
        """Returns 0-100 WCS score"""
```

### Before/After Metrics Dashboard

```python
class QualityDashboard:
    """Visual quality comparison"""

    def generate_report(self, original: Path, restored: Path) -> HTMLReport:
        return HTMLReport(
            side_by_side_slider=True,
            metrics={
                "sharpness_improvement": "+42%",
                "noise_reduction": "-65%",
                "temporal_consistency": "98.5 WCS",
                "face_quality": "+38%",
            },
            frame_samples=self._select_best_samples(),
        )
```

---

## Part 5: Implementation Timeline

### ✅ PHASE 1 COMPLETE: Foundation (Months 1-3)

#### Architecture Cleanup
- [x] Consolidate error handling → `core/errors.py` (31 exception classes)
- [x] Unify GPU management → `infrastructure/gpu/detector.py` (1,000 lines)
- [x] Create `infrastructure/` layer with GPU, memory, backends
- [x] Simplify configuration system
- [x] External preset YAML → `presets/presets.yaml` (507 lines)

#### Foundation Features
- [x] Temporal consistency metrics → `engine/temporal_consistency.py` (800 lines)
- [x] Smart preset auto-selection → `presets/registry.py`
- [x] Grain preservation mode → `processors/restoration/grain_manager.py` (1,485 lines)
- [x] TensorRT acceleration → `infrastructure/gpu/backends/tensorrt.py` (1,214 lines)

#### Unified Processors (All Backends Preserved)
- [x] Unified Denoiser → `processors/enhancement/denoising.py` (1,100 lines, 8 backends)
- [x] Unified Super Resolution → `processors/enhancement/super_resolution.py` (1,650 lines, 11 backends)
- [x] Unified Colorizer → `processors/restoration/colorization.py` (1,000 lines, 4 backends)
- [x] Unified Face Restorer → `processors/restoration/faces.py` (1,000 lines, 4 backends)
- [x] Unified Audio Enhancer → `processors/audio_unified/enhancer.py` (700 lines, 4 backends)
- [x] Content Analyzer → `processors/analysis/content_analyzer.py` (750 lines)

#### CLI & Documentation
- [x] Simplified CLI → `ui/cli/main.py` (938 lines)
- [x] Documentation → `docs/quickstart.md`, `hardware.md`, `presets.md`, `troubleshooting.md`

**Phase 1 Total: ~13,000 lines of new unified infrastructure**

---

### ✅ PHASE 2 COMPLETE: Core Differentiation (Months 4-6)

- [x] FlashSR-style diffusion upscaling → `processors/enhancement/diffusion_sr.py` (900 lines)
  - 4 backends: FlashVSR (16GB+), StableSR (12GB+), SwinIR (8GB+), Fallback (CPU)
  - Sparse attention, tiling, temporal consistency
- [x] DeepFilterNet audio integration → `processors/audio_deepfilter/deepfilter.py` (1,015 lines)
  - 4 backends: DeepFilterNet3, Lite, SpeechBrain, Traditional
  - 10ms latency, streaming support, audio analysis
- [x] Real-time preview server → `ui/preview/server.py` (950 lines)
  - Web UI at localhost:8080, before/after slider
  - Segment caching, keyboard shortcuts
- [x] One-click wizard → `ui/cli/wizard.py` (620 lines)
  - 5-step guided flow, color terminal support
  - `framewright video.mp4 --wizard`

**Phase 2 Total: ~3,500 lines**

---

### ✅ PHASE 3 COMPLETE: Advanced Features (Months 7-9)

- [x] Cross-attention temporal processor → `processors/enhancement/temporal_vae.py` (1,632 lines)
  - TemporalEncoder3D, CrossFrameAttention, TemporalDecoder3D
  - ConsistencyEnforcer for lightweight mode
  - Chunked processing for 7k+ frames
- [x] HDR Dolby Vision export → `processors/enhancement/hdr_export.py` (1,130 lines)
  - HDR10, HDR10+, Dolby Vision, HLG support
  - ToneMapper (Reinhard, ACES, Hable, BT2390)
  - ColorSpaceConverter (BT.709 ↔ BT.2020, PQ, HLG)
  - SDR to HDR expansion
- [x] Cloud rendering integration → `infrastructure/cloud/coordinator.py` (2,498 lines)
  - RunPod, Vast.ai, Lambda Labs providers
  - Cost estimation, data transfer, job monitoring
  - Auto provider selection

**Phase 3 Total: ~5,260 lines**

---

### ✅ PHASE 4 COMPLETE: Innovation (Months 10-12)

- [x] Text-guided upscaling → `processors/enhancement/guided_sr.py` (1,591 lines)
  - CLIP text encoder, style presets (cinematic, anime, vintage, etc.)
  - Reference image style transfer
  - Classifier-free guidance
- [x] Generative frame extension → `processors/restoration/frame_generator.py` (1,200 lines)
  - MotionEstimator (RAFT optical flow)
  - FrameInterpolator (RIFE, FILM, diffusion)
  - FrameExtender (SVD-based forward/backward)
  - GapFiller, DamagedFrameRestorer
- [x] NPU/Neural Engine support → `infrastructure/gpu/backends/apple_silicon.py` (685 lines)
  - CoreML conversion (PyTorch → CoreML)
  - ANE optimization, M1/M2/M3/M4 detection
  - Performance profiling per compute unit

**Phase 4 Total: ~3,476 lines**

---

### ✅ REMAINING INFRASTRUCTURE COMPLETE

#### Core Layer
- [x] `core/config.py` - Unified configuration system (430 lines)
- [x] `core/events.py` - Event bus for pub/sub (300 lines)
- [x] `core/types.py` - Shared type definitions (200 lines)

#### Infrastructure Cache
- [x] `infrastructure/cache/frame_cache.py` - LRU frame caching (500 lines)
- [x] `infrastructure/cache/model_cache.py` - Model VRAM management (450 lines)
- [x] `infrastructure/cache/eviction.py` - Eviction policies (560 lines)

#### Infrastructure Models
- [x] `infrastructure/models/registry.py` - Model registry (852 lines)
- [x] `infrastructure/models/downloader.py` - Resume-capable downloads (664 lines)
- [x] `infrastructure/models/manager.py` - Model lifecycle management (680 lines)

#### Engine
- [x] `engine/pipeline.py` - Pipeline orchestrator with fluent API (1,686 lines)
- [x] `engine/scheduler.py` - Job queue with priorities (1,156 lines)
- [x] `engine/checkpoint.py` - Crash recovery & resume (932 lines)

#### GPU Backends
- [x] `infrastructure/gpu/backends/amd.py` - ROCm/HIP backend (528 lines)
- [x] `infrastructure/gpu/backends/intel.py` - oneAPI/OpenVINO backend (549 lines)
- [x] `infrastructure/gpu/distributor.py` - Multi-GPU distribution (419 lines)

#### Analysis Processors
- [x] `processors/analysis/quality_scorer.py` - Quality metrics & reports (1,488 lines)
- [x] `processors/analysis/degradation_detector.py` - Degradation detection (1,637 lines)

#### Restoration Processors
- [x] `processors/restoration/defects.py` - Scratch/dust/damage repair (1,540 lines)
- [x] `processors/restoration/stabilization.py` - Video stabilization (1,203 lines)

#### Format Processors
- [x] `processors/format/film.py` - Film-specific processing (850 lines)
- [x] `processors/format/vhs.py` - VHS artifact removal (1,010 lines)
- [x] `processors/format/interlace.py` - Deinterlacing (850 lines)
- [x] `processors/format/aspect.py` - Aspect ratio handling (965 lines)

#### UI
- [x] `ui/dashboard/server.py` - Web dashboard (600 lines)
- [x] `ui/dashboard/templates.py` - Dashboard UI templates (400 lines)
- [x] `ui/api/server.py` - REST API with OpenAPI spec (500 lines)
- [x] `ui/api/client.py` - Python API client (250 lines)

**Remaining Infrastructure Total: ~18,699 lines**

---

### 🎉 FULL IMPLEMENTATION COMPLETE

| Phase | Status | New Code | Key Components |
|-------|--------|----------|----------------|
| Phase 1 | ✅ Complete | ~13,000 lines | Infrastructure, unified processors, presets |
| Phase 2 | ✅ Complete | ~3,500 lines | Diffusion SR, DeepFilter, preview, wizard |
| Phase 3 | ✅ Complete | ~5,260 lines | Temporal VAE, HDR export, cloud rendering |
| Phase 4 | ✅ Complete | ~3,476 lines | Text-guided SR, frame generation, Apple Silicon |
| Remaining | ✅ Complete | ~18,699 lines | Core, cache, models, engine, GPU backends, format handlers, dashboard |
| **Total** | **✅ Complete** | **~43,935 lines** | **Full implementation** |

---

## Part 6: Success Metrics

### User Experience
| Metric | Current | Target |
|--------|---------|--------|
| Commands to restore a video | 5-10 flags | 1 command |
| Time to first successful restore | 30+ minutes | 5 minutes |
| Error resolution rate | Unknown | 80%+ self-service |
| User satisfaction | Unknown | 4.5+ stars |

### Technical Performance
| Metric | Current | Target |
|--------|---------|--------|
| 4K upscaling speed | ~2 FPS | ~15 FPS (TensorRT + Flash) |
| Temporal consistency | Manual | WCS > 95 |
| Memory efficiency | Varies | <8GB for 1080p |
| Multi-GPU scaling | Linear | Near-linear |

### Code Quality
| Metric | Current | Target |
|--------|---------|--------|
| Total lines of code | ~47,000 | ~40,000 |
| Code duplication | ~3,300 lines | <500 lines |
| Test coverage | ~70% | >85% |
| Documentation | Partial | Complete |

---

## Part 7: Design Philosophy Summary

### Apple Principles Applied

1. **Simplicity**: One command does the right thing
2. **Progressive Disclosure**: Advanced options exist but don't intrude
3. **Consistency**: Same patterns everywhere
4. **Feedback**: Always know what's happening
5. **Aesthetic Integrity**: Beautiful even in CLI

### The FrameWright Promise

> "Drop in any video. Get extraordinary results. That's it."

For advanced users:
> "Every parameter is there when you need it. None when you don't."

---

## Appendix A: Technical Specifications

### Hardware Requirements - ALL TIERS SUPPORTED

FrameWright works on ANY hardware - from Raspberry Pi to data center GPUs.
Quality and speed scale with available resources.

#### Tier 1: CPU Only (No GPU / Integrated Graphics)
- **Hardware:** Any x64 CPU, Intel HD Graphics, etc.
- **RAM:** 8GB minimum
- **Disk:** 10GB free space
- **What Works:**
  - NCNN CPU inference (slow but functional)
  - Traditional denoising (FFmpeg-based)
  - Basic colorization
  - Audio enhancement (full quality)
  - 2x upscaling recommended
- **Speed:** ~0.5-2 FPS for 1080p
- **Use Case:** Laptops, older PCs, servers without GPU

#### Tier 2: Entry GPU (4GB VRAM)
- **Hardware:** GTX 1050 Ti, GTX 1650, RX 570, RX 580
- **RAM:** 8GB minimum, 16GB recommended
- **What Works:**
  - Real-ESRGAN (all models)
  - GFPGAN v1.3/v1.4
  - RIFE frame interpolation
  - Basic temporal denoising
  - 4x upscaling
- **Speed:** ~5-15 FPS for 1080p
- **Use Case:** Budget gaming PCs, older workstations

#### Tier 3: Mid GPU (8GB VRAM)
- **Hardware:** RTX 3060, RTX 3070, RX 6700 XT, RX 6800
- **RAM:** 16GB minimum, 32GB recommended
- **What Works:**
  - HAT (Small/Base models)
  - TAP denoising
  - CodeFormer
  - All colorization methods
  - Scene intelligence
- **Speed:** ~10-25 FPS for 1080p
- **Use Case:** Most home users, content creators

#### Tier 4: High GPU (12-16GB VRAM)
- **Hardware:** RTX 3080, RTX 4070 Ti, RX 6900 XT, RX 7800 XT
- **RAM:** 32GB minimum
- **What Works:**
  - HAT-Large
  - Extended temporal windows (16 frames)
  - Basic diffusion SR
  - Ensemble face restoration
  - Full VHS restoration pipeline
- **Speed:** ~15-35 FPS for 1080p
- **Use Case:** Enthusiasts, semi-professional

#### Tier 5: Extreme GPU (24GB+ VRAM)
- **Hardware:** RTX 4090, RTX 5090, A6000, multiple GPUs
- **RAM:** 64GB recommended
- **What Works:**
  - EVERYTHING at maximum quality
  - Full diffusion SR (30 steps)
  - Ensemble SR (HAT + VRT + Diffusion voting)
  - Extended temporal windows (32+ frames)
  - 8K processing
- **Speed:** ~30-60 FPS for 1080p, ~10-20 FPS for 4K
- **Use Case:** Professionals, archives, studios

#### Tier 6: Apple Silicon
- **Hardware:** M1/M2/M3 (any variant), M1/M2/M3 Pro/Max/Ultra
- **Unified Memory:** 8GB-192GB
- **What Works:**
  - CoreML optimized models (78x faster than CPU)
  - Metal Performance Shaders
  - All features scale with unified memory
- **Speed:** M3 Max ~20-40 FPS for 1080p
- **Use Case:** Mac users, mobile workflows

#### Tier 7: Cloud GPU
- **Hardware:** RunPod A100 (40GB), Vast.ai A100 (80GB), Lambda H100
- **What Works:** Everything, no local limits
- **Speed:** Limited by upload/download
- **Use Case:** Anyone who needs maximum quality without local hardware

### Model Registry (By Hardware Tier)

| Model | Purpose | Min VRAM | Tier | Speed |
|-------|---------|----------|------|-------|
| **Super Resolution** |||||
| realesrgan-x4plus | General upscaling | 2GB | 2+ | Fast |
| realesrgan-ncnn | General (no CUDA) | CPU | 1+ | Slow |
| realesr-animevideov3 | Animation | 2GB | 2+ | Fast |
| hat-s | High quality SR | 6GB | 3+ | Medium |
| hat-l | Maximum quality SR | 14GB | 5+ | Slow |
| flashvsr | Diffusion SR | 8GB | 3+ | Medium |
| diffusion-sr-full | Best diffusion | 16GB | 5+ | Slow |
| **Face Restoration** |||||
| gfpgan-v1.3 | Lightweight face | 2GB | 2+ | Fast |
| gfpgan-v1.4 | Better face | 2GB | 2+ | Fast |
| codeformer | High quality face | 3GB | 3+ | Medium |
| restoreformer | Alternative face | 4GB | 3+ | Medium |
| aesrgan-face | Attention face | 6GB | 3+ | Medium |
| **Denoising** |||||
| traditional | FFmpeg-based | CPU | 1+ | Fast |
| temporal-basic | Optical flow | 4GB | 2+ | Medium |
| tap-denoise | Neural temporal | 8GB | 3+ | Slow |
| tap-extended | Extended window | 12GB | 4+ | Slow |
| **Audio** |||||
| ffmpeg-audio | Basic processing | CPU | 1+ | Fast |
| deepfilter-v3 | AI restoration | CPU | 1+ | Fast |
| resemble-enhance | Full restoration | CPU | 1+ | Medium |
| **Frame Interpolation** |||||
| rife-v4.6 | Standard | 2GB | 2+ | Fast |
| rife-v4.6-anime | Animation | 2GB | 2+ | Fast |
| **Colorization** |||||
| deoldify | Lightweight | 2GB | 2+ | Fast |
| ddcolor | Better quality | 4GB | 2+ | Medium |
| swintexco | Reference-based | 8GB | 3+ | Slow |

---

## Appendix B: CLI Command Reference

### Primary Commands

```bash
# Basic restoration (auto-detect everything)
framewright video.mp4

# Quality preference
framewright video.mp4 --quality fast|balanced|best

# Time preference
framewright video.mp4 --speed 30m|1h|unlimited

# Content style
framewright video.mp4 --style film|animation|home-video|archive

# Output location
framewright video.mp4 --output restored/
```

### Discovery Commands

```bash
# Interactive wizard
framewright wizard

# Analyze without processing
framewright analyze video.mp4

# Preview before processing
framewright preview video.mp4 --from 0:30 --duration 15

# Compare settings
framewright compare video.mp4 --presets fast,balanced,best
```

### Advanced Commands

```bash
# Custom configuration
framewright video.mp4 --config myconfig.yaml

# Batch processing
framewright batch input_folder/ --output output_folder/

# Cloud processing
framewright cloud video.mp4 --provider runpod

# Resume interrupted job
framewright resume job_id
```

### Help System

```bash
# Topic help
framewright help getting-started
framewright help presets
framewright help gpu-memory
framewright help troubleshooting

# Search help
framewright help --search "VRAM"
```

---

## Appendix C: Key Principles Summary

### What This Plan DOES

1. **Preserves ALL hardware tiers** - From CPU-only to RTX 5090
2. **Keeps ALL processing backends** - PyTorch, NCNN, TensorRT, CoreML
3. **Maintains ALL model options** - Light models for 4GB, heavy for 24GB+
4. **Unifies interfaces** - One API, automatic backend selection
5. **Adds long-form consistency** - Works correctly over 7,000+ frames
6. **Improves UX** - Apple-like simplicity with power underneath

### What This Plan Does NOT Do

1. **Does NOT remove features** - All current capabilities preserved
2. **Does NOT require high-end hardware** - CPU-only still works
3. **Does NOT force specific backends** - Auto-selection with manual override
4. **Does NOT sacrifice flexibility** - Advanced users keep full control
5. **Does NOT break existing workflows** - Backward compatible

### The Core Philosophy

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  "Every user gets the best possible result for THEIR hardware" │
│                                                                 │
│  - GTX 1050 Ti user: Great results, slower processing          │
│  - RTX 4090 user: Maximum quality, fast processing             │
│  - M2 MacBook user: Optimized for Apple Silicon                │
│  - Cloud user: No limits                                       │
│                                                                 │
│  Same simple command: framewright video.mp4                    │
│  Automatic optimization for each system                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Long-Form Temporal Consistency Summary

For videos with thousands of frames (4+ minutes):

1. **Pre-analyze entire video** before processing
2. **Extract global anchors** (color palette, grain profile, face identities)
3. **Process in overlapping chunks** (50 frames, 4-frame overlap)
4. **Enforce consistency corrections** at each chunk
5. **Blend chunk boundaries** for seamless transitions
6. **Track and correct drift** over entire video length

Result: Frame 1 and Frame 7000 have consistent:
- Color grading
- Face appearances
- Grain characteristics
- Enhancement "style"

---

## Appendix D: New Files Created

### Phase 1 - Foundation
| File | Lines | Purpose |
|------|-------|---------|
| `core/errors.py` | 300 | Unified 31 exception classes |
| `infrastructure/gpu/detector.py` | 1,000 | Hardware detection (all tiers) |
| `infrastructure/gpu/memory.py` | 200 | Memory management |
| `infrastructure/gpu/backends/base.py` | 150 | Backend base class |
| `infrastructure/gpu/backends/tensorrt.py` | 1,214 | TensorRT acceleration |
| `engine/temporal_consistency.py` | 800 | Long-form consistency (7k+ frames) |
| `processors/enhancement/denoising.py` | 1,100 | Unified denoiser (8 backends) |
| `processors/enhancement/super_resolution.py` | 1,650 | Unified SR (11 backends) |
| `processors/restoration/colorization.py` | 1,000 | Unified colorizer (4 backends) |
| `processors/restoration/faces.py` | 1,000 | Unified face restorer (4 backends) |
| `processors/restoration/grain_manager.py` | 1,485 | Film grain extraction/restoration |
| `processors/audio_unified/enhancer.py` | 700 | Unified audio (4 backends) |
| `processors/analysis/content_analyzer.py` | 750 | Content analysis |
| `presets/presets.yaml` | 507 | Hardware-aware presets |
| `presets/registry.py` | 300 | Preset management |
| `ui/cli/main.py` | 938 | Simplified CLI |
| `docs/quickstart.md` | 100 | Quick start guide |
| `docs/hardware.md` | 100 | Hardware guide |
| `docs/presets.md` | 100 | Preset guide |
| `docs/troubleshooting.md` | 100 | Troubleshooting guide |

### Phase 2 - Core Differentiation
| File | Lines | Purpose |
|------|-------|---------|
| `processors/enhancement/diffusion_sr.py` | 900 | FlashVSR one-step diffusion |
| `processors/audio_deepfilter/deepfilter.py` | 1,015 | DeepFilterNet audio |
| `ui/preview/server.py` | 950 | Live preview server |
| `ui/cli/wizard.py` | 620 | Interactive wizard |

### Phase 3 - Advanced Features
| File | Lines | Purpose |
|------|-------|---------|
| `processors/enhancement/temporal_vae.py` | 1,632 | Cross-attention temporal VAE |
| `processors/enhancement/hdr_export.py` | 1,130 | HDR/Dolby Vision export |
| `infrastructure/cloud/coordinator.py` | 2,498 | Cloud rendering (RunPod, Vast.ai, Lambda) |

### Phase 4 - Innovation
| File | Lines | Purpose |
|------|-------|---------|
| `processors/enhancement/guided_sr.py` | 1,591 | Text-guided upscaling |
| `processors/restoration/frame_generator.py` | 1,200 | Generative frame extension |
| `infrastructure/gpu/backends/apple_silicon.py` | 685 | Apple Neural Engine backend |

### Remaining Infrastructure
| File | Lines | Purpose |
|------|-------|---------|
| `core/config.py` | 430 | Unified configuration system |
| `core/events.py` | 300 | Event bus pub/sub |
| `core/types.py` | 200 | Shared type definitions |
| `infrastructure/cache/frame_cache.py` | 500 | LRU frame caching |
| `infrastructure/cache/model_cache.py` | 450 | Model VRAM management |
| `infrastructure/cache/eviction.py` | 560 | Eviction policies (LRU, LFU, FIFO, etc.) |
| `infrastructure/models/registry.py` | 852 | Model registry with built-in models |
| `infrastructure/models/downloader.py` | 664 | Resume-capable downloads |
| `infrastructure/models/manager.py` | 680 | Model lifecycle management |
| `engine/pipeline.py` | 1,686 | Pipeline orchestrator with fluent API |
| `engine/scheduler.py` | 1,156 | Job queue with priorities |
| `engine/checkpoint.py` | 932 | Crash recovery & resume |
| `infrastructure/gpu/backends/amd.py` | 528 | ROCm/HIP backend |
| `infrastructure/gpu/backends/intel.py` | 549 | oneAPI/OpenVINO backend |
| `infrastructure/gpu/distributor.py` | 419 | Multi-GPU distribution |
| `processors/analysis/quality_scorer.py` | 1,488 | Quality metrics & HTML reports |
| `processors/analysis/degradation_detector.py` | 1,637 | 25+ degradation types detection |
| `processors/restoration/defects.py` | 1,540 | Scratch/dust/damage repair |
| `processors/restoration/stabilization.py` | 1,203 | Video stabilization |
| `processors/format/film.py` | 850 | Film-specific (gate weave, flicker) |
| `processors/format/vhs.py` | 1,010 | VHS artifacts (tracking, dropout) |
| `processors/format/interlace.py` | 850 | Deinterlacing (YADIF, BWDIF, etc.) |
| `processors/format/aspect.py` | 965 | Aspect ratio handling |
| `ui/dashboard/server.py` | 600 | Web dashboard server |
| `ui/dashboard/templates.py` | 400 | Dashboard HTML/CSS/JS templates |
| `ui/api/server.py` | 500 | REST API with OpenAPI spec |
| `ui/api/client.py` | 250 | Python API client |

---

*Document Version: 4.0*
*Updated: 2026-02-02*
*Status: ✅ FULLY IMPLEMENTED - ALL FEATURES COMPLETE*
*Total New Code: ~43,935 lines*
*All hardware tiers preserved (CPU-only to RTX 5090 + Apple Silicon + Cloud)*
*Long-form temporal consistency for 7,000+ frames*
*Complete infrastructure: caching, model management, job scheduling, checkpoints*
*Full format support: Film, VHS, interlacing, aspect ratio*
*Web dashboard and REST API included*
*Based on deep research and implementation by parallel agent swarms*
