# FrameWright Video Restoration Pipeline - Improvements & Roadmap

**Generated:** 2025-12-29
**Version:** 1.0.0
**Analysis Scope:** Complete codebase review
**Last Updated:** 2025-12-30

---

## Executive Summary

The FrameWright video restoration pipeline is a production-quality system with comprehensive error handling, checkpointing, and multiple interfaces. This document tracks completed features and outlines remaining potential improvements organized by priority and category.

---

## Completed Features (v1.0.0)

### Core Pipeline
- **CLI Implementation** - Full VideoRestorer integration with all commands
- **Web UI (Gradio)** - Browser-based interface with real-time progress
- **Python API** - Programmatic access via `VideoRestorer` class
- **Parallel Frame Processing** - ThreadPoolExecutor-based concurrent processing
- **Checkpointing System** - Resume interrupted processing
- **Configuration Presets** - fast, quality, archive, anime, film_restoration

### AI Enhancement Models
- **AI Upscaling** - Real-ESRGAN 2x/4x enhancement
- **Face Restoration** - GFPGAN/CodeFormer integration
- **Frame Interpolation** - RIFE smooth motion interpolation
- **Colorization Support** - DeOldify and DDColor integration
- **Watermark Removal** - LaMA inpainting with auto-detect and mask support
- **Burnt-in Subtitle Removal** - OCR-based detection with LaMA inpainting

### Audio & Video Processing
- **Audio Enhancement** - AudioProcessor with FFmpeg fallback
- **Audio Sync** - AI-powered audio-video synchronization
- **Video Stabilization** - FFmpeg vidstab and OpenCV integration
- **HDR Conversion** - SDR/HDR format conversion support
- **Scene Detection** - Automatic scene boundary detection
- **Defect Repair** - Scratch, dust, grain removal
- **Advanced Temporal Denoising** - Multi-frame context with optical flow and flicker reduction

### Hardware & Performance
- **GPU Memory Pre-check** - VRAM validation and tile size optimization
- **Multi-GPU Support** - Distribute processing across multiple GPUs
- **Benchmark Suite** - Performance profiling and metrics collection
- **Caching System** - Intelligent result caching for repeated operations

### Model Management
- **Model Download Manager** - `~/.framewright/models/` with progress bars
- **RIFE Model Download Automation** - Auto-download on first use
- **Advanced Model Support** - Extended model selection for specialized use cases

### Output & Configuration
- **Configurable Output Filename** - `--output-dir`, `--format` support
- **Output Directory Configuration** - Configurable frames/enhanced/output paths
- **Batch Processing Mode** - `framewright batch` with `--continue-on-error`
- **Web UI Output Format Selection** - Full dropdown and folder support

### Cloud & Streaming
- **Cloud Provider Integration** - RunPod and Vast.ai GPU cloud support
- **Cloud Storage** - Upload/download to cloud storage providers
- **Streaming Mode** - Real-time progressive processing
- **Preview Generation** - Quick preview before full processing

### Developer Features
- **Dry Run Mode** - Test configuration without processing
- **Watch Mode** - Monitor folders for automatic processing
- **Structured Logging** - Configurable logging with multiple backends
- **Security Utilities** - Input validation and sanitization
- **Progress Tracking** - Detailed progress with ETA estimation

### Integration
- **YouTube Download** - yt-dlp integration for source acquisition
- **GitHub Actions CI/CD** - Automated testing and deployment
- **Pre-commit Hooks** - Code quality enforcement
- **Docker Support** - Containerized deployment

---

## High Priority Improvements (v2.0 Roadmap)

### TAP Denoising Framework

**Current:** Temporal denoising with optical flow-guided filtering.

**Enhancement:** Integrate the **TAP framework** (ECCV 2024) which adds tunable temporal modules to pre-trained image denoisers.

**Benefits:**
- Uses existing high-quality image denoisers as spatial priors
- Progressive fine-tuning with pseudo-clean frames
- Unsupervised learning - no paired video data needed
- Superior performance on both sRGB and raw video

**Reference:** [TAP GitHub](https://github.com/zfu006/TAP) | [ECCV 2024 Paper](https://link.springer.com/chapter/10.1007/978-3-031-72992-8_20)

---

### Exemplar-Based Colorization (SwinTExCo/BiSTNet)

**Current:** DeOldify (archived 2024) and DDColor for automatic colorization.

**Enhancement:** Add **SwinTExCo** or **BiSTNet** (TPAMI 2024) for user-guided reference-based colorization.

**Benefits:**
- User provides reference color image for guided colorization
- Swin Transformer backbone for better feature extraction
- Bidirectional temporal fusion eliminates flickering
- Superior quality for archival restoration work

**Reference:** [SwinTExCo Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417424023042)

---

### AESRGAN Face Enhancement

**Current:** GFPGAN v1.3/v1.4 and CodeFormer for face restoration.

**Enhancement:** Add **AESRGAN** (Attention-Enhanced ESRGAN) for better facial detail preservation.

**Benefits:**
- Explicit attention modulation preserves subtle facial details
- Reduces artifacts in high-frequency regions
- More computationally efficient than transformer approaches
- Drop-in enhancement to existing face pipeline

**Reference:** [AESRGAN Analysis](https://nhsjs.com/2025/enhancing-super-resolution-models-a-comparative-analysis-of-real-esrgan-aesrgan-and-esrgan/)

---

## Medium Priority Improvements

### Diffusion-Based Video Super-Resolution

**Current:** Real-ESRGAN, BasicVSR++, and VRT for upscaling.

**Enhancement:** Integrate diffusion-based VSR models like **Upscale-A-Video** (CVPR 2024) or **SeedVR** (CVPR 2025).

**Benefits:**
- Superior texture generation vs GAN-based methods
- Better handling of diverse real-world degradations
- State-of-the-art perceptual quality metrics

**Reference:** [DiffVSR](https://arxiv.org/html/2501.10110v1) | [RealisVSR](https://arxiv.org/html/2507.19138v1)

---

### QP-Aware Codec Artifact Removal

**Current:** No specific handling for codec compression artifacts.

**Enhancement:** Implement **QP-aware restoration** (WACV 2025) for reversing codec compression damage.

**Benefits:**
- Specifically addresses 8K video codec compression artifacts
- QP (Quantization Parameter) aware - adapts to compression level
- Combines transformer and diffusion approaches
- Critical for streaming/web video restoration

---

### Missing Frame Generation

**Current:** RIFE interpolation between existing frames.

**Enhancement:** Integrate **generative AI frame reconstruction** for damaged/missing frames.

**Benefits:**
- Reconstruct completely missing frames in damaged film reels
- Essential for archival restoration with physical damage
- Goes beyond interpolation to actual content generation

**Reference:** [AI in Film Archives - Pulitzer Center](https://pulitzercenter.org/stories/saving-cinema-ais-starring-role-preserving-film-archives)

---

### Unified Multi-Task Model (BCell RNN)

**Current:** Separate models for denoising, deblurring, and super-resolution.

**Enhancement:** Implement a **unified BCell-based RNN** handling multiple restoration tasks.

**Benefits:**
- Single model for denoising, deblurring, AND super-resolution
- Bi-directional hidden states leverage past AND future frames
- 1-4 dB PSNR improvement OR several times less computation
- Reduces model management overhead

**Reference:** [Versatile RNN for Video Restoration](https://www.sciencedirect.com/science/article/abs/pii/S0031320323000614)

---

## Why These Improvements Matter

This section explains the gap between current implementations and proposed enhancements, including implementation barriers and expected quality improvements.

### High Priority Analysis

#### TAP Denoising Framework

**Current Implementation:** Our `temporal_denoise.py` uses classical algorithms (NLMeans, bilateral filtering) with optical flow alignment.

**Why Not Implemented Yet:**
- TAP requires **pre-trained image denoiser weights** (e.g., Restormer, NAFNet) as the backbone
- Needs PyTorch model fine-tuning infrastructure
- Requires training on video sequences (even if unsupervised)

**Quality Comparison:**

| Aspect | Current | TAP |
|--------|---------|-----|
| Spatial denoising | OpenCV filters | State-of-art neural denoisers |
| Temporal learning | Hand-crafted flow warping | Learned temporal attention |
| Adaptability | Fixed parameters | Self-tuning from data |
| Quality (PSNR) | ~30-32 dB | ~34-38 dB |

---

#### Exemplar-Based Colorization (SwinTExCo/BiSTNet)

**Current Implementation:** DeOldify/DDColor auto-colorize based on learned priors - they "guess" colors.

**Why Not Implemented Yet:**
- Requires Swin Transformer architecture (heavy model)
- Needs reference image matching pipeline
- BiSTNet uses bidirectional propagation requiring full video in memory

**Quality Comparison:**

| Aspect | Current (DDColor) | SwinTExCo |
|--------|-------------------|-----------|
| Color accuracy | AI's best guess | User-guided with reference |
| Temporal consistency | Per-frame (flickers) | Bidirectional fusion |
| Historical accuracy | Generic colors | Match actual source photos |
| Use case | Quick auto-color | Archival restoration |

*Example: Colorizing 1920s footage - DDColor guesses skin tones. SwinTExCo lets you provide a color photo from that era as reference.*

---

#### AESRGAN Face Enhancement

**Current Implementation:** GFPGAN/CodeFormer restore faces using GAN-based priors.

**Why Not Implemented Yet:**
- AESRGAN model weights aren't as widely distributed
- Requires attention mechanism integration
- Would need comparative testing against CodeFormer

**Quality Comparison:**

| Aspect | Current (GFPGAN) | AESRGAN |
|--------|------------------|---------|
| Detail preservation | Sometimes over-smooths | Attention preserves subtle features |
| Artifacts | Can hallucinate features | Better high-frequency control |
| Speed | ~50ms/face | Similar or faster |
| Eye/mouth detail | Good | Better fine detail |

---

### Medium Priority Analysis

#### Diffusion-Based Video Super-Resolution

**Current Implementation:** Real-ESRGAN uses GAN-based upscaling.

**Why Not Implemented Yet:**
- Diffusion models are **10-100x slower** than GANs
- Require significant VRAM (12GB+ for video)
- Complex inference pipeline with iterative denoising

**Quality Comparison:**

| Aspect | Current (Real-ESRGAN) | Diffusion VSR |
|--------|----------------------|---------------|
| Texture generation | Can look "plastic" | Realistic fine detail |
| Handling unknowns | Struggles with severe damage | Generates plausible content |
| Perceptual quality | Good | State-of-art |
| Speed | ~0.1s/frame | ~2-10s/frame |

*Trade-off: Quality vs speed. Diffusion wins on quality but loses badly on processing time.*

---

#### QP-Aware Codec Artifact Removal

**Current Implementation:** No specific codec artifact handling - we treat all degradation generically.

**Why Not Implemented Yet:**
- Requires QP extraction from video bitstream (complex)
- Model must be trained on compression-specific artifacts
- WACV 2025 paper is very recent (no public weights yet)

**Quality Comparison:**

| Aspect | Current | QP-Aware |
|--------|---------|----------|
| Blocking artifacts | Generic denoising | Targeted removal |
| Banding | Not addressed | Specifically handled |
| Mosquito noise | Partially removed | Precisely targeted |
| Adaptation | One-size-fits-all | Adapts to compression level |

*Critical for YouTube sources which are heavily compressed.*

---

#### Missing Frame Generation

**Current Implementation:** RIFE interpolates **between existing frames** - requires 2 valid frames.

**Why Not Implemented Yet:**
- Requires generative model (diffusion/transformer)
- Training data for damaged film is scarce
- High computational cost
- Risk of hallucinating incorrect content

**Quality Comparison:**

| Aspect | Current (RIFE) | Generative |
|--------|----------------|------------|
| Missing single frame | Interpolate from neighbors | Generate from context |
| Missing sequence | Cannot handle | Can reconstruct |
| Damaged frames | Must use as-is | Can inpaint damage |
| Use case | Smooth motion | Archival recovery |

*Example: Film reel with 10 consecutive damaged frames - RIFE can't help, generative AI could reconstruct the scene.*

---

#### Unified Multi-Task Model (BCell RNN)

**Current Implementation:** Separate models for each task:
- Real-ESRGAN for upscaling
- Temporal denoiser for noise
- Separate deblurring (if implemented)

**Why Not Implemented Yet:**
- Requires custom RNN architecture
- Training from scratch needed
- Model complexity high

**Quality Comparison:**

| Aspect | Current (Separate) | BCell Unified |
|--------|-------------------|---------------|
| Model count | 3-5 models | 1 model |
| Memory usage | High (load all) | Lower |
| Processing | Sequential passes | Single pass |
| Cross-task optimization | None | Joint optimization |
| PSNR improvement | Baseline | +1-4 dB |

---

### Implementation Effort Summary

| Enhancement | Main Barrier | Effort | Quick Win? |
|-------------|--------------|--------|------------|
| TAP Denoising | Need pretrained backbone + training | High | No |
| SwinTExCo | Heavy transformer model | Medium | Yes |
| AESRGAN | Model availability | Low | **Yes** |
| Diffusion VSR | 10-100x slower processing | Medium | No |
| QP-Aware | No public weights yet | High | No |
| Missing Frame Gen | Generative model training | Very High | No |
| BCell RNN | Custom architecture + training | High | No |

**Recommended Implementation Order:**
1. **AESRGAN** - Drop-in replacement, lowest effort
2. **SwinTExCo** - Clear user value for archival work
3. **TAP** - Significant quality jump for denoising

---

## Low Priority Improvements

### Plugin Architecture

**Issue:** Adding new enhancement processors requires code changes.

**Recommendation:** Implement plugin architecture:
```python
# Plugin interface
class ProcessorPlugin(Protocol):
    name: str
    version: str

    def process(self, frame: np.ndarray, config: Dict) -> np.ndarray: ...
    def get_requirements(self) -> List[str]: ...
    def is_available(self) -> bool: ...

# Plugin discovery
# ~/.framewright/plugins/my_processor/
```

**Impact:** Low - Future extensibility for third-party processors

---

### REST API Mode

**Issue:** No programmatic access for integration with other systems.

**Recommendation:** Add REST API server mode:
```bash
framewright serve --port 8080
```

**API Endpoints:**
```
POST /api/v1/jobs          - Submit restoration job
GET  /api/v1/jobs/{id}     - Get job status
GET  /api/v1/jobs/{id}/logs - Stream job logs
DELETE /api/v1/jobs/{id}   - Cancel job
GET  /api/v1/health        - Health check
GET  /api/v1/gpus          - List available GPUs
```

**Features:**
- Job queue with priority support
- Webhook callbacks on completion
- Multiple concurrent job support
- OpenAPI/Swagger documentation
- Authentication and rate limiting

**Impact:** Medium - Important for integration workflows

---

### Mobile Companion App

**Issue:** No way to monitor/control jobs from mobile devices.

**Recommendation:** Progressive Web App (PWA) for:
- Job monitoring and notifications
- Remote start/stop control
- Preview viewing
- Push notifications on completion

**Impact:** Low - Nice-to-have feature

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | TBD | Initial release with all features |

### Development Changelog
- 2025-12-30: Added advanced temporal denoising with optical flow, flicker reduction
- 2025-12-29: Added cloud support, streaming, multi-GPU, enhanced audio
- 2025-12-28: Added subtitle removal, scene detection, stabilization
- 2025-12-27: Added colorization, watermark removal, batch processing
- 2025-12-26: Initial development with core features

---

*This document should be reviewed and updated as improvements are implemented.*

*Last updated: 2025-12-30*
