# FrameWright Roadmap

**Version:** 1.4.0
**Last Updated:** 2026-01-07

---

## Completed Features (v1.4.0)

### Core Pipeline
- CLI with full VideoRestorer integration
- Web UI (Gradio) with real-time progress
- Python API via `VideoRestorer` class
- Parallel frame processing (ThreadPoolExecutor)
- Checkpointing system (resume interrupted jobs)
- Configuration presets: fast, quality, archive, anime, film_restoration

### AI Enhancement Models
- Real-ESRGAN 2x/4x upscaling
- GFPGAN/CodeFormer face restoration
- RIFE frame interpolation (v2.3, v4.0, v4.6)
- DeOldify and DDColor colorization
- LaMA watermark removal (auto-detect + mask)
- OCR-based burnt-in subtitle removal

### Audio & Video
- Audio enhancement with FFmpeg fallback
- AI-powered audio-video sync
- Video stabilization (vidstab + OpenCV)
- HDR/SDR conversion
- Scene detection
- Defect repair (scratch, dust, grain)
- Temporal denoising with optical flow

### Hardware & Performance
- GPU memory pre-check with VRAM validation
- Multi-GPU distribution
- Performance profiling
- Intelligent result caching

### Cloud & Storage
- Vast.ai GPU cloud integration
- RunPod support
- Google Drive storage (rclone)
- Streaming mode
- Preview generation

### Developer Features
- Dry run mode
- Watch mode (folder monitoring)
- Structured logging
- YouTube download (yt-dlp)
- GitHub Actions CI/CD

---

## v2.0 Roadmap (High Priority)

### TAP Denoising Framework
**Status:** Planned

Integrate TAP (ECCV 2024) which adds tunable temporal modules to pre-trained image denoisers.

| Current | With TAP |
|---------|----------|
| OpenCV filters | Neural denoisers (Restormer, NAFNet) |
| Hand-crafted flow | Learned temporal attention |
| ~30-32 dB PSNR | ~34-38 dB PSNR |

**Barrier:** Requires pretrained backbone + training infrastructure
**Reference:** [TAP GitHub](https://github.com/zfu006/TAP)

---

### Exemplar-Based Colorization (SwinTExCo)
**Status:** Planned

User-guided colorization with reference images instead of AI guessing.

| Current (DDColor) | SwinTExCo |
|-------------------|-----------|
| AI's best guess | User provides reference |
| Per-frame (flickers) | Bidirectional fusion |
| Generic colors | Match actual source photos |

**Use case:** Provide a color photo from 1920s as reference for 1920s footage.

**Barrier:** Heavy Swin Transformer model
**Reference:** [SwinTExCo Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417424023042)

---

### AESRGAN Face Enhancement
**Status:** Planned (Quick Win)

Attention-Enhanced ESRGAN for better facial detail preservation.

| Current (GFPGAN) | AESRGAN |
|------------------|---------|
| Sometimes over-smooths | Preserves subtle features |
| Can hallucinate | Better high-frequency control |

**Barrier:** Model weights not widely distributed
**Effort:** Low - drop-in replacement

---

## v2.x Roadmap (Medium Priority)

### Diffusion-Based Video Super-Resolution
Models like Upscale-A-Video (CVPR 2024) or SeedVR (CVPR 2025).

| Real-ESRGAN | Diffusion VSR |
|-------------|---------------|
| Can look "plastic" | Realistic fine detail |
| ~0.1s/frame | ~2-10s/frame |
| Good quality | State-of-art quality |

**Barrier:** 10-100x slower than GANs

---

### QP-Aware Codec Artifact Removal
WACV 2025 approach for reversing codec compression damage.

| Current | QP-Aware |
|---------|----------|
| Generic denoising | Targeted blocking removal |
| One-size-fits-all | Adapts to compression level |

**Critical for:** YouTube sources (heavily compressed)
**Barrier:** No public weights yet

---

### Missing Frame Generation
Generative AI for reconstructing damaged/missing frames.

| RIFE | Generative |
|------|------------|
| Needs 2 valid frames | Generate from context |
| Can't handle gaps | Reconstruct sequences |

**Use case:** Film reel with 10 consecutive damaged frames
**Barrier:** High compute, risk of hallucination

---

### Unified Multi-Task Model (BCell RNN)
Single model for denoising + deblurring + super-resolution.

| Separate Models | BCell Unified |
|-----------------|---------------|
| 3-5 models | 1 model |
| Sequential passes | Single pass |
| Baseline | +1-4 dB PSNR |

**Barrier:** Custom architecture + training from scratch

---

## v3.0 Vision (Low Priority)

### Plugin Architecture
```python
class ProcessorPlugin(Protocol):
    name: str
    version: str
    def process(self, frame: np.ndarray, config: Dict) -> np.ndarray: ...
```
Location: `~/.framewright/plugins/`

---

### REST API Mode
```bash
framewright serve --port 8080
```

Endpoints:
- `POST /api/v1/jobs` - Submit job
- `GET /api/v1/jobs/{id}` - Status
- `GET /api/v1/jobs/{id}/logs` - Stream logs
- `DELETE /api/v1/jobs/{id}` - Cancel
- `GET /api/v1/health` - Health check

Features: Job queue, webhooks, OpenAPI docs, auth

---

### Mobile Companion PWA
- Job monitoring
- Push notifications
- Remote start/stop
- Preview viewing

---

## Developer Experience (Ongoing)

- [ ] Full type hint coverage
- [ ] 90%+ test coverage
- [ ] Sphinx/MkDocs API documentation
- [ ] Standardized benchmarking suite
- [ ] Windows one-click installer
- [ ] Pre-built Docker images

---

## Implementation Priority

| Enhancement | Effort | Impact | Priority |
|-------------|--------|--------|----------|
| AESRGAN faces | Low | Medium | **Do First** |
| SwinTExCo colorization | Medium | High | Second |
| TAP denoising | High | High | Third |
| Diffusion VSR | Medium | High | When speed improves |
| Plugin architecture | Medium | Low | When needed |
| REST API | Medium | Medium | On request |

---

## Ideas & Research

### Potential Additions
- Real-time preview during cloud processing
- Cost estimation before job submission
- Automatic quality presets based on source analysis
- Batch job templates (JSON/YAML)
- Integration with DaVinci Resolve / Premiere
- Neural audio enhancement (separate from video)

### Papers to Watch
- Any CVPR/ECCV 2025 video restoration papers
- Efficient diffusion models (speed improvements)
- Zero-shot video editing techniques

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-07 | 1.4.0 | Fixed rclone cloud bug, added cloud guide |
| 2025-12-30 | 1.3.0 | Temporal denoising, flicker reduction |
| 2025-12-29 | 1.2.0 | Cloud support, streaming, multi-GPU |
| 2025-12-28 | 1.1.0 | Subtitle removal, stabilization |
| 2025-12-27 | 1.0.0 | Initial release |

---

*This roadmap is updated as features are implemented or priorities change.*
