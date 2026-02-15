# FrameWright Model Selection Guide

## 🎬 Choose the Right Model for Your Content

### Quick Decision Tree

```
Is your content animated/anime?
├─ YES → Use "realesrgan-x4plus-anime" or "realesr-animevideov3"
└─ NO → Continue...

Is it real footage?
├─ YES → Continue...
└─ NO (CGI/Game) → Use "realesrgan-x4plus"

What decade is your footage from?
├─ Pre-1960 (Old film/historical) → Special settings ⭐
├─ 1960-1990 (Vintage) → Standard + grain preservation
└─ 1990+ (Modern) → Standard settings
```

---

## 📺 Model Recommendations by Content Type

### 🎞️ Historical Film (Pre-1960) - **YOUR CASE**

**Example:** 1909 B&W film, newsreel footage, early cinema

**Optimal Configuration:**
- **Upscale Model:** `realesrgan-x4plus`
- **Denoise:** `restormer` (medium strength)
- **Face Restore:** `gfpgan-v1.4` (if faces present)
- **Reference Enhance:** ✅ **ENABLE** (use historical photos)
- **Colorize:** ❌ **DISABLE** (stay B&W)
- **Preserve Grain:** ✅ **ENABLE**
- **Interlace Fix:** ✅ **ENABLE**

**Why:**
- Old film has heavy degradation
- Reference images help fill in lost details
- Grain is part of the aesthetic
- Interlace from telecine conversions common

**Reference Images:**
Upload 3-5 high-quality B&W photos from the same era to guide enhancement.

---

### 🎭 Vintage Film (1960-1990)

**Example:** Classic movies, home videos, old TV shows

**Optimal Configuration:**
- **Upscale Model:** `realesrgan-x4plus`
- **Denoise:** `restormer` (light-medium)
- **Face Restore:** `aesrgan-face`
- **Colorize:** Based on source (B&W → colorize, color → keep)
- **Preserve Grain:** ✅ for film, ❌ for video
- **Interpolate:** Optional (24fps → 60fps for smoothness)

---

### 🎬 Modern Footage (1990+)

**Example:** Digital video, recent films, YouTube uploads

**Optimal Configuration:**
- **Upscale Model:** `realesrgan-x4plus`
- **Denoise:** `nafnet` (light) or none
- **Face Restore:** Only if needed
- **Interpolate:** ✅ (increase frame rate)
- **Preserve Grain:** ❌

---

### 🎨 Animation / Anime

**Example:** Anime, cartoons, animated films

**Optimal Configuration:**
- **Upscale Model:** `realesrgan-x4plus-anime` or `realesr-animevideov3`
- **Denoise:** Light or none
- **Colorize:** Only for B&W animation
- **Interpolate:** ⚠️ Use with caution (can blur motion lines)

**Model Differences:**
- `realesrgan-x4plus-anime`: General anime, 6B parameters
- `realesr-animevideov3`: Best for anime **video** (temporal consistency)

---

## 🎨 Feature Guide

### When to Use Reference Enhancement

✅ **Use When:**
- Historical footage with heavy degradation
- You have reference photos from the same era
- You want to "guide" the AI with style examples
- Archival restoration projects

❌ **Don't Use When:**
- Modern, clean footage
- No reference images available
- Fictional/fantasy content (no historical references)

**How It Works:**
1. Upload 3-5 reference images (same era/style)
2. AI uses references to guide detail reconstruction
3. Strength slider: 0.5 = subtle, 0.8 = strong guidance

---

### When to Enable Colorization

✅ **Colorize When:**
- B&W historical footage that would benefit from color
- Family home movies/photos
- Educational content (better engagement)
- Source is definitely grayscale

❌ **Don't Colorize When:**
- B&W is intentional (artistic noir, Schindler's List style)
- Historical accuracy required
- Film grain/texture is important
- User preference for B&W aesthetic

**Models:**
- `deoldify-artistic`: Creative, vibrant colors
- `deoldify-stable`: Conservative, realistic colors
- `ddcolor`: Highest quality, modern approach

---

### When to Preserve Film Grain

✅ **Preserve Grain When:**
- Pre-1980s film footage
- Artistic/cinematic look desired
- Film texture is part of the aesthetic
- Archival/historical projects

❌ **Remove Grain When:**
- Modern digital video
- Clean, polished look desired
- Source grain is degradation (not intentional)
- Preparing for further editing

---

### When to Interpolate (Increase FPS)

✅ **Interpolate When:**
- Low frame rate source (24fps → 60fps)
- Smooth motion desired
- Modern viewing (TVs, displays)
- Sports/action content

❌ **Don't Interpolate When:**
- Artistic/cinematic 24fps look desired
- Animation (can blur motion lines)
- Already high frame rate (60fps+)
- Historical accuracy required

**Settings:**
- **24fps → 30fps:** Subtle smoothness
- **24fps → 60fps:** Very smooth (modern look)
- **30fps → 60fps:** Interpolate between frames

---

## 🎯 Preset Recommendations

### Historical Archival (1909 B&W Film)
```
Preset: Custom
├─ Scale: 4x
├─ Model: realesrgan-x4plus
├─ Denoise: restormer (medium)
├─ Reference Enhance: ON (with period photos)
├─ Colorize: OFF
├─ Preserve Grain: ON
├─ Interlace Fix: ON
└─ CRF: 18 (archival quality)
```

### Classic Film Restoration (1940s-1960s)
```
Preset: Maximum Quality
├─ Scale: 4x
├─ Model: realesrgan-x4plus
├─ Denoise: restormer (light)
├─ Face Restore: gfpgan-v1.4
├─ Colorize: Optional (deoldify-stable)
├─ Preserve Grain: ON
└─ Interpolate: Optional (24→30fps)
```

### Home Video Enhancement (1980s-1990s VHS)
```
Preset: Balanced
├─ Scale: 4x
├─ Model: realesrgan-x4plus
├─ Denoise: nafnet (medium)
├─ Face Restore: aesrgan-face
├─ Colorize: OFF (already color)
├─ VHS Artifact Removal: ON
└─ Audio Enhance: ON
```

### Anime Upscaling
```
Preset: Custom
├─ Scale: 4x
├─ Model: realesr-animevideov3
├─ Denoise: Light or OFF
├─ Face Restore: OFF
├─ Colorize: Only if B&W
└─ Interpolate: Use caution
```

---

## 📊 Model Performance Comparison

| Model | Best For | Quality | Speed | VRAM |
|-------|----------|---------|-------|------|
| `realesrgan-x4plus` | Real footage, photos | ⭐⭐⭐⭐ | Medium | 4-8 GB |
| `realesrgan-x4plus-anime` | Anime, animation | ⭐⭐⭐⭐ | Fast | 2-4 GB |
| `realesr-animevideov3` | Anime video | ⭐⭐⭐⭐⭐ | Medium | 4-6 GB |
| `hat-l-srx4` | Highest quality SR | ⭐⭐⭐⭐⭐ | Slow | 8-12 GB |
| `hat-srx4` | Balanced quality/speed | ⭐⭐⭐⭐ | Medium | 6-8 GB |

**Denoise Models:**
| Model | Best For | Quality | Speed | VRAM |
|-------|----------|---------|-------|------|
| `restormer` | Heavy degradation | ⭐⭐⭐⭐⭐ | Slow | 4-6 GB |
| `nafnet` | Light/medium noise | ⭐⭐⭐⭐ | Fast | 2-3 GB |
| `dncnn-deblock` | Compression artifacts | ⭐⭐⭐ | Fast | 1-2 GB |

---

## 💡 Pro Tips

### For Best Results:

1. **Analyze First:**
   Click "Analyze Video" to get AI recommendations

2. **Use References:**
   For historical content, provide era-appropriate reference images

3. **Preserve Originals:**
   Keep grain/texture for pre-1980s film

4. **Test Settings:**
   Process 10 seconds first, then full video

5. **CRF Quality:**
   - 15-18: Archival/master copy
   - 20-23: High quality (recommended)
   - 24-28: Web/streaming

### Common Mistakes:

❌ Using anime model on real footage
❌ Heavy denoising on film grain
❌ Interpolating animation
❌ Colorizing intentional B&W
❌ Skipping interlace detection

---

## 🎓 Example Workflows

### 1909 Silent Film → 4K Restored
1. Analyze video → detects B&W, heavy degradation
2. Upload reference: Historical photos from 1900s-1910s
3. Settings:
   - Scale: 4x, Model: realesrgan-x4plus
   - Denoise: restormer medium
   - Reference enhance: 70% strength
   - Preserve grain: ON
   - No colorization
4. Export: 4K ProRes or H.265 CRF 18

### 1960s Color Film → Modern HD
1. Analyze video → detects vintage color film
2. Settings:
   - Scale: 4x, Model: realesrgan-x4plus
   - Denoise: restormer light
   - Face restore: gfpgan-v1.4
   - Preserve grain: ON
   - Color correction: Auto-fix fading
3. Export: 1080p H.265 CRF 20

### 1990s Anime VHS → 1080p60
1. Analyze video → detects animation
2. Settings:
   - Scale: 4x, Model: realesr-animevideov3
   - Denoise: nafnet light
   - Interpolate: 30fps → 60fps
   - Deinterlace: ON
3. Export: 1080p60 H.264 CRF 23

---

This guide helps users make informed decisions about which models and settings to use based on their specific content type and restoration goals.
