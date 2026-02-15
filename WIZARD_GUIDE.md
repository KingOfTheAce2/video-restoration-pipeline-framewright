# FrameWright Interactive Wizard Guide

## 🧙 What is the Wizard?

The **Interactive Wizard** is a step-by-step command-line tool that guides you through configuring video restoration settings. It asks questions, analyzes your video, and recommends optimal settings for your specific content.

## 🚀 How to Launch

### Option 1: Windows Command Prompt (Recommended)

1. Open **File Explorer**
2. Navigate to: `G:\GitHub\video-restoration-pipeline-framewright`
3. Double-click **`wizard.bat`**

Or from Command Prompt:
```cmd
cd G:\GitHub\video-restoration-pipeline-framewright
wizard.bat
```

### Option 2: PowerShell

```powershell
cd G:\GitHub\video-restoration-pipeline-framewright
.\wizard.ps1
```

### Option 3: Python Directly (any console)

```cmd
cd G:\GitHub\video-restoration-pipeline-framewright
python run_wizard.py
```

Or with a video file already selected:
```cmd
python run_wizard.py "path\to\your\video.mp4"
```

## 📋 What the Wizard Does (Step-by-Step)

### 1. **Welcome Screen**
   - Shows FrameWright banner and instructions
   - Press Ctrl+C anytime to cancel

### 2. **Select Input Video**
   - Browse to select your video file
   - Or provide path as command-line argument

### 3. **Analyze Video** (Automatic)
   - Detects content type (film, animation, home video, documentary)
   - Identifies era (silent era, early sound, golden age, modern)
   - Checks if B&W or color
   - Detects faces
   - Analyzes degradation level
   - Finds primary issues (noise, blur, compression)

### 4. **Scan for Issues** (NEW!)
   - **Interlacing**: Old VHS/DVD comb effect → Deinterlace it?
   - **Black bars**: Letterbox/pillarbox → Crop them?
   - **Color fading**: Faded film stock → Fix colors?
   - **Audio sync**: A/V drift → Re-sync it?
   - **Watermark/logo**: → Remove it?

### 5. **Confirm Content Type**
   - Classic film or movie
   - Animation or cartoons
   - Home videos
   - Documentary/archival
   - Other / Let AI decide

### 6. **Quality Priority**
   - **Speed**: Fast processing, good quality
   - **Balanced**: Good balance (Recommended)
   - **Quality**: Higher quality, slower
   - **Maximum**: Best possible, much slower

### 7. **Feature Selection**
   - **Upscaling factor**: 2x or 4x (or 1x for no upscaling)
   - **Face enhancement**: Improve facial details?
   - **Frame interpolation**: Smoother motion (increase FPS)?
     - If yes: Target FPS (24, 30, or 60)
   - **Frame generation**: AI reconstruct missing/damaged frames?

### 8. **Colorization** (if B&W detected)
   - Would you like to colorize this video?
   - Do you have reference color images?
     - Add 1-5 reference images for better results

### 9. **Output Settings**
   - Default: `{filename}_restored.mp4` in same folder
   - Or choose custom output location

### 10. **Summary & Confirmation**
   - Shows restoration plan with all stages
   - Lists all settings
   - Asks: "Start restoration with these settings?"

### 11. **Done!**
   - Shows configuration saved
   - Displays command to start restoration

## 🎯 Example: 1909 B&W Historical Film

**Your Use Case:**
- Heavily degraded B&W film from 1909
- Should stay B&W (no colorization)
- Fill in details using reference pictures

**Wizard Answers:**

```
Select your video file:
→ [Browse to your 1909 film]

[Analysis detects: B&W, silent era, heavy degradation]

Which issues would you like to fix?
→ ✓ Interlacing (if detected)
→ ✓ Black bars (if detected)

What type of content is this?
→ Classic film or movie

What's your priority?
→ Quality - Higher quality, slower processing

Upscaling factor?
→ 4x (Quadruple)

Enable face enhancement?
→ Yes (if faces present)

Enable frame interpolation (smoother motion)?
→ No (preserve original cinematic feel)
  OR
→ Yes → 24 fps (restore original film speed)

[B&W detected!]
Would you like to colorize this video?
→ No (keep B&W as you requested)

Save to film_1909_restored.mp4?
→ Yes

[Summary shows all settings]
Start restoration with these settings?
→ Yes
```

## ⚙️ Technical Details

**What You Get:**
- Intelligent defaults based on analysis
- Clear explanations for each option
- Visual progress indicators
- Colorful terminal output (with Rich library)
- Interactive prompts (with Questionary library)

**Dependencies:**
- `rich`: Beautiful terminal formatting
- `questionary`: Interactive prompts
- Both are included in `requirements.txt`

**Note on Console Compatibility:**
- ✅ Works in: **cmd.exe**, **PowerShell**, **Windows Terminal**
- ❌ Doesn't work in: Git Bash (questionary limitation)
- If you get errors, use `wizard.bat` or PowerShell

## 🔍 Understanding Video Issues

### Interlacing / Deinterlacing
- **What**: Old analog video drew frames in two passes (odd lines, then even lines)
- **Problem**: Creates "comb" effect on modern screens
- **Fix**: Deinterlacing combines both passes into smooth frames
- **When**: VHS tapes, old DVDs, TV broadcasts
- **NOT**: Frame duplication (that's different)

### Letterbox/Pillarbox
- **Letterbox**: Black bars on top/bottom (widescreen on old TVs)
- **Pillarbox**: Black bars on left/right (old content on widescreen)
- **Fix**: Auto-crops to content region
- **NOT**: YouTube padding (that's frame duplication)

### Frame Duplication (YouTube Padding)
- **What**: Same frame repeated to increase FPS (16fps → 24fps)
- **When**: Old silent films uploaded to YouTube
- **Fix**: Frame Deduplication feature (auto-enabled for silent era content)
- **NOT**: Interlacing or black bars

## 📚 Next Steps

After wizard completes, you can:

1. **Start restoration immediately** (wizard can do this automatically)
2. **Save settings** for later use
3. **Edit settings** in generated config file
4. **Batch process** multiple videos with same settings

## 🆘 Troubleshooting

**"NoConsoleScreenBufferError"**
- Solution: Run in cmd.exe or PowerShell, not Git Bash
- Use: `wizard.bat` (easiest)

**"Wizard dependencies not installed"**
```bash
pip install rich questionary
```

**Wizard starts but freezes**
- You might be in an incompatible terminal
- Try: `wizard.bat` or PowerShell

**Want to skip video analysis (faster)?**
```python
python run_wizard.py --skip-analysis your_video.mp4
```

## 💡 Pro Tips

1. **First time?** Use wizard instead of manual CLI
2. **Same settings for multiple videos?** Run wizard once, save settings, batch process
3. **Not sure about settings?** Wizard recommends based on analysis
4. **Testing?** Use "Speed" priority first, then re-run with "Quality"
5. **Reference images matter!** For historical B&W, provide era-appropriate photos

---

**Questions?** See the main README.md or run `framewright --help`
