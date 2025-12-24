"""Web-based UI for FrameWright video restoration pipeline.

Provides a user-friendly Gradio interface for non-technical users.
"""
import logging
import tempfile
import threading
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Check for Gradio availability
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    gr = None


def check_gradio_installed() -> bool:
    """Check if Gradio is installed."""
    return HAS_GRADIO


def install_gradio_instructions() -> str:
    """Get instructions for installing Gradio."""
    return """
Gradio is required for the web UI but is not installed.

Install it with:
    pip install gradio

Or install FrameWright with UI support:
    pip install framewright[ui]
"""


# =============================================================================
# UI Components (only defined if Gradio is available)
# =============================================================================

if HAS_GRADIO:
    from .config import Config
    from .restorer import VideoRestorer
    from .hardware import check_hardware, print_hardware_report, HardwareReport

    def create_ui() -> gr.Blocks:
        """Create the Gradio web interface.

        Returns:
            Gradio Blocks application
        """
        # State for tracking progress
        progress_state = {"current": 0, "total": 100, "stage": "idle"}

        def run_hardware_check() -> str:
            """Run hardware compatibility check."""
            try:
                report = check_hardware()
                return print_hardware_report(report)
            except Exception as e:
                return f"Error checking hardware: {e}"

        def get_hardware_summary() -> Tuple[str, str, str, str]:
            """Get hardware summary for display."""
            try:
                report = check_hardware()

                # Status badge
                status_map = {
                    "ready": "🟢 Ready",
                    "limited": "🟡 Limited",
                    "incompatible": "🔴 Incompatible",
                }
                status = status_map.get(report.overall_status, "❓ Unknown")

                # GPU info
                if report.gpu.has_gpu:
                    gpu = f"{report.gpu.gpu_name} ({report.gpu.vram_total_mb}MB VRAM)"
                else:
                    gpu = "No GPU detected (CPU mode)"

                # RAM
                ram = f"{report.system.ram_available_gb:.1f} GB available"

                # Disk
                disk = f"{report.disk_free_gb:.1f} GB free"

                return status, gpu, ram, disk
            except Exception as e:
                return "❓ Error", str(e), "Unknown", "Unknown"

        def restore_video(
            input_video: str,
            youtube_url: str,
            scale_factor: int,
            model: str,
            crf: int,
            enhance_audio: bool,
            interpolate: bool,
            target_fps: int,
            progress: gr.Progress = gr.Progress(),
        ) -> Tuple[Optional[str], str]:
            """Run video restoration pipeline.

            Returns:
                Tuple of (output_video_path, log_messages)
            """
            logs = []

            def log(msg: str):
                logs.append(msg)
                logger.info(msg)

            try:
                # Determine input source
                if youtube_url and youtube_url.strip():
                    source = youtube_url.strip()
                    log(f"📥 Downloading from: {source}")
                elif input_video:
                    source = input_video
                    log(f"📁 Using local file: {source}")
                else:
                    return None, "❌ Please provide a video file or YouTube URL"

                # Create temp directory for output
                output_dir = Path(tempfile.mkdtemp(prefix="framewright_"))
                log(f"📂 Working directory: {output_dir}")

                # Configure
                config = Config(
                    project_dir=output_dir,
                    scale_factor=scale_factor,
                    model_name=model,
                    crf=crf,
                    enable_checkpointing=True,
                    enable_validation=True,
                )

                # Progress callback
                def on_progress(current: int, total: int, stage: str):
                    progress((current / total) if total > 0 else 0, desc=stage)
                    log(f"  {stage}: {current}/{total}")

                # Create restorer
                log("🔧 Initializing pipeline...")
                restorer = VideoRestorer(config)

                # Run restoration
                log("🎬 Starting restoration...")
                progress(0.1, desc="Downloading/Loading video")

                result = restorer.restore_video(
                    source=source,
                    enhance_audio=enhance_audio,
                    cleanup=False,
                )

                log(f"✅ Restoration complete!")
                log(f"📹 Output: {result}")

                return str(result), "\n".join(logs)

            except Exception as e:
                error_msg = f"❌ Error: {e}"
                logs.append(error_msg)
                logger.exception("Restoration failed")
                return None, "\n".join(logs)

        # Build the UI
        with gr.Blocks(
            title="FrameWright - Video Restoration",
        ) as app:
            gr.Markdown(
                """
                # 🎬 FrameWright Video Restoration

                Restore and enhance vintage or degraded video footage using AI.
                Perfect for 100-year-old film fragments, home videos, or any low-quality footage.

                ---
                """
            )

            with gr.Tabs():
                # Tab 1: Hardware Check
                with gr.TabItem("🔧 Hardware Check"):
                    gr.Markdown("### Check if your system can run FrameWright")

                    with gr.Row():
                        with gr.Column(scale=1):
                            status_display = gr.Textbox(
                                label="Status",
                                interactive=False,
                                elem_classes=["status-badge"],
                            )
                        with gr.Column(scale=2):
                            gpu_display = gr.Textbox(label="GPU", interactive=False)
                        with gr.Column(scale=1):
                            ram_display = gr.Textbox(label="RAM", interactive=False)
                        with gr.Column(scale=1):
                            disk_display = gr.Textbox(label="Disk", interactive=False)

                    check_btn = gr.Button("🔍 Check Hardware", variant="primary", size="lg")
                    hardware_report = gr.Textbox(
                        label="Detailed Report",
                        lines=25,
                        interactive=False,
                    )

                    check_btn.click(
                        fn=run_hardware_check,
                        outputs=hardware_report,
                    ).then(
                        fn=get_hardware_summary,
                        outputs=[status_display, gpu_display, ram_display, disk_display],
                    )

                    # Auto-check on load
                    app.load(
                        fn=get_hardware_summary,
                        outputs=[status_display, gpu_display, ram_display, disk_display],
                    )

                # Tab 2: Restore Video
                with gr.TabItem("🎥 Restore Video"):
                    gr.Markdown("### Upload a video or provide a YouTube URL")

                    with gr.Row():
                        with gr.Column(scale=2):
                            input_video = gr.Video(
                                label="Upload Video",
                                sources=["upload"],
                            )
                            gr.Markdown("**OR**")
                            youtube_url = gr.Textbox(
                                label="YouTube URL",
                                placeholder="https://www.youtube.com/watch?v=...",
                            )

                        with gr.Column(scale=1):
                            gr.Markdown("### Settings")

                            scale_factor = gr.Radio(
                                choices=[2, 4],
                                value=4,
                                label="Upscale Factor",
                                info="4x for heavily degraded, 2x for mild enhancement",
                            )

                            model = gr.Dropdown(
                                choices=[
                                    "realesrgan-x4plus",
                                    "realesrgan-x4plus-anime",
                                    "realesr-animevideov3",
                                    "realesrnet-x4plus",
                                ],
                                value="realesrgan-x4plus",
                                label="AI Model",
                                info="x4plus recommended for film",
                            )

                            crf = gr.Slider(
                                minimum=15,
                                maximum=28,
                                value=18,
                                step=1,
                                label="Quality (CRF)",
                                info="Lower = better quality, larger file",
                            )

                            enhance_audio = gr.Checkbox(
                                value=True,
                                label="Enhance Audio",
                                info="Apply noise reduction and normalization",
                            )

                            with gr.Accordion("Advanced Options", open=False):
                                interpolate = gr.Checkbox(
                                    value=False,
                                    label="Frame Interpolation (RIFE)",
                                    info="Increase frame rate (requires RIFE)",
                                )

                                target_fps = gr.Slider(
                                    minimum=24,
                                    maximum=60,
                                    value=60,
                                    step=6,
                                    label="Target FPS",
                                    visible=False,
                                )

                                interpolate.change(
                                    fn=lambda x: gr.update(visible=x),
                                    inputs=interpolate,
                                    outputs=target_fps,
                                )

                    restore_btn = gr.Button(
                        "🚀 Start Restoration",
                        variant="primary",
                        size="lg",
                    )

                    with gr.Row():
                        with gr.Column():
                            output_video = gr.Video(label="Restored Video")
                        with gr.Column():
                            log_output = gr.Textbox(
                                label="Processing Log",
                                lines=15,
                                interactive=False,
                            )

                    restore_btn.click(
                        fn=restore_video,
                        inputs=[
                            input_video,
                            youtube_url,
                            scale_factor,
                            model,
                            crf,
                            enhance_audio,
                            interpolate,
                            target_fps,
                        ],
                        outputs=[output_video, log_output],
                    )

                # Tab 3: Help
                with gr.TabItem("❓ Help"):
                    gr.Markdown(
                        """
                        ## Quick Start Guide

                        ### 1. Check Your Hardware
                        - Go to the **Hardware Check** tab
                        - Click "Check Hardware" to see if your system is ready
                        - For best results, you need:
                          - **NVIDIA GPU** with 4GB+ VRAM (or be patient with CPU processing)
                          - **16GB RAM** recommended
                          - **50GB+ free disk space**

                        ### 2. Restore a Video
                        - Go to the **Restore Video** tab
                        - Either upload a video file or paste a YouTube URL
                        - Adjust settings (defaults work well for most footage)
                        - Click "Start Restoration" and wait

                        ### Settings Explained

                        | Setting | What it does |
                        |---------|--------------|
                        | **Upscale Factor** | 4x doubles size twice (best for old film), 2x for newer video |
                        | **AI Model** | `x4plus` works best for real footage, `anime` for animation |
                        | **Quality (CRF)** | 15-18 for archival, 20-23 for web sharing |
                        | **Enhance Audio** | Removes hiss, rumble, normalizes volume |

                        ### Tips for Old Film (1890s-1920s)

                        - Use **4x upscaling** - old film is usually low resolution
                        - Keep **enhance audio** on - old recordings are noisy
                        - For silent films, the pipeline handles missing audio gracefully
                        - Processing time: expect 2-10x the video length

                        ### Troubleshooting

                        **"Out of memory" errors:**
                        - Your GPU doesn't have enough VRAM
                        - Try using 2x upscaling instead of 4x
                        - Close other applications

                        **Processing is very slow:**
                        - You may be running on CPU (no GPU detected)
                        - Check the Hardware tab to confirm GPU status

                        **Video looks worse:**
                        - Try a different AI model
                        - Some heavily compressed videos don't improve much

                        ---

                        For more help, visit: [GitHub Issues](https://github.com/your-repo/framewright/issues)
                        """
                    )

        return app


def launch_ui(
    share: bool = False,
    server_port: int = 7860,
    server_name: str = "127.0.0.1",
) -> None:
    """Launch the FrameWright web UI.

    Args:
        share: Create a public shareable link
        server_port: Port to run server on
        server_name: Server hostname (use "0.0.0.0" for network access)
    """
    if not HAS_GRADIO:
        print(install_gradio_instructions())
        return

    print("\n" + "=" * 50)
    print("  🎬 FrameWright Video Restoration UI")
    print("=" * 50)
    print(f"\n  Starting web interface...")
    print(f"  Open your browser to: http://{server_name}:{server_port}")
    if share:
        print("  (A public link will be generated...)")
    print("\n  Press Ctrl+C to stop the server")
    print("=" * 50 + "\n")

    app = create_ui()
    app.launch(
        share=share,
        server_port=server_port,
        server_name=server_name,
        show_error=True,
        theme=gr.themes.Soft(),
    )
