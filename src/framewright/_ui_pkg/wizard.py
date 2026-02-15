"""Interactive wizard for FrameWright.

Provides a guided, Apple-quality setup experience that walks users
through the restoration process with intelligent defaults and
clear explanations.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import sys

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.table import Table
    from rich.text import Text
    from rich.box import ROUNDED
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import questionary
    from questionary import Style as QStyle
    QUESTIONARY_AVAILABLE = True
except ImportError:
    QUESTIONARY_AVAILABLE = False

from .auto_detect import SmartAnalyzer, AnalysisResult, ContentType, Era
from .terminal import Console as FWConsole


# Custom questionary style matching FrameWright theme
WIZARD_STYLE = None
if QUESTIONARY_AVAILABLE:
    WIZARD_STYLE = QStyle([
        ('qmark', 'fg:cyan bold'),
        ('question', 'bold'),
        ('answer', 'fg:cyan bold'),
        ('pointer', 'fg:cyan bold'),
        ('highlighted', 'fg:cyan bold'),
        ('selected', 'fg:cyan'),
        ('separator', 'fg:cyan'),
        ('instruction', 'fg:gray'),
        ('text', ''),
    ])


class QualityPriority(Enum):
    """User's quality priority preference."""
    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"
    MAXIMUM = "maximum"


class ContentChoice(Enum):
    """Content type choices for wizard."""
    FILM = "Classic film or movie footage"
    ANIMATION = "Animation or cartoons"
    HOME_VIDEO = "Home videos or personal recordings"
    DOCUMENTARY = "Documentary or archival footage"
    OTHER = "Other / Let AI decide"


@dataclass
class WizardResult:
    """Result from the interactive wizard."""
    # Input
    input_path: Path
    output_path: Optional[Path] = None
    # Detected/confirmed settings
    content_type: ContentType = ContentType.UNKNOWN
    quality_priority: QualityPriority = QualityPriority.BALANCED
    # Features
    enable_colorization: bool = False
    colorization_references: List[Path] = field(default_factory=list)
    enable_face_enhancement: bool = True
    enable_frame_generation: bool = False
    enable_interpolation: bool = False
    target_fps: float = 30.0
    # Scaling
    scale_factor: int = 2
    # Preset
    preset: str = "balanced"
    # Pre-processing fixes (new features)
    fix_interlacing: bool = False
    deinterlace_method: str = "auto"
    crop_black_bars: bool = False
    remove_watermark: bool = False
    watermark_position: Optional[str] = None
    fix_film_colors: bool = False
    detected_film_stock: Optional[str] = None
    fix_audio_sync: bool = False
    detected_drift_ms: float = 0.0
    # All settings as dict
    settings: Dict[str, Any] = field(default_factory=dict)
    # Was wizard completed?
    completed: bool = False
    cancelled: bool = False

    def to_config_dict(self) -> Dict[str, Any]:
        """Convert to config dictionary."""
        config = {
            "preset": self.preset,
            "scale_factor": self.scale_factor,
            "enable_interpolation": self.enable_interpolation,
            "target_fps": self.target_fps,
            "auto_face_restore": self.enable_face_enhancement,
            "enable_frame_generation": self.enable_frame_generation,
        }

        if self.enable_colorization and self.colorization_references:
            config["colorization_reference_images"] = [
                str(p) for p in self.colorization_references
            ]

        # Pre-processing fixes
        if self.fix_interlacing:
            config["enable_deinterlace"] = True
            config["deinterlace_method"] = self.deinterlace_method

        if self.crop_black_bars:
            config["enable_crop_letterbox"] = True

        if self.remove_watermark:
            config["enable_watermark_removal"] = True
            if self.watermark_position:
                config["watermark_position"] = self.watermark_position

        if self.fix_film_colors:
            config["enable_film_color_correction"] = True
            config["detected_film_stock"] = self.detected_film_stock

        if self.fix_audio_sync:
            config["enable_audio_sync_fix"] = True
            config["audio_drift_ms"] = self.detected_drift_ms

        # Merge any additional settings
        config.update(self.settings)

        return config


class InteractiveWizard:
    """Interactive setup wizard for FrameWright.

    Provides a guided experience for configuring video restoration
    with intelligent defaults and clear explanations.

    Example:
        >>> wizard = InteractiveWizard()
        >>> result = wizard.run("input.mp4")
        >>> if result.completed:
        ...     # Use result.to_config_dict() for restoration
    """

    def __init__(self, console: Optional[FWConsole] = None):
        """Initialize wizard.

        Args:
            console: FrameWright console instance
        """
        self.console = console or FWConsole()
        self._analyzer = SmartAnalyzer()
        self._analysis: Optional[AnalysisResult] = None

    def run(
        self,
        input_path: Optional[Path] = None,
        skip_analysis: bool = False,
    ) -> WizardResult:
        """Run the interactive wizard.

        Args:
            input_path: Optional pre-selected input path
            skip_analysis: Skip video analysis step

        Returns:
            WizardResult with user's choices
        """
        result = WizardResult(input_path=input_path or Path())

        try:
            # Welcome
            self._show_welcome()

            # Step 1: Input selection
            if input_path is None:
                input_path = self._ask_input_path()
                if input_path is None:
                    result.cancelled = True
                    return result
            result.input_path = Path(input_path)

            # Step 2: Analyze video
            if not skip_analysis:
                self._analysis = self._analyze_video(result.input_path)
                self._show_analysis_results()

            # Step 3: Pre-processing fixes (new!)
            self._ask_preprocessing(result)

            # Step 4: Confirm/adjust content type
            result.content_type = self._ask_content_type()

            # Step 5: Quality priority
            result.quality_priority = self._ask_quality_priority()

            # Step 6: Feature selection
            self._ask_features(result)

            # Step 7: Colorization (if B&W detected)
            if self._analysis and self._analysis.content.is_black_and_white:
                self._ask_colorization(result)

            # Step 8: Output settings
            self._ask_output_settings(result)

            # Step 9: Show summary and confirm
            if not self._confirm_settings(result):
                result.cancelled = True
                return result

            result.completed = True
            return result

        except KeyboardInterrupt:
            self.console.print("\n")
            self.console.warning("Wizard cancelled")
            result.cancelled = True
            return result

    def _show_welcome(self) -> None:
        """Show welcome message."""
        self.console.print_compact_banner()

        if RICH_AVAILABLE:
            from rich.panel import Panel
            welcome = (
                "[bold]Welcome to the FrameWright Setup Wizard![/bold]\n\n"
                "This wizard will help you configure the optimal restoration\n"
                "settings for your video. Answer a few questions and we'll\n"
                "take care of the rest.\n\n"
                "[dim]Press Ctrl+C at any time to cancel.[/dim]"
            )
            self.console.print(Panel(welcome, border_style="cyan", padding=(1, 2)))
        else:
            print("\nWelcome to the FrameWright Setup Wizard!")
            print("This wizard will help configure optimal restoration settings.\n")

        self.console.print()

    def _ask_input_path(self) -> Optional[Path]:
        """Ask for input video path."""
        if QUESTIONARY_AVAILABLE:
            path = questionary.path(
                "Select your video file:",
                style=WIZARD_STYLE,
            ).ask()
            return Path(path) if path else None
        else:
            path = input("Enter path to video file: ").strip()
            return Path(path) if path else None

    def _analyze_video(self, path: Path) -> AnalysisResult:
        """Analyze video and show progress."""
        self.console.info("Analyzing video...")

        if RICH_AVAILABLE:
            from rich.progress import Progress, SpinnerColumn, TextColumn
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task("Analyzing content and degradation...", total=None)
                analysis = self._analyzer.analyze(path, quick=False)
        else:
            print("Analyzing video...", end="", flush=True)
            analysis = self._analyzer.analyze(path, quick=False)
            print(" done")

        return analysis

    def _show_analysis_results(self) -> None:
        """Display analysis results."""
        if not self._analysis:
            return

        a = self._analysis

        self.console.print()
        self.console.video_summary(
            path=a.video_path,
            resolution=a.resolution,
            fps=a.fps,
            duration=a.duration_formatted,
            codec=a.codec,
            size_mb=a.bitrate_kbps * a.duration_seconds / 8000,
        )

        # Show detected characteristics
        if RICH_AVAILABLE:
            table = Table(
                title="Detected Characteristics",
                box=ROUNDED,
                show_header=False,
            )
            table.add_column("Property", style="dim")
            table.add_column("Value", style="cyan")

            table.add_row("Content Type", a.content.content_type.value.replace("_", " ").title())
            table.add_row("Era", a.content.era.value.replace("_", " ").title())
            table.add_row("Color", "Black & White" if a.content.is_black_and_white else "Color")
            table.add_row("Faces Detected", f"{a.content.face_percentage:.0f}% of frames")
            table.add_row("Degradation", a.degradation.severity.title())
            if a.degradation.primary_issues:
                table.add_row("Main Issues", ", ".join(a.degradation.primary_issues))

            self.console.print(table)
        else:
            print(f"\nDetected: {a.content.summary()}")
            print(f"Degradation: {a.degradation.severity}")
            if a.degradation.primary_issues:
                print(f"Issues: {', '.join(a.degradation.primary_issues)}")

        # Show warnings
        for warning in a.warnings:
            self.console.warning(warning)

        self.console.print()

    def _ask_preprocessing(self, result: WizardResult) -> None:
        """Ask about pre-processing fixes (interlace, letterbox, watermark, etc.)."""
        issues_found = []

        self.console.print()
        self.console.info("Scanning for common issues...")

        # Check for interlacing
        try:
            from ..processors.interlace_handler import InterlaceDetector
            detector = InterlaceDetector(sample_count=30)
            interlace = detector.analyze(result.input_path)
            if interlace.is_interlaced:
                issues_found.append({
                    "name": "Interlacing",
                    "description": f"{interlace.interlace_type.value} detected",
                    "key": "interlace",
                    "data": interlace,
                })
        except Exception:
            pass

        # Check for letterbox/black bars
        try:
            from ..processors.letterbox_handler import LetterboxDetector
            detector = LetterboxDetector(sample_count=10)
            letterbox = detector.analyze(result.input_path)
            if letterbox.has_letterbox or letterbox.has_pillarbox:
                issues_found.append({
                    "name": "Black bars",
                    "description": f"{letterbox.bar_percentage:.0f}% of frame",
                    "key": "letterbox",
                    "data": letterbox,
                })
        except Exception:
            pass

        # Check film stock (for color footage)
        try:
            if self._analysis and not self._analysis.content.is_black_and_white:
                from ..processors.film_stock_detector import FilmStockDetector
                detector = FilmStockDetector(sample_count=20)
                stock = detector.analyze(result.input_path)
                if stock.fading_detected:
                    issues_found.append({
                        "name": "Color fading",
                        "description": f"{stock.detected_stock.value} ({stock.fading_pattern})",
                        "key": "filmstock",
                        "data": stock,
                    })
        except Exception:
            pass

        # Check audio sync
        try:
            from ..processors.audio_sync import AudioSyncAnalyzer
            analyzer = AudioSyncAnalyzer()
            sync = analyzer.analyze_sync(result.input_path)
            if abs(sync.drift_ms) > 40:
                issues_found.append({
                    "name": "Audio sync",
                    "description": f"{sync.drift_ms:.0f}ms drift",
                    "key": "audiosync",
                    "data": sync,
                })
        except Exception:
            pass

        if not issues_found:
            self.console.info("✓ No pre-processing issues detected")
            return

        # Show detected issues
        if RICH_AVAILABLE:
            from rich.table import Table
            table = Table(title="Issues Detected", box=ROUNDED)
            table.add_column("#", style="dim", width=3)
            table.add_column("Issue", style="yellow")
            table.add_column("Details", style="cyan")

            for i, issue in enumerate(issues_found, 1):
                table.add_row(str(i), issue["name"], issue["description"])

            self.console.print(table)
        else:
            self.console.print("\nIssues detected:")
            for i, issue in enumerate(issues_found, 1):
                print(f"  {i}. {issue['name']}: {issue['description']}")

        self.console.print()

        # Ask which to fix
        if QUESTIONARY_AVAILABLE:
            fix_choices = [f"{issue['name']}: {issue['description']}" for issue in issues_found]

            selected = questionary.checkbox(
                "Which issues would you like to fix?",
                choices=fix_choices,
                style=WIZARD_STYLE,
            ).ask() or []

            for issue in issues_found:
                label = f"{issue['name']}: {issue['description']}"
                if label in selected:
                    if issue["key"] == "interlace":
                        result.fix_interlacing = True
                        result.deinterlace_method = issue["data"].recommended_method.value
                    elif issue["key"] == "letterbox":
                        result.crop_black_bars = True
                    elif issue["key"] == "filmstock":
                        result.fix_film_colors = True
                        result.detected_film_stock = issue["data"].detected_stock.value
                    elif issue["key"] == "audiosync":
                        result.fix_audio_sync = True
                        result.detected_drift_ms = issue["data"].drift_ms
        else:
            # Simple yes/no for each
            for issue in issues_found:
                fix = input(f"Fix {issue['name']}? [Y/n]: ").lower() != 'n'
                if fix:
                    if issue["key"] == "interlace":
                        result.fix_interlacing = True
                    elif issue["key"] == "letterbox":
                        result.crop_black_bars = True
                    elif issue["key"] == "filmstock":
                        result.fix_film_colors = True
                    elif issue["key"] == "audiosync":
                        result.fix_audio_sync = True

        # Ask about watermark removal (always offer)
        if QUESTIONARY_AVAILABLE:
            result.remove_watermark = questionary.confirm(
                "Does this video have a watermark or logo to remove?",
                default=False,
                style=WIZARD_STYLE,
            ).ask()

            if result.remove_watermark:
                position = questionary.select(
                    "Where is the watermark?",
                    choices=[
                        "Auto-detect (recommended)",
                        "Top-left corner",
                        "Top-right corner",
                        "Bottom-left corner",
                        "Bottom-right corner",
                    ],
                    style=WIZARD_STYLE,
                ).ask()

                pos_map = {
                    "Top-left corner": "top-left",
                    "Top-right corner": "top-right",
                    "Bottom-left corner": "bottom-left",
                    "Bottom-right corner": "bottom-right",
                }
                result.watermark_position = pos_map.get(position)
        else:
            result.remove_watermark = input("Remove watermark/logo? [y/N]: ").lower() == 'y'

        self.console.print()

    def _ask_content_type(self) -> ContentType:
        """Ask user to confirm/select content type."""
        detected = ContentType.UNKNOWN
        if self._analysis:
            detected = self._analysis.content.content_type

        choices = [
            ("Classic film or movie", ContentType.FILM),
            ("Animation or cartoons", ContentType.ANIMATION),
            ("Home videos", ContentType.HOME_VIDEO),
            ("Documentary/archival", ContentType.DOCUMENTARY),
            ("Other", ContentType.UNKNOWN),
        ]

        if QUESTIONARY_AVAILABLE:
            # Find default based on detection
            default = "Other"
            for label, ct in choices:
                if ct == detected:
                    default = label
                    break

            answer = questionary.select(
                "What type of content is this?",
                choices=[c[0] for c in choices],
                default=default,
                style=WIZARD_STYLE,
            ).ask()

            for label, ct in choices:
                if label == answer:
                    return ct
            return ContentType.UNKNOWN
        else:
            print("\nContent type:")
            for i, (label, _) in enumerate(choices, 1):
                print(f"  {i}. {label}")
            choice = input(f"Select [1-{len(choices)}]: ").strip()
            try:
                idx = int(choice) - 1
                return choices[idx][1]
            except (ValueError, IndexError):
                return detected

    def _ask_quality_priority(self) -> QualityPriority:
        """Ask for quality/speed priority."""
        choices = [
            ("Speed - Fast processing, good quality", QualityPriority.SPEED),
            ("Balanced - Good balance of speed and quality (Recommended)", QualityPriority.BALANCED),
            ("Quality - Higher quality, slower processing", QualityPriority.QUALITY),
            ("Maximum - Best possible quality, much slower", QualityPriority.MAXIMUM),
        ]

        if QUESTIONARY_AVAILABLE:
            answer = questionary.select(
                "What's your priority?",
                choices=[c[0] for c in choices],
                default=choices[1][0],  # Balanced
                style=WIZARD_STYLE,
            ).ask()

            for label, priority in choices:
                if label == answer:
                    return priority
            return QualityPriority.BALANCED
        else:
            print("\nQuality priority:")
            for i, (label, _) in enumerate(choices, 1):
                print(f"  {i}. {label}")
            choice = input("Select [1-4]: ").strip()
            try:
                return choices[int(choice) - 1][1]
            except (ValueError, IndexError):
                return QualityPriority.BALANCED

    def _ask_features(self, result: WizardResult) -> None:
        """Ask about feature enablement."""
        # Map priority to preset
        preset_map = {
            QualityPriority.SPEED: "fast",
            QualityPriority.BALANCED: "balanced",
            QualityPriority.QUALITY: "quality",
            QualityPriority.MAXIMUM: "ultimate",
        }
        result.preset = preset_map[result.quality_priority]

        # Scale factor based on resolution
        if self._analysis:
            if self._analysis.width < 720:
                default_scale = 4
            elif self._analysis.width < 1080:
                default_scale = 2
            else:
                default_scale = 2
        else:
            default_scale = 2

        if QUESTIONARY_AVAILABLE:
            # Scale factor
            scale_choices = ["2x (Double)", "4x (Quadruple)", "1x (No upscaling)"]
            scale_answer = questionary.select(
                "Upscaling factor?",
                choices=scale_choices,
                default=scale_choices[0] if default_scale == 2 else scale_choices[1],
                style=WIZARD_STYLE,
            ).ask()
            result.scale_factor = 4 if "4x" in scale_answer else (1 if "1x" in scale_answer else 2)

            # Face enhancement
            has_faces = self._analysis and self._analysis.content.face_percentage > 10
            result.enable_face_enhancement = questionary.confirm(
                "Enable face enhancement?",
                default=has_faces,
                style=WIZARD_STYLE,
            ).ask()

            # Frame interpolation
            low_fps = self._analysis and self._analysis.fps < 25
            result.enable_interpolation = questionary.confirm(
                "Enable frame interpolation (smoother motion)?",
                default=low_fps,
                style=WIZARD_STYLE,
            ).ask()

            if result.enable_interpolation:
                fps_choices = ["24 fps", "30 fps", "60 fps"]
                fps_answer = questionary.select(
                    "Target frame rate?",
                    choices=fps_choices,
                    default="30 fps",
                    style=WIZARD_STYLE,
                ).ask()
                result.target_fps = float(fps_answer.split()[0])

            # Missing frame generation (for damaged footage)
            has_damage = self._analysis and self._analysis.degradation.frame_damage_ratio > 0
            if has_damage or result.quality_priority == QualityPriority.MAXIMUM:
                result.enable_frame_generation = questionary.confirm(
                    "Generate missing/damaged frames? (AI reconstruction)",
                    default=has_damage,
                    style=WIZARD_STYLE,
                ).ask()

        else:
            # Fallback to simple prompts
            result.scale_factor = int(input(f"Scale factor [2/4, default={default_scale}]: ") or default_scale)
            result.enable_face_enhancement = input("Enable face enhancement? [Y/n]: ").lower() != 'n'
            result.enable_interpolation = input("Enable frame interpolation? [y/N]: ").lower() == 'y'
            if result.enable_interpolation:
                result.target_fps = float(input("Target FPS [30]: ") or 30)

    def _ask_colorization(self, result: WizardResult) -> None:
        """Ask about colorization options."""
        self.console.info("Black & white footage detected!")

        if QUESTIONARY_AVAILABLE:
            result.enable_colorization = questionary.confirm(
                "Would you like to colorize this video?",
                default=False,
                style=WIZARD_STYLE,
            ).ask()

            if result.enable_colorization:
                self.console.print(
                    "\n[dim]For best results, provide 1-5 reference color images\n"
                    "showing the same or similar subjects in color.[/dim]\n"
                )

                add_refs = questionary.confirm(
                    "Do you have reference color images?",
                    default=False,
                    style=WIZARD_STYLE,
                ).ask()

                if add_refs:
                    while True:
                        ref_path = questionary.path(
                            "Select reference image (or leave empty to finish):",
                            style=WIZARD_STYLE,
                        ).ask()

                        if not ref_path:
                            break

                        result.colorization_references.append(Path(ref_path))

                        if len(result.colorization_references) >= 5:
                            break

                if not result.colorization_references:
                    self.console.warning(
                        "No references provided - using automatic colorization "
                        "(results may vary)"
                    )
        else:
            colorize = input("Colorize B&W footage? [y/N]: ").lower() == 'y'
            result.enable_colorization = colorize

    def _ask_output_settings(self, result: WizardResult) -> None:
        """Ask about output settings."""
        # Generate default output path
        input_stem = result.input_path.stem
        default_output = result.input_path.parent / f"{input_stem}_restored.mp4"

        if QUESTIONARY_AVAILABLE:
            use_default = questionary.confirm(
                f"Save to {default_output.name}?",
                default=True,
                style=WIZARD_STYLE,
            ).ask()

            if not use_default:
                output = questionary.path(
                    "Select output location:",
                    style=WIZARD_STYLE,
                ).ask()
                result.output_path = Path(output) if output else default_output
            else:
                result.output_path = default_output
        else:
            result.output_path = default_output

    def _confirm_settings(self, result: WizardResult) -> bool:
        """Show summary and get confirmation."""
        # Build settings dict
        result.settings = self._build_settings(result)

        self.console.print()

        # Show restoration plan
        stages = self._get_stages_list(result)

        self.console.restoration_plan(
            preset=result.preset.title(),
            stages=stages,
            estimated_time="Depends on video length and hardware",
            quality_target=result.quality_priority.value.title(),
        )

        if RICH_AVAILABLE:
            table = Table(
                title="Settings Summary",
                box=ROUNDED,
                show_header=False,
            )
            table.add_column("Setting", style="dim")
            table.add_column("Value", style="cyan")

            table.add_row("Input", str(result.input_path.name))
            table.add_row("Output", str(result.output_path.name if result.output_path else "auto"))
            table.add_row("Scale", f"{result.scale_factor}x")
            table.add_row("Preset", result.preset)
            table.add_row("Face Enhancement", "Yes" if result.enable_face_enhancement else "No")
            table.add_row("Interpolation", f"{result.target_fps} fps" if result.enable_interpolation else "No")
            table.add_row("Colorization", "Yes" if result.enable_colorization else "No")
            if result.colorization_references:
                table.add_row("Color References", f"{len(result.colorization_references)} images")

            self.console.print(table)

        self.console.print()

        if QUESTIONARY_AVAILABLE:
            return questionary.confirm(
                "Start restoration with these settings?",
                default=True,
                style=WIZARD_STYLE,
            ).ask()
        else:
            return input("Start restoration? [Y/n]: ").lower() != 'n'

    def _build_settings(self, result: WizardResult) -> Dict[str, Any]:
        """Build complete settings dictionary."""
        settings = {
            "preset": result.preset,
            "scale_factor": result.scale_factor,
            "enable_interpolation": result.enable_interpolation,
            "target_fps": result.target_fps,
            "auto_face_restore": result.enable_face_enhancement,
            "enable_frame_generation": result.enable_frame_generation,
        }

        # Quality-specific settings
        if result.quality_priority == QualityPriority.MAXIMUM:
            settings.update({
                "enable_tap_denoise": True,
                "enable_qp_artifact_removal": True,
                "temporal_method": "hybrid",
                "sr_model": "diffusion" if result.scale_factor >= 2 else "realesrgan",
                "face_model": "aesrgan",
            })
        elif result.quality_priority == QualityPriority.QUALITY:
            settings.update({
                "enable_tap_denoise": True,
                "temporal_method": "optical_flow",
            })

        # Colorization
        if result.enable_colorization:
            if result.colorization_references:
                settings["colorization_reference_images"] = [
                    str(p) for p in result.colorization_references
                ]
            else:
                settings["enable_colorization"] = True

        # Archive-specific optimizations
        if self._analysis:
            if self._analysis.content.era in (Era.SILENT_ERA, Era.EARLY_SOUND):
                settings["enable_deduplication"] = True

            if self._analysis.degradation.compression_level > 0.3:
                settings["enable_qp_artifact_removal"] = True

        return settings

    def _get_stages_list(self, result: WizardResult) -> List[str]:
        """Get list of processing stages."""
        stages = []

        if result.settings.get("enable_qp_artifact_removal"):
            stages.append("QP Artifact Removal")

        if result.settings.get("enable_tap_denoise"):
            stages.append("TAP Neural Denoising")

        if result.enable_frame_generation:
            stages.append("Missing Frame Generation")

        sr_model = result.settings.get("sr_model", "realesrgan")
        stages.append(f"Super-Resolution ({sr_model.title()}) {result.scale_factor}x")

        if result.enable_face_enhancement:
            face_model = result.settings.get("face_model", "gfpgan")
            stages.append(f"Face Enhancement ({face_model.upper()})")

        if result.enable_interpolation:
            stages.append(f"Frame Interpolation (RIFE) → {result.target_fps}fps")

        if result.settings.get("temporal_method"):
            stages.append("Temporal Consistency")

        if result.enable_colorization:
            stages.append("AI Colorization")

        stages.append("Video Reassembly")

        return stages


def run_wizard(
    input_path: Optional[Path] = None,
    skip_analysis: bool = False,
) -> WizardResult:
    """Run the interactive wizard.

    Convenience function for running the wizard.

    Args:
        input_path: Optional pre-selected input path
        skip_analysis: Skip video analysis

    Returns:
        WizardResult with user's choices
    """
    wizard = InteractiveWizard()
    return wizard.run(input_path, skip_analysis)
