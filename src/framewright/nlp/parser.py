"""Natural language command parser for video restoration.

Parses plain English descriptions into restoration settings.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CommandIntent(Enum):
    """User command intent types."""
    RESTORE = "restore"
    UPSCALE = "upscale"
    DENOISE = "denoise"
    COLORIZE = "colorize"
    STABILIZE = "stabilize"
    ENHANCE = "enhance"
    FIX = "fix"
    REMOVE = "remove"
    INTERPOLATE = "interpolate"
    ANALYZE = "analyze"
    COMPARE = "compare"
    PREVIEW = "preview"
    BATCH = "batch"
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass
class ParsedCommand:
    """Parsed command from natural language input."""
    raw_input: str
    intent: CommandIntent = CommandIntent.UNKNOWN
    confidence: float = 0.0

    # Input/Output
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None

    # Quality preferences
    quality_preset: str = "balanced"  # draft, fast, balanced, quality, ultimate
    target_quality: Optional[str] = None  # "best", "good enough", "fast"

    # Resolution
    scale_factor: Optional[float] = None
    target_resolution: Optional[Tuple[int, int]] = None
    target_resolution_name: Optional[str] = None  # "4K", "HD", "1080p"

    # Frame rate
    target_fps: Optional[float] = None
    fps_multiplier: Optional[float] = None

    # Format/Era
    source_era: Optional[str] = None  # "1920s", "silent", "VHS", etc.
    source_format: Optional[str] = None  # "film", "VHS", "digital"

    # Specific fixes
    fix_issues: List[str] = field(default_factory=list)  # "scratches", "grain", "flicker"
    preserve_aspects: List[str] = field(default_factory=list)  # "grain", "color", "aesthetic"

    # Processing preferences
    use_ai: bool = True
    preserve_authenticity: bool = True
    aggressive_processing: bool = False

    # Additional settings
    settings: Dict[str, Any] = field(default_factory=dict)

    # Explanation for user
    explanation: str = ""
    suggestions: List[str] = field(default_factory=list)


class NLPCommandParser:
    """Parse natural language commands for video restoration."""

    def __init__(self):
        self._patterns = self._build_patterns()

    def _build_patterns(self) -> Dict[str, List[Tuple[str, float]]]:
        """Build regex patterns for intent detection."""
        return {
            "restore": [
                (r"\brestore\b", 1.0),
                (r"\bfix\s+(up|this|my|the)\b", 0.9),
                (r"\brepair\b", 0.9),
                (r"\brescue\b", 0.8),
                (r"\bbring\s+back\b", 0.8),
                (r"\brevive\b", 0.7),
                (r"\bremaster\b", 0.9),
                (r"\bclean\s+up\b", 0.8),
            ],
            "upscale": [
                (r"\bupscale\b", 1.0),
                (r"\bincrease\s+(resolution|res)\b", 0.9),
                (r"\b(make|get)\s+(it\s+)?(bigger|larger|higher\s+res)\b", 0.8),
                (r"\b4k\b", 0.7),
                (r"\bhd\b", 0.6),
                (r"\bsuper\s*res(olution)?\b", 0.9),
                (r"\benhance\s+(resolution|detail)\b", 0.8),
            ],
            "denoise": [
                (r"\bdenoise\b", 1.0),
                (r"\bremove\s+noise\b", 1.0),
                (r"\bnoise\s+reduct(ion)?\b", 0.9),
                (r"\bclean\b", 0.7),
                (r"\bremove\s+grain\b", 0.8),
                (r"\breduce\s+grain\b", 0.7),
                (r"\bgrainy\b", 0.6),
                (r"\bnoisy\b", 0.6),
            ],
            "colorize": [
                (r"\bcoloriz(e|ation)\b", 1.0),
                (r"\badd\s+color\b", 1.0),
                (r"\bcolor\s+(this|it|the)\b", 0.9),
                (r"\bblack\s+and\s+white\s+to\s+color\b", 1.0),
                (r"\bb&?w\s+to\s+color\b", 0.9),
                (r"\bconvert\s+to\s+color\b", 0.9),
            ],
            "stabilize": [
                (r"\bstabiliz(e|ation)\b", 1.0),
                (r"\bremove\s+shake\b", 1.0),
                (r"\bfix\s+shak(y|ing)\b", 0.9),
                (r"\bsteady\b", 0.8),
                (r"\bsmooth\s+(out\s+)?(motion|camera)\b", 0.8),
                (r"\bjitter\b", 0.7),
            ],
            "enhance": [
                (r"\benhance\b", 1.0),
                (r"\bimprove\b", 0.9),
                (r"\bmake\s+(it\s+)?(look\s+)?better\b", 0.8),
                (r"\bbeautify\b", 0.7),
                (r"\bpolish\b", 0.7),
            ],
            "fix": [
                (r"\bfix\b", 0.8),
                (r"\brepair\b", 0.8),
                (r"\bcorrect\b", 0.7),
                (r"\bremove\s+(scratches|damage|artifacts)\b", 0.9),
            ],
            "interpolate": [
                (r"\binterpolat(e|ion)\b", 1.0),
                (r"\bframe\s+(rate\s+)?(increase|boost|up)\b", 0.9),
                (r"\bsmooth\s+(motion|video)\b", 0.8),
                (r"\b(60|120)\s*fps\b", 0.7),
                (r"\bslow\s*mo(tion)?\b", 0.6),
            ],
            "analyze": [
                (r"\banalyz(e|is)\b", 1.0),
                (r"\bcheck\s+quality\b", 0.9),
                (r"\bassess\b", 0.8),
                (r"\bevaluat(e|ion)\b", 0.8),
                (r"\bwhat('s| is)\s+(wrong|the\s+issue)\b", 0.7),
            ],
        }

    def parse(self, text: str) -> ParsedCommand:
        """Parse natural language text into a command."""
        text_lower = text.lower().strip()

        command = ParsedCommand(raw_input=text)

        # Detect intent
        command.intent, command.confidence = self._detect_intent(text_lower)

        # Extract paths
        command.input_path, command.output_path = self._extract_paths(text)

        # Extract quality preferences
        command.quality_preset = self._extract_quality_preset(text_lower)
        command.target_quality = self._extract_target_quality(text_lower)

        # Extract resolution
        command.scale_factor = self._extract_scale_factor(text_lower)
        command.target_resolution_name = self._extract_resolution_name(text_lower)
        command.target_resolution = self._resolution_name_to_size(command.target_resolution_name)

        # Extract frame rate
        command.target_fps = self._extract_fps(text_lower)

        # Extract era/format
        command.source_era = self._extract_era(text_lower)
        command.source_format = self._extract_format(text_lower)

        # Extract issues to fix
        command.fix_issues = self._extract_issues(text_lower)

        # Extract preservation preferences
        command.preserve_aspects = self._extract_preserve(text_lower)

        # Processing preferences
        command.use_ai = self._should_use_ai(text_lower)
        command.preserve_authenticity = self._should_preserve_authenticity(text_lower)
        command.aggressive_processing = self._is_aggressive(text_lower)

        # Generate explanation
        command.explanation = self._generate_explanation(command)
        command.suggestions = self._generate_suggestions(command)

        return command

    def _detect_intent(self, text: str) -> Tuple[CommandIntent, float]:
        """Detect the primary intent from text."""
        best_intent = CommandIntent.UNKNOWN
        best_score = 0.0

        for intent_name, patterns in self._patterns.items():
            for pattern, weight in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score = weight
                    if score > best_score:
                        best_score = score
                        best_intent = CommandIntent(intent_name)

        # Default to restore if no clear intent
        if best_intent == CommandIntent.UNKNOWN and best_score < 0.5:
            # Check if it mentions video/file at all
            if re.search(r"\b(video|film|footage|movie|clip)\b", text):
                best_intent = CommandIntent.RESTORE
                best_score = 0.5

        return best_intent, best_score

    def _extract_paths(self, text: str) -> Tuple[Optional[Path], Optional[Path]]:
        """Extract file paths from text."""
        # Look for quoted paths
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        paths = [Path(p) for p in quoted if self._looks_like_path(p)]

        # Look for common video extensions
        ext_pattern = r'(\S+\.(?:mp4|mkv|avi|mov|webm|m4v|wmv|flv))'
        for match in re.finditer(ext_pattern, text, re.IGNORECASE):
            path = Path(match.group(1).strip('"\''))
            if path not in paths:
                paths.append(path)

        input_path = paths[0] if len(paths) > 0 else None
        output_path = paths[1] if len(paths) > 1 else None

        return input_path, output_path

    def _looks_like_path(self, text: str) -> bool:
        """Check if text looks like a file path."""
        return any([
            '/' in text,
            '\\' in text,
            text.endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm')),
            text.startswith(('C:', 'D:', '~', './', '../')),
        ])

    def _extract_quality_preset(self, text: str) -> str:
        """Extract quality preset from text."""
        if re.search(r"\b(draft|preview|quick\s+look)\b", text):
            return "draft"
        elif re.search(r"\b(fast|quick|speed)\b", text):
            return "fast"
        elif re.search(r"\b(best|ultimate|maximum|highest|premium)\b", text):
            return "ultimate"
        elif re.search(r"\b(quality|high|good)\b", text):
            return "quality"
        return "balanced"

    def _extract_target_quality(self, text: str) -> Optional[str]:
        """Extract target quality description."""
        if re.search(r"\b(best|perfect|pristine|flawless)\b", text):
            return "best"
        elif re.search(r"\b(good\s+enough|acceptable|decent)\b", text):
            return "good_enough"
        elif re.search(r"\b(as\s+fast\s+as|quick|speedy)\b", text):
            return "fast"
        return None

    def _extract_scale_factor(self, text: str) -> Optional[float]:
        """Extract upscaling factor."""
        # Look for explicit "Nx" or "x times"
        match = re.search(r"\b(\d+)\s*[xX]\b", text)
        if match:
            return float(match.group(1))

        match = re.search(r"\b(\d+)\s+times\b", text)
        if match:
            return float(match.group(1))

        # Resolution-based inference
        if re.search(r"\b4k\b", text, re.IGNORECASE):
            return 4.0
        elif re.search(r"\b(2k|1440p)\b", text, re.IGNORECASE):
            return 2.0
        elif re.search(r"\b(1080p|full\s*hd)\b", text, re.IGNORECASE):
            return 2.0

        return None

    def _extract_resolution_name(self, text: str) -> Optional[str]:
        """Extract target resolution name."""
        if re.search(r"\b(8k|7680\s*x\s*4320)\b", text, re.IGNORECASE):
            return "8K"
        elif re.search(r"\b(4k|uhd|2160p|3840\s*x\s*2160)\b", text, re.IGNORECASE):
            return "4K"
        elif re.search(r"\b(2k|1440p|2560\s*x\s*1440)\b", text, re.IGNORECASE):
            return "2K"
        elif re.search(r"\b(1080p|full\s*hd|1920\s*x\s*1080)\b", text, re.IGNORECASE):
            return "1080p"
        elif re.search(r"\b(720p|hd)\b", text, re.IGNORECASE):
            return "720p"
        return None

    def _resolution_name_to_size(self, name: Optional[str]) -> Optional[Tuple[int, int]]:
        """Convert resolution name to pixel dimensions."""
        resolutions = {
            "8K": (7680, 4320),
            "4K": (3840, 2160),
            "2K": (2560, 1440),
            "1080p": (1920, 1080),
            "720p": (1280, 720),
            "480p": (854, 480),
        }
        return resolutions.get(name)

    def _extract_fps(self, text: str) -> Optional[float]:
        """Extract target frame rate."""
        match = re.search(r"(\d+(?:\.\d+)?)\s*fps\b", text, re.IGNORECASE)
        if match:
            return float(match.group(1))

        if re.search(r"\b60\s*(?:frames|hertz|hz)\b", text, re.IGNORECASE):
            return 60.0
        elif re.search(r"\b120\s*(?:frames|hertz|hz)\b", text, re.IGNORECASE):
            return 120.0

        return None

    def _extract_era(self, text: str) -> Optional[str]:
        """Extract source era from text."""
        # Decades
        match = re.search(r"\b(19\d0)s?\b", text)
        if match:
            return match.group(1) + "s"

        match = re.search(r"\b(20[012]\d)s?\b", text)
        if match:
            return match.group(1) + "s"

        # Era names
        if re.search(r"\bsilent\s+(film|era|movie)\b", text, re.IGNORECASE):
            return "silent"
        elif re.search(r"\b(golden\s+age|classic\s+hollywood)\b", text, re.IGNORECASE):
            return "golden_age"
        elif re.search(r"\b(early\s+color|technicolor)\b", text, re.IGNORECASE):
            return "early_color"
        elif re.search(r"\b(home\s+video|camcorder)\b", text, re.IGNORECASE):
            return "home_video"

        return None

    def _extract_format(self, text: str) -> Optional[str]:
        """Extract source format from text."""
        formats = {
            r"\bvhs\b": "vhs",
            r"\bbeta(max)?\b": "betamax",
            r"\bhi-?8\b": "hi8",
            r"\bsuper\s*8\b": "super8",
            r"\b16\s*mm\b": "16mm",
            r"\b35\s*mm\b": "35mm",
            r"\b8\s*mm\b": "8mm",
            r"\blaser\s*disc\b": "laserdisc",
            r"\bdvd\b": "dvd",
            r"\bdigital\b": "digital",
            r"\bfilm\b": "film",
            r"\bnitrate\b": "nitrate",
        }

        for pattern, format_name in formats.items():
            if re.search(pattern, text, re.IGNORECASE):
                return format_name

        return None

    def _extract_issues(self, text: str) -> List[str]:
        """Extract issues to fix from text."""
        issues = []

        issue_patterns = {
            r"\bscratches?\b": "scratches",
            r"\bgrain(y)?\b": "grain",
            r"\bnois(e|y)\b": "noise",
            r"\bflicker(ing)?\b": "flicker",
            r"\bshak(e|y|ing)\b": "shake",
            r"\bblur(ry)?\b": "blur",
            r"\bfade[ds]?\b": "fading",
            r"\bdamage[ds]?\b": "damage",
            r"\bartifacts?\b": "artifacts",
            r"\bdust\b": "dust",
            r"\bspots?\b": "spots",
            r"\btears?\b": "tears",
            r"\bdropout\b": "dropout",
            r"\btracking\b": "tracking",
            r"\bjitter\b": "jitter",
            r"\bcompression\b": "compression",
            r"\bblocking\b": "blocking",
            r"\bcolor\s+bleed\b": "color_bleed",
            r"\binterlac(ed|ing)\b": "interlacing",
        }

        for pattern, issue in issue_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(issue)

        return issues

    def _extract_preserve(self, text: str) -> List[str]:
        """Extract aspects to preserve from text."""
        preserve = []

        if re.search(r"\b(keep|preserve|maintain)\s+(the\s+)?grain\b", text, re.IGNORECASE):
            preserve.append("grain")

        if re.search(r"\b(keep|preserve|maintain)\s+(the\s+)?color\b", text, re.IGNORECASE):
            preserve.append("color")

        if re.search(r"\b(authentic|original|period|vintage)\b", text, re.IGNORECASE):
            preserve.append("authenticity")

        if re.search(r"\b(look|feel|aesthetic|character)\b", text, re.IGNORECASE):
            preserve.append("aesthetic")

        if re.search(r"\bdon'?t\s+(make|want)\s+it\s+look\s+(too\s+)?(modern|new|polished)\b", text, re.IGNORECASE):
            preserve.append("authenticity")

        return preserve

    def _should_use_ai(self, text: str) -> bool:
        """Determine if AI processing should be used."""
        if re.search(r"\b(no\s+ai|without\s+ai|non-ai|traditional)\b", text, re.IGNORECASE):
            return False
        return True

    def _should_preserve_authenticity(self, text: str) -> bool:
        """Determine if authenticity should be preserved."""
        if re.search(r"\b(modern|polished|pristine|flawless|perfect)\b", text, re.IGNORECASE):
            return False

        if re.search(r"\b(authentic|original|period|vintage|era|character)\b", text, re.IGNORECASE):
            return True

        # Default to preserve for old content
        if re.search(r"\b(old|vintage|classic|historic|archiv)\b", text, re.IGNORECASE):
            return True

        return True

    def _is_aggressive(self, text: str) -> bool:
        """Determine if aggressive processing is requested."""
        if re.search(r"\b(aggressive|maximum|extreme|heavy|strong)\b", text, re.IGNORECASE):
            return True
        if re.search(r"\b(subtle|gentle|light|mild|careful)\b", text, re.IGNORECASE):
            return False
        return False

    def _generate_explanation(self, command: ParsedCommand) -> str:
        """Generate human-readable explanation of parsed command."""
        parts = []

        intent_desc = {
            CommandIntent.RESTORE: "restore",
            CommandIntent.UPSCALE: "upscale",
            CommandIntent.DENOISE: "denoise",
            CommandIntent.COLORIZE: "colorize",
            CommandIntent.STABILIZE: "stabilize",
            CommandIntent.ENHANCE: "enhance",
            CommandIntent.INTERPOLATE: "interpolate frames for",
            CommandIntent.ANALYZE: "analyze",
        }

        parts.append(f"I'll {intent_desc.get(command.intent, 'process')} your video")

        if command.target_resolution_name:
            parts.append(f"to {command.target_resolution_name}")

        if command.target_fps:
            parts.append(f"at {command.target_fps:.0f}fps")

        if command.quality_preset == "ultimate":
            parts.append("with maximum quality settings")
        elif command.quality_preset == "draft":
            parts.append("as a quick preview")

        if command.preserve_authenticity:
            parts.append("while preserving authentic character")

        if command.fix_issues:
            parts.append(f"fixing {', '.join(command.fix_issues[:3])}")

        return " ".join(parts) + "."

    def _generate_suggestions(self, command: ParsedCommand) -> List[str]:
        """Generate helpful suggestions for the user."""
        suggestions = []

        if command.source_era and not command.preserve_authenticity:
            suggestions.append(
                f"Since this is {command.source_era} footage, consider enabling "
                "authenticity preservation to maintain period character."
            )

        if command.intent == CommandIntent.UPSCALE and not command.scale_factor:
            suggestions.append(
                "Tip: You can specify '2x' or '4x' for upscaling, "
                "or target resolution like '4K' or '1080p'."
            )

        if "grain" in command.fix_issues and command.preserve_authenticity:
            suggestions.append(
                "Note: Preserving authenticity will keep some film grain. "
                "For aggressive grain removal, try 'remove all grain'."
            )

        return suggestions
