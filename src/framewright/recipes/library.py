"""Restoration recipe library with step-by-step guides."""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RecipeCategory(Enum):
    """Categories for restoration recipes."""
    FILM = "film"
    VIDEO = "video"
    ARCHIVE = "archive"
    BROADCAST = "broadcast"
    HOME_MOVIE = "home_movie"
    SURVEILLANCE = "surveillance"
    TUTORIAL = "tutorial"


@dataclass
class RecipeStep:
    """A single step in a restoration recipe."""
    order: int
    name: str
    description: str
    processor: str  # Processor to use
    config: Dict[str, Any]  # Configuration for the processor
    optional: bool = False
    skip_condition: Optional[str] = None  # Python expression to evaluate
    notes: str = ""


@dataclass
class Recipe:
    """A complete restoration recipe."""
    name: str
    title: str
    description: str
    category: RecipeCategory
    steps: List[RecipeStep]

    # Metadata
    author: str = "FrameWright"
    version: str = "1.0"
    difficulty: str = "intermediate"  # beginner, intermediate, advanced
    estimated_time: str = ""  # e.g., "2x realtime"
    vram_required: str = ""  # e.g., "8GB"

    # Requirements
    requires_gpu: bool = True
    recommended_preset: str = ""

    # Tags for searchability
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "steps": [
                {
                    "order": s.order,
                    "name": s.name,
                    "description": s.description,
                    "processor": s.processor,
                    "config": s.config,
                    "optional": s.optional,
                    "notes": s.notes,
                }
                for s in self.steps
            ],
            "author": self.author,
            "version": self.version,
            "difficulty": self.difficulty,
            "estimated_time": self.estimated_time,
            "vram_required": self.vram_required,
            "requires_gpu": self.requires_gpu,
            "recommended_preset": self.recommended_preset,
            "tags": self.tags,
        }


# Built-in recipes
BUILTIN_RECIPES: Dict[str, Recipe] = {
    "vhs_family_video": Recipe(
        name="vhs_family_video",
        title="VHS Family Video Restoration",
        description="Complete restoration workflow for VHS family recordings from the 80s and 90s.",
        category=RecipeCategory.HOME_MOVIE,
        difficulty="beginner",
        estimated_time="3x realtime",
        vram_required="8GB",
        recommended_preset="vhs_standard",
        tags=["vhs", "family", "home_video", "analog"],
        steps=[
            RecipeStep(
                order=1,
                name="Capture Assessment",
                description="Analyze the VHS capture for common issues",
                processor="video_analyzer",
                config={"detect_vhs_artifacts": True, "detect_tracking_issues": True},
                notes="Check for head switching noise, tracking errors, and color bleeding",
            ),
            RecipeStep(
                order=2,
                name="Deinterlacing",
                description="Convert interlaced video to progressive",
                processor="deinterlacer",
                config={"method": "yadif", "mode": "send_frame"},
                notes="VHS is typically interlaced at 59.94i",
            ),
            RecipeStep(
                order=3,
                name="VHS Artifact Removal",
                description="Remove VHS-specific artifacts",
                processor="vhs_restoration",
                config={
                    "head_switch_repair": True,
                    "tracking_repair": True,
                    "color_bleed_fix": True,
                    "dropout_repair": True,
                },
            ),
            RecipeStep(
                order=4,
                name="Noise Reduction",
                description="Remove tape noise and grain",
                processor="denoise",
                config={"method": "nafnet", "strength": 0.6},
                notes="VHS has significant noise - use moderate strength",
            ),
            RecipeStep(
                order=5,
                name="Color Correction",
                description="Fix faded colors and white balance",
                processor="color_correction",
                config={"auto_white_balance": True, "saturation_boost": 1.2},
                optional=True,
            ),
            RecipeStep(
                order=6,
                name="Upscaling",
                description="Upscale to HD resolution",
                processor="upscale",
                config={"model": "realesrgan", "scale": 4},
                notes="VHS is roughly 240-250 lines, upscale to 1080p",
            ),
            RecipeStep(
                order=7,
                name="Face Enhancement",
                description="Enhance faces for better clarity",
                processor="face_restore",
                config={"model": "gfpgan", "strength": 0.7},
                optional=True,
                notes="Use moderate strength to avoid plastic look",
            ),
            RecipeStep(
                order=8,
                name="Final Encoding",
                description="Encode to modern format",
                processor="encoder",
                config={"codec": "libx264", "crf": 18, "preset": "slow"},
            ),
        ],
    ),

    "silent_film_restoration": Recipe(
        name="silent_film_restoration",
        title="Silent Film Restoration",
        description="Careful restoration of silent era films (pre-1930) with authenticity preservation.",
        category=RecipeCategory.FILM,
        difficulty="advanced",
        estimated_time="5x realtime",
        vram_required="12GB",
        recommended_preset="film_silent",
        tags=["silent", "film", "vintage", "authentic", "grain"],
        steps=[
            RecipeStep(
                order=1,
                name="Film Analysis",
                description="Analyze film characteristics and defects",
                processor="video_analyzer",
                config={"detect_grain": True, "detect_scratches": True, "detect_flicker": True},
            ),
            RecipeStep(
                order=2,
                name="Stabilization",
                description="Reduce gate weave and jitter",
                processor="stabilizer",
                config={"method": "vidstab", "smoothing": 10, "crop": "auto"},
                notes="Silent films often have significant gate weave",
            ),
            RecipeStep(
                order=3,
                name="Scratch Removal",
                description="Remove vertical scratches and dust",
                processor="defect_repair",
                config={"scratch_removal": True, "dust_removal": True, "sensitivity": 0.8},
                notes="Be careful not to remove intentional film grain",
            ),
            RecipeStep(
                order=4,
                name="Flicker Reduction",
                description="Reduce brightness flicker",
                processor="deflicker",
                config={"method": "histogram", "strength": 0.7},
                notes="Early projectors had inconsistent lighting",
            ),
            RecipeStep(
                order=5,
                name="Gentle Denoising",
                description="Light noise reduction preserving grain",
                processor="denoise",
                config={"method": "tap", "preserve_grain": True, "strength": 0.3},
                notes="CRITICAL: Preserve film grain for authenticity",
            ),
            RecipeStep(
                order=6,
                name="Limited Upscaling",
                description="Modest upscale to preserve character",
                processor="upscale",
                config={"model": "realesrgan", "scale": 2},
                notes="Avoid excessive upscaling - 2x max for silent films",
            ),
            RecipeStep(
                order=7,
                name="Face Enhancement",
                description="Subtle face enhancement",
                processor="face_restore",
                config={"model": "codeformer", "fidelity": 0.9, "strength": 0.3},
                optional=True,
                notes="Very light touch - faces should look period-appropriate",
            ),
            RecipeStep(
                order=8,
                name="Grain Synthesis",
                description="Add back authentic film grain if needed",
                processor="grain_synthesis",
                config={"intensity": 0.1, "grain_type": "film"},
                optional=True,
                skip_condition="context.get('grain_preserved', False)",
            ),
            RecipeStep(
                order=9,
                name="Archive Encoding",
                description="High-quality archival encoding",
                processor="encoder",
                config={"codec": "libx265", "crf": 14, "preset": "veryslow"},
            ),
        ],
    ),

    "dvd_upscale": Recipe(
        name="dvd_upscale",
        title="DVD to HD Upscale",
        description="Upscale DVD-quality video (480p/576p) to 1080p HD.",
        category=RecipeCategory.VIDEO,
        difficulty="beginner",
        estimated_time="2x realtime",
        vram_required="6GB",
        recommended_preset="dvd_upscale",
        tags=["dvd", "upscale", "compression", "digital"],
        steps=[
            RecipeStep(
                order=1,
                name="Source Analysis",
                description="Detect interlacing and compression",
                processor="video_analyzer",
                config={"detect_interlace": True, "detect_compression": True},
            ),
            RecipeStep(
                order=2,
                name="Deinterlacing",
                description="Deinterlace if needed",
                processor="deinterlacer",
                config={"method": "yadif", "mode": "send_frame"},
                skip_condition="not context.get('is_interlaced', False)",
            ),
            RecipeStep(
                order=3,
                name="Artifact Removal",
                description="Remove MPEG-2 compression artifacts",
                processor="qp_artifact_removal",
                config={"auto_detect": True, "strength": 0.5},
            ),
            RecipeStep(
                order=4,
                name="Light Denoising",
                description="Remove compression noise",
                processor="denoise",
                config={"method": "nafnet", "strength": 0.3},
            ),
            RecipeStep(
                order=5,
                name="Upscaling",
                description="Upscale to 1080p",
                processor="upscale",
                config={"model": "realesrgan", "scale": 2},
            ),
            RecipeStep(
                order=6,
                name="Face Enhancement",
                description="Enhance faces",
                processor="face_restore",
                config={"model": "gfpgan", "strength": 0.5},
                optional=True,
            ),
            RecipeStep(
                order=7,
                name="Sharpening",
                description="Add subtle sharpening",
                processor="sharpen",
                config={"method": "unsharp", "amount": 0.5},
                optional=True,
            ),
            RecipeStep(
                order=8,
                name="Encoding",
                description="Encode to H.264",
                processor="encoder",
                config={"codec": "libx264", "crf": 18, "preset": "slow"},
            ),
        ],
    ),

    "youtube_cleanup": Recipe(
        name="youtube_cleanup",
        title="YouTube Video Cleanup",
        description="Clean up heavily compressed YouTube videos.",
        category=RecipeCategory.VIDEO,
        difficulty="beginner",
        estimated_time="1.5x realtime",
        vram_required="4GB",
        recommended_preset="youtube_cleanup",
        tags=["youtube", "compression", "web", "cleanup"],
        steps=[
            RecipeStep(
                order=1,
                name="Compression Analysis",
                description="Analyze compression level",
                processor="video_analyzer",
                config={"detect_compression": True},
            ),
            RecipeStep(
                order=2,
                name="Artifact Removal",
                description="Remove compression artifacts",
                processor="qp_artifact_removal",
                config={"auto_detect": True, "strength": 0.7},
            ),
            RecipeStep(
                order=3,
                name="Denoising",
                description="Remove compression noise",
                processor="denoise",
                config={"method": "nafnet", "strength": 0.4},
            ),
            RecipeStep(
                order=4,
                name="Encoding",
                description="Re-encode with better quality",
                processor="encoder",
                config={"codec": "libx264", "crf": 18, "preset": "slow"},
            ),
        ],
    ),

    "broadcast_archive": Recipe(
        name="broadcast_archive",
        title="Broadcast TV Archive",
        description="Restore and archive broadcast TV recordings.",
        category=RecipeCategory.BROADCAST,
        difficulty="intermediate",
        estimated_time="2.5x realtime",
        vram_required="8GB",
        tags=["broadcast", "tv", "archive", "interlaced"],
        steps=[
            RecipeStep(
                order=1,
                name="Source Analysis",
                description="Analyze broadcast characteristics",
                processor="video_analyzer",
                config={"detect_interlace": True, "detect_telecine": True},
            ),
            RecipeStep(
                order=2,
                name="Inverse Telecine",
                description="Remove 3:2 pulldown if present",
                processor="ivtc",
                config={"pattern": "auto"},
                skip_condition="not context.get('has_telecine', False)",
            ),
            RecipeStep(
                order=3,
                name="Deinterlacing",
                description="Convert to progressive",
                processor="deinterlacer",
                config={"method": "yadif", "mode": "send_frame"},
            ),
            RecipeStep(
                order=4,
                name="Logo Removal",
                description="Remove broadcast logo",
                processor="logo_removal",
                config={"detect_auto": True},
                optional=True,
            ),
            RecipeStep(
                order=5,
                name="Denoising",
                description="Remove broadcast noise",
                processor="denoise",
                config={"method": "nafnet", "strength": 0.5},
            ),
            RecipeStep(
                order=6,
                name="Upscaling",
                description="Upscale to 1080p",
                processor="upscale",
                config={"model": "realesrgan", "scale": 2},
            ),
            RecipeStep(
                order=7,
                name="Encoding",
                description="High-quality encoding",
                processor="encoder",
                config={"codec": "libx265", "crf": 16, "preset": "slower"},
            ),
        ],
    ),
}


class RecipeLibrary:
    """Library of restoration recipes."""

    def __init__(self, user_recipes_path: Optional[Path] = None):
        self._recipes: Dict[str, Recipe] = dict(BUILTIN_RECIPES)
        self._user_recipes_path = user_recipes_path or Path.home() / ".framewright" / "recipes.json"
        self._load_user_recipes()

    def _load_user_recipes(self) -> None:
        """Load user-defined recipes."""
        if not self._user_recipes_path.exists():
            return

        try:
            with open(self._user_recipes_path, "r") as f:
                data = json.load(f)

            for name, recipe_data in data.items():
                recipe = self._recipe_from_dict(recipe_data)
                if recipe:
                    self._recipes[name] = recipe

            logger.info(f"Loaded {len(data)} user recipes")
        except Exception as e:
            logger.warning(f"Failed to load user recipes: {e}")

    def _recipe_from_dict(self, data: Dict) -> Optional[Recipe]:
        """Create recipe from dictionary."""
        try:
            steps = [
                RecipeStep(
                    order=s["order"],
                    name=s["name"],
                    description=s["description"],
                    processor=s["processor"],
                    config=s.get("config", {}),
                    optional=s.get("optional", False),
                    skip_condition=s.get("skip_condition"),
                    notes=s.get("notes", ""),
                )
                for s in data.get("steps", [])
            ]

            return Recipe(
                name=data["name"],
                title=data["title"],
                description=data["description"],
                category=RecipeCategory(data.get("category", "video")),
                steps=steps,
                author=data.get("author", "User"),
                version=data.get("version", "1.0"),
                difficulty=data.get("difficulty", "intermediate"),
                estimated_time=data.get("estimated_time", ""),
                vram_required=data.get("vram_required", ""),
                requires_gpu=data.get("requires_gpu", True),
                recommended_preset=data.get("recommended_preset", ""),
                tags=data.get("tags", []),
            )
        except Exception as e:
            logger.error(f"Failed to parse recipe: {e}")
            return None

    def get(self, name: str) -> Optional[Recipe]:
        """Get a recipe by name."""
        return self._recipes.get(name)

    def list_all(self) -> List[Recipe]:
        """List all recipes."""
        return list(self._recipes.values())

    def list_by_category(self, category: RecipeCategory) -> List[Recipe]:
        """List recipes by category."""
        return [r for r in self._recipes.values() if r.category == category]

    def search(self, query: str) -> List[Recipe]:
        """Search recipes by name, title, or tags."""
        query = query.lower()
        return [
            r for r in self._recipes.values()
            if query in r.name.lower()
            or query in r.title.lower()
            or any(query in tag for tag in r.tags)
        ]

    def add_recipe(self, recipe: Recipe) -> None:
        """Add a user recipe."""
        self._recipes[recipe.name] = recipe
        self._save_user_recipes()

    def remove_recipe(self, name: str) -> bool:
        """Remove a user recipe."""
        if name in BUILTIN_RECIPES:
            logger.warning(f"Cannot remove built-in recipe: {name}")
            return False
        if name in self._recipes:
            del self._recipes[name]
            self._save_user_recipes()
            return True
        return False

    def _save_user_recipes(self) -> None:
        """Save user recipes to file."""
        user_recipes = {
            name: recipe.to_dict()
            for name, recipe in self._recipes.items()
            if name not in BUILTIN_RECIPES
        }

        self._user_recipes_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._user_recipes_path, "w") as f:
            json.dump(user_recipes, f, indent=2)

    def print_recipe(self, name: str) -> str:
        """Generate printable recipe guide."""
        recipe = self.get(name)
        if not recipe:
            return f"Recipe not found: {name}"

        lines = [
            "=" * 60,
            recipe.title,
            "=" * 60,
            "",
            f"Category: {recipe.category.value}",
            f"Difficulty: {recipe.difficulty}",
            f"Estimated Time: {recipe.estimated_time}",
            f"VRAM Required: {recipe.vram_required}",
            "",
            "DESCRIPTION",
            "-" * 40,
            recipe.description,
            "",
            "STEPS",
            "-" * 40,
        ]

        for step in recipe.steps:
            optional = " (optional)" if step.optional else ""
            lines.append(f"\n{step.order}. {step.name}{optional}")
            lines.append(f"   {step.description}")
            if step.notes:
                lines.append(f"   Note: {step.notes}")

        lines.extend(["", "=" * 60])

        return "\n".join(lines)
