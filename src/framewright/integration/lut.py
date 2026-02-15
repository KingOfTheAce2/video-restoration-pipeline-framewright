"""LUT (Look-Up Table) support for color grading workflows.

Supports 1D LUT, 3D LUT (.cube, .3dl), and generation of analysis LUTs.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import struct

logger = logging.getLogger(__name__)


class LUTFormat(Enum):
    """Supported LUT formats."""
    CUBE = "cube"      # Adobe/Resolve .cube
    THREEDL = "3dl"    # Autodesk .3dl
    CSP = "csp"        # Cinespace
    ICC = "icc"        # ICC color profile
    CLF = "clf"        # Common LUT Format (ACES)


class LUTType(Enum):
    """LUT dimensionality."""
    LUT_1D = "1d"
    LUT_3D = "3d"


@dataclass
class LUT:
    """Represents a Look-Up Table."""
    name: str = "Untitled"
    lut_type: LUTType = LUTType.LUT_3D
    size: int = 33  # Grid size (17, 33, 65 common)

    # Domain (input range)
    domain_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    domain_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Data
    data_1d: Optional[List[Tuple[float, float, float]]] = None
    data_3d: Optional[List[List[List[Tuple[float, float, float]]]]] = None

    # Metadata
    title: str = ""
    comments: List[str] = field(default_factory=list)

    def apply_to_rgb(self, r: float, g: float, b: float) -> Tuple[float, float, float]:
        """Apply LUT to normalized RGB values (0-1)."""
        # Clamp to domain
        r = max(self.domain_min[0], min(self.domain_max[0], r))
        g = max(self.domain_min[1], min(self.domain_max[1], g))
        b = max(self.domain_min[2], min(self.domain_max[2], b))

        # Normalize to 0-1 within domain
        r_norm = (r - self.domain_min[0]) / (self.domain_max[0] - self.domain_min[0])
        g_norm = (g - self.domain_min[1]) / (self.domain_max[1] - self.domain_min[1])
        b_norm = (b - self.domain_min[2]) / (self.domain_max[2] - self.domain_min[2])

        if self.lut_type == LUTType.LUT_1D and self.data_1d:
            return self._apply_1d(r_norm, g_norm, b_norm)
        elif self.lut_type == LUTType.LUT_3D and self.data_3d:
            return self._apply_3d(r_norm, g_norm, b_norm)
        else:
            return (r, g, b)

    def _apply_1d(self, r: float, g: float, b: float) -> Tuple[float, float, float]:
        """Apply 1D LUT with linear interpolation."""
        if not self.data_1d:
            return (r, g, b)

        size = len(self.data_1d)

        def interpolate_1d(val: float, channel: int) -> float:
            idx = val * (size - 1)
            idx_low = int(idx)
            idx_high = min(idx_low + 1, size - 1)
            frac = idx - idx_low
            return (1 - frac) * self.data_1d[idx_low][channel] + frac * self.data_1d[idx_high][channel]

        return (
            interpolate_1d(r, 0),
            interpolate_1d(g, 1),
            interpolate_1d(b, 2)
        )

    def _apply_3d(self, r: float, g: float, b: float) -> Tuple[float, float, float]:
        """Apply 3D LUT with trilinear interpolation."""
        if not self.data_3d:
            return (r, g, b)

        size = self.size

        # Get indices and fractions
        r_idx = r * (size - 1)
        g_idx = g * (size - 1)
        b_idx = b * (size - 1)

        r0 = int(r_idx)
        g0 = int(g_idx)
        b0 = int(b_idx)

        r1 = min(r0 + 1, size - 1)
        g1 = min(g0 + 1, size - 1)
        b1 = min(b0 + 1, size - 1)

        r_frac = r_idx - r0
        g_frac = g_idx - g0
        b_frac = b_idx - b0

        # Trilinear interpolation
        def trilinear(c: int) -> float:
            c000 = self.data_3d[r0][g0][b0][c]
            c001 = self.data_3d[r0][g0][b1][c]
            c010 = self.data_3d[r0][g1][b0][c]
            c011 = self.data_3d[r0][g1][b1][c]
            c100 = self.data_3d[r1][g0][b0][c]
            c101 = self.data_3d[r1][g0][b1][c]
            c110 = self.data_3d[r1][g1][b0][c]
            c111 = self.data_3d[r1][g1][b1][c]

            c00 = c000 * (1 - r_frac) + c100 * r_frac
            c01 = c001 * (1 - r_frac) + c101 * r_frac
            c10 = c010 * (1 - r_frac) + c110 * r_frac
            c11 = c011 * (1 - r_frac) + c111 * r_frac

            c0 = c00 * (1 - g_frac) + c10 * g_frac
            c1 = c01 * (1 - g_frac) + c11 * g_frac

            return c0 * (1 - b_frac) + c1 * b_frac

        return (trilinear(0), trilinear(1), trilinear(2))


class LUTParser:
    """Parse LUT files."""

    def parse(self, path: Path) -> LUT:
        """Parse LUT file, auto-detecting format."""
        suffix = path.suffix.lower()

        if suffix == ".cube":
            return self._parse_cube(path)
        elif suffix == ".3dl":
            return self._parse_3dl(path)
        elif suffix == ".csp":
            return self._parse_csp(path)
        else:
            raise ValueError(f"Unsupported LUT format: {suffix}")

    def _parse_cube(self, path: Path) -> LUT:
        """Parse .cube format (Resolve/Adobe)."""
        lut = LUT(name=path.stem)
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")

        data_lines = []
        size_1d = None
        size_3d = None

        for line in lines:
            line = line.strip()

            if not line or line.startswith("#"):
                if line.startswith("#"):
                    lut.comments.append(line[1:].strip())
                continue

            # Parse header
            if line.startswith("TITLE"):
                match = re.match(r'TITLE\s+"?([^"]+)"?', line)
                if match:
                    lut.title = match.group(1)
                continue

            if line.startswith("LUT_1D_SIZE"):
                match = re.match(r"LUT_1D_SIZE\s+(\d+)", line)
                if match:
                    size_1d = int(match.group(1))
                    lut.lut_type = LUTType.LUT_1D
                    lut.size = size_1d
                continue

            if line.startswith("LUT_3D_SIZE"):
                match = re.match(r"LUT_3D_SIZE\s+(\d+)", line)
                if match:
                    size_3d = int(match.group(1))
                    lut.lut_type = LUTType.LUT_3D
                    lut.size = size_3d
                continue

            if line.startswith("DOMAIN_MIN"):
                values = line.split()[1:4]
                lut.domain_min = tuple(float(v) for v in values)
                continue

            if line.startswith("DOMAIN_MAX"):
                values = line.split()[1:4]
                lut.domain_max = tuple(float(v) for v in values)
                continue

            # Data line
            parts = line.split()
            if len(parts) >= 3:
                try:
                    r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
                    data_lines.append((r, g, b))
                except ValueError:
                    pass

        # Convert data to appropriate structure
        if lut.lut_type == LUTType.LUT_1D:
            lut.data_1d = data_lines
        else:
            # Convert flat list to 3D array
            size = lut.size
            lut.data_3d = [[[None for _ in range(size)] for _ in range(size)] for _ in range(size)]

            idx = 0
            for b in range(size):
                for g in range(size):
                    for r in range(size):
                        if idx < len(data_lines):
                            lut.data_3d[r][g][b] = data_lines[idx]
                            idx += 1
                        else:
                            lut.data_3d[r][g][b] = (
                                r / (size - 1),
                                g / (size - 1),
                                b / (size - 1)
                            )

        return lut

    def _parse_3dl(self, path: Path) -> LUT:
        """Parse .3dl format (Autodesk)."""
        lut = LUT(name=path.stem, lut_type=LUTType.LUT_3D)
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")

        data_lines = []
        mesh_line = None

        for line in lines:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            # First non-comment line with single numbers defines mesh
            parts = line.split()

            if mesh_line is None and len(parts) <= 4:
                # This might be the mesh definition
                try:
                    mesh_line = [int(p) for p in parts]
                    if len(mesh_line) >= 1:
                        # Determine size from mesh
                        lut.size = mesh_line[0] + 1 if mesh_line[0] < 100 else 33
                    continue
                except ValueError:
                    pass

            # Data line (R G B as integers 0-4095 typically)
            if len(parts) >= 3:
                try:
                    r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                    # Normalize to 0-1
                    max_val = 4095
                    data_lines.append((r / max_val, g / max_val, b / max_val))
                except ValueError:
                    pass

        # Convert to 3D array
        size = lut.size
        lut.data_3d = [[[None for _ in range(size)] for _ in range(size)] for _ in range(size)]

        idx = 0
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    if idx < len(data_lines):
                        lut.data_3d[r][g][b] = data_lines[idx]
                        idx += 1
                    else:
                        lut.data_3d[r][g][b] = (
                            r / (size - 1),
                            g / (size - 1),
                            b / (size - 1)
                        )

        return lut

    def _parse_csp(self, path: Path) -> LUT:
        """Parse .csp format (Cinespace)."""
        # CSP format is similar to cube
        return self._parse_cube(path)


class LUTWriter:
    """Write LUT files."""

    def write(self, lut: LUT, path: Path, format: Optional[LUTFormat] = None) -> None:
        """Write LUT to file."""
        format = format or LUTFormat.CUBE

        if format == LUTFormat.CUBE:
            content = self._write_cube(lut)
        elif format == LUTFormat.THREEDL:
            content = self._write_3dl(lut)
        else:
            raise ValueError(f"Unsupported format: {format}")

        path.write_text(content, encoding="utf-8")
        logger.info(f"Written LUT to {path}")

    def _write_cube(self, lut: LUT) -> str:
        """Write .cube format."""
        lines = []

        # Header
        if lut.title:
            lines.append(f'TITLE "{lut.title}"')
        else:
            lines.append(f'TITLE "{lut.name}"')

        for comment in lut.comments:
            lines.append(f"# {comment}")

        lines.append("")

        if lut.domain_min != (0.0, 0.0, 0.0):
            lines.append(f"DOMAIN_MIN {lut.domain_min[0]} {lut.domain_min[1]} {lut.domain_min[2]}")

        if lut.domain_max != (1.0, 1.0, 1.0):
            lines.append(f"DOMAIN_MAX {lut.domain_max[0]} {lut.domain_max[1]} {lut.domain_max[2]}")

        if lut.lut_type == LUTType.LUT_1D:
            lines.append(f"LUT_1D_SIZE {len(lut.data_1d) if lut.data_1d else lut.size}")
            lines.append("")

            if lut.data_1d:
                for r, g, b in lut.data_1d:
                    lines.append(f"{r:.10f} {g:.10f} {b:.10f}")
        else:
            lines.append(f"LUT_3D_SIZE {lut.size}")
            lines.append("")

            if lut.data_3d:
                size = lut.size
                for b in range(size):
                    for g in range(size):
                        for r in range(size):
                            rgb = lut.data_3d[r][g][b]
                            lines.append(f"{rgb[0]:.10f} {rgb[1]:.10f} {rgb[2]:.10f}")

        return "\n".join(lines)

    def _write_3dl(self, lut: LUT) -> str:
        """Write .3dl format."""
        lines = []

        size = lut.size

        # Mesh definition
        lines.append(" ".join(str(i * (4095 // (size - 1))) for i in range(size)))
        lines.append("")

        if lut.data_3d:
            for b in range(size):
                for g in range(size):
                    for r in range(size):
                        rgb = lut.data_3d[r][g][b]
                        # Convert to 12-bit values
                        r_int = int(rgb[0] * 4095)
                        g_int = int(rgb[1] * 4095)
                        b_int = int(rgb[2] * 4095)
                        lines.append(f"{r_int} {g_int} {b_int}")

        return "\n".join(lines)


class LUTManager:
    """Manage LUT loading, caching, and application."""

    def __init__(self):
        self.parser = LUTParser()
        self.writer = LUTWriter()
        self._cache: Dict[str, LUT] = {}
        self._numpy_available = self._check_numpy()

    def _check_numpy(self) -> bool:
        """Check if numpy is available."""
        try:
            import numpy as np
            return True
        except ImportError:
            return False

    def load(self, path: Path) -> LUT:
        """Load a LUT file with caching."""
        cache_key = str(path.resolve())

        if cache_key in self._cache:
            return self._cache[cache_key]

        lut = self.parser.parse(path)
        self._cache[cache_key] = lut
        return lut

    def save(self, lut: LUT, path: Path, format: Optional[LUTFormat] = None) -> None:
        """Save a LUT to file."""
        self.writer.write(lut, path, format)

    def clear_cache(self) -> None:
        """Clear the LUT cache."""
        self._cache.clear()

    def apply_to_image(self, image: Any, lut: LUT) -> Any:
        """Apply LUT to numpy image array."""
        if not self._numpy_available:
            return image

        import numpy as np

        if image is None or not isinstance(image, np.ndarray):
            return image

        # Normalize to 0-1
        if image.dtype == np.uint8:
            normalized = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            normalized = image.astype(np.float32) / 65535.0
        else:
            normalized = image.astype(np.float32)

        result = np.zeros_like(normalized)
        height, width = image.shape[:2]

        # Apply LUT to each pixel
        for y in range(height):
            for x in range(width):
                if len(image.shape) == 3:
                    r, g, b = normalized[y, x, 2], normalized[y, x, 1], normalized[y, x, 0]
                    r_out, g_out, b_out = lut.apply_to_rgb(r, g, b)
                    result[y, x] = [b_out, g_out, r_out]
                else:
                    val = normalized[y, x]
                    out, _, _ = lut.apply_to_rgb(val, val, val)
                    result[y, x] = out

        # Convert back to original dtype
        if image.dtype == np.uint8:
            return (np.clip(result, 0, 1) * 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            return (np.clip(result, 0, 1) * 65535).astype(np.uint16)
        else:
            return result

    def apply_to_image_fast(self, image: Any, lut: LUT) -> Any:
        """Apply LUT to image using vectorized operations (faster)."""
        if not self._numpy_available:
            return image

        import numpy as np

        if lut.lut_type != LUTType.LUT_3D or lut.data_3d is None:
            return self.apply_to_image(image, lut)

        # Convert LUT to 3D numpy array for indexing
        size = lut.size
        lut_array = np.zeros((size, size, size, 3), dtype=np.float32)

        for r in range(size):
            for g in range(size):
                for b in range(size):
                    lut_array[r, g, b] = lut.data_3d[r][g][b]

        # Normalize image
        if image.dtype == np.uint8:
            normalized = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            normalized = image.astype(np.float32) / 65535.0
        else:
            normalized = image.astype(np.float32)

        # Scale to LUT indices
        scaled = normalized * (size - 1)

        # Get integer indices and fractions
        indices_low = np.floor(scaled).astype(np.int32)
        indices_high = np.minimum(indices_low + 1, size - 1)
        fractions = scaled - indices_low

        # Clamp indices
        indices_low = np.clip(indices_low, 0, size - 1)
        indices_high = np.clip(indices_high, 0, size - 1)

        # BGR to RGB (OpenCV convention)
        r_low, g_low, b_low = indices_low[:, :, 2], indices_low[:, :, 1], indices_low[:, :, 0]
        r_high, g_high, b_high = indices_high[:, :, 2], indices_high[:, :, 1], indices_high[:, :, 0]
        r_frac, g_frac, b_frac = fractions[:, :, 2:3], fractions[:, :, 1:2], fractions[:, :, 0:1]

        # Trilinear interpolation
        c000 = lut_array[r_low, g_low, b_low]
        c001 = lut_array[r_low, g_low, b_high]
        c010 = lut_array[r_low, g_high, b_low]
        c011 = lut_array[r_low, g_high, b_high]
        c100 = lut_array[r_high, g_low, b_low]
        c101 = lut_array[r_high, g_low, b_high]
        c110 = lut_array[r_high, g_high, b_low]
        c111 = lut_array[r_high, g_high, b_high]

        c00 = c000 * (1 - r_frac) + c100 * r_frac
        c01 = c001 * (1 - r_frac) + c101 * r_frac
        c10 = c010 * (1 - r_frac) + c110 * r_frac
        c11 = c011 * (1 - r_frac) + c111 * r_frac

        c0 = c00 * (1 - g_frac) + c10 * g_frac
        c1 = c01 * (1 - g_frac) + c11 * g_frac

        result = c0 * (1 - b_frac) + c1 * b_frac

        # Convert RGB back to BGR
        result = result[:, :, ::-1]

        # Convert back to original dtype
        if image.dtype == np.uint8:
            return (np.clip(result, 0, 1) * 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            return (np.clip(result, 0, 1) * 65535).astype(np.uint16)
        else:
            return result

    def create_identity_lut(self, size: int = 33, lut_type: LUTType = LUTType.LUT_3D) -> LUT:
        """Create an identity (no-op) LUT."""
        lut = LUT(name="Identity", lut_type=lut_type, size=size)

        if lut_type == LUTType.LUT_1D:
            lut.data_1d = [(i / (size - 1), i / (size - 1), i / (size - 1)) for i in range(size)]
        else:
            lut.data_3d = [[[None for _ in range(size)] for _ in range(size)] for _ in range(size)]
            for r in range(size):
                for g in range(size):
                    for b in range(size):
                        lut.data_3d[r][g][b] = (
                            r / (size - 1),
                            g / (size - 1),
                            b / (size - 1)
                        )

        return lut

    def create_contrast_lut(self, contrast: float = 1.2, size: int = 33) -> LUT:
        """Create a contrast adjustment LUT."""
        lut = self.create_identity_lut(size, LUTType.LUT_1D)
        lut.name = f"Contrast_{contrast:.1f}"

        # S-curve for contrast
        import math

        def apply_contrast(val: float) -> float:
            # Center around 0.5, apply power, recenter
            centered = val - 0.5
            sign = 1 if centered >= 0 else -1
            adjusted = sign * (abs(centered) ** (1 / contrast))
            return max(0, min(1, adjusted + 0.5))

        lut.data_1d = [
            (apply_contrast(i / (size - 1)),) * 3
            for i in range(size)
        ]

        return lut

    def create_film_emulation_lut(
        self,
        film_stock: str = "kodak_vision3",
        size: int = 33
    ) -> LUT:
        """Create a film stock emulation LUT."""
        lut = self.create_identity_lut(size, LUTType.LUT_3D)
        lut.name = f"Film_{film_stock}"
        lut.title = f"Film Emulation: {film_stock}"

        # Film stock characteristics (simplified)
        stocks = {
            "kodak_vision3": {
                "r_lift": 0.02, "r_gamma": 1.05, "r_gain": 0.98,
                "g_lift": 0.01, "g_gamma": 1.0, "g_gain": 1.0,
                "b_lift": 0.0, "b_gamma": 0.95, "b_gain": 1.02,
            },
            "fuji_eterna": {
                "r_lift": 0.015, "r_gamma": 1.02, "r_gain": 0.97,
                "g_lift": 0.02, "g_gamma": 1.0, "g_gain": 0.99,
                "b_lift": 0.025, "b_gamma": 0.98, "b_gain": 1.01,
            },
            "kodachrome": {
                "r_lift": 0.03, "r_gamma": 1.1, "r_gain": 1.0,
                "g_lift": 0.02, "g_gamma": 1.05, "g_gain": 0.98,
                "b_lift": 0.01, "b_gamma": 0.95, "b_gain": 0.95,
            },
            "ektachrome": {
                "r_lift": 0.01, "r_gamma": 1.08, "r_gain": 0.99,
                "g_lift": 0.015, "g_gamma": 1.02, "g_gain": 1.0,
                "b_lift": 0.02, "b_gamma": 1.0, "b_gain": 1.02,
            },
        }

        params = stocks.get(film_stock, stocks["kodak_vision3"])

        def apply_channel(val: float, lift: float, gamma: float, gain: float) -> float:
            val = val * gain + lift
            val = val ** (1 / gamma)
            return max(0, min(1, val))

        for r in range(size):
            for g in range(size):
                for b in range(size):
                    r_in = r / (size - 1)
                    g_in = g / (size - 1)
                    b_in = b / (size - 1)

                    r_out = apply_channel(r_in, params["r_lift"], params["r_gamma"], params["r_gain"])
                    g_out = apply_channel(g_in, params["g_lift"], params["g_gamma"], params["g_gain"])
                    b_out = apply_channel(b_in, params["b_lift"], params["b_gamma"], params["b_gain"])

                    lut.data_3d[r][g][b] = (r_out, g_out, b_out)

        return lut

    def create_seasonal_lut(
        self,
        season: str = "winter",
        strength: float = 1.0,
        size: int = 33,
    ) -> LUT:
        """Create a seasonal color grading LUT.

        Adjusts color palette to match the visual character of a season.
        Use after colorization to ensure colors feel appropriate for the
        time of year depicted in the footage.

        Args:
            season: One of 'winter', 'spring', 'summer', 'autumn'.
            strength: How strongly to apply the grade (0.0-1.0).
            size: LUT grid size.

        Returns:
            A 3D LUT with the seasonal color shift baked in.
        """
        import math

        lut = self.create_identity_lut(size, LUTType.LUT_3D)
        lut.name = f"Seasonal_{season}"
        lut.title = f"Seasonal Color Grade: {season} (strength={strength:.1f})"

        # Each season defines per-channel lift/gamma/gain plus a saturation
        # multiplier.  Values are the *maximum* shift; ``strength`` scales
        # them linearly so 0.0 == identity and 1.0 == full effect.
        presets = {
            "winter": {
                # Cool blue shadows, desaturated, bright whites
                "r_lift": -0.02, "r_gamma": 0.97, "r_gain": 0.95,
                "g_lift": -0.01, "g_gamma": 0.98, "g_gain": 0.96,
                "b_lift":  0.04, "b_gamma": 1.04, "b_gain": 1.02,
                "saturation": 0.75,
            },
            "spring": {
                # Soft pastels, light greens, warm highlights
                "r_lift": 0.01, "r_gamma": 1.02, "r_gain": 1.00,
                "g_lift": 0.02, "g_gamma": 1.04, "g_gain": 1.02,
                "b_lift": 0.01, "b_gamma": 1.00, "b_gain": 0.98,
                "saturation": 0.90,
            },
            "summer": {
                # Warm golden tones, vibrant greens, high saturation
                "r_lift": 0.02, "r_gamma": 1.06, "r_gain": 1.02,
                "g_lift": 0.02, "g_gamma": 1.04, "g_gain": 1.01,
                "b_lift": -0.01, "b_gamma": 0.96, "b_gain": 0.96,
                "saturation": 1.15,
            },
            "autumn": {
                # Orange/amber shadows, warm reds, muted greens
                "r_lift": 0.03, "r_gamma": 1.08, "r_gain": 1.02,
                "g_lift": 0.01, "g_gamma": 1.00, "g_gain": 0.96,
                "b_lift": -0.02, "b_gamma": 0.94, "b_gain": 0.92,
                "saturation": 1.05,
            },
        }

        if season not in presets:
            raise ValueError(
                f"Unknown season '{season}'. "
                f"Valid seasons: {list(presets.keys())}"
            )

        p = presets[season]

        # Blend preset toward identity based on strength
        def lerp(a: float, b: float, t: float) -> float:
            return a + (b - a) * t

        r_lift = lerp(0, p["r_lift"], strength)
        g_lift = lerp(0, p["g_lift"], strength)
        b_lift = lerp(0, p["b_lift"], strength)
        r_gamma = lerp(1.0, p["r_gamma"], strength)
        g_gamma = lerp(1.0, p["g_gamma"], strength)
        b_gamma = lerp(1.0, p["b_gamma"], strength)
        r_gain = lerp(1.0, p["r_gain"], strength)
        g_gain = lerp(1.0, p["g_gain"], strength)
        b_gain = lerp(1.0, p["b_gain"], strength)
        sat = lerp(1.0, p["saturation"], strength)

        def apply_grade(val: float, lift: float, gamma: float, gain: float) -> float:
            val = val * gain + lift
            val = max(0.0, min(1.0, val))
            if gamma != 1.0 and val > 0:
                val = val ** (1.0 / gamma)
            return max(0.0, min(1.0, val))

        def adjust_saturation(r: float, g: float, b: float, sat_mult: float):
            luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            r = luma + (r - luma) * sat_mult
            g = luma + (g - luma) * sat_mult
            b = luma + (b - luma) * sat_mult
            return (
                max(0.0, min(1.0, r)),
                max(0.0, min(1.0, g)),
                max(0.0, min(1.0, b)),
            )

        for ri in range(size):
            for gi in range(size):
                for bi in range(size):
                    r_in = ri / (size - 1)
                    g_in = gi / (size - 1)
                    b_in = bi / (size - 1)

                    r_out = apply_grade(r_in, r_lift, r_gamma, r_gain)
                    g_out = apply_grade(g_in, g_lift, g_gamma, g_gain)
                    b_out = apply_grade(b_in, b_lift, b_gamma, b_gain)

                    r_out, g_out, b_out = adjust_saturation(r_out, g_out, b_out, sat)

                    lut.data_3d[ri][gi][bi] = (r_out, g_out, b_out)

        return lut

    def combine_luts(self, luts: List[LUT], size: int = 33) -> LUT:
        """Combine multiple LUTs into one by sequential application."""
        result = self.create_identity_lut(size, LUTType.LUT_3D)
        result.name = "Combined"
        result.title = " + ".join(lut.name for lut in luts)

        for r in range(size):
            for g in range(size):
                for b in range(size):
                    rgb = (r / (size - 1), g / (size - 1), b / (size - 1))

                    for lut in luts:
                        rgb = lut.apply_to_rgb(*rgb)

                    result.data_3d[r][g][b] = rgb

        return result

    def get_ffmpeg_filter(self, lut_path: Path) -> str:
        """Get FFmpeg filter string for applying this LUT."""
        return f"lut3d='{lut_path}'"
