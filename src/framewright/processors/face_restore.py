"""Face restoration processor using GFPGAN or CodeFormer.

Automatically detects faces in frames and applies specialized
face restoration for improved quality on portraits and group scenes.
"""

import logging
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class FaceModel(Enum):
    """Available face restoration models."""
    GFPGAN_V1_3 = "GFPGANv1.3"
    GFPGAN_V1_4 = "GFPGANv1.4"
    CODEFORMER = "CodeFormer"
    RESTOREFORMER = "RestoreFormer"


@dataclass
class FaceRestorationResult:
    """Result of face restoration processing."""
    frames_processed: int = 0
    faces_detected: int = 0
    faces_restored: int = 0
    failed_frames: int = 0
    output_dir: Optional[Path] = None


class FaceRestorer:
    """Face restoration using GFPGAN or CodeFormer.

    This processor detects faces in video frames and applies
    specialized AI restoration to improve facial details.

    Supports multiple backends:
    - GFPGAN (via gfpgan command or Python API)
    - CodeFormer (via codeformer command or Python API)

    Falls back gracefully if face restoration tools aren't installed.
    """

    def __init__(
        self,
        model: FaceModel = FaceModel.GFPGAN_V1_4,
        upscale: int = 2,
        bg_upsampler: str = "realesrgan",
        only_center_face: bool = False,
        aligned: bool = False,
        weight: float = 0.5,  # CodeFormer fidelity weight
    ):
        """Initialize face restorer.

        Args:
            model: Face restoration model to use
            upscale: Upscaling factor for output
            bg_upsampler: Background upsampler (realesrgan, None)
            only_center_face: Only restore center/largest face
            aligned: Input faces are already aligned
            weight: CodeFormer fidelity weight (0=quality, 1=fidelity)
        """
        self.model = model
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler
        self.only_center_face = only_center_face
        self.aligned = aligned
        self.weight = weight
        self._backend = self._detect_backend()

    def _detect_backend(self) -> Optional[str]:
        """Detect available face restoration backend."""
        # Check for GFPGAN
        if shutil.which('gfpgan'):
            logger.info("Found GFPGAN CLI backend")
            return 'gfpgan_cli'

        # Check for inference_gfpgan.py script
        gfpgan_script = Path.home() / "GFPGAN" / "inference_gfpgan.py"
        if gfpgan_script.exists():
            logger.info("Found GFPGAN Python backend")
            return 'gfpgan_python'

        # Check for CodeFormer
        if shutil.which('codeformer'):
            logger.info("Found CodeFormer CLI backend")
            return 'codeformer_cli'

        # Check Python imports
        try:
            import gfpgan
            logger.info("Found GFPGAN Python module")
            return 'gfpgan_module'
        except ImportError:
            pass

        logger.warning(
            "No face restoration backend found. "
            "Install GFPGAN: pip install gfpgan"
        )
        return None

    def is_available(self) -> bool:
        """Check if face restoration is available."""
        return self._backend is not None

    def restore_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> FaceRestorationResult:
        """Restore faces in all frames in a directory.

        Args:
            input_dir: Directory containing input frames
            output_dir: Directory for output frames
            progress_callback: Optional progress callback (0.0 to 1.0)

        Returns:
            FaceRestorationResult with statistics
        """
        result = FaceRestorationResult(output_dir=output_dir)

        if not self._backend:
            logger.warning("Face restoration not available, copying frames")
            self._copy_frames(input_dir, output_dir)
            result.frames_processed = len(list(input_dir.glob("*.png")))
            return result

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            logger.warning("No frames found in input directory")
            return result

        logger.info(f"Restoring faces in {len(frames)} frames using {self.model.value}")

        if progress_callback:
            progress_callback(0.0)

        # Process based on backend
        if self._backend == 'gfpgan_cli':
            result = self._restore_gfpgan_cli(input_dir, output_dir, frames)
        elif self._backend == 'gfpgan_python':
            result = self._restore_gfpgan_python(input_dir, output_dir, frames)
        elif self._backend == 'gfpgan_module':
            result = self._restore_gfpgan_module(input_dir, output_dir, frames, progress_callback)
        elif self._backend == 'codeformer_cli':
            result = self._restore_codeformer_cli(input_dir, output_dir, frames)
        else:
            logger.warning(f"Unknown backend: {self._backend}")
            self._copy_frames(input_dir, output_dir)
            result.frames_processed = len(frames)

        if progress_callback:
            progress_callback(1.0)

        return result

    def _restore_gfpgan_cli(
        self,
        input_dir: Path,
        output_dir: Path,
        frames: List[Path]
    ) -> FaceRestorationResult:
        """Restore using GFPGAN CLI."""
        result = FaceRestorationResult(output_dir=output_dir)

        cmd = [
            'gfpgan',
            '-i', str(input_dir),
            '-o', str(output_dir),
            '-v', self.model.value.replace('GFPGAN', ''),
            '-s', str(self.upscale),
        ]

        if self.bg_upsampler:
            cmd.extend(['--bg_upsampler', self.bg_upsampler])
        else:
            cmd.extend(['--bg_upsampler', 'none'])

        if self.only_center_face:
            cmd.append('--only_center_face')

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=3600)
            result.frames_processed = len(frames)
            result.faces_restored = len(frames)  # Assume all processed
        except subprocess.CalledProcessError as e:
            logger.error(f"GFPGAN CLI failed: {e.stderr}")
            self._copy_frames(input_dir, output_dir)
            result.frames_processed = len(frames)
            result.failed_frames = len(frames)
        except FileNotFoundError:
            logger.error("GFPGAN CLI not found")
            self._copy_frames(input_dir, output_dir)
            result.frames_processed = len(frames)
            result.failed_frames = len(frames)

        return result

    def _restore_gfpgan_python(
        self,
        input_dir: Path,
        output_dir: Path,
        frames: List[Path]
    ) -> FaceRestorationResult:
        """Restore using GFPGAN Python script."""
        result = FaceRestorationResult(output_dir=output_dir)

        gfpgan_dir = Path.home() / "GFPGAN"
        script = gfpgan_dir / "inference_gfpgan.py"

        cmd = [
            'python', str(script),
            '-i', str(input_dir),
            '-o', str(output_dir),
            '-v', self.model.value.replace('GFPGAN', ''),
            '-s', str(self.upscale),
        ]

        if self.bg_upsampler:
            cmd.extend(['--bg_upsampler', self.bg_upsampler])
        if self.only_center_face:
            cmd.append('--only_center_face')

        try:
            subprocess.run(cmd, cwd=gfpgan_dir, capture_output=True, check=True, timeout=3600)
            result.frames_processed = len(frames)
            result.faces_restored = len(frames)
        except Exception as e:
            logger.error(f"GFPGAN Python failed: {e}")
            self._copy_frames(input_dir, output_dir)
            result.failed_frames = len(frames)

        return result

    def _restore_gfpgan_module(
        self,
        input_dir: Path,
        output_dir: Path,
        frames: List[Path],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> FaceRestorationResult:
        """Restore using GFPGAN Python module directly."""
        result = FaceRestorationResult(output_dir=output_dir)

        try:
            import cv2
            import numpy as np
            from gfpgan import GFPGANer

            # Initialize GFPGAN
            restorer = GFPGANer(
                model_path=self._get_model_path(),
                upscale=self.upscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self._get_bg_upsampler(),
            )

            for i, frame_path in enumerate(frames):
                try:
                    # Read image
                    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)

                    # Restore faces
                    _, _, output = restorer.enhance(
                        img,
                        has_aligned=self.aligned,
                        only_center_face=self.only_center_face,
                        paste_back=True,
                    )

                    # Save output
                    output_path = output_dir / frame_path.name
                    cv2.imwrite(str(output_path), output)

                    result.frames_processed += 1
                    result.faces_restored += 1

                except Exception as e:
                    logger.debug(f"Failed to restore {frame_path.name}: {e}")
                    # Copy original on failure
                    shutil.copy(frame_path, output_dir / frame_path.name)
                    result.frames_processed += 1
                    result.failed_frames += 1

                if progress_callback:
                    progress_callback((i + 1) / len(frames))

        except ImportError:
            logger.error("GFPGAN module import failed")
            self._copy_frames(input_dir, output_dir)
            result.frames_processed = len(frames)
            result.failed_frames = len(frames)

        return result

    def _restore_codeformer_cli(
        self,
        input_dir: Path,
        output_dir: Path,
        frames: List[Path]
    ) -> FaceRestorationResult:
        """Restore using CodeFormer CLI."""
        result = FaceRestorationResult(output_dir=output_dir)

        cmd = [
            'codeformer',
            '-i', str(input_dir),
            '-o', str(output_dir),
            '-w', str(self.weight),
            '--face_upsample',
            '-s', str(self.upscale),
        ]

        if self.bg_upsampler:
            cmd.extend(['--bg_upsampler', self.bg_upsampler])

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=3600)
            result.frames_processed = len(frames)
            result.faces_restored = len(frames)
        except Exception as e:
            logger.error(f"CodeFormer CLI failed: {e}")
            self._copy_frames(input_dir, output_dir)
            result.failed_frames = len(frames)

        return result

    def _get_model_path(self) -> str:
        """Get path to GFPGAN model weights."""
        model_paths = {
            FaceModel.GFPGAN_V1_3: "GFPGANv1.3.pth",
            FaceModel.GFPGAN_V1_4: "GFPGANv1.4.pth",
        }
        return model_paths.get(self.model, "GFPGANv1.4.pth")

    def _get_bg_upsampler(self):
        """Get background upsampler instance."""
        if not self.bg_upsampler or self.bg_upsampler == 'none':
            return None

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)

            return RealESRGANer(
                scale=4,
                model_path='RealESRGAN_x4plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True,
            )
        except Exception:
            return None

    def _copy_frames(self, input_dir: Path, output_dir: Path) -> None:
        """Copy frames when restoration fails."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for frame in input_dir.glob("*.png"):
            shutil.copy(frame, output_dir / frame.name)
        for frame in input_dir.glob("*.jpg"):
            shutil.copy(frame, output_dir / frame.name)


class AutoFaceRestorer:
    """Automatic face restoration that only processes frames with faces.

    More efficient than processing all frames - first detects which
    frames contain faces, then only processes those.
    """

    def __init__(self, restorer: Optional[FaceRestorer] = None):
        """Initialize with optional custom restorer."""
        self.restorer = restorer or FaceRestorer()

    def restore_selective(
        self,
        input_dir: Path,
        output_dir: Path,
        face_frame_indices: Optional[List[int]] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> FaceRestorationResult:
        """Selectively restore only frames containing faces.

        Args:
            input_dir: Input frames directory
            output_dir: Output frames directory
            face_frame_indices: Pre-computed indices of frames with faces
                              (if None, all frames are processed)
            progress_callback: Progress callback

        Returns:
            FaceRestorationResult
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = sorted(input_dir.glob("*.png"))

        if not self.restorer.is_available():
            # No face restoration available, just copy
            for frame in frames:
                shutil.copy(frame, output_dir / frame.name)
            return FaceRestorationResult(
                frames_processed=len(frames),
                output_dir=output_dir
            )

        if face_frame_indices is None:
            # Process all frames
            return self.restorer.restore_frames(input_dir, output_dir, progress_callback)

        # Process only face frames, copy others
        result = FaceRestorationResult(output_dir=output_dir)
        import tempfile

        face_frames_dir = Path(tempfile.mkdtemp(prefix="face_frames_"))

        try:
            # Copy face frames to temp directory
            for idx in face_frame_indices:
                if idx < len(frames):
                    shutil.copy(frames[idx], face_frames_dir / frames[idx].name)

            # Restore face frames
            face_result = self.restorer.restore_frames(face_frames_dir, output_dir)

            # Copy non-face frames directly
            face_set = set(face_frame_indices)
            for i, frame in enumerate(frames):
                if i not in face_set:
                    shutil.copy(frame, output_dir / frame.name)
                    result.frames_processed += 1

            result.faces_detected = len(face_frame_indices)
            result.faces_restored = face_result.faces_restored
            result.frames_processed += face_result.frames_processed

        finally:
            shutil.rmtree(face_frames_dir, ignore_errors=True)

        return result
