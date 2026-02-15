"""Video assembly mixin for VideoRestorer.

Contains video and audio extraction/reassembly logic:
- extract_audio: Extract audio track from video
- extract_frames: Extract frames from video as PNG
- reassemble_video: Reassemble enhanced frames with audio
"""

import asyncio
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class VideoAssemblyMixin:
    """Mixin providing video/audio assembly capabilities to VideoRestorer.

    This mixin encapsulates all FFmpeg-based video assembly operations,
    keeping them separate from frame processing and core orchestration.

    Methods included:
    - extract_audio: Extract audio track as WAV
    - extract_frames: Extract frames as PNG files
    - reassemble_video: Reassemble video from frames + audio
    """

    def extract_audio(self, video_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """Extract audio track as WAV with PCM encoding.

        Args:
            video_path: Path to video file
            output_path: Optional output path (defaults to config.temp_dir/audio.wav)

        Returns:
            Path to extracted audio file, or None if no audio track exists

        Raises:
            AudioExtractionError: If audio extraction fails
        """
        from ..utils.async_io import AsyncSubprocess
        from ..errors import AudioExtractionError
        from ..restorer import get_ffmpeg_path

        if not self.metadata.get('has_audio', False):
            logger.info("No audio track found in video")
            return None

        if output_path is None:
            output_path = self.config.temp_dir / "audio.wav"

        self._update_progress("extract_audio", 0.0)
        logger.info(f"Extracting audio to {output_path}")

        cmd = [
            get_ffmpeg_path(),
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s24le',  # PCM 24-bit little-endian
            '-ar', '48000',  # 48kHz sample rate
            '-y',  # Overwrite output file
            str(output_path)
        ]

        try:
            # Use async I/O if enabled for better performance
            if self.config.enable_async_io:
                async def _extract_audio_async():
                    async with AsyncSubprocess(timeout=600) as proc:
                        stdout, stderr = await proc.run_checked(cmd)
                        return stderr

                stderr = asyncio.run(_extract_audio_async())
                logger.debug(f"Audio extraction output: {stderr}")
            else:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                logger.debug(f"Audio extraction output: {result.stderr}")

            self._update_progress("extract_audio", 1.0)
            logger.info(f"Audio extracted to {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Audio extraction failed: {e.stderr}")
            raise AudioExtractionError(f"Failed to extract audio: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise AudioExtractionError("Audio extraction timed out")

    def extract_frames(self, video_path: Path) -> int:
        """Extract all frames from video as PNG files.

        Args:
            video_path: Path to video file

        Returns:
            Number of frames extracted

        Raises:
            FrameExtractionError: If frame extraction fails
        """
        from ..utils.async_io import AsyncSubprocess
        from ..errors import FrameExtractionError
        from ..validators import validate_frame_sequence
        from ..restorer import get_ffmpeg_path

        self._update_progress("extract_frames", 0.0)
        logger.info(f"Extracting frames to {self.config.frames_dir}")

        # Check for existing checkpoint
        if self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint and checkpoint.stage in ("extract", "enhance"):
                # Frames already extracted, verify
                existing_frames = list(self.config.frames_dir.glob("frame_*.png"))
                if len(existing_frames) == checkpoint.total_frames:
                    logger.info(f"Resuming from checkpoint: {len(existing_frames)} frames exist")
                    self._update_progress("extract_frames", 1.0)
                    return len(existing_frames)

        # Clear existing frames for fresh extraction
        if self.config.frames_dir.exists():
            shutil.rmtree(self.config.frames_dir)
        self.config.frames_dir.mkdir(parents=True)

        output_pattern = self.config.frames_dir / "frame_%08d.png"

        cmd = [
            get_ffmpeg_path(),
            '-i', str(video_path),
            '-qscale:v', '1',  # Highest quality
            '-qmin', '1',
            '-qmax', '1',
            str(output_pattern)
        ]

        try:
            # Use async I/O if enabled for better performance
            if self.config.enable_async_io:
                async def _extract_async():
                    async with AsyncSubprocess(timeout=3600) as proc:
                        stdout, stderr = await proc.run_checked(cmd)
                        return stderr

                stderr = asyncio.run(_extract_async())
                logger.debug(f"Frame extraction output: {stderr}")
            else:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                logger.debug(f"Frame extraction output: {result.stderr}")

            # Count extracted frames
            frame_count = len(list(self.config.frames_dir.glob("frame_*.png")))

            if frame_count == 0:
                raise FrameExtractionError("No frames were extracted")

            # Validate frame sequence
            seq_report = validate_frame_sequence(self.config.frames_dir)
            if seq_report.has_issues:
                logger.warning(
                    f"Frame sequence issues: {seq_report.missing_count} missing, "
                    f"{len(seq_report.duplicate_frames)} duplicates"
                )

            # Create checkpoint
            if self.checkpoint_manager:
                self.checkpoint_manager.create_checkpoint(
                    stage="extract",
                    total_frames=frame_count,
                    source_path=str(video_path),
                    metadata=self.metadata,
                )

            self._update_progress("extract_frames", 1.0)
            logger.info(f"Extracted {frame_count} frames")
            return frame_count

        except subprocess.CalledProcessError as e:
            logger.error(f"Frame extraction failed: {e.stderr}")
            raise FrameExtractionError(f"Failed to extract frames: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise FrameExtractionError("Frame extraction timed out")

    def reassemble_video(
        self,
        audio_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        frames_dir: Optional[Path] = None,
    ) -> Path:
        """Reassemble video from enhanced/interpolated frames with audio.

        Args:
            audio_path: Path to audio file (optional)
            output_path: Output video path (defaults to config.project_dir/output.mkv)
            frames_dir: Directory with frames (default: interpolated if exists, else enhanced)

        Returns:
            Path to output video file

        Raises:
            ReassemblyError: If video reassembly fails
        """
        from ..utils.async_io import AsyncSubprocess
        from ..errors import ReassemblyError
        from ..validators import validate_frame_sequence
        from ..utils.ffmpeg import get_best_video_codec
        from ..restorer import get_ffmpeg_path

        if output_path is None:
            output_path = self.config.project_dir / f"output.{self.config.output_format}"

        # Determine which frames to use
        if frames_dir is None:
            # Prefer interpolated frames if they exist
            if self.config.interpolated_dir.exists() and \
               list(self.config.interpolated_dir.glob("*.png")):
                frames_dir = self.config.interpolated_dir
                logger.info("Using interpolated frames for reassembly")
            else:
                frames_dir = self.config.enhanced_dir
                logger.info("Using enhanced frames for reassembly")

        self._update_progress("reassemble_video", 0.0)
        logger.info(f"Reassembling video to {output_path}")

        # Verify we have frames
        frame_files = sorted(frames_dir.glob("*.png"))
        if not frame_files:
            raise ReassemblyError(f"No frames found in {frames_dir} for reassembly")

        # Validate frame sequence
        seq_report = validate_frame_sequence(frames_dir)
        if seq_report.missing_count > 0:
            logger.warning(f"Missing {seq_report.missing_count} frames in sequence")

        # Get framerate - use interpolated fps if available, otherwise source fps
        framerate = self.metadata.get('interpolated_fps') or self.metadata.get('framerate', 30)
        logger.info(f"Output framerate: {framerate}fps")

        # Base ffmpeg command for video encoding
        input_pattern = frames_dir / "frame_%08d.png"

        cmd = [
            get_ffmpeg_path(),
            '-framerate', str(framerate),
            '-i', str(input_pattern),
        ]

        # Add audio if available
        if audio_path and audio_path.exists():
            cmd.extend([
                '-i', str(audio_path),
                '-c:a', 'flac',  # FLAC audio codec
            ])

        # Video encoding settings - use best available codec with fallback
        codec, pix_fmt = get_best_video_codec('libx265')
        logger.info(f"Using video codec: {codec} with pixel format: {pix_fmt}")

        cmd.extend([
            '-c:v', codec,
            '-crf', str(self.config.crf),  # Quality
            '-preset', self.config.preset,  # Encoding preset
            '-pix_fmt', pix_fmt,
            '-y',  # Overwrite output
            str(output_path)
        ])

        try:
            # Use async I/O if enabled for better performance
            if self.config.enable_async_io:
                async def _reassemble_async():
                    async with AsyncSubprocess(timeout=7200) as proc:
                        stdout, stderr = await proc.run_checked(cmd)
                        return stderr

                stderr = asyncio.run(_reassemble_async())
                logger.debug(f"Video reassembly output: {stderr}")
            else:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=7200  # 2 hour timeout
                )
                logger.debug(f"Video reassembly output: {result.stderr}")

            if not output_path.exists():
                raise ReassemblyError("Output video file was not created")

            # Update checkpoint
            if self.checkpoint_manager:
                self.checkpoint_manager.update_stage("reassemble")

            self._update_progress("reassemble_video", 1.0)
            logger.info(f"Video reassembled to {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Video reassembly failed: {e.stderr}")
            raise ReassemblyError(f"Failed to reassemble video: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise ReassemblyError("Video reassembly timed out")
