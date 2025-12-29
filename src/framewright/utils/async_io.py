"""Async I/O operations for improved performance.

Enables:
- Download next video while processing current one
- Read frames from disk while GPU processes previous batch
- Write enhanced frames while GPU works on next batch
- Better CPU/GPU utilization = faster overall processing
"""
import asyncio
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import shutil

logger = logging.getLogger(__name__)

# Thread pool for blocking I/O operations
_io_executor: Optional[ThreadPoolExecutor] = None


def get_io_executor(max_workers: int = 4) -> ThreadPoolExecutor:
    """Get or create the I/O thread pool executor.

    Args:
        max_workers: Maximum worker threads

    Returns:
        ThreadPoolExecutor for I/O operations
    """
    global _io_executor
    if _io_executor is None:
        _io_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="async_io")
    return _io_executor


def shutdown_executor() -> None:
    """Shutdown the I/O executor gracefully."""
    global _io_executor
    if _io_executor:
        _io_executor.shutdown(wait=True)
        _io_executor = None


@dataclass
class AsyncDownloadResult:
    """Result of an async download operation."""
    success: bool
    path: Optional[Path] = None
    error: Optional[str] = None
    size_bytes: int = 0
    duration_seconds: float = 0.0


@dataclass
class AsyncReadResult:
    """Result of an async file read operation."""
    success: bool
    data: Optional[bytes] = None
    path: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class AsyncWriteResult:
    """Result of an async file write operation."""
    success: bool
    path: Optional[Path] = None
    error: Optional[str] = None
    bytes_written: int = 0


class AsyncFileOperations:
    """Async file I/O operations using thread pools.

    Provides async wrappers for file operations that would
    otherwise block the event loop.

    Example:
        >>> async with AsyncFileOperations() as ops:
        ...     # Read multiple files concurrently
        ...     results = await asyncio.gather(
        ...         ops.read_file(path1),
        ...         ops.read_file(path2),
        ...         ops.read_file(path3),
        ...     )
    """

    def __init__(self, max_workers: int = 4):
        """Initialize async file operations.

        Args:
            max_workers: Maximum concurrent I/O operations
        """
        self.executor = get_io_executor(max_workers)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def __aenter__(self) -> "AsyncFileOperations":
        """Async context manager entry."""
        self._loop = asyncio.get_event_loop()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        pass

    async def read_file(self, path: Path) -> AsyncReadResult:
        """Read file contents asynchronously.

        Args:
            path: Path to file

        Returns:
            AsyncReadResult with file data
        """
        loop = self._loop or asyncio.get_event_loop()

        def _read():
            try:
                data = path.read_bytes()
                return AsyncReadResult(success=True, data=data, path=path)
            except Exception as e:
                return AsyncReadResult(success=False, path=path, error=str(e))

        return await loop.run_in_executor(self.executor, _read)

    async def write_file(self, path: Path, data: bytes) -> AsyncWriteResult:
        """Write file contents asynchronously.

        Args:
            path: Path to write to
            data: Bytes to write

        Returns:
            AsyncWriteResult with status
        """
        loop = self._loop or asyncio.get_event_loop()

        def _write():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                return AsyncWriteResult(success=True, path=path, bytes_written=len(data))
            except Exception as e:
                return AsyncWriteResult(success=False, path=path, error=str(e))

        return await loop.run_in_executor(self.executor, _write)

    async def copy_file(self, src: Path, dst: Path) -> AsyncWriteResult:
        """Copy file asynchronously.

        Args:
            src: Source path
            dst: Destination path

        Returns:
            AsyncWriteResult with status
        """
        loop = self._loop or asyncio.get_event_loop()

        def _copy():
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                return AsyncWriteResult(success=True, path=dst, bytes_written=dst.stat().st_size)
            except Exception as e:
                return AsyncWriteResult(success=False, path=dst, error=str(e))

        return await loop.run_in_executor(self.executor, _copy)

    async def read_frames_batch(
        self,
        frame_paths: List[Path],
        max_concurrent: int = 8,
    ) -> List[AsyncReadResult]:
        """Read multiple frame files concurrently.

        Args:
            frame_paths: List of frame paths to read
            max_concurrent: Maximum concurrent reads

        Returns:
            List of AsyncReadResult in same order as input
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def read_with_limit(path: Path) -> AsyncReadResult:
            async with semaphore:
                return await self.read_file(path)

        return await asyncio.gather(*[read_with_limit(p) for p in frame_paths])

    async def write_frames_batch(
        self,
        frames: List[Tuple[Path, bytes]],
        max_concurrent: int = 8,
    ) -> List[AsyncWriteResult]:
        """Write multiple frame files concurrently.

        Args:
            frames: List of (path, data) tuples
            max_concurrent: Maximum concurrent writes

        Returns:
            List of AsyncWriteResult in same order as input
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def write_with_limit(path: Path, data: bytes) -> AsyncWriteResult:
            async with semaphore:
                return await self.write_file(path, data)

        return await asyncio.gather(*[write_with_limit(p, d) for p, d in frames])


class AsyncSubprocess:
    """Async subprocess execution for external tools.

    Wraps subprocess calls to be non-blocking, allowing
    other operations to continue while waiting.

    Example:
        >>> async with AsyncSubprocess() as proc:
        ...     result = await proc.run(['ffmpeg', '-i', 'input.mp4', ...])
    """

    def __init__(self, timeout: float = 3600.0):
        """Initialize async subprocess runner.

        Args:
            timeout: Default timeout in seconds
        """
        self.timeout = timeout
        self.executor = get_io_executor()

    async def __aenter__(self) -> "AsyncSubprocess":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        pass

    async def run(
        self,
        cmd: List[str],
        timeout: Optional[float] = None,
        cwd: Optional[Path] = None,
    ) -> Tuple[int, str, str]:
        """Run subprocess command asynchronously.

        Args:
            cmd: Command and arguments
            timeout: Override timeout
            cwd: Working directory

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        timeout = timeout or self.timeout

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            return process.returncode or 0, stdout.decode(), stderr.decode()
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(f"Command timed out after {timeout}s: {' '.join(cmd)}")

    async def run_checked(
        self,
        cmd: List[str],
        timeout: Optional[float] = None,
        cwd: Optional[Path] = None,
    ) -> Tuple[str, str]:
        """Run subprocess and raise on non-zero exit.

        Args:
            cmd: Command and arguments
            timeout: Override timeout
            cwd: Working directory

        Returns:
            Tuple of (stdout, stderr)

        Raises:
            subprocess.CalledProcessError: On non-zero exit
        """
        returncode, stdout, stderr = await self.run(cmd, timeout, cwd)

        if returncode != 0:
            raise subprocess.CalledProcessError(
                returncode, cmd, stdout.encode(), stderr.encode()
            )

        return stdout, stderr


class AsyncDownloader:
    """Async video downloader using yt-dlp.

    Enables downloading next video while current one processes.

    Example:
        >>> async with AsyncDownloader() as dl:
        ...     # Start download in background
        ...     download_task = asyncio.create_task(
        ...         dl.download(url, output_dir)
        ...     )
        ...     # Do other work while downloading
        ...     await process_current_video()
        ...     # Now get download result
        ...     result = await download_task
    """

    def __init__(self, timeout: float = 3600.0):
        """Initialize async downloader.

        Args:
            timeout: Download timeout in seconds
        """
        self.subprocess = AsyncSubprocess(timeout=timeout)

    async def __aenter__(self) -> "AsyncDownloader":
        """Async context manager entry."""
        await self.subprocess.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.subprocess.__aexit__(exc_type, exc_val, exc_tb)

    async def download(
        self,
        url: str,
        output_dir: Path,
        filename: str = "video",
        format_spec: str = "bestvideo[ext=webm]+bestaudio[ext=webm]/best",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> AsyncDownloadResult:
        """Download video asynchronously.

        Args:
            url: Video URL
            output_dir: Output directory
            filename: Base filename (without extension)
            format_spec: yt-dlp format specification
            progress_callback: Optional progress callback (0.0-1.0)

        Returns:
            AsyncDownloadResult with download status
        """
        import time
        start_time = time.time()

        output_path = output_dir / f"{filename}.%(ext)s"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            'yt-dlp',
            '--format', format_spec,
            '--merge-output-format', 'mkv',
            '--output', str(output_path),
            '--no-playlist',
            '--newline',  # Progress on new lines
            url,
        ]

        try:
            returncode, stdout, stderr = await self.subprocess.run(cmd)

            if returncode != 0:
                return AsyncDownloadResult(
                    success=False,
                    error=f"yt-dlp failed: {stderr}",
                )

            # Find actual output file
            actual_path = None
            for ext in ['.mkv', '.webm', '.mp4']:
                candidate = output_dir / f"{filename}{ext}"
                if candidate.exists():
                    actual_path = candidate
                    break

            if not actual_path:
                return AsyncDownloadResult(
                    success=False,
                    error="Downloaded file not found",
                )

            duration = time.time() - start_time
            size = actual_path.stat().st_size

            return AsyncDownloadResult(
                success=True,
                path=actual_path,
                size_bytes=size,
                duration_seconds=duration,
            )

        except TimeoutError as e:
            return AsyncDownloadResult(success=False, error=str(e))
        except Exception as e:
            return AsyncDownloadResult(success=False, error=f"Download failed: {e}")


class AsyncFrameProcessor:
    """Async frame processing coordinator.

    Orchestrates concurrent frame I/O with GPU processing
    for maximum throughput.

    Example:
        >>> processor = AsyncFrameProcessor(enhance_fn)
        >>> results = await processor.process_frames(
        ...     input_dir, output_dir, max_concurrent=4
        ... )
    """

    def __init__(
        self,
        process_fn: Callable[[Path, Path], bool],
        max_io_workers: int = 4,
    ):
        """Initialize async frame processor.

        Args:
            process_fn: Function(input_path, output_path) -> success
            max_io_workers: Maximum I/O worker threads
        """
        self.process_fn = process_fn
        self.file_ops = AsyncFileOperations(max_workers=max_io_workers)
        self._processed = 0
        self._failed = 0

    async def process_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        max_concurrent: int = 2,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """Process all frames with async I/O coordination.

        Uses a pipeline approach:
        1. Async read ahead of frames
        2. GPU processing
        3. Async write behind of results

        Args:
            input_dir: Directory with input frames
            output_dir: Directory for output frames
            max_concurrent: Max concurrent GPU operations
            progress_callback: Optional progress callback

        Returns:
            Dict with processing statistics
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(input_dir.glob("frame_*.png"))
        total = len(frames)

        if total == 0:
            return {"success": False, "error": "No frames found", "processed": 0}

        semaphore = asyncio.Semaphore(max_concurrent)
        loop = asyncio.get_event_loop()

        async def process_one(frame: Path) -> bool:
            async with semaphore:
                output_path = output_dir / frame.name

                # Run GPU processing in thread pool to not block event loop
                def _process():
                    return self.process_fn(frame, output_path)

                try:
                    success = await loop.run_in_executor(
                        self.file_ops.executor,
                        _process
                    )

                    if success:
                        self._processed += 1
                    else:
                        self._failed += 1

                    # Update progress
                    if progress_callback:
                        progress = (self._processed + self._failed) / total
                        progress_callback(progress, f"Processed {self._processed}/{total}")

                    return success

                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    self._failed += 1
                    return False

        # Process all frames concurrently (up to semaphore limit)
        results = await asyncio.gather(
            *[process_one(f) for f in frames],
            return_exceptions=True,
        )

        successes = sum(1 for r in results if r is True)
        failures = sum(1 for r in results if r is not True)

        return {
            "success": failures == 0,
            "processed": successes,
            "failed": failures,
            "total": total,
        }


async def run_pipeline_async(
    download_url: Optional[str],
    input_path: Optional[Path],
    output_dir: Path,
    process_fn: Callable[[Path], Path],
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Path:
    """Run complete pipeline with async I/O.

    Coordinates async downloads, processing, and I/O for
    maximum throughput.

    Args:
        download_url: Optional URL to download from
        input_path: Optional local input path
        output_dir: Output directory
        process_fn: Processing function (input) -> output
        progress_callback: Optional callback(stage, progress)

    Returns:
        Path to final output

    Example:
        >>> result = await run_pipeline_async(
        ...     download_url="https://...",
        ...     output_dir=Path("./output"),
        ...     process_fn=my_enhance_function,
        ... )
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download if needed
    if download_url:
        if progress_callback:
            progress_callback("download", 0.0)

        async with AsyncDownloader() as dl:
            result = await dl.download(download_url, output_dir)

            if not result.success:
                raise RuntimeError(f"Download failed: {result.error}")

            input_path = result.path

        if progress_callback:
            progress_callback("download", 1.0)

    if not input_path or not input_path.exists():
        raise ValueError("No input path available")

    # Step 2: Process
    if progress_callback:
        progress_callback("process", 0.0)

    loop = asyncio.get_event_loop()
    output_path = await loop.run_in_executor(
        get_io_executor(),
        process_fn,
        input_path,
    )

    if progress_callback:
        progress_callback("process", 1.0)

    return output_path


# Convenience function for sync code to use async features
def run_async(coro):
    """Run async coroutine from synchronous code.

    Args:
        coro: Coroutine to run

    Returns:
        Coroutine result
    """
    try:
        loop = asyncio.get_running_loop()
        # Already in async context, create task
        return asyncio.ensure_future(coro)
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(coro)
