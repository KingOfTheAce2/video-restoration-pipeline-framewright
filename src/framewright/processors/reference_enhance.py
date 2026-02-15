"""Reference-guided frame enhancement using IP-Adapter + ControlNet.

Uses high-quality reference photos to guide detail generation in degraded frames.
The reference images teach the model what the scene should look like,
while ControlNet preserves the original frame structure.
"""

import logging
import shutil
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def is_reference_enhance_available() -> bool:
    """Check if reference enhancement dependencies are installed."""
    try:
        from diffusers import (  # noqa: F401
            StableDiffusionControlNetImg2ImgPipeline,
            ControlNetModel,
        )
        return True
    except ImportError:
        return False


def load_reference_images(ref_dir: Path, max_refs: int = 5) -> list[Image.Image]:
    """Load reference images from a directory."""
    ref_dir = Path(ref_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = []
    for f in sorted(ref_dir.iterdir()):
        if f.suffix.lower() in exts:
            img = Image.open(f).convert("RGB")
            # Resize to reasonable size for IP-Adapter (max 1024px)
            img.thumbnail((1024, 1024), Image.LANCZOS)
            images.append(img)
            if len(images) >= max_refs:
                break
    return images


def composite_references(images: list[Image.Image], target_size: int = 512) -> Image.Image:
    """Combine multiple reference images into a single composite for IP-Adapter.

    Creates a grid collage that IP-Adapter can use as a single style reference.
    """
    if len(images) == 1:
        img = images[0].copy()
        img = img.resize((target_size, target_size), Image.LANCZOS)
        return img

    # Determine grid layout (e.g., 2x2 for 3-4 images, 3x2 for 5-6)
    n = len(images)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    cell_w = target_size // cols
    cell_h = target_size // rows

    composite = Image.new("RGB", (target_size, target_size), (128, 128, 128))
    for i, img in enumerate(images):
        row, col = divmod(i, cols)
        resized = img.resize((cell_w, cell_h), Image.LANCZOS)
        composite.paste(resized, (col * cell_w, row * cell_h))

    return composite


def enhance_with_references(
    frames_dir: str,
    output_dir: str,
    references_dir: str,
    strength: float = 0.35,
    guidance_scale: float = 7.5,
    ip_adapter_scale: float = 0.6,
    batch_size: int = 1,
    resume: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None,
):
    """Enhance frames using reference images via IP-Adapter + ControlNet.

    Args:
        frames_dir: Directory with input frames (PNG)
        output_dir: Directory for enhanced output frames
        references_dir: Directory with reference photos
        strength: How much to change from original (0.0=no change, 1.0=full regeneration).
                  Lower values preserve more of the original. 0.3-0.4 is good.
        guidance_scale: How strongly to follow the prompt/references (5-10 typical)
        ip_adapter_scale: How much the reference images influence output (0.0-1.0).
                         Higher = more reference influence, lower = more original.
        batch_size: Frames per batch (1 for safety, higher if VRAM allows)
        resume: Skip frames that already exist in output_dir
        progress_callback: Optional callback for progress reporting (0.0 to 1.0)
    """
    from diffusers import (
        StableDiffusionControlNetImg2ImgPipeline,
        ControlNetModel,
    )

    frames_path = Path(frames_dir)
    output_path = Path(output_dir)
    refs_path = Path(references_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load reference images
    logger.info(f"Loading reference images from {refs_path}...")
    ref_images = load_reference_images(refs_path)
    if not ref_images:
        logger.warning(f"No reference images found in {refs_path}")
        return
    logger.info(f"  Loaded {len(ref_images)} reference images")

    # Get frame list
    frames = sorted(frames_path.glob("*.png"))
    if not frames:
        logger.warning("No PNG frames found")
        return

    total_frames = len(frames)

    # Skip already processed frames if resuming
    if resume:
        todo = [f for f in frames if not (output_path / f.name).exists()]
        logger.info(
            f"  {total_frames} total frames, {total_frames - len(todo)} already done, "
            f"{len(todo)} to process"
        )
        frames = todo

    if not frames:
        logger.info("All frames already processed!")
        if progress_callback:
            progress_callback(1.0)
        return

    # Load ControlNet (Canny edge detection to preserve structure)
    logger.info("Loading ControlNet (Canny)...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=torch.float16,
    )

    # Load Stable Diffusion pipeline with ControlNet
    logger.info("Loading Stable Diffusion 1.5 + IP-Adapter...")
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.to("cuda")

    # Load IP-Adapter for reference image guidance
    logger.info("Loading IP-Adapter...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter_sd15.bin",
    )
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    # Enable memory optimizations
    pipe.enable_model_cpu_offload()

    # Composite all references into a single image for IP-Adapter
    # (IP-Adapter expects one image per adapter, not a list)
    ip_adapter_image = composite_references(ref_images)
    logger.info(f"  Created composite reference image ({ip_adapter_image.size[0]}x{ip_adapter_image.size[1]})")

    logger.info(f"Processing {len(frames)} frames (strength={strength}, ip_scale={ip_adapter_scale})...")

    processed = 0
    failed = 0

    for i, frame_path in enumerate(frames):
        try:
            # Load frame
            frame_img = Image.open(frame_path).convert("RGB")
            w, h = frame_img.size

            # Generate canny edges for ControlNet (preserves structure)
            frame_np = np.array(frame_img)
            canny = cv2.Canny(frame_np, 50, 150)
            canny = np.stack([canny] * 3, axis=-1)
            control_image = Image.fromarray(canny)

            # Generate enhanced frame
            result = pipe(
                prompt="high quality detailed photograph, sharp, clear",
                negative_prompt="blurry, noisy, artifacts, distorted, low quality",
                image=frame_img,
                control_image=control_image,
                ip_adapter_image=ip_adapter_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=20,
                width=w - (w % 8),
                height=h - (h % 8),
            ).images[0]

            # Resize back to exact original size if needed
            if result.size != (w, h):
                result = result.resize((w, h), Image.LANCZOS)

            # Save
            result.save(output_path / frame_path.name)
            processed += 1

        except Exception as e:
            logger.warning(f"Failed on {frame_path.name}: {e}")
            # Copy original as fallback
            shutil.copy2(frame_path, output_path / frame_path.name)
            failed += 1

        if progress_callback:
            progress_callback((i + 1) / len(frames))

    logger.info(f"Done! {processed} enhanced, {failed} failed. Output: {output_path}")
    return processed, failed


def main():
    """CLI entry point for reference-guided enhancement."""
    import argparse

    parser = argparse.ArgumentParser(description="Reference-guided frame enhancement")
    parser.add_argument("--frames", required=True, help="Input frames directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--references", required=True, help="Reference photos directory")
    parser.add_argument("--strength", type=float, default=0.35,
                        help="Denoising strength (0.0-1.0, lower=more original, default: 0.35)")
    parser.add_argument("--ip-scale", type=float, default=0.6,
                        help="IP-Adapter influence (0.0-1.0, higher=more reference, default: 0.6)")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="Guidance scale (default: 7.5)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Reprocess all frames even if output exists")
    args = parser.parse_args()

    enhance_with_references(
        frames_dir=args.frames,
        output_dir=args.output,
        references_dir=args.references,
        strength=args.strength,
        ip_adapter_scale=args.ip_scale,
        guidance_scale=args.guidance,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
