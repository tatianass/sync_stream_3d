import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import imageio

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Context manager to track processing throughput and enforce performance SLAs.
    """

    def __init__(self, total_size_gb: float, threshold_mb_s: float = 15.0):
        self.total_size_mb = total_size_gb * 1024
        self.threshold = threshold_mb_s
        self.start_time: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return

        duration = time.perf_counter() - self.start_time
        throughput = self.total_size_mb / duration if duration > 0 else 0

        logger.info("--- Performance Report ---")
        logger.info(f"Total Time: {duration:.2f} seconds")
        logger.info(f"Throughput: {throughput:.2f} MB/s")

        if throughput < self.threshold:
            logger.warning(
                f"PERFORMANCE ALERT: {throughput:.2f} MB/s < {self.threshold} MB/s"
            )


def create_preview_gif(
    output_dir: Path, index_data: List[Dict[str, Any]], max_frames: int = 550
):
    """
    Stitches processed frames into a preview GIF for quick visual verification.
    """
    gif_path = output_dir / "preview_stream.gif"
    images = []

    # Take the first N samples from the index
    sample_frames = index_data[:max_frames]

    logger.info(f"Generating preview GIF at {gif_path}...")

    try:
        for frame_meta in sample_frames:
            # Reconstruct the full path using the relative path in the index
            img_path = output_dir / frame_meta["image_path"]

            if img_path.exists():
                # Read with OpenCV (BGR) and convert to RGB
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Downscale to 480p to keep GIF size small for sharing/email
                img_small = cv2.resize(img_rgb, (480, 270))
                images.append(img_small)

        if images:
            # duration is in seconds per frame (0.1s = 10fps)
            imageio.mimsave(str(gif_path), images, fps=10, loop=0)
            logger.info(f"Preview GIF created successfully: {gif_path}")
        else:
            logger.warning("No valid images found to create GIF.")

    except Exception as e:
        logger.error(f"Failed to generate preview GIF: {e}")
