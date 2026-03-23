import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from syncstream.chunker import DataChunker
from syncstream.converter import DatasetConverter
from syncstream.utils import PerformanceMonitor  # Assuming the monitor is in utils

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SyncStream3DEngine")


def main():
    # 1. Parse CLI arguments
    parser = argparse.ArgumentParser(description="SyncStream3D Data Engineering Pipeline")
    parser.add_argument(
        "--input", type=str, default="data/kitti.mcap", help="Path to input MCAP file"
    )
    parser.add_argument(
        "--output", type=str, default="output/", help="Output directory"
    )
    parser.add_argument(
        "--threshold", type=float, default=15.0, help="Performance threshold (MB/s)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel I/O workers"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # 2. Initialize Components
    output_path.mkdir(parents=True, exist_ok=True)
    chunker = DataChunker(input_path)
    converter = DatasetConverter(output_path)

    file_size_gb = input_path.stat().st_size / (1024**3)

    logger.info(f"Starting pipeline for {input_path.name} ({file_size_gb:.2f} GB)")

    # 3. Execute with Performance Monitoring
    try:
        with PerformanceMonitor(
            total_size_gb=file_size_gb, threshold_mb_s=args.threshold
        ):
            # Using context manager for the executor
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                logger.info(f"Streaming data with {args.workers} workers...")

                for chunk in chunker.stream_synchronized_data():
                    executor.submit(converter.process_and_save, chunk)
            # The context manager automatically 'waits' (joins) for workers here.
            # ONLY now is it safe to write the index.
            logger.info("Finalizing index...")
            converter.write_index()

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
