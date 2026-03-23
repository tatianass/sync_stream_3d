import argparse
import logging
from pathlib import Path
from typing import Any, Dict, cast

import pandas as pd

from syncstream.utils import create_preview_gif

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualizer():
    parser = argparse.ArgumentParser(
        description="Generate a preview GIF from processed data."
    )
    parser.add_argument(
        "--dir", type=str, required=True, help="Path to the processed output directory"
    )
    args = parser.parse_args()

    output_path = Path(args.dir)
    index_path = output_path / "index.parquet"

    if not index_path.exists():
        logger.error(
            f"No index.parquet found in {output_path}. Did you run the main pipeline first?"
        )
        return

    # Load the index and convert to the list of dicts the util expects
    logger.info(f"Loading index from {index_path}...")
    df = pd.read_parquet(index_path)
    index_data = cast(list[Dict[str, Any]], df.to_dict("records"))

    # Generate the GIF
    create_preview_gif(output_path, index_data)


if __name__ == "__main__":
    visualizer()
