import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd

from syncstream.chunker import ChunkData
from syncstream.schemas import FrameMetadata, SemanticTags
from syncstream.tagging import PerceptionAnalyzer
from syncstream.validation import QualityGuard

logger = logging.getLogger(__name__)


class DatasetConverter:
    """
    Handles conversion of ROS2 sensor messages (Compressed Images, LiDAR, TF)
    to disk formats and maintains the global Parquet index.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.image_dir = output_dir / "images"
        self.lidar_dir = output_dir / "lidar"

        # Create subdirectories
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.lidar_dir.mkdir(parents=True, exist_ok=True)

        self.index_data: List[Dict[str, Any]] = []
        self._lock = Lock()

    def process_and_save(self, chunk: ChunkData):
        """
        Processes a synchronized chunk.
        1. Decodes CompressedImage -> JPG
        2. Saves PointCloud2 -> NPY
        3. Updates Thread-safe index
        """
        cid = chunk["chunk_id"]
        ts = chunk["timestamp"]

        try:
            # --- 1. Process Image ---
            img = self._ros_to_numpy_image(chunk["image_data"])
            image_rel_path: Optional[str] = None

            if img is not None:
                img_filename = f"frame_{cid:06d}.jpg"
                img_path = self.image_dir / img_filename
                # Explicitly cast path to string for cv2.imwrite
                if cv2.imwrite(str(img_path), img):
                    image_rel_path = str(img_path.relative_to(self.output_dir))
                else:
                    logger.error(f"Failed to write image file: {img_path}")
            else:
                logger.warning(f"Skipping image for chunk {cid}: Decoding failed.")

            # --- 2. Process LiDAR ---
            lidar_filename = f"cloud_{cid:06d}.npy"
            lidar_path = self.lidar_dir / lidar_filename
            points = self._ros_to_numpy_pc2(chunk["lidar_data"])
            np.save(lidar_path, points)
            lidar_rel_path = str(lidar_path.relative_to(self.output_dir))

            tags_json_str = PerceptionAnalyzer.get_automated_tags(
                image_msg=chunk["image_data"],
                pose=chunk["pose"],
                is_discontinuous=chunk["is_discontinuous"],
            )
            tags_dict = json.loads(tags_json_str)

            # Validate via Pydantic
            # Using 'timestamp_ns=ts' here maps the local variable to the Model
            FrameMetadata(
                chunk_id=cid,
                timestamp_ns=ts,
                pose=chunk["pose"],
                image_path=str(image_rel_path),
                lidar_path=str(lidar_rel_path),
                tags=SemanticTags(**tags_dict),
            )

            # --- 3. Update Index (Thread-safely) ---
            with self._lock:
                self.index_data.append(
                    {
                        "chunk_id": cid,
                        "timestamp_ns": ts,
                        "image_path": image_rel_path,
                        "lidar_path": lidar_rel_path,
                        "pose_x": chunk["pose"][0],
                        "pose_y": chunk["pose"][1],
                        "pose_z": chunk["pose"][2],
                        "pose_qx": chunk["pose"][3],
                        "pose_qy": chunk["pose"][4],
                        "pose_qz": chunk["pose"][5],
                        "pose_qw": chunk["pose"][6],
                        "semantic_tags": tags_json_str,
                    }
                )

        except Exception as e:
            logger.error(f"Failed to process chunk {cid}: {e}")

    def write_index(self):
        """Finalize dataset by validating and writing the index."""
        if not self.index_data:
            logger.warning("No data found to index.")
            return

        df = pd.DataFrame(self.index_data)
        df = df.sort_values("timestamp_ns").reset_index(drop=True)

        # --- THE QUALITY GATE ---
        success, report = QualityGuard.validate_index(df)

        if success:
            logger.info(f"Quality Check Passed: {report}")
            index_path = self.output_dir / "index.parquet"
            df.to_parquet(index_path, index=False)
        else:
            logger.error(f"Quality Check FAILED: {report}")
            # Save with a warning suffix so data isn't lost
            fail_path = self.output_dir / "index_FAILED_QC.parquet"
            df.to_parquet(fail_path, index=False)

    def _ros_to_numpy_image(self, ros_compressed_img) -> Optional[npt.NDArray]:
        """Decodes a ROS2 CompressedImage (JPEG) into an OpenCV BGR array."""
        try:
            # CompressedImage.data is a bytes/uint8 array of the JPEG file
            np_arr = np.frombuffer(ros_compressed_img.data, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Image decoding error: {e}")
            return None

    def _ros_to_numpy_pc2(self, ros_pc2) -> npt.NDArray:
        """
        Decodes PointCloud2 using the exact metadata from the SyncStream3D MCAP.
        point_step = 16 (4 fields per point: x, y, z, intensity)
        """
        raw_data = np.array([])  # Initialize to avoid reference before assignment
        try:
            # Convert raw bytes to float32
            # Use frombuffer for speed, and only take the relevant data
            raw_data = np.frombuffer(ros_pc2.data, dtype=np.float32)

            # Calculate fields per point dynamically (16 / 4 = 4)
            fields_per_point = ros_pc2.point_step // 4

            # Reshape based on the width/height provided in the message
            # On reference image: width = 122766, height = 1
            return raw_data.reshape(-1, fields_per_point)

        except Exception as e:
            logger.error(f"LiDAR Reshape Error: {e} (Data size: {len(raw_data)})")
            return np.array([])
