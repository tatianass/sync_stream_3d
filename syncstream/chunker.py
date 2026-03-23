import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, TypedDict

# Modern MCAP imports
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from syncstream.constants import SyncStream3DTopic

logger = logging.getLogger(__name__)


class ChunkData(TypedDict):
    chunk_id: int
    timestamp: int
    image_data: Any  # ROS2 Message
    lidar_data: Any  # ROS2 Message
    pose: List[float]
    is_discontinuous: bool


class DataChunker:
    """
    Handles streaming MCAP messages and grouping them into synchronized chunks.
    Validates the 150ms gap requirement between sensor messages.
    """

    def __init__(self, mcap_path: Path, max_gap_ns: int = 150_000_000):
        self.mcap_path = mcap_path
        self.max_gap_ns = max_gap_ns
        self.current_chunk_id = 0

        # Cache topic strings to avoid Enum overhead in tight loop
        self.TOPIC_IMG = SyncStream3DTopic.CAMERA_LEFT.value
        self.TOPIC_LIDAR = SyncStream3DTopic.LIDAR_FRONT.value
        self.TOPIC_TF = SyncStream3DTopic.TRANSFORM.value

        # Tracking for continuity validation
        self._last_ts: Dict[str, Optional[int]] = {
            self.TOPIC_IMG: None,
            self.TOPIC_LIDAR: None,
        }

    def stream_synchronized_data(self) -> Generator[ChunkData, None, None]:
        """
        Streams and groups messages using raw reader access for efficiency.
        Logic:
        1. Checks if a message belongs to a new time window (Sync Logic).
        2. Checks if there was a sensor dropout > 150ms (Continuity Logic).
        """
        active_chunk = self._new_chunk()
        max_gap = self.max_gap_ns
        tracked_topics = self._last_ts

        try:
            with open(self.mcap_path, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])

                for _, channel, message, ros_msg in reader.iter_decoded_messages():
                    topic = channel.topic

                    # Early exit for irrelevant topics
                    if topic not in (self.TOPIC_IMG, self.TOPIC_LIDAR, self.TOPIC_TF):
                        continue

                    current_ts = message.log_time

                    # 1. SYNCHRONIZATION WINDOW LOGIC
                    # If this message is too far from the start of the chunk, finalize the old one.
                    if active_chunk["ref_ts"] is not None:
                        if (current_ts - active_chunk["ref_ts"]) > max_gap:
                            if self._is_complete(active_chunk):
                                yield self._finalize(active_chunk)

                            # Start fresh for the new window
                            active_chunk = self._new_chunk(current_ts)

                    # 2. CONTINUITY LOGIC (Project Requirement)
                    # Check gap between previous and current message of this specific sensor.
                    if topic in tracked_topics:
                        prev = tracked_topics[topic]
                        if prev is not None:
                            if (current_ts - prev) > max_gap:
                                logger.warning(
                                    f"Sensor gap > 150ms for chunk_id: {self.current_chunk_id}"
                                )
                                # Mark the current (new) chunk as the one containing the discontinuity
                                active_chunk["is_discontinuous"] = True
                        tracked_topics[topic] = current_ts

                    # 3. DATA ASSIGNMENT
                    # Ensure first message of a chunk sets the reference timestamp
                    if active_chunk["ref_ts"] is None:
                        active_chunk["ref_ts"] = current_ts

                    if topic == self.TOPIC_IMG:
                        active_chunk["image"] = ros_msg
                    elif topic == self.TOPIC_LIDAR:
                        active_chunk["lidar"] = ros_msg
                    elif topic == self.TOPIC_TF:
                        active_chunk["pose"] = self._extract_pose(ros_msg)

                # Final flush for the last buffered data
                if self._is_complete(active_chunk):
                    yield self._finalize(active_chunk)

        except Exception as e:
            logger.error(f"Error streaming MCAP: {e}")
            raise

    def _new_chunk(self, ts: Optional[int] = None) -> Dict[str, Any]:
        """Creates a fresh state for a synchronization window."""
        return {
            "image": None,
            "lidar": None,
            "pose": None,
            "ref_ts": ts,
            "is_discontinuous": False,
        }

    def _is_complete(self, chunk: Dict) -> bool:
        """Verifies all required sensors are present in the window."""
        return (
            chunk["image"] is not None
            and chunk["lidar"] is not None
            and chunk["pose"] is not None
        )

    def _finalize(self, chunk: Dict) -> ChunkData:
        """Increments ID and packages the chunk for output."""
        self.current_chunk_id += 1
        return {
            "chunk_id": self.current_chunk_id,
            "timestamp": chunk["ref_ts"],
            "image_data": chunk["image"],
            "lidar_data": chunk["lidar"],
            "pose": chunk["pose"],
            "is_discontinuous": chunk["is_discontinuous"],
        }

    def _extract_pose(self, tf_msg: Any) -> list:
        """Flattens TransformStamped into [x, y, z, qx, qy, qz, qw]."""
        if not tf_msg.transforms:
            return [0.0] * 7

        t = tf_msg.transforms[0].transform
        return [
            float(t.translation.x),
            float(t.translation.y),
            float(t.translation.z),
            float(t.rotation.x),
            float(t.rotation.y),
            float(t.rotation.z),
            float(t.rotation.w),
        ]
