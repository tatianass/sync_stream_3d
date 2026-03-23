import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from main import main
from syncstream.constants import SyncStream3DTopic

# Ensure the project root is in the system path so we can import 'main'
root_dir = Path(__file__).resolve().parents[1]
root_str = str(root_dir)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


def create_mcap_tuple(topic, ts):
    """
    Creates the (schema, channel, message, ros_msg) tuple
    that the modern MCAP reader yields.
    """
    schema = MagicMock()
    channel = MagicMock()
    channel.topic = topic

    # Acessing log_time directly on the message object, as expected by chunker.py
    message = MagicMock()
    message.log_time = ts

    ros_msg = MagicMock()
    if topic == "/tf":
        # Structure required by chunker._extract_pose
        t_obj = MagicMock()
        t_obj.transform.translation.x = 1.0
        t_obj.transform.translation.y = 2.0
        t_obj.transform.translation.z = 3.0
        t_obj.transform.rotation.x = 0.0
        t_obj.transform.rotation.y = 0.0
        t_obj.transform.rotation.z = 0.0
        t_obj.transform.rotation.w = 1.0
        ros_msg.transforms = [t_obj]
    elif "/points" in topic:
        # Structure required by converter._ros_to_numpy_pc2
        ros_msg.data = b"\x00" * 32
        ros_msg.point_step = 16
    elif "/compressed" in topic:
        # Structure required by converter._ros_to_numpy_image
        ros_msg.data = b"fake_jpg_bytes"

    return (schema, channel, message, ros_msg)


def test_full_pipeline_flow(tmp_path):
    # 1. Setup mock environment
    input_file = tmp_path / "test.mcap"
    input_file.write_bytes(b"mcap_header_mock")
    output_dir = tmp_path / "output"

    # Use a specific timestamp
    now_ns = 1600000000000000000
    # Add a message 200ms later to exceed the 150ms gap threshold
    later_ns = now_ns + 200_000_000

    # 2. Create the data stream (3 messages for a chunk + 1 to trigger the gap)
    mock_tuples = [
        create_mcap_tuple(SyncStream3DTopic.CAMERA_LEFT.value, now_ns),
        create_mcap_tuple(SyncStream3DTopic.LIDAR_FRONT.value, now_ns),
        create_mcap_tuple(SyncStream3DTopic.TRANSFORM.value, now_ns),
        create_mcap_tuple(SyncStream3DTopic.TRANSFORM.value, later_ns),
    ]

    # CLI arguments matching main.py
    test_args = [
        "main.py",
        "--input",
        str(input_file),
        "--output",
        str(output_dir),
        "--workers",
        "1",  # Use 1 worker to ensure synchronous processing during test
    ]

    # 3. Patch and execute
    # We patch PerformanceMonitor to avoid any timing or telemetry issues in the test
    with (
        patch("sys.argv", test_args),
        patch("syncstream.chunker.make_reader") as mock_make_reader,
        patch("main.PerformanceMonitor", MagicMock()),
        patch("cv2.imdecode", return_value=np.zeros((10, 10, 3), dtype=np.uint8)),
        patch("cv2.imwrite", return_value=True),
    ):

        mock_reader = MagicMock()
        mock_reader.iter_decoded_messages.return_value = mock_tuples

        # FIX: make_reader returns the reader object directly,
        # it is not a context manager in your chunker.py implementation.
        mock_make_reader.return_value = mock_reader

        main()

    # 4. Assertions
    index_file = output_dir / "index.parquet"
    assert (
        index_file.exists()
    ), f"Failed to create index.parquet. Found: {list(output_dir.rglob('*'))}"

    import pandas as pd

    df = pd.read_parquet(index_file)
    assert len(df) >= 1, "The index was created but contains no samples."
    assert df.iloc[0]["pose_x"] == 1.0
