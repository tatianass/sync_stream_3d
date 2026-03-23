from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from syncstream.chunker import DataChunker
from syncstream.constants import SyncStream3DTopic


# Helper to create the tuple returned by mcap reader.iter_decoded_messages()
def create_mock_mcap_entry(topic, log_time_ns):
    channel = MagicMock()
    channel.topic = topic

    message = MagicMock()
    message.log_time = log_time_ns

    ros_msg = MagicMock()
    if topic == SyncStream3DTopic.TRANSFORM.value:
        t_stamped = MagicMock()
        t = t_stamped.transform
        t.translation.x, t.translation.y, t.translation.z = 1.0, 2.0, 3.0
        t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w = 0.0, 0.0, 0.0, 1.0
        ros_msg.transforms = [t_stamped]

    return (None, channel, message, ros_msg)


@pytest.fixture
def chunker():
    return DataChunker(Path("dummy.mcap"), max_gap_ns=150_000_000)


def test_synchronized_chunk_generation(chunker):
    """Tests that a complete set of sensors within 150ms produces a chunk."""
    now = 1_000_000_000

    mock_data = [
        create_mock_mcap_entry(SyncStream3DTopic.CAMERA_LEFT.value, now),
        create_mock_mcap_entry(SyncStream3DTopic.LIDAR_FRONT.value, now + 10_000),
        create_mock_mcap_entry(SyncStream3DTopic.TRANSFORM.value, now + 20_000),
        # This next message is far away, triggering the finalization of the first chunk
        create_mock_mcap_entry(SyncStream3DTopic.CAMERA_LEFT.value, now + 200_000_000),
    ]

    with (
        patch("syncstream.chunker.open", mock_open(read_data=b"data")),
        patch("syncstream.chunker.make_reader") as mock_make_reader,
    ):

        reader_inst = MagicMock()
        reader_inst.iter_decoded_messages.return_value = mock_data
        mock_make_reader.return_value = reader_inst

        chunks = list(chunker.stream_synchronized_data())

        assert len(chunks) >= 1
        assert chunks[0]["chunk_id"] == 1
        assert chunks[0]["is_discontinuous"] is False
        assert chunks[0]["pose"][0] == 1.0


def test_discontinuity_sticky_flag(chunker):
    """Tests that a gap between windows correctly marks the subsequent chunk."""
    t1 = 1_000_000_000
    gap = 200_000_000
    t2 = t1 + gap

    mock_data = [
        # --- Window 1: Fully Satisfied ---
        create_mock_mcap_entry(SyncStream3DTopic.CAMERA_LEFT.value, t1),
        create_mock_mcap_entry(SyncStream3DTopic.LIDAR_FRONT.value, t1 + 1000),
        create_mock_mcap_entry(SyncStream3DTopic.TRANSFORM.value, t1 + 2000),
        # --- The Trigger: This message is 200ms late ---
        # It closes Window 1 and sets the sticky gap flag to True
        create_mock_mcap_entry(SyncStream3DTopic.LIDAR_FRONT.value, t2),
        # --- Window 2: Now we must satisfy this chunk to see it yielded ---
        create_mock_mcap_entry(SyncStream3DTopic.CAMERA_LEFT.value, t2 + 1000),
        create_mock_mcap_entry(SyncStream3DTopic.TRANSFORM.value, t2 + 2000),
        # One final message far in the future to flush Window 2
        create_mock_mcap_entry(SyncStream3DTopic.TRANSFORM.value, t2 + 300_000_000),
    ]

    with (
        patch("syncstream.chunker.open", mock_open(read_data=b"data")),
        patch("syncstream.chunker.make_reader") as mock_make_reader,
    ):

        reader_inst = MagicMock()
        reader_inst.iter_decoded_messages.return_value = mock_data
        mock_make_reader.return_value = reader_inst

        chunks = list(chunker.stream_synchronized_data())

        # Now we should have 2 chunks
        assert len(chunks) == 2
        assert chunks[0]["chunk_id"] == 1
        assert chunks[0]["is_discontinuous"] is False  # First window was tight

        assert chunks[1]["chunk_id"] == 2
        assert chunks[1]["is_discontinuous"] is True  # Second window followed the gap
