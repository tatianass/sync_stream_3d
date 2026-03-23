from enum import Enum


class SyncStream3DTopic(Enum):
    """Enumeration of supported ROS2 topics for the SyncStream3D pipeline."""

    CAMERA_LEFT = "/sensor/camera/left/image_raw/compressed"
    LIDAR_FRONT = "/sensor/lidar/front/points"
    TRANSFORM = "/tf"
