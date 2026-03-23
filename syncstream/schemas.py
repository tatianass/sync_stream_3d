from typing import List, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


# 1. Define the internal structure of your tags
class SemanticTags(BaseModel):
    lighting: Literal[
        "very_dark", "dark", "nominal", "bright", "overexposed", "unknown"
    ]
    is_blurry: bool
    motion_state: Literal["moving", "static"]
    data_integrity: Literal["nominal", "discontinuous_jump"]

    # Example of adding a new safety check
    @field_validator("lighting")
    @classmethod
    def check_lighting_known(cls, v):
        if v == "unknown":
            # You could raise a warning or handle it here
            pass
        return v


# 2. Main Metadata Model
class FrameMetadata(BaseModel):
    # Enforce strict types and documentation
    chunk_id: int = Field(..., ge=0, description="Sequential ID of the data chunk")
    timestamp_ns: int = Field(..., description="UNIX timestamp in nanoseconds")

    # Pose validation
    pose: List[float] = Field(..., min_length=7, max_length=7)

    # File pointers
    image_path: str
    lidar_path: str

    # Nested validation: This is the "Better" way
    tags: SemanticTags

    # For performance with NumPy types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("pose")
    @classmethod
    def validate_quaternion(cls, v):
        # Convert to numpy array for vector math
        # Assuming v[3:7] is [qx, qy, qz, qw]
        q = np.array(v[3:7])

        # Calculate the squared norm: (qx^2 + qy^2 + qz^2 + qw^2)
        squared_norm = np.dot(q, q)

        # Use a tighter tolerance (1% deviation)
        # A norm of 1.0 is a perfect rotation.
        if not (0.99 <= squared_norm <= 1.01):
            raise ValueError(
                f"Quaternion normalization failed. "
                f"Expected ~1.0, got {squared_norm:.4f}. "
                f"Data may be corrupted or incorrectly scaled."
            )

        return v
