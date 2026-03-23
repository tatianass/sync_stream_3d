import json
from typing import Any

import cv2
import numpy as np


class PerceptionAnalyzer:
    @staticmethod
    def get_automated_tags(
        image_msg: Any, pose: list, is_discontinuous: bool = False
    ) -> str:
        """
        Analyzes lighting, blur, motion, and data integrity.
        """
        try:
            lighting = "unknown"
            is_blurry = False

            # 1. Image Processing
            if image_msg is not None and hasattr(image_msg, "data"):
                raw_bytes = np.frombuffer(image_msg.data, np.uint8)
                img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

                    # Brightness Tiers
                    avg_b = float(np.mean(gray)) * 100 / 255
                    if avg_b < 15:
                        lighting = "very_dark"
                    elif avg_b < 30:
                        lighting = "dark"
                    elif avg_b < 70:
                        lighting = "nominal"
                    elif avg_b < 85:
                        lighting = "bright"
                    else:
                        lighting = "overexposed"

                    # Blur detection (Laplacian Variance)
                    blur_v = cv2.Laplacian(gray, cv2.CV_64F).var()
                    is_blurry = bool(blur_v < 100.0)

            # 2. Motion State (Thresholding noise)
            is_moving = any(abs(v) > 0.1 for v in pose[:3]) if pose else False

            # 3. Final JSON Package
            return json.dumps(
                {
                    "lighting": lighting,
                    "is_blurry": is_blurry,
                    "motion_state": "moving" if is_moving else "static",
                    "data_integrity": (
                        "nominal" if not is_discontinuous else "discontinuous_jump"
                    ),
                }
            )

        except Exception as e:
            return json.dumps({"error": "analysis_failed", "details": str(e)})
