import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SyncStream3DDataset(Dataset):
    def __init__(self, index_file: Path, transform=None):
        self.df = pd.read_parquet(index_file)
        self.root_dir = index_file.parent
        # Default transform to ensure images are Tensors if no transform is provided
        self.transform = transform or transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        img_path = self.root_dir / row["image_path"]
        image = cv2.imread(str(img_path))
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image)  # Now a Tensor [3, H, W]

        # 2. Load LiDAR
        lidar_path = self.root_dir / row["lidar_path"]
        lidar_data = np.load(lidar_path)

        # 3. Load Pose [x, y, z, qx, qy, qz, qw]
        pose = torch.tensor(
            [
                row["pose_x"],
                row["pose_y"],
                row["pose_z"],
                row["pose_qx"],
                row["pose_qy"],
                row["pose_qz"],
                row["pose_qw"],
            ],
            dtype=torch.float32,
        )

        return {
            "image": image,
            "lidar": torch.from_numpy(lidar_data).float(),
            "pose": pose,
            "tags": json.loads(row["semantic_tags"]),
            "timestamp": row["timestamp_ns"],
        }


def collate_fn(batch):
    """
    Handles batching of synchronized robotics data.
    Stacks images and poses into contiguous tensors.
    Keeps LiDAR as a list of tensors to handle variable point cloud density.
    """
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "pose": torch.stack([item["pose"] for item in batch]),
        "lidar": [item["lidar"] for item in batch],
        "tags": [item["tags"] for item in batch],
        "timestamps": [item["timestamp"] for item in batch],
    }


if __name__ == "__main__":
    from pathlib import Path

    from torch.utils.data import DataLoader

    # 1. Initialize Dataset
    dataset = SyncStream3DDataset(index_file=Path("output/index.parquet"))

    # 2. Setup DataLoader with your custom collate
    train_loader = DataLoader(
        dataset,
        batch_size=4,  # Small batch for clear printing
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 3. Grab the first batch
    batch = next(iter(train_loader))

    print("--- Batch Tensor Shapes ---")
    print(f"Images Tensor: {batch['image'].shape}")  # Should be [4, 3, H, W]
    print(f"Poses Tensor:  {batch['pose'].shape}")  # Should be [4, 7]

    print("\n--- Variable Length Data (Lists) ---")
    print(f"LiDAR List Length: {len(batch['lidar'])}")
    print(f"First LiDAR Scan Shape: {batch['lidar'][0].shape}")

    print("\n--- Metadata Example ---")
    print(f"First Sample Timestamp: {batch['timestamps'][0]}")
    print(f"First Sample Tags:      {batch['tags'][0]}")
