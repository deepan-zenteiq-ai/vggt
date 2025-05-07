from typing import List
import numpy as np
from dataclasses import dataclass

@dataclass
class TimestampedFrameSet:
    """Data class for a set of frames with a timestamp."""
    frames: List[np.ndarray]
    timestamp: float
    batch_id: int

@dataclass
class PointCloudResult:
    """Data class for a processed point cloud result."""
    points: np.ndarray
    colors: np.ndarray
    timestamp: float
    batch_id: int
    
    def __lt__(self, other):
        """For sorting by timestamp."""
        return self.timestamp < other.timestamp
