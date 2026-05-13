"""Write joint trajectories to CSV files."""

from typing import Dict
import numpy as np
from .trajectory_generator import JOINT_NAMES

_EXTRA_COLUMNS = ["left_foot_step", "right_foot_step"]


def write_csv(
    filepath: str,
    sample_times: np.ndarray,
    trajectories: Dict[str, np.ndarray],
):
    """Write trajectories to a CSV file with header.

    Format: timestamp, joint_1, ..., joint_N [, left_foot_step, right_foot_step]
    """
    columns = list(JOINT_NAMES)
    for col in _EXTRA_COLUMNS:
        if col in trajectories:
            columns.append(col)

    header = ["timestamp"] + columns
    data = np.column_stack([sample_times] + [trajectories[n] for n in columns])

    np.savetxt(filepath, data, delimiter=",", header=",".join(header),
               fmt="%.6f", comments="")
