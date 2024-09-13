from typing import TypedDict, Optional
import numpy.typing as npt

# DRAFT. Not in use yet, see todo in CarTrackEnvironment.py
class F1tenthGroundTruthState(TypedDict):
    position: Optional[npt.ArrayLike]
    lin_vel: Optional[float]
    ang_vel: Optional [float]
    raw_lidar: Optional[npt.ArrayLike]