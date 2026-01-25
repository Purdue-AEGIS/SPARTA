# from enum import Enum
from enum import StrEnum
import numpy as np


# type of study
class StudyType(StrEnum):
    DYNAMIC = "dynamic"
    POWERFLOW = "powerflow"


# edge at which a node
class NodeSide(StrEnum):
    FROM = "from"
    TO = "to"
    AT = "at"


phase_angle_map = {"A": 0, "B": -2 * np.pi / 3, "C": 2 * np.pi / 3}


w_nominal = None