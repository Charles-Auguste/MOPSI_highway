from itertools import repeat, product
from typing import Tuple

from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle

class MopsiEnv(AbstractEnv):
    pass



register(
    id='mopsi-env-v0',
    entry_point='highway_env.envs:MopsiEnv',
)