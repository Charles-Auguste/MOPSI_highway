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
    """
    Our own environment, built from the racetrack_env.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "OccupancyGrid",
                "features": ['presence', 'on_road'],
                "grid_size": [[-65, 65], [-65, 65]],
                "grid_step": [3, 3],
                "as_image": False,
                "align_to_vehicle_axes": True
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": False,
                "lateral": True,
                "target_speeds": [0, 5, 10]
            },
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 300,
            "number_of_lane" : 2,
            "collision_reward": -1,
            "lane_centering_cost": 4,
            "action_reward": -0.3,
            "controlled_vehicles": 1,
            "other_vehicles": 20,
            "screen_width": 1500,
            "screen_height": 1000,
            "centering_position": [0.5, 0.5],
        })
        return config

    def _reward(self, action: np.ndarray) -> float:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        lane_centering_reward = 1/(1+self.config["lane_centering_cost"]*lateral**2)
        action_reward = self.config["action_reward"]*np.linalg.norm(action)
        reward = lane_centering_reward \
            + action_reward \
            + self.config["collision_reward"] * self.vehicle.crashed
        reward = reward if self.vehicle.on_road else self.config["collision_reward"]
        return utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.steps >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:

        nb_lane = self.config["number_of_lane"]
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10]

        #===============================================================================================================
        # Straight Lane #1
        if nb_lane == 1:
            lane1 = StraightLane([0, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS), width=5,
                                 speed_limit=speedlimits[1])
            self.lane1 = lane1
            net.add_lane("ENPC", "ESIEE", lane1)
        else :
            lane1 = StraightLane([0, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5,
                                 speed_limit=speedlimits[1])
            self.lane1 = lane1
            net.add_lane("ENPC", "ESIEE", lane1)
            for i in range(nb_lane-2):
                net.add_lane("ENPC", "ESIEE",
                             StraightLane([0, 5 * (i + 1)], [100, 5 * (i + 1)],
                                          line_types=(LineType.STRIPED, LineType.STRIPED), width=5,
                                          speed_limit=speedlimits[1]))
            net.add_lane("ENPC", "ESIEE",
                         StraightLane([0, 5 * (nb_lane-1)], [100, 5 * (nb_lane-1)],
                                      line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                      speed_limit=speedlimits[1]))
        # ===============================================================================================================
        # 2 - Circular Arc #1
        center1 = [100, -30]
        radii1 = 30
        if nb_lane == 1:
            net.add_lane("ESIEE", "TV",
                         CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-1), width=5, clockwise=False,
                                      line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS), speed_limit=speedlimits[2]))
        else :
            net.add_lane("ESIEE", "TV",
                         CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-1), width=5, clockwise=False,
                                      line_types=(LineType.CONTINUOUS, LineType.NONE),
                                      speed_limit=speedlimits[2]))
            for i in range(nb_lane - 2):
                net.add_lane("ESIEE", "TV",
                             CircularLane(center1, radii1 + 5*(i+1), np.deg2rad(90), np.deg2rad(-1), width=5, clockwise=False,
                                          line_types=(LineType.STRIPED, LineType.NONE), speed_limit=speedlimits[2]))
            net.add_lane("ESIEE", "TV",
                         CircularLane(center1, radii1 + 5*(nb_lane - 1), np.deg2rad(90), np.deg2rad(-1), width=5, clockwise=False,
                                      line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=speedlimits[2]))
        # ===============================================================================================================
        # 3 - Circular Arc #2
        center2 = center1
        radii2 = radii1
        if nb_lane == 1 :
            net.add_lane("TV", "BOISDELET",
                         CircularLane(center2, radii2, np.deg2rad(-1), np.deg2rad(-90), width=5,
                                      clockwise=False, line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                                      speed_limit=speedlimits[3]))
        else :
            net.add_lane("TV", "BOISDELET",
                         CircularLane(center2, radii2, np.deg2rad(-1), np.deg2rad(-90), width=5,
                                      clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                      speed_limit=speedlimits[3]))
            for i in range(nb_lane - 2):
                net.add_lane("TV", "BOISDELET",
                             CircularLane(center2, radii2 + 5*(i+1), np.deg2rad(-1), np.deg2rad(-90), width=5,
                                          clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                          speed_limit=speedlimits[3]))
            net.add_lane("TV", "BOISDELET",
                         CircularLane(center2, radii2 + 5 * (nb_lane - 1), np.deg2rad(-1), np.deg2rad(-90), width=5,
                                      clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                      speed_limit=speedlimits[3]))
        # ===============================================================================================================
        # 4 - Straight Line #2
        if nb_lane == 1:
            net.add_lane("BOISDELET", "BOULANGERIE",
                         StraightLane([100, -60], [0, -60], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS), width=5,
                                      speed_limit=speedlimits[4]))
        else :
            net.add_lane("BOISDELET", "BOULANGERIE",
                         StraightLane([100, -60], [0, -60], line_types=(LineType.CONTINUOUS, LineType.NONE),
                                      width=5,
                                      speed_limit=speedlimits[4]))
            for i in range(nb_lane - 2):
                net.add_lane("BOISDELET", "BOULANGERIE",
                             StraightLane([100, -60 - 5*(i+1)], [0, -60 - 5*(i+1)], line_types=(LineType.STRIPED, LineType.NONE),
                                          width=5,
                                          speed_limit=speedlimits[4]))
            net.add_lane("BOISDELET", "BOULANGERIE",
                         StraightLane([100, -60 - 5*(nb_lane - 1)], [0, -60 - 5*(nb_lane - 1)], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                      speed_limit=speedlimits[4]))

        # ===============================================================================================================
        # 5 - Circular Arc #3
        center3 = [0, -30]
        radii3 = 30
        if nb_lane == 1 :
            net.add_lane("BOULANGERIE", "RER",
                         CircularLane(center3, radii3, np.deg2rad(270), np.deg2rad(181), width=5,
                                      clockwise=False, line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                                      speed_limit=speedlimits[5]))
        else :
            net.add_lane("BOULANGERIE", "RER",
                         CircularLane(center3, radii3, np.deg2rad(270), np.deg2rad(181), width=5,
                                      clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                      speed_limit=speedlimits[5]))
            for i in range (nb_lane - 2):
                net.add_lane("BOULANGERIE", "RER",
                             CircularLane(center3, radii3 + 5*(i+1), np.deg2rad(270), np.deg2rad(181), width=5,
                                          clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                          speed_limit=speedlimits[5]))
            net.add_lane("BOULANGERIE", "RER",
                         CircularLane(center3, radii3 + 5*(nb_lane - 1), np.deg2rad(270), np.deg2rad(181), width=5,
                                      clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                      speed_limit=speedlimits[5]))

        # ===============================================================================================================
        # 6 - Circular Arc #4
        center4 = center3
        radii4 = radii3
        if nb_lane == 1 :
            net.add_lane("RER", "ENPC",
                         CircularLane(center4, radii4 , np.deg2rad(90), np.deg2rad(180), width=5,
                                      clockwise=True, line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                                      speed_limit=speedlimits[6]))
        else:
            net.add_lane("RER", "ENPC",
                         CircularLane(center4, radii4 , np.deg2rad(90), np.deg2rad(180), width=5,
                                      clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                      speed_limit=speedlimits[6]))
            for i in range (nb_lane - 2):
                net.add_lane("RER", "ENPC",
                             CircularLane(center4, radii4 + 5*(i+1), np.deg2rad(90), np.deg2rad(180), width=5,
                                          clockwise=True, line_types=(LineType.NONE, LineType.STRIPED),
                                          speed_limit=speedlimits[6]))
            net.add_lane("RER", "ENPC",
                         CircularLane(center4, radii4 + 5 * (nb_lane - 1), np.deg2rad(90), np.deg2rad(180), width=5,
                                      clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                      speed_limit=speedlimits[6]))
        # ===============================================================================================================
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random
        nb_lane = self.config["number_of_lane"]

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = ("ENPC", "ESIEE", rng.randint(1)) if i == 0 else \
                self.road.network.random_lane_index(rng)
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=None,
                                                                             longitudinal=rng.uniform(20, 50))

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        # Front vehicle
        vehicle = IDMVehicle.make_on_lane(self.road, ("ESIEE", "TV", lane_index[-1]),
                                          longitudinal=rng.uniform(
                                              low=0,
                                              high=self.road.network.get_lane(("ESIEE", "TV", 0)).length
                                          ),
                                          speed=6+rng.uniform(high=3))
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in range(rng.randint(self.config["other_vehicles"])):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=rng.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=6+rng.uniform(high=3))
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)



register(
    id='mopsi-env-v0',
    entry_point='highway_env.envs:MopsiEnv',
)