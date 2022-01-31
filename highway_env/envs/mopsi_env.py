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
            "collision_reward": -1,
            "lane_centering_cost": 4,
            "action_reward": -0.3,
            "controlled_vehicles": 1,
            "other_vehicles": 1,
            "screen_width": 1000,
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
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # Straight Lane #1
        lane1 = StraightLane([0, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5, speed_limit=speedlimits[1])
        self.lane1 = lane1

        net.add_lane("ENPC", "ESIEE", lane1)
        net.add_lane("ENPC", "ESIEE", StraightLane([0, 5], [100, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[1]))


        # 2 - Circular Arc #1
        center1 = [100, -30]
        radii1 = 30
        net.add_lane("ESIEE", "TV",
                     CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("ESIEE", "TV",
                     CircularLane(center1, radii1 + 5, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))

        # 3 - Circular Arc #2
        center2 = center1
        radii2 = radii1
        net.add_lane("TV", "BOISDELET",
                     CircularLane(center2, radii2, np.deg2rad(-1), np.deg2rad(-90), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))

        net.add_lane("TV", "BOISDELET",
                     CircularLane(center2, radii2 + 5, np.deg2rad(-1), np.deg2rad(-90), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))

        # 4 - Straight Line #2
        lane2 = StraightLane([100, -60], [0, -60], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5, speed_limit=speedlimits[1])
        self.lane2 = lane2

        # Add Lanes to Road Network - Straight Section
        net.add_lane("BOISDELET", "BOULANGERIE", lane2)
        net.add_lane("BOISDELET", "BOULANGERIE", StraightLane([100, -65], [0, -65], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[1]))


        # 5 - Circular Arc #3
        center3 = [50, -30]
        radii3 = 30
        net.add_lane("BOULANGERIE", "RER",
                     CircularLane(center3, radii3, np.deg2rad(-270), np.deg2rad(181), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[4]))
        net.add_lane("BOULANGERIE", "RER",
                     CircularLane(center3, radii3+5, np.deg2rad(-270), np.deg2rad(181), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[4]))


        # 6 - Circular Arc #4
        center4 = center3
        radii4 = radii3
        net.add_lane("RER", "ENPC",
                     CircularLane(center4, radii4+5, np.deg2rad(180), np.deg2rad(90), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[5]))
        net.add_lane("RER", "ENPC",
                     CircularLane(center4, radii4, np.deg2rad(180), np.deg2rad(90), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[5]))

        # # 6 - Slant
        # net.add_lane("f", "g", StraightLane([55.7, -15.7], [35.7, -35.7],
        #                                     line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
        #                                     speed_limit=speedlimits[6]))
        # net.add_lane("f", "g", StraightLane([59.3934, -19.2], [39.3934, -39.2],
        #                                     line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
        #                                     speed_limit=speedlimits[6]))
        #
        # # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
        # center4 = [18.1, -18.1]
        # radii4 = 25
        # net.add_lane("g", "h",
        #              CircularLane(center4, radii4, np.deg2rad(315), np.deg2rad(170), width=5,
        #                           clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
        #                           speed_limit=speedlimits[7]))
        # net.add_lane("g", "h",
        #              CircularLane(center4, radii4+5, np.deg2rad(315), np.deg2rad(165), width=5,
        #                           clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
        #                           speed_limit=speedlimits[7]))
        # net.add_lane("h", "i",
        #              CircularLane(center4, radii4, np.deg2rad(170), np.deg2rad(56), width=5,
        #                           clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
        #                           speed_limit=speedlimits[7]))
        # net.add_lane("h", "i",
        #              CircularLane(center4, radii4+5, np.deg2rad(170), np.deg2rad(58), width=5,
        #                           clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
        #                           speed_limit=speedlimits[7]))
        #
        # # 8 - Circular Arc #5 - Reconnects to Start
        # center5 = [43.2, 23.4]
        # radii5 = 18.5
        # net.add_lane("i", "a",
        #              CircularLane(center5, radii5+5, np.deg2rad(240), np.deg2rad(270), width=5,
        #                           clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
        #                           speed_limit=speedlimits[8]))
        # net.add_lane("i", "a",
        #              CircularLane(center5, radii5, np.deg2rad(238), np.deg2rad(268), width=5,
        #                           clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
        #                           speed_limit=speedlimits[8]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = ("ENPC", "ESIEE", rng.randint(2)) if i == 0 else \
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