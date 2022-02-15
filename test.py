import random
import numpy as np
from gym.utils import seeding

class Test_random ():
    def __init__(self):
        self.np_random = None
        self.seed()

    def seed(self, seed: int = None) -> list[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == "__main__":
    rand1 = Test_random()
    rng = rand1.np_random
    print(rng.uniform(0,20))


