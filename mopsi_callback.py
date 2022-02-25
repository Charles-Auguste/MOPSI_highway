from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from stable_baselines3.common.results_plotter import ts2xy


class MopsiCallback_multi_core(BaseCallback):
    def __init__(self, nb_step: int, log_dir: str, verbose: int = 1, title: str = ""):
        super(MopsiCallback_multi_core, self).__init__(verbose)
        self.nb_step = nb_step
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'last_save_state_'+title)

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.nb_step == 0 :
            self.model.save(self.save_path)
        return True


class MopsiCallback_single_core(BaseCallback):
    def __init__(self, nb_step: int, log_dir: str, env = None, verbose: int = 1, title: str = ""):
        super(MopsiCallback_single_core, self).__init__(verbose)
        self.nb_step = nb_step
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'last_save_state_'+title)
        self.env = env

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.nb_step == 0 :
            self.model.save(self.save_path)
        action_dict = self.env.action_reward
        self.logger.record("lane_centering_reward",action_dict["lane_centering"])
        self.logger.record("speed_reward", action_dict["speed_reward"])
        self.logger.record("action_reward", action_dict["action_reward"])
        self.logger.record("total_reward", action_dict["reward"])
        return True

if __name__ == "__main__" :
    pass
