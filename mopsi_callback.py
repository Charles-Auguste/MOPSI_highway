from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from stable_baselines3.common.results_plotter import ts2xy

"""
class MopsiCallback_multi_core(BaseCallback):
    def __init__(self, nb_step: int, log_dir: str, verbose: int = 1):
        super(MopsiCallback_multi_core, self).__init__(verbose)
        self.nb_step = nb_step
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'last_save_state_')

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.nb_step == 0:
            self.save_path = os.path.join(self.log_dir, 'last_save_state' + str(self.n_calls) + "it")
            self.model.save(self.save_path)
        return True
"""

class MopsiCallback_single_core(BaseCallback):
    def __init__(self, nb_step: int, log_dir: str, env = None, verbose: int = 1):
        super(MopsiCallback_single_core, self).__init__(verbose)
        self.nb_step = nb_step
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'last_save_state')
        self.current_path = self.save_path
        self.env = env

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.nb_step == 0 :
            list_dir = os.listdir(self.save_path)
            if len(list_dir) >= 1:
                os.remove(os.path.join(self.save_path,list_dir[-1]))
            self.current_path = os.path.join(self.save_path, "model_" + str(self.n_calls) +"it")
            self.model.save(self.current_path)
        action_dict = self.env.action_reward
        self.logger.record("lane_centering_reward",action_dict["lane_centering"])
        self.logger.record("speed_reward", action_dict["speed_reward"])
        self.logger.record("action_reward", action_dict["action_reward"])
        self.logger.record("total_reward", action_dict["reward"])
        return True

if __name__ == "__main__" :
    pass
