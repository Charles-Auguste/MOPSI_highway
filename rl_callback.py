from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from stable_baselines3.common.results_plotter import ts2xy


class SaveCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, nb_step: int, log_dir: str, verbose: int = 1):
        super(SaveCallback, self).__init__(verbose)
        self.nb_step = nb_step
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'model_after_'+str(self.nb_step))

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls == self.nb_step :
            self.model.save(self.save_path)
