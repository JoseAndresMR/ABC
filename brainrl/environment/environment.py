import os
from distutils import util
from torch.utils.tensorboard import SummaryWriter

class Environment(object):
    """ Parent class for every type of environment. Manages their common information and functions. """

    def __init__(self, id, log_path: str):
        self.tensorboard_writer = SummaryWriter(os.path.join(log_path, "envs", "{}".format(id)))

    def get_environment_info(self):
        return self.env_info

    def set_action(self, action):
        self.actions = action
