import os
from torch.utils.tensorboard import SummaryWriter

class Environment(object):

    def __init__(self, id, log_path):
        self.tensorboard_writer = SummaryWriter(os.path.join(log_path,"envs", "{}".format(id)))

    def getEnvironmentInfo(self):
        return self.env_info

    def setAction(self, action):
        self.actions = action