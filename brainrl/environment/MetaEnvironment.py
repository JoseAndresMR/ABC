import json, os

from .UnityEpisodicEnvironment import UnityEpisodicEnvironment
from .GymEpisodicEnvironment import GymEpisodicEnvironment
from copy import deepcopy

class MetaEnvironment(object):
    """
    MetaEnvironment gathers and manages all the environments currently active in the current experience.
    It serves as a common interface with brain.
    """

    def __init__(self,config, log_path):
        """MetaEnvironment gathers and manages all the environments currently active in the current experience.
        It serves as a common interface with brain.

        Args:
            config (list of dicts): Configuration of the environments present in the experience. Contents of each env:
                - origin (string): Wether it is a unity or gym environment.
                - id (string): Identification of the experience.
                - temporality (string): Whether it is an episodic or continuous env.
                - others depending on the origin.
            log_path (string): Path on disk to store gathered information about the experience
        """
        self.config = config
        self.log_path = log_path
        self.environments = {}
        self.active_envs = 0
        self.addEnvironments()

    def addEnvironments(self):
        """
        Create and start all the environments required in this experience given the configuration.
        TODO: Dynamically create and close the envs when more complex schedulres are required.
        """
        empty_env = {"env" : None, "state" : None, "action" : None, "reward" : None, "done" : False, "finished" : False, "info" : {}, "active" : False}
        
        envs = self.config["environments"]
        for env in envs:
            self.environments[env["id"]] = deepcopy(empty_env) ### Test if deepcopy can be deleted
            if env["origin"] == "unity":
                if env["temporality"] == "episodic":
                    self.environments[env["id"]]["env"] = UnityEpisodicEnvironment(env["file_path"], env["id"], self.log_path)
            if env["origin"] == "gym":
                if env["temporality"] == "episodic":
                    self.environments[env["id"]]["env"] = GymEpisodicEnvironment(env["id"], env["name"], self.log_path)
            self.environments[env["id"]]["info"] = self.environments[env["id"]]["env"].getEnvironmentInfo()

    def startEnvironmentsEpisodes(self):
        """ Begin the first step of currently active environments. """
        envs_config= self.config["schedule"]
        for env_config in envs_config:
            if env_config["active"]:
                self.environments[env_config["env"]]["active"] = True
                self.environments[env_config["env"]]["state"] = self.environments[env_config["env"]]["env"].startEpisodes(env_config["max_episodes"], env_config["max_t"], env_config["success_avg"])

    def runSteps(self):
        """ Take one more step in the currently active environments. """
        for _, env in self.environments.items():
            if env["active"]:
                env_output = env["env"].step()
                env["reward"] = env_output[0]
                env["state"] = env_output[1]
                env["done"] = env_output[2]
                env["finished"] = env_output[3]

    def setAction(self):
        """ Transports the action information from the information object in this class to inside the environment classes. """
        [env["env"].setAction(env["action"]) for _, env in self.environments.items()]

    def closeEnvironments(self):
        """ Closes the environments once they are finished. """
        envs = self.config["environments"]
        for env in envs:
            if self.environments[env["id"]]["finished"]:
                self.environments[env["id"]]["env"].finishEnvironment()
                self.environments.pop(env["id"])

        return not bool(self.environments)
