import json, os

from environments.UnityEpisodicEnvironment import UnityEpisodicEnvironment
from environments.GymEpisodicEnvironment import GymEpisodicEnvironment
from copy import deepcopy

class MetaEnvironment(object):

    def __init__(self,):
        with open(os.path.join(os.path.dirname(__file__),'envs_config.json'), 'r') as j:
            self.config = json.load(j)
        self.environments = {}
        self.addEnvironments()

    
    def addEnvironments(self):
        empty_env = {"env" : None, "state" : [], "action" : [], "reward" : [], "done" : False, "finished" : False, "info" : {}, "active" : False}
        
        envs = self.config["environments"]
        for env in envs:
            self.environments[env["id"]] = deepcopy(empty_env)
            if env["origin"] == "unity":
                if env["temporality"] == "episodic":
                    self.environments[env["id"]]["env"] = UnityEpisodicEnvironment(env["file_path"], env["id"])
            if env["origin"] == "gym":
                if env["temporality"] == "episodic":
                    self.environments[env["id"]]["env"] = GymEpisodicEnvironment(env["id"], env["name"])
            self.environments[env["id"]]["info"] = self.environments[env["id"]]["env"].getEnvironmentInfo()

    def startEnvironmentsEpisodes(self):
        envs = self.config["schedule"]
        for env in envs:
            if env["active"]:
                self.environments[env["env"]]["active"] = True
                self.environments[env["env"]]["state"] = self.environments[env["env"]]["env"].startEpisodes(env["max_episodes"], env["max_t"], env["success_avg"])

    def runSteps(self):
        for _, env in self.environments.items():
            if env["active"]:
                env_output = env["env"].step()
                env["reward"] = env_output[0]
                env["state"] = env_output[1]
                env["done"] = env_output[2]
                env["finished"] = env_output[3]

    def setAction(self):
        [env["env"].setAction(env["action"]) for _, env in self.environments.items()]

    def checkAllEnvsHave(self, var):
        return all(env[var] for env in self.environments.values())

    def closeEnvironments(self):
        envs = self.config["environments"]
        for env in envs:
            if self.environments[env["id"]]["finished"]:
                self.environments[env["id"]]["env"].finishEnvironment()
                self.environments.pop(env["id"])
