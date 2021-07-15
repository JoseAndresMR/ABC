from environments.UnityEnvironment import ABCUnityEpisodicEnvironment
from copy import deepcopy

class MetaEnvironment(object):

    def __init__(self,):
        self.environments = {}
        self.addEnvironments()

    
    def addEnvironments(self):
        empty_env = {"env" : None, "state" : [], "action" : [], "reward" : [], "done" : False, "finished" : False, "info" : {}}
        
        envs = ["reacher"]
        for env in envs:
            self.environments[env] = deepcopy(empty_env)
            self.environments[env]["env"] = ABCUnityEpisodicEnvironment()
            self.environments[env]["info"] = self.environments["reacher"]["env"].getEnvironmentInfo()

    def startEnvironmentsEpisodes(self):
        envs = ["reacher"]
        for env in envs: 
            self.environments[env]["state"] = self.environments[env]["env"].startEpisodes()

    def runSteps(self):
        envs = ["reacher"]
        for env in envs:
            env_output = self.environments[env]["env"].step()
            self.environments[env]["reward"] = env_output[0]
            self.environments[env]["state"] = env_output[1]
            self.environments[env]["done"] = env_output[2]
            self.environments[env]["finished"] = env_output[3]

    def setAction(self):
        [env["env"].setAction(env["action"]) for _, env in self.environments.items()]

    def checkAllEnvsHave(self, var):
        return all(env[var] for env in self.environments.values())

    def closeEnvironments(self):
        envs = ["reacher"]
        for env in envs:
            if self.environments[env]["finished"]:
                self.environments["reacher"]["env"].finishEnvironment()
                self.environments.pop(env)
