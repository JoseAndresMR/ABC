import json, os
import numpy as np

from environments.MetaEnvironment import MetaEnvironment
from brain.Brain import Brain

class Experience(object):

    def __init__(self):

        self.meta_environment = MetaEnvironment()
        self.brain = Brain()
        with open(os.path.join(os.path.dirname(__file__),'config.json'), 'r') as j:
            self.config = json.load(j)
        self.loop()

    def loop(self):
        print("Experience: Starting loop")
        try:
            self.meta_environment.startEnvironmentsEpisodes() ### dynamically start environments on env schedule
            self.allocateEnvironementOutput()
            for spin in range(999999999999):
                # print("experience loop spin:", spin)
                self.brain.forward()
                self.allocateBrainOutput()
                self.meta_environment.runSteps()
                self.allocateEnvironementOutput()
                # self.meta_environment.closeEnvironments()
        except AssertionError as error:
            print(error)
            self.meta_environment.closeEnvironments()

    def allocateEnvironementOutput(self):
        for env_conf in self.meta_environment.config["schedule"]:
            if env_conf["active"]:
                first_dim = self.meta_environment.environments[env_conf["env"]]["state"].shape[0]
                for map in env_conf["signals_map"]["state"]:
                    env_output = map["env_output"]
                    neuron_input = map["neuron_input"]
                    self.brain.neurons["sensory"][map["neuron"]-1]["state"] = addMatrixToTarget(self.brain.neurons["sensory"][map["neuron"]-1]["state"],
                                                                                                neuron_input,
                                                                                                self.meta_environment.environments[env_conf["env"]]["state"][:, env_output[0]-1: env_output[1]])
                for map in env_conf["signals_map"]["action"]:
                    if self.meta_environment.environments[env_conf["env"]]["reward"] != []:
                        neuron_output = map["neuron_output"]                    
                        self.brain.neurons["motor"][map["neuron"]-1]["reward"] = addMatrixToTarget(self.brain.neurons["motor"][map["neuron"]-1]["reward"],
                                                                                                    neuron_output,
                                                                                                    np.ones((first_dim, neuron_output[1]-neuron_output[0] + 1))*self.meta_environment.environments[env_conf["env"]]["reward"][0])
        self.brain.setStateAndReward()

    def allocateBrainOutput(self):
        for env_conf in self.meta_environment.config["schedule"]:
            for map in env_conf["signals_map"]["action"]:
                if self.brain.neurons["motor"][map["neuron"]-1]["action"] != []:
                    neuron_output = map["neuron_output"]
                    env_input = map["env_input"]
                    self.meta_environment.environments[env_conf["env"]]["action"] = addMatrixToTarget(self.meta_environment.environments[env_conf["env"]]["action"],
                                                                                        env_input,
                                                                                        self.brain.neurons["motor"][map["neuron"]-1]["action"][:, neuron_output[0]-1: neuron_output[1]])
        self.meta_environment.setAction()

def addMatrixToTarget(target_matrix, target_dim, added_matrix):
    if target_matrix == []:
        target_matrix = added_matrix
    elif target_dim[1] <= target_matrix.shape[1]:
        target_matrix[:,target_dim[0]-1:target_dim[1]] = added_matrix
    else:
        if target_dim[0]-1 == target_matrix.shape[1]:
            target_matrix = np.concatenate((target_matrix, added_matrix),1)
        else:
            target_matrix = np.concatenate((target_matrix, np.zeros((target_matrix.shape[0], target_dim[0]-1 - target_matrix.shape[1])), added_matrix),1)

    return target_matrix

Experience()