"""
One timeline exposition of a brain to different environments following a schedule
"""
import sys,json, os
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np

from environments.MetaEnvironment import MetaEnvironment
from brain.Brain import Brain

class Experience(object):

    def __init__(self, config, log_path):
        """One timeline exposition of a brain to different environments following a schedule

        Args:
            config (dict): Configuration of the whole experience. Contents:
                - base_configs (dict): config not altered. Changed by alterations key.
                    - envs: configuration of environments present on the experience
                    - brain: configuration of the single brain
                    - schedule: timeline of the exposition and connections
                - alterations (list): temporal sequence of changes produced to base_configs
                - ID (string): identification of the experience
            log_path (string): Path on disk to store gathered information about the experience
        """
        self.meta_environment = MetaEnvironment({"environments": config["envs"], "schedule": config["schedule"]}, log_path)
        self.brain = Brain(config["brain"], log_path)
        self.config = config

    def loop(self):
        """
        Perform the experience. Every spin, the metaenvironment takes a step and the observation is passed to the brain.
        It learns from prior experiences and decides new actions, whick are transferred to the metaenvironment.
        """
        print("Experience: Starting loop")
        try:
            self.meta_environment.startEnvironmentsEpisodes() ### TODO: dynamically start environments on env schedule
            for spin in range(999999999999):
                self.allocateEnvironementOutput()
                self.brain.forward()
                self.allocateBrainOutput()
                self.meta_environment.runSteps()
                self.meta_environment.closeEnvironments()
        except AssertionError as error:
            print(error)
            self.meta_environment.closeEnvironments()

    def allocateEnvironementOutput(self):
        """
        Takes the observation from the metaenvionment - environments information object and maps it into brain information object.
        """
        for env_conf in self.config["schedule"]:
            if env_conf["active"]:
                first_dim = self.meta_environment.environments[env_conf["env"]]["state"].shape[0]
                for map in env_conf["signals_map"]["state"]:
                    env_output = map["env_output"]
                    neuron_input = map["neuron_input"]
                    self.brain.neurons[map["neuron_type"]][map["neuron"]-1]["state"] = addMatrixToTarget(self.brain.neurons[map["neuron_type"]][map["neuron"]-1]["state"],
                                                                                                neuron_input,
                                                                                                self.meta_environment.environments[env_conf["env"]]["state"][:, env_output[0]-1: env_output[1]])
                for map in env_conf["signals_map"]["action"]:
                    if self.meta_environment.environments[env_conf["env"]]["reward"] != []:
                        neuron_output = map["neuron_output"]
                        self.brain.neurons[map["neuron_type"]][map["neuron"]-1]["reward"] = addMatrixToTarget(self.brain.neurons[map["neuron_type"]][map["neuron"]-1]["reward"],
                                                                                                    neuron_output,
                                                                                                    np.ones((first_dim, neuron_output[1]-neuron_output[0] + 1))*self.meta_environment.environments[env_conf["env"]]["reward"][0])
        self.brain.setStateAndReward()

    def allocateBrainOutput(self):
        """
        Takes the actions decided by the brain in the information object and maps it into the metaenvionment - environments information object.
        """
        for env_conf in self.config["schedule"]:
            if env_conf["active"]:
                for map in env_conf["signals_map"]["action"]:
                    if self.brain.neurons[map["neuron_type"]][map["neuron"]-1]["action"].size != 0:
                        neuron_output = map["neuron_output"]
                        env_input = map["env_input"]
                        self.meta_environment.environments[env_conf["env"]]["action"] = addMatrixToTarget(self.meta_environment.environments[env_conf["env"]]["action"],
                                                                                            env_input,
                                                                                            self.brain.neurons[map["neuron_type"]][map["neuron"]-1]["action"][:, neuron_output[0]-1: neuron_output[1]])
        self.meta_environment.setAction()

def addMatrixToTarget(target_matrix, target_dim, added_matrix):
    """ Adds a matrix in the desired cooredinates of a bigger matrix

    Args:
        target_matrix (np.array): Big matrix in which added_matrix is included
        target_dim (list): Coordinates to place added_matrix in target_matrix
        added_matrix (np.array): Small matrix included in target_matrix

    Returns:
        target_matrix with added_matrix included
    """
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