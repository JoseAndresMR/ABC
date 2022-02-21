"""
One timeline exposition of a brain to different environments following a schedule
"""
import numpy as np
from brainrl.environment import MetaEnvironment
from brainrl.brain import Brain


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
        self.meta_environment = MetaEnvironment(config={"environments": config["envs"],
                                                        "schedule": config["schedule"]},
                                                log_path=log_path)
        self.brain = Brain(config["brain"], log_path)
        self.config = config

    def loop(self, max_iterations=999999999999):
        """
        Perform the experience. Every spin, the metaenvironment takes a step and the observation is passed to the brain.
        It learns from prior experiences and decides new actions, whick are transferred to the metaenvironment.
        """
        print("Experience: Starting loop")
        metaenv_finished = False
        try:
            # TODO: dynamically start environments on env schedule
            self.meta_environment.start_environments_episodes()
            for spin in range(max_iterations):
                if spin % 10000 == 0:
                    debug_flag = True
                self.allocate_environement_output()
                metaenv_finished = self.meta_environment.close_environments()
                if metaenv_finished:
                    break
                self.brain.forward()
                self.allocate_brain_output()
                self.meta_environment.run_steps()
        except AssertionError as error:
            print(error)
            self.meta_environment.close_environments()

        return spin

    def allocate_environement_output(self):
        """
        Takes the observation from the metaenvionment - environments information object and maps it into brain information object.
        """
        for env_conf in self.config["schedule"]:
            if env_conf["active"]:
                first_dim = self.meta_environment.environments[env_conf["env"]
                                                               ]["state"].shape[0]
                for map in env_conf["signals_map"]["state"]:
                    env_output = map["env_output"]
                    neuron_input = map["neuron_input"]
                    self.brain.neurons[map["neuron_type"]][map["neuron"]-1]["state"] = add_matrix_to_target(self.brain.neurons[map["neuron_type"]][map["neuron"]-1]["state"],
                                                                                                            neuron_input,
                                                                                                            self.meta_environment.environments[env_conf["env"]]["state"][:, env_output[0]-1: env_output[1]])
                for map in env_conf["signals_map"]["action"]:
                    if self.meta_environment.environments[env_conf["env"]]["reward"] != []:
                        neuron_output = map["neuron_output"]
                        self.brain.neurons[map["neuron_type"]][map["neuron"]-1]["reward"] = add_matrix_to_target(self.brain.neurons[map["neuron_type"]][map["neuron"]-1]["reward"],
                                                                                                                 neuron_output,
                                                                                                                 np.ones((first_dim, neuron_output[1]-neuron_output[0] + 1))*self.meta_environment.environments[env_conf["env"]]["reward"][0])
        self.brain.set_state_and_reward()

    def allocate_brain_output(self):
        """
        Takes the actions decided by the brain in the information object and maps it into the metaenvionment - environments information object.
        """
        for env_conf in self.config["schedule"]:
            if env_conf["active"]:
                for map in env_conf["signals_map"]["action"]:
                    if self.brain.neurons[map["neuron_type"]][map["neuron"]-1]["action"].size != 0:
                        neuron_output = map["neuron_output"]
                        env_input = map["env_input"]
                        self.meta_environment.environments[env_conf["env"]]["action"] = add_matrix_to_target(self.meta_environment.environments[env_conf["env"]]["action"],
                                                                                                             env_input,
                                                                                                             self.brain.neurons[map["neuron_type"]][map["neuron"]-1]["action"][:, neuron_output[0]-1: neuron_output[1]])
        self.meta_environment.set_action()

    def finish(self):
        del self.meta_environment
        del self.brain
        del self.config


def add_matrix_to_target(target_matrix, target_dim, added_matrix):
    """ Adds a matrix in the desired cooredinates of a bigger matrix

    Args:
        target_matrix (np.array): Big matrix in which added_matrix is included
        target_dim (list): Coordinates to place added_matrix in target_matrix
        added_matrix (np.array): Small matrix included in target_matrix

    Returns:
        target_matrix with added_matrix included
    """
    if len(target_matrix) == 0:
        target_matrix = added_matrix
    elif target_dim[1] <= target_matrix.shape[1]:
        target_matrix[:, target_dim[0]-1:target_dim[1]] = added_matrix
    else:
        if target_dim[0]-1 == target_matrix.shape[1]:
            target_matrix = np.concatenate((target_matrix, added_matrix), 1)
        else:
            target_matrix = np.concatenate((target_matrix,
                                            np.zeros((target_matrix.shape[0],
                                                      target_dim[0] - 1 - target_matrix.shape[1])),
                                            added_matrix),
                                           1)

    return target_matrix
