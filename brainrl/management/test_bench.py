import json
import os
import copy
import shutil
from typing import NewType
import optuna

from brainrl.management.experience import Experience


def set_value_in_dict_with_path(dic, keys, value):
    for key in keys[:-1]:
        if type(dic) == dict:
            dic = dic.setdefault(key, {})
        elif type(dic) == list:
            dic = dic[key]
    dic[keys[-1]] = value


class TestBench(object):
    """
    Class to run batches of experiments following different settings.
    """

    def __init__(self, config_folder_path, data_folder_path):
        """
        Define initial configurations.
        """

        self.config_path = config_folder_path
        with open(os.path.join(self.config_path,"management",'config.json'), 'r') as j:
            self.base_config = json.load(j)
        self.log_path = os.path.join(data_folder_path,"runs", self.base_config["id"])

        self.expandJsons(self.base_config)

    def expandJsons(self, nested_dict, prepath=()):
        if type(nested_dict) == list:
            for i,v in enumerate(nested_dict):
                path = prepath + (i,)
                self.expandJsons(v, path) # recursive call
        elif type(nested_dict) == dict:
            for k, v in nested_dict.items():
                path = prepath + (k,)
                if type(v) == str and len(v) >= 5 and v[-5:] == ".json": # found json
                    with open(os.path.join(self.config_path,v), 'r') as j:
                        set_value_in_dict_with_path(self.base_config, path, json.load(j))
                    v = nested_dict[k]
                if hasattr(v, 'items') or type(v) == list: # v is a dict or list
                    self.expandJsons(v, path) # recursive call

    def experiences(self, max_iterations=999999999999, n_trials=15):
        """ Run all experiments in sequence, changing the alterations from the prior base configuration. """
        i = 0
        for experiment_config in self.base_config["experiments"]:
            def objective(trial):
                for variable_name, variable_params in self.test_variables.items():
                    suggested_values = variable_params["suggested_values"]
                    x = trial.suggest_float(variable_name, suggested_values[0], suggested_values[1])
                    set_value_in_dict_with_path(current_config, variable_params["path"], x)

                exp = Experience(current_config, current_log_path)
                tol_spins = exp.loop(max_iterations=max_iterations)
                if experiment_config["optim_fn"] == "tot_spins":
                    optim_fn = tol_spins
                elif experiment_config["optim_fn"] == "avg_reward":
                    optim_fn = exp.brain.get_performance()
                elif experiment_config["optim_fn"] == "tot_episodes":
                    optim_fn = exp.meta_environment.total_episodes_finished
                else:
                    print("TestBench: optim_fn param not set properly. Value is {}".format(experiment_config["optim_fn"]))
                    
                exp.finish()
                return optim_fn

            current_log_path = os.path.join(self.log_path, str(i))
            if os.path.isdir(current_log_path):
                shutil.rmtree(current_log_path)
            os.makedirs(current_log_path)
            current_config = copy.deepcopy(self.base_config["base_configs"])
            for field, value in experiment_config["config_alterations"].items():
                current_config[field] = value
            with open(os.path.join(current_log_path,"config.json"), 'w') as j:
                json.dump(current_config, j)
            self.test_variables = {}
            for stack in experiment_config["values"]:
                for variable, suggested_values in stack["values"].items():
                    self.test_variables[stack["prefix"] + variable] = {
                        "current_value" : None,
                        "path" : stack["path"] + [variable],
                        "suggested_values" : suggested_values
                    }
            study = optuna.create_study(direction= experiment_config["optim_dir"])
            study.optimize(objective, n_trials= experiment_config["n_trials"])
            study.best_params
            print(study.best_params)
            i += 1
