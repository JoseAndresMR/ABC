import json
import os
import copy
import shutil
import optuna

from brainrl.management.experience import Experience


def setValueInDictWithPath(dic, keys, value):
    for key in keys[:-1]:
        if type(dic) == dict:
            dic = dic.setdefault(key, {})
        elif type(dic) == list:
            dic = dic[key]
    dic[keys[-1]] = value


class Testbench(object):
    """
    Class to run batches of experiments following different settings.
    """

    def __init__(self):
        """
        Define initial configurations.
        """

        with open(os.path.join(os.path.dirname(__file__), "configs", 'config.jsonc'), 'r') as j:
            self.config = json.load(j)
        self.configs_paths = {"envs": os.path.join(
            os.path.dirname(__file__), "..", "environments", "configs")}
        self.configs_paths["brain"] = os.path.join(
            os.path.dirname(__file__), "..", "brain", "configs")
        self.configs_paths["schedule"] = os.path.join(
            os.path.dirname(__file__), "schedule_configs")
        self.configs_paths["predefined_agents"] = os.path.join(
            os.path.dirname(__file__), "..", "brain", "rl_agents", "predefined_agents")
        self.configs_paths["predefined_models"] = os.path.join(
            os.path.dirname(__file__), "..", "brain", "rl_agents", "predefined_models")
        self.base_config = {}
        with open(os.path.join(self.configs_paths["envs"], '{}.json'.format(self.config["base_configs"]["envs"])), 'r') as j:
            self.base_config["envs"] = json.load(j)
        with open(os.path.join(self.configs_paths["brain"], '{}.json'.format(self.config["base_configs"]["brain"])), 'r') as j:
            self.base_config["brain"] = json.load(j)
        with open(os.path.join(self.configs_paths["schedule"], '{}.json'.format(self.config["base_configs"]["schedule"])), 'r') as j:
            self.base_config["schedule"] = json.load(j)

        self.expandJsons(self.base_config)

        self.log_path = os.path.join(os.path.dirname(
            __file__), '..', '..', "data/runs", self.config["id"])

    def expandJsons(self, nested_dict, prepath=()):
        if type(nested_dict) == list:
            for i, v in enumerate(nested_dict):
                path = prepath + (i,)
                self.expandJsons(v, path)  # recursive call
        elif type(nested_dict) == dict:
            for k, v in nested_dict.items():
                path = prepath + (k,)
                # found json
                if type(v) == str and len(v) >= 5 and v[-5:] == ".json":
                    folder, file = v.split("/")
                    with open(os.path.join(self.configs_paths[folder], file), 'r') as j:
                        setValueInDictWithPath(
                            self.base_config, path, json.load(j))
                elif hasattr(v, 'items') or type(v) == list:  # v is a dict or list
                    self.expandJsons(v, path)  # recursive call

    def experiences(self):
        """ Run all experiments in sequence, changing the alterations from the prior base configuration. """
        i = 0
        for experiment_config in self.config["experiments"]:
            def objective(trial):
                for variable_name, variable_params in self.test_variables.items():
                    suggested_values = variable_params["suggested_values"]
                    x = trial.suggest_float(
                        variable_name, suggested_values[0], suggested_values[1])
                    setValueInDictWithPath(
                        current_config, variable_params["path"], x)

                exp = Experience(current_config, current_log_path)
                performance = exp.loop()
                exp.finish()
                return performance

            current_log_path = os.path.join(self.log_path, str(i))
            if os.path.isdir(current_log_path):
                shutil.rmtree(current_log_path)
            os.makedirs(current_log_path)
            current_config = copy.deepcopy(self.base_config)
            for field, value in experiment_config["config_alterations"].items():
                with open(os.path.join(self.configs_paths[field], '{}.json'.format(value)), 'r') as j:
                    current_config[field] = json.load(j)
            with open(os.path.join(current_log_path, "config.json"), 'w') as j:
                json.dump(current_config, j)
            self.test_variables = {}
            for stack in experiment_config["values"]:
                for variable, suggested_values in stack["values"].items():
                    self.test_variables[stack["prefix"] + variable] = {
                        "current_value": None,
                        "path": stack["path"] + [variable],
                        "suggested_values": suggested_values
                    }
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=15)
            study.best_params  # E.g. {'x': 2.002108042}
            print(study.best_params)
            i += 1


if __name__ == "__main__":
    tb = Testbench()
    tb.experiences()
