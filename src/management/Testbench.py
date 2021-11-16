import json, os
import numpy as np
import copy
import shutil

from Experience import Experience

"""
fjgnlskdfjgnlskfdjgn
"""

class Testbench(object):
    """
    sifuhgsifduhgsidfuhg
    """

    def __init__(self):
        """
        sfgksfdgnksdfg
        """

        with open(os.path.join(os.path.dirname(__file__),"configs",'config.jsonc'), 'r') as j:
            self.config = json.load(j)
        self.configs_paths = {"envs" : os.path.join(os.path.dirname(__file__),"..","environments","configs")}
        self.configs_paths["brain"] = os.path.join(os.path.dirname(__file__),"..","brain","configs")
        self.configs_paths["schedule"] = os.path.join(os.path.dirname(__file__),"schedule_configs")
        self.base_config = {}
        with open(os.path.join(self.configs_paths["envs"],'{}.json'.format(self.config["base_configs"]["envs"])), 'r') as j:
            self.base_config["envs"] = json.load(j)
        with open(os.path.join(self.configs_paths["brain"],'{}.json'.format(self.config["base_configs"]["brain"])), 'r') as j:
            self.base_config["brain"] = json.load(j)
        with open(os.path.join(self.configs_paths["schedule"],'{}.json'.format(self.config["base_configs"]["schedule"])), 'r') as j:
            self.base_config["schedule"] = json.load(j)

        self.log_path = os.path.join(os.path.dirname(__file__),'..','..',"data/runs", self.config["ID"])
        self.experiences()

    def experiences(self):
        i = 0
        for alteration in self.config["alterations"]:
            current_log_path = os.path.join(self.log_path, str(i))
            if os.path.isdir(current_log_path):
                shutil.rmtree(current_log_path)
            os.makedirs(current_log_path)
            current_config = copy.deepcopy(self.base_config)
            for field, value in alteration.items():
                with open(os.path.join(self.configs_paths[field],'{}.json'.format(value)), 'r') as j:
                    current_config[field] = json.load(j)

            with open(os.path.join(current_log_path,"config.json"), 'w') as j:
                json.dump(current_config, j)
            exp = Experience(current_config, current_log_path)
            exp.loop()
            i += 1

Testbench()