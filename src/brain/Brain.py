import numpy as np
from copy import deepcopy
import json
import os

from brain.Neuron import Neuron
from brain.AttentionField import AttentionField

class Brain(object):

    def __init__(self):

        self.neurons = {"all": [], "sensory" : [], "intern" : [], "motor" : []}
        with open(os.path.join(os.path.dirname(__file__),'config.json'), 'r') as j:
            self.config = json.load(j)
        self.k_dim, self.v_dim = self.config["attention_field"]["key_dim"], self.config["attention_field"]["value_dim"]
        self.startAttentionField()
        self.spawnNeurons()

    def spawnNeurons(self): ### SIMPLIFY FUNCTION
        for neuron_type, type_neuron_config in self.config["neurons"].items():
            if neuron_type == "sensory":
                for neuron_config in type_neuron_config["neurons"]:
                    self.spawnOneNeuron(neuron_type, self.k_dim, self.v_dim, neuron_config["agent"]["additional_dim"])

            elif neuron_type == "intern":
                for _ in range(type_neuron_config["quantity"]):
                    self.spawnOneNeuron(neuron_type, self.k_dim, self.v_dim)

            elif neuron_type == "sensory" or neuron_type == "motor":
                for neuron_config in type_neuron_config["neurons"]:
                    self.spawnOneNeuron(neuron_type, self.k_dim, self.v_dim, neuron_config["agent"]["additional_dim"])

    def spawnOneNeuron(self, neuron_type, k_min, v_min, additional_dim):
        empty_neuron = {"neuron" : None, "state" : [], "next_state" : [], "action" : [], "reward" : [], "attended" : [], "info" : {"type" : ""}}
        neuron = deepcopy(empty_neuron)
        neuron["neuron"] = Neuron(neuron_type, k_min, v_min, additional_dim)
        neuron["state"] = neuron["neuron"].state
        neuron["info"]["type"] = neuron_type
        self.neurons["all"].append(neuron)
        self.neurons[neuron_type].append(neuron)

    def forward(self):
        for neuron in self.neurons["sensory"]:
            neuron["action"] = neuron["neuron"].forward()
            self.attention_field.addEntries(None, neuron["action"][:,:self.k_dim], neuron["action"][:,self.k_dim:])

        self.runAttentionFieldStep()

        for neuron in self.neurons["intern"]:
            neuron["action"] = neuron["neuron"].forward()
            self.attention_field.addEntries(neuron["action"][:,:self.k_dim],
                                            neuron["action"][:,self.k_dim:self.k_dim*2],
                                            neuron["action"][:,self.k_dim*2:])

        for neuron in self.neurons["motor"]:
            neuron["action"] = neuron["neuron"].forward()
            self.attention_field.addEntries(neuron["action"][:,:self.k_dim], None, None)

    def runAttentionFieldStep(self):

        values, attended = self.attention_field.runStep()
        neurons = self.neurons["intern"] + self.neurons["motor"]
        for i, value in enumerate(values):
            if i < len(self.neurons["intern"]):
                neurons[i]["state"] = neurons[i]["actions"][:self.k_dim*2].append(value)
            else:
                neurons[i-len(self.neurons["intern"])]["state"] = neurons[i-len(self.neurons["intern"])]["actions"][:self.k_dim].append(value)
            neurons[i]["attended"] = attended

    def backprop(self):
        [neuron["neuron"].backprop() for neuron in self.neurons["motor"]]
        [neuron["neuron"].backprop() for neuron in self.neurons["intern"]]
        [neuron["neuron"].backprop() for neuron in self.neurons["sensory"]]

    def setStateAndReward(self):
        [neuron["neuron"].setNextState(neuron["state"]) for neuron in self.neurons["sensory"]]
        [neuron["neuron"].setReward(neuron["reward"]) for neuron in self.neurons["motor"]]

    def startAttentionField(self):
        af_config = self.config["attention_field"]
        self.attention_field = AttentionField(af_config["key_dim"], af_config["value_dim"])