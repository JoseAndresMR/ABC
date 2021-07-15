import numpy as np
from copy import deepcopy

from brain.Neuron import Neuron

class Brain(object):

    def __init__(self):

        self.neurons = {"all": [], "sensory" : [], "intern" : [], "motor" : [], "temporal_mix" : []}  ### Delete temporal mix
        self.spawnNeurons()

    def spawnNeurons(self):
        empty_neuron = {"neuron" : None, "state" : [], "next_state" : [], "action" : [], "reward" : [], "info" : {"type" : ""}}

        for i in range(1):
            neuron = deepcopy(empty_neuron)
            neuron["neuron"] = Neuron("temporal_mix")
            neuron["info"]["type"] = "temporal_mix"
            self.neurons["all"].append(neuron)
            self.neurons["temporal_mix"].append(neuron) ### cambiar "intern" a neuron["neuron"]["type"]

    def forward(self):
        [neuron["neuron"].forward() for neuron in self.neurons["sensory"]]
        [neuron["neuron"].forward() for neuron in self.neurons["intern"]]
        for neuron in self.neurons["motor"]:
            neuron["action"] = neuron["neuron"].forward()

        for neuron in self.neurons["temporal_mix"]:
            neuron["action"] = neuron["neuron"].forward()

    def backprop(self):
        [neuron["neuron"].backprop() for neuron in self.neurons["motor"]]
        [neuron["neuron"].backprop() for neuron in self.neurons["intern"]]
        [neuron["neuron"].backprop() for neuron in self.neurons["sensory"]]
        [neuron["neuron"].backprop() for neuron in self.neurons["temporal_mix"]]  ### Delete

    def setStateAndReward(self):
        [neuron["neuron"].setNextState(neuron["state"]) for neuron in self.neurons["temporal_mix"]]  ### cambiar "intern" a "sensory"
        [neuron["neuron"].setReward(neuron["reward"]) for neuron in self.neurons["temporal_mix"]]  ### cambiar "intern" a "motor"