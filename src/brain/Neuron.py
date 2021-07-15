from brain.rl_agents.DdpgAgent import DdpgAgent
import numpy as np

class Neuron(object):

    def __init__(self, neuron_type, non_intern_signal_size = None):

        self.state, self.next_state, self.action, self.reward = None, None, None, None
        self.neuron_type = neuron_type
        self.non_intern_signal_size = non_intern_signal_size

        self.buildRlAgent()

    def buildRlAgent(self):

        self.state_size, self.action_size = 80, 80

        if self.neuron_type == "sensory":
            self.state_size = self.non_intern_signal_size

        elif self.neuron_type == "motor":
            self.action_size = self.non_intern_signal_size

        elif self.neuron_type == "temporal_mix":
            self.state_size = 33
            self.action_size = 4

        self.rl_agent = DdpgAgent(self.state_size, self.action_size, random_seed = 2)

    def setNextState(self, state):
        if type(self.state) != type(np.array(1)):  
            self.state = state
        else:
            self.next_state = state
        
    def setReward(self, reward):
        self.reward = reward

    def forward(self):
        self.action = self.rl_agent.act(self.state)
        return self.action

    def backprop(self):
        if type(self.next_state) != type(np.array(1)):
            done = True
        else:
            done = False
        self.rl_agent.step(self.state, self.action, self.reward, self.next_state, [done])
        self.state = self.next_state
        self.next_state = None

