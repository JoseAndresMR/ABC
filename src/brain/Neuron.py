from brain.rl_agents.DdpgAgent import DdpgAgent
import numpy as np

class Neuron(object):

    def __init__(self, neuron_type, k_dim, v_dim, environment_signal_size = None):

        self.state, self.next_state, self.action, self.reward = None, None, None, None
        self.neuron_type = neuron_type
        self.k_dim, self. v_dim, self.environment_signal_size = k_dim, v_dim, environment_signal_size
        self.attended = []

        self.buildRlAgent()

    def buildRlAgent(self):
        if self.neuron_type == "sensory":
            self.state_size = self.environment_signal_size
            self.action_size = self.k_dim + self.v_dim
            self.key = np.random.rand(1,self.k_dim)
            
        elif self.neuron_type == "intern":
            self.state_size = self.v_dim
            self.action_size = self.k_dim*2 + self.v_dim
            self.query = np.random.rand(1,self.k_dim)
            self.key, self.output_value = None, None


        elif self.neuron_type == "motor":
            self.state_size = self.v_dim
            self.action_size = self.k_dim + self.environment_signal_size
            self.query = np.random.rand(1,self.k_dim)

        print("Neuron: Building a {} neuron with {} state size and {} action size".format(self.neuron_type, self.state_size, self.action_size))

        self.rl_agent = DdpgAgent(self.state_size, self.action_size, random_seed = 2)

    def setNextInputValue(self, state):
        if type(self.state) != type(np.array(1)):  
            self.state = state
        else:
            self.next_state = state
        
    def setReward(self, reward):
        self.reward = reward

    def forward(self):
        self.action = self.rl_agent.act(self.state)
        self.decomposeAction()
        return self.action

    def backprop(self):
        if type(self.next_state) != type(np.array(1)):
            done = True
        else:
            done = False
        self.rl_agent.step(self.state, self.action, self.reward, self.next_state, [done])
        self.state = self.next_state
        self.next_state = None

    def decomposeAction(self):
        if self.neuron_type == "sensory":
            self.key = self.action[:,:self.k_dim]
            self.output_value = self.action[:,self.k_dim:]
        elif self.neuron_type == "intern":
            self.query = self.action[:,:self.k_dim]
            self.key = self.action[:,self.k_dim:self.k_dim*2]
            self.output_value = self.action[:,self.k_dim*2:]
        elif self.neuron_type == "motor":
            self.query = self.action[:,:self.k_dim]
            self.output_value = self.action[:,self.k_dim:]
