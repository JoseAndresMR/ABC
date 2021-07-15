from environments.MetaEnvironment import MetaEnvironment
from brain.Brain import Brain

class Experience(object):

    def __init__(self):

        self.meta_environment = MetaEnvironment()
        self.brain = Brain()
        self.loop()

    def loop(self):
        self.meta_environment.startEnvironmentsEpisodes()
        self.allocateEnvironementOutput()
        for spin in range(999999999999):
            # print("experience loop spin:", spin)
            self.brain.forward()
            self.allocateBrainOutput()
            self.meta_environment.runSteps()
            self.allocateEnvironementOutput()
            self.meta_environment.closeEnvironments()
            self.brain.backprop()

    def allocateEnvironementOutput(self):
        self.brain.neurons["temporal_mix"][0]["state"] = self.meta_environment.environments["reacher"]["state"]  ### cambiar "temporal_mix" a "sensory"
        self.brain.neurons["temporal_mix"][0]["reward"] = self.meta_environment.environments["reacher"]["reward"]  ### cambiar "temporal_mix" a "motor"
        self.brain.setStateAndReward()

    def allocateBrainOutput(self):
        self.meta_environment.environments["reacher"]["action"] = self.brain.neurons["temporal_mix"][0]["action"] ### cambiar "temporal_mix" a "motor"  
        self.meta_environment.setAction()

Experience()