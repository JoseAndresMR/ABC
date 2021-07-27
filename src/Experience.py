from environments.MetaEnvironment import MetaEnvironment
from brain.Brain import Brain

class Experience(object):

    def __init__(self):

        self.meta_environment = MetaEnvironment()
        self.brain = Brain()
        self.loop()

    def loop(self):
        try:
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
        except AssertionError as error:
            print(error)
            self.meta_environment.closeEnvironments()

    def allocateEnvironementOutput(self):
        self.brain.neurons["sensory"][0]["state"] = self.meta_environment.environments["reacher"]["state"]
        self.brain.neurons["motor"][0]["reward"] = self.meta_environment.environments["reacher"]["reward"]
        self.brain.setStateAndReward()

    def allocateBrainOutput(self):
        self.meta_environment.environments["reacher"]["action"] = self.brain.neurons["motor"][0]["action"] 
        self.meta_environment.setAction()

Experience()