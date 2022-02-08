from brainrl.brain import Brain
import numpy as np
import unittest


class TestBrain(unittest.TestCase):
    @staticmethod
    def get_config():
        """
        Same config for all the tests.
        """
        return {
            "neurons" : {
                "sensory-motor" : {
                    "neurons" : [
                        {
                            "agent" : {
                                "type" : "DDPG",
                                "additional_dim" : [3,1],
                                "definition" : {
                                    "metaparameters": {
                                        "buffer_size" : 100000,
                                        "batch_size" : 256,
                                        "gamma" : 0.99,
                                        "tau" : 0.01,
                                        "lr_actor" : 0.002,
                                        "lr_critic" : 0.002,
                                        "learn_every" : 4,
                                        "learn_steps" : 2
                                    }
                                },
                                "models": {
                                    "actor" : {
                                        "layers" : [
                                            {
                                                "type" : "BatchNorm1d",
                                                "size" : "state"
                                            },
                                            {
                                                "type" : "linear",
                                                "size" : 256,
                                                "features" : ["relu"]
                                            },
                                            {
                                                "type" : "linear",
                                                "size" : "action",
                                                "features" : ["tanh"]
                                            }
                                        ]
                                    },
                                    "critic" : {
                                        "layers" : [
                                            {
                                                "type" : "BatchNorm1d",
                                                "size" : "state"
                                            },
                                            {
                                                "type" : "linear",
                                                "size" : 256,
                                                "features" : ["leaky_relu"]
                                            },
                                            {
                                                "type" : "linear",
                                                "size" : 256,
                                                "features" : ["leaky_relu"],
                                                "concat" : ["action"]
                                            },
                                            {
                                                "type" : "linear",
                                                "size" : 128,
                                                "features" : ["leaky_relu"]
                                            },
                                            {
                                                "type" : "linear",
                                                "size" : 1
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    ]
                }
            },
            "attention_field":
                {
                    "key_dim" : 3,
                    "value_dim" : 10,
                    "reward_backprop_thr" : 0.01
                }
        }

    def test_instance(self):
        brain = Brain(config=self.get_config(),
                       log_path='')
        self.assertIsInstance(brain, Brain)

    def test_state_and_reward(self):
        brain = Brain(config=self.get_config(),
                       log_path='')
        brain.set_state_and_reward()

    def test_forward(self):
        brain = Brain(config=self.get_config(),
                       log_path='')
        for neuron in brain.neurons["sensory-motor"]:
            neuron["state"] = np.random.random((1, 3))
        brain.set_state_and_reward()
        brain.forward()

    def test_performance(self):
        brain = Brain(config=self.get_config(),
                      log_path='')
        self.assertTrue(brain.get_performance()==-99999)

    def test_update_plots(self):
        brain = Brain(config=self.get_config(),
                      log_path='')
        brain.update_plots()


if __name__ == '__main__':
    unittest.main()
