from brainrl.brain import Brain
import numpy as np
import unittest
import os
import pathlib

class TestBrain(unittest.TestCase):
    @staticmethod
    def get_config():
        """
        Same config for all the tests.
        """
        return {
            "neurons" : {
                "sensory" : {
                    "neurons" : [
                        {
                            "agent" : {
                                "type" : "DDPG",
                                "additional_dim" : 3,
                                "definition" : {
                                "metaparameters":
                                    {
                                        "buffer_size" : 100000,
                                        "batch_size" : 256,
                                        "gamma" : 0.9853,
                                        "tau" : 0.0122,
                                        "lr_actor" : 0.00263,
                                        "lr_critic" : 0.00323,
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
                                                "size" : 361,
                                                "features" : ["leaky_relu"]
                                            },
                                            {
                                                "type" : "linear",
                                                "size" : 113,
                                                "features" : ["leaky_relu"],
                                                "concat" : ["action"]
                                            },
                                            {
                                                "type" : "linear",
                                                "size" : 186,
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
                },
                "motor" : {
                    "neurons" : [
                        {
                            "agent" : {
                                "type" : "DDPG",
                                "additional_dim" : 1,
                                "definition" : {
                                    "metaparameters":
                                    {
                                        "buffer_size" : 100000,
                                        "batch_size" : 256,
                                        "gamma" : 0.9853,
                                        "tau" : 0.0122,
                                        "lr_actor" : 0.00263,
                                        "lr_critic" : 0.00323,
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
                                                "size" : 361,
                                                "features" : ["leaky_relu"]
                                            },
                                            {
                                                "type" : "linear",
                                                "size" : 113,
                                                "features" : ["leaky_relu"],
                                                "concat" : ["action"]
                                            },
                                            {
                                                "type" : "linear",
                                                "size" : 186,
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
                },
                "intern" : {
                    "quantity": 1,
                    "agent" : {
                        "type" : "DDPG",
                        "definition" : {
                            "metaparameters":
                            {
                                "buffer_size" : 100000,
                                "batch_size" : 256,
                                "gamma" : 0.9853,
                                "tau" : 0.0122,
                                "lr_actor" : 0.00263,
                                "lr_critic" : 0.00323,
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
                                        "size" : 361,
                                        "features" : ["leaky_relu"]
                                    },
                                    {
                                        "type" : "linear",
                                        "size" : 113,
                                        "features" : ["leaky_relu"],
                                        "concat" : ["action"]
                                    },
                                    {
                                        "type" : "linear",
                                        "size" : 186,
                                        "features" : ["leaky_relu"]
                                    }
                                ]
                            }
                        }
                    }
                }
            },
            "attention_field":
                {
                    "key_dim" : 3,
                    "value_dim" : 10,
                    "reward_backprop_thr" : 0.01
                }
        }

    @staticmethod
    def get_log_path():
        return  os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "data")

    def test_instance(self):
        brain = Brain(config=self.get_config(),
                       log_path= "data")
        self.assertIsInstance(brain, Brain)

    def test_state_and_reward(self):
        brain = Brain(config=self.get_config(),
                       log_path= "data")
        brain.set_state_and_reward()

    def test_forward(self):
        brain = Brain(config=self.get_config(),
                       log_path= "data")
        brain.neurons["sensory"][0]["state"] = np.random.random((1, 3))
        brain.set_state_and_reward()
        brain.forward()

    def test_performance(self):
        brain = Brain(config=self.get_config(),
                      log_path= "data")
        self.assertTrue(brain.get_performance()==-99999)

if __name__ == '__main__':
    unittest.main()
