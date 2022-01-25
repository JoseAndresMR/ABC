from genericpath import exists
import unittest
import os
import numpy as np

from brainrl.management.experience import Experience


class TestExperience(unittest.TestCase):
    @staticmethod
    def get_config():
        return {'brain': {
            "neurons": {
                "sensory-motor": {
                    "neurons": [
                        {
                            "agent": {
                                "type": "DDPG",
                                "additional_dim": [
                                    3,
                                    1
                                ],
                                "definition": {
                                    "metaparameters": {
                                        "buffer_size": 100000,
                                        "batch_size": 256,
                                        "gamma": 0.99,
                                        "tau": 0.01,
                                        "lr_actor": 0.002,
                                        "lr_critic": 0.002,
                                        "learn_every": 4,
                                        "learn_steps": 2
                                    }
                                },
                                "models": {
                                    "actor": {
                                        "layers": [
                                            {
                                                "type": "BatchNorm1d",
                                                "size": "state"
                                            },
                                            {
                                                "type": "linear",
                                                "size": 256,
                                                "features": ["relu"]
                                            },
                                            {
                                                "type": "linear",
                                                "size": "action",
                                                "features": ["tanh"]
                                            }
                                        ]
                                    },
                                    "critic": {
                                        "layers": [
                                            {
                                                "type": "BatchNorm1d",
                                                "size": "state"
                                            },
                                            {
                                                "type": "linear",
                                                "size": 256,
                                                "features": ["leaky_relu"]
                                            },
                                            {
                                                "type": "linear",
                                                "size": 256,
                                                "features": ["leaky_relu"],
                                                "concat": ["action"]
                                            },
                                            {
                                                "type": "linear",
                                                "size": 128,
                                                "features": ["leaky_relu"]
                                            },
                                            {
                                                "type": "linear",
                                                "size": 1
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    ]
                }
            },
            "attention_field": {
                "key_dim": 3,
                "value_dim": 10,
                "reward_backprop_thr": 0.01
            }
        },
            'envs': [
            {
                "origin": "gym",
                "id": "gym0",
                "temporality": "episodic",
                "name": "Pendulum-v1",
                "use_kb_render" : False
            },
            {
                "origin": "gym",
                "id": "gym1",
                "temporality": "episodic",
                "name": "Pendulum-v1",
                "use_kb_render" : False
            }
        ],
            'schedule': [
            {
                "env": "gym0",
                "active": True,
                "play_ID": 2,
                "start_type": "step",
                "start_value": 0,
                "max_episodes": 600,
                "max_t": 3000,
                "success_avg": -300,
                "signals_map": {
                    "state": [
                        {
                            "env_output": [
                                1,
                                3
                            ],
                            "neuron_type": "sensory-motor",
                            "neuron": 1,
                            "neuron_input": [
                                1,
                                3
                            ]
                        }
                    ],
                    "action": [
                        {
                            "env_input": [
                                1,
                                1
                            ],
                            "neuron_type": "sensory-motor",
                            "neuron": 1,
                            "neuron_output": [
                                1,
                                1
                            ]
                        }
                    ]
                }
            },
            {
                "env": "gym1",
                "active": False
            }
        ]
        }

    @staticmethod
    def create_log_folder(path=''):
        os.makedirs(os.path.join(path, 'log'),
                    exist_ok=True)

    def test_instance(self):
        self.create_log_folder()
        config = self.get_config()
        exp = Experience(config=config, log_path='log')
        self.assertIsInstance(exp, Experience)

    def test_loop(self):
        self.create_log_folder()
        config = self.get_config()
        exp = Experience(config=config, log_path='log')
        result = exp.loop(max_iterations=3)
        self.assertIsInstance(result, int)

    def test_finish(self):
        self.create_log_folder()
        config = self.get_config()
        exp = Experience(config=config, log_path='log')
        exp.finish()


if __name__ == '__main__':
    unittest.main()
