from genericpath import exists
import unittest
import os
import numpy as np

from brainrl.management.Experience import Experience


class TestExperience(unittest.TestCase):
    @staticmethod
    def get_config():
        return {'brain': {
            "neurons": {
                "sensory-motor": {
                    "neurons": [
                        {
                            "agent": {
                                "type": "DQN",
                                "additional_dim": [
                                    37,
                                    4
                                ],
                                "eps": [1.0, 0.99999, 0.1],
                                "models": {
                                    "actor": "actor_discrete_1"
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
                "name": "CartPole-v0"
            },
            {
                "origin": "gym",
                "id": "gym1",
                "temporality": "episodic",
                "name": "CartPole-v1"
            }
        ],
            'schedule': [
            {
                "active": True,
                "env": "gym0",
                "max_episodes": 1000,
                "max_t": 3000,
                "success_avg": 30,
                "name": "CartPole-v0"
            },
            {
                "active": False,
                "name": "CartPole-v1"
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
        env = Experience(config=config, log_path='log')
        self.assertIsInstance(env, Experience)


if __name__ == '__main__':
    unittest.main()
