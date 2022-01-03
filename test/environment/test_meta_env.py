from genericpath import exists
from brainrl.environment import MetaEnvironment
import unittest
import os
import numpy as np


class TestMetaEnvironment(unittest.TestCase):
    @staticmethod
    def get_config():
        return {'environments': [
            {
                "origin": "gym",
                "id": "gym",
                "temporality": "episodic",
                "name": "CartPole-v0"
            },
            {
                "origin": "gym",
                "id": "gym",
                "temporality": "episodic",
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
        env = MetaEnvironment(config=config, log_path='log')
        self.assertIsInstance(env, MetaEnvironment)
        
    def test_close(self):
        self.create_log_folder()
        config = self.get_config()
        env = MetaEnvironment(config=config, log_path='log')
        env.close_environments()


if __name__ == '__main__':
    unittest.main()
