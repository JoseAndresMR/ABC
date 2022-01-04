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
                "id": "gym0",
                "temporality": "episodic",
                "name": "Pendulum-v1"
            },
            {
                "origin": "gym",
                "id": "gym1",
                "temporality": "episodic",
                "name": "Pendulum-v1"
            }
        ],
            'schedule': [
            {
                "active": True,
                "env": "gym0",
                "max_episodes": 1000,
                "max_t": 3000,
                "success_avg": 30,
                "name": "Pendulum-v1"
            },
            {
                "active": False,
                "name": "Pendulum-v1"
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
        
    def test_steps(self):
        self.create_log_folder()
        config = self.get_config()
        env = MetaEnvironment(config=config, log_path='log')
        env.run_steps()
    
    def test_start(self):
        self.create_log_folder()
        config = self.get_config()
        env = MetaEnvironment(config=config, log_path='log')
        env.start_environments_episodes()

    def test_close(self):
        self.create_log_folder()
        config = self.get_config()
        env = MetaEnvironment(config=config, log_path='log')
        env.close_environments()


if __name__ == '__main__':
    unittest.main()
