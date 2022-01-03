from genericpath import exists
from brainrl.environment import Environment, GymEpisodicEnvironment
from gym.spaces.box import Box
import unittest
import os
import numpy as np


class TestEnvironment(unittest.TestCase):
    @staticmethod
    def create_log_folder(path=''):
        os.makedirs(os.path.join(path, 'log'),
                    exist_ok=True)

    def test_instance(self):
        self.create_log_folder()
        env = Environment(id='gym', log_path='log')
        self.assertIsInstance(env, Environment)
    
    def test_action(self):
        self.create_log_folder()
        env = Environment(id='gym', log_path='log')
        action = np.random.random((1, 2))
        env.set_action(action)
        self.assertTrue((action == env.actions).all())
        
    def test_get_info(self):
        self.create_log_folder()
        env = GymEpisodicEnvironment(id='gym', log_path='log', name="Pendulum-v1")
        env_info = env.get_environment_info()

        # Expect: {'num_agents': 1,
        #          'state_type': <class 'gym.spaces.box.Box'>,
        #          'state_size': (3,),
        #          'action_size': (1,),
        #          'action_type': <class 'gym.spaces.box.Box'>}

        self.assertIsInstance(env_info['num_agents'], int)
        self.assertIsInstance(env_info['state_size'], tuple)
        self.assertIsInstance(env_info['action_size'], tuple)
        self.assertEqual(env_info['state_type'], Box)
        self.assertEqual(env_info['action_type'], Box)


if __name__ == '__main__':
    unittest.main()
