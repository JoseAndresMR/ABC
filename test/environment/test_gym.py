from genericpath import exists
from brainrl.environment import GymEpisodicEnvironment
import unittest
import os


class TestGymEpisodicEnvironment(unittest.TestCase):
    @staticmethod
    def create_log_folder(path=''):
        os.makedirs(os.path.join(path, 'log'),
                    exist_ok=True)

    def test_instance(self):
        self.create_log_folder()
        env = GymEpisodicEnvironment(id='gym', log_path='log', name="Pendulum-v1")
        self.assertIsInstance(env, GymEpisodicEnvironment)
    
    def test_finish(self):
        self.create_log_folder()
        env = GymEpisodicEnvironment(id='gym', log_path='log', name="Pendulum-v1")
        env.finish_environment()


if __name__ == '__main__':
    unittest.main()
