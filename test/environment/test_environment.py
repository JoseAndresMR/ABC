from genericpath import exists
from brainrl.environment.environment import Environment
import unittest
import os


class TestEnvironment(unittest.TestCase):
    @staticmethod
    def create_log_folder(path=''):
        os.makedirs(os.path.join(path, 'log'),
                    exist_ok=True)

    def test_instance(self):
        self.create_log_folder()
        env = Environment(id='1', log_path='log')
        self.assertIsInstance(env, Environment)


if __name__ == '__main__':
    unittest.main()
