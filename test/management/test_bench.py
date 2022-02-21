from genericpath import exists
import unittest
import os
import numpy as np

from brainrl.management.test_bench import TestBench


class TestTestBench(unittest.TestCase):
    def test_instance(self):
        tb = TestBench(config_folder_path='config', data_folder_path='data')    
        self.assertIsInstance(tb, TestBench)

    def test_experiences(self):
        tb = TestBench(config_folder_path='config', data_folder_path='data')  
        tb.experiences(max_iterations=2000, n_trials=1)


if __name__ == '__main__':
    unittest.main()
