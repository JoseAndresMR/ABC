from brainrl.brain import Neuron
import unittest

class TestNeuron(unittest.TestCase):
    def test_instance(self):
        neuron = Neuron(neuron_type='intern',
                 config={'ID': '1',
                        'agent': {'ID': 1,
                                  'type': 'Other'}
                        },
                 log_path='',
                 k_dim=2,
                 v_dim=2,
                 environment_signal_size=None)
        self.assertIsInstance(neuron, Neuron)
    
    def test_error_type(self):
        with self.assertRaises(ValueError):
            neuron = Neuron(neuron_type='non_valid_type',
                 config={'ID': '1',
                        'agent': {'ID': 1,
                                  'type': 'Other'}
                        },
                 log_path='',
                 k_dim=2,
                 v_dim=2,
                 environment_signal_size=None)


if __name__ == '__main__':
    unittest.main()