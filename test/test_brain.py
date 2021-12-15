from brainrl.brain import Neuron
import numpy as np
import unittest

class TestNeuron(unittest.TestCase):
    @staticmethod
    def get_config():
        """
        Same config for all the tests.
        """
        return {'ID': '1',
                "agent" : {
                "type" : "DDPG",
                "additional_dim" : [3,1],
                "definition" : "config/predefined_agents/ddpg_1.json",
                "models": {
                    "actor" : "config/predefined_models/actor_udacity.json",
                    "critic" : "config/predefined_models/critic_udacity.json"
                }
                        }
        }

    def test_instance(self):
        neuron = Neuron(neuron_type='sensory-motor',
                 config=self.get_config(),
                 log_path='',
                 k_dim=2,
                 v_dim=2,
                 environment_signal_size=[2, 2])
        self.assertIsInstance(neuron, Neuron)
    
    def test_error_type(self):
        with self.assertRaises(ValueError):
            neuron = Neuron(neuron_type='non_valid_type',
                 config=self.get_config(),
                 log_path='',
                 k_dim=2,
                 v_dim=2,
                 environment_signal_size=[2, 2])

    def test_next_input_value(self):
        neuron = Neuron(neuron_type='sensory-motor',
                        config=self.get_config(),
                        log_path='',
                        k_dim=2,
                        v_dim=2,
                        environment_signal_size=[2, 2])
        neuron.set_next_input_value(np.random.random(2))
        self.assertIsInstance(neuron.state, np.ndarray)

    def test_reward(self):
        neuron = Neuron(neuron_type='sensory-motor',
                        config=self.get_config(),
                        log_path='',
                        k_dim=2,
                        v_dim=2,
                        environment_signal_size=[2, 2])
        reward: int = 3
        neuron.set_reward(reward)
        self.assertEqual(neuron.scores[-1], reward)

    def test_forward(self):
        neuron = Neuron(neuron_type='sensory-motor',
                        config=self.get_config(),
                        log_path='',
                        k_dim=2,
                        v_dim=2,
                        environment_signal_size=[2, 2])
        neuron.set_next_input_value(np.random.random(2))
        action = neuron.forward()

    def test_decompose(self):
        neuron_sensory_motor = Neuron(neuron_type='sensory-motor',
                                      config=self.get_config(),
                                      log_path='',
                                      k_dim=2,
                                      v_dim=2,
                                      environment_signal_size=[2, 2])
        neuron_sensory_motor.decompose_action()

        neuron_sensory = Neuron(neuron_type='sensory',
                                      config=self.get_config(),
                                      log_path='',
                                      k_dim=2,
                                      v_dim=2,
                                      environment_signal_size=[2, 2])
        neuron_sensory.decompose_action()
        
        neuron_intern = Neuron(neuron_type='intern',
                                      config=self.get_config(),
                                      log_path='',
                                      k_dim=2,
                                      v_dim=2,
                                      environment_signal_size=[2, 2])
        neuron_intern.decompose_action()

        neuron_motor = Neuron(neuron_type='motor',
                                      config=self.get_config(),
                                      log_path='',
                                      k_dim=2,
                                      v_dim=2,
                                      environment_signal_size=[2, 2])
        neuron_motor.decompose_action()
    

if __name__ == '__main__':
    unittest.main()
