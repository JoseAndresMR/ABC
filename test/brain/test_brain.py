from brainrl.brain import Brain
import numpy as np
import unittest


class TestBrain(unittest.TestCase):
    @staticmethod
    def get_config():
        """
        Same config for all the tests.
        """
        return {
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
                        "eps" : [1.0, 0.99999,0.1],
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
}

    # def test_instance(self):
    #     brain = Brain(config=self.get_config(),
    #                    log_path='')
    #     self.assertIsInstance(brain, Brain)
    
    # def test_forward(self):
    #     brain = Brain(config=self.get_config(),
    #                   log_path='')
    #     action = brain.forward()

    def test_run_attention_field_step(self):
        brain = Brain(config=self.get_config(),
                      log_path='')
        brain.run_attention_field_step(0)
    
    # def test_set_state_and_reward(self):
    #     brain = Brain(config=self.get_config(),
    #                   log_path='')
    
    # def test_allocate_reward(self):
    #     brain = Brain(config=self.get_config(),
    #                   log_path='')
    
    # def test_make_plots(self):
    #     brain = Brain(config=self.get_config(),
    #                   log_path='')


if __name__ == '__main__':
    unittest.main()