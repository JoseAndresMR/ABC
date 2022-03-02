from genericpath import exists
import unittest
import os
import shutil
import numpy as np
import pathlib


from brainrl.management.experience import Experience


class TestExperience(unittest.TestCase):
    @staticmethod
    def get_config():
        return {'brain':
                {
                    'neurons': {
                        'sensory': {
                            'neurons': [
                                {
                                    'agent': {
                                        'type': 'DDPG',
                                        'additional_dim': 3,
                                        'definition': {
                                            'metaparameters':
                                            {
                                                'buffer_size': 100000,
                                                'batch_size': 256,
                                                'gamma': 0.9853,
                                                'tau': 0.0122,
                                                'lr_actor': 0.00263,
                                                'lr_critic': 0.00323,
                                                'learn_every': 4,
                                                'learn_steps': 2
                                            }
                                        },
                                        'models': {
                                            'actor': {
                                                'layers': [
                                                    {
                                                        'type': 'BatchNorm1d',
                                                        'size': 'state'
                                                    },
                                                    {
                                                        'type': 'linear',
                                                        'size': 256,
                                                        'features': ['relu']
                                                    },
                                                    {
                                                        'type': 'linear',
                                                        'size': 'action',
                                                        'features': ['tanh']
                                                    }
                                                ]
                                            },
                                            'critic': {
                                                'layers': [
                                                    {
                                                        'type': 'BatchNorm1d',
                                                        'size': 'state'
                                                    },
                                                    {
                                                        'type': 'linear',
                                                        'size': 361,
                                                        'features': ['leaky_relu']
                                                    },
                                                    {
                                                        'type': 'linear',
                                                        'size': 113,
                                                        'features': ['leaky_relu'],
                                                        'concat': ['action']
                                                    },
                                                    {
                                                        'type': 'linear',
                                                        'size': 186,
                                                        'features': ['leaky_relu']
                                                    },
                                                    {
                                                        'type': 'linear',
                                                        'size': 1
                                                    }
                                                ]
                                            }
                                        }
                                    }
                                }
                            ]
                        },
                        'motor': {
                            'neurons': [
                                {
                                    'agent': {
                                        'type': 'DDPG',
                                        'additional_dim': 1,
                                        'definition': {
                                            'metaparameters':
                                            {
                                                'buffer_size': 100000,
                                                'batch_size': 256,
                                                'gamma': 0.9853,
                                                'tau': 0.0122,
                                                'lr_actor': 0.00263,
                                                'lr_critic': 0.00323,
                                                'learn_every': 4,
                                                'learn_steps': 2
                                            }
                                        },
                                        'models': {
                                            'actor': {
                                                'layers': [
                                                    {
                                                        'type': 'BatchNorm1d',
                                                        'size': 'state'
                                                    },
                                                    {
                                                        'type': 'linear',
                                                        'size': 256,
                                                        'features': ['relu']
                                                    },
                                                    {
                                                        'type': 'linear',
                                                        'size': 'action',
                                                        'features': ['tanh']
                                                    }
                                                ]
                                            },
                                            'critic': {
                                                'layers': [
                                                    {
                                                        'type': 'BatchNorm1d',
                                                        'size': 'state'
                                                    },
                                                    {
                                                        'type': 'linear',
                                                        'size': 361,
                                                        'features': ['leaky_relu']
                                                    },
                                                    {
                                                        'type': 'linear',
                                                        'size': 113,
                                                        'features': ['leaky_relu'],
                                                        'concat': ['action']
                                                    },
                                                    {
                                                        'type': 'linear',
                                                        'size': 186,
                                                        'features': ['leaky_relu']
                                                    },
                                                    {
                                                        'type': 'linear',
                                                        'size': 1
                                                    }
                                                ]
                                            }
                                        }
                                    }
                                }
                            ]
                        },
                        'intern': {
                            'quantity': 1,
                            'agent': {
                                'type': 'DDPG',
                                'definition': {
                                    'metaparameters':
                                    {
                                        'buffer_size': 100000,
                                        'batch_size': 256,
                                        'gamma': 0.9853,
                                        'tau': 0.0122,
                                        'lr_actor': 0.00263,
                                        'lr_critic': 0.00323,
                                        'learn_every': 4,
                                        'learn_steps': 2
                                    }
                                },
                                'models': {
                                    'actor': {
                                        'layers': [
                                            {
                                                'type': 'BatchNorm1d',
                                                'size': 'state'
                                            },
                                            {
                                                'type': 'linear',
                                                'size': 256,
                                                'features': ['relu']
                                            },
                                            {
                                                'type': 'linear',
                                                'size': 'action',
                                                'features': ['tanh']
                                            }
                                        ]
                                    },
                                    'critic': {
                                        'layers': [
                                            {
                                                'type': 'BatchNorm1d',
                                                'size': 'state'
                                            },
                                            {
                                                'type': 'linear',
                                                'size': 361,
                                                'features': ['leaky_relu']
                                            },
                                            {
                                                'type': 'linear',
                                                'size': 113,
                                                'features': ['leaky_relu'],
                                                'concat': ['action']
                                            },
                                            {
                                                'type': 'linear',
                                                'size': 186,
                                                'features': ['leaky_relu']
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    },
                    'attention_field':
                    {
                        'key_dim': 3,
                        'value_dim': 10,
                        'reward_backprop_thr': 0.01
                    }
                },
                'envs': [
                    {
                        'origin': 'gym',
                        'id': 'gym',
                        'temporality': 'episodic',
                        'name': 'Pendulum-v1',
                        'use_kb_render': False,
                        'render_mp4':
                        {
                            'active': False
                        }
                    }
                ],
                'schedule': [
                    {
                        'env': 'gym',
                        'active': True,
                        'play_ID': 2,
                        'start_type': 'step',
                        'start_value': 0,
                        'max_episodes': 1000,
                        'max_t': 300000,
                        'success_avg': -150,
                        'signals_map': {
                            'state': [
                                {
                                    'env_output': [
                                        1,
                                        3
                                    ],
                                    'neuron_type': 'sensory',
                                    'neuron': 1,
                                    'neuron_input': [
                                        1,
                                        3
                                    ]
                                }
                            ],
                            'action': [
                                {
                                    'env_input': [
                                        1,
                                        1
                                    ],
                                    'neuron_type': 'motor',
                                    'neuron': 1,
                                    'neuron_output': [
                                        1,
                                        1
                                    ]
                                }
                            ]
                        }
                    }
                ]
                }

    @staticmethod
    def create_log_folder(path=''):
        os.makedirs(os.path.join(path, 'log'),
                    exist_ok=True)

    @staticmethod
    def get_log_path():
        return os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'data')

    def test_instance(self):
        self.create_log_folder()
        config = self.get_config()
        exp = Experience(config=config, log_path='log')
        self.assertIsInstance(exp, Experience)

    def test_loop(self):
        self.create_log_folder()
        config = self.get_config()
        exp = Experience(config=config, log_path='log')
        result = exp.loop(max_iterations=510)
        self.assertIsInstance(result, int)
        print(os.path.isfile(os.path.join(exp.brain.log_path,
              'plots', 'Rewards + attention 500.png')))
        self.assertTrue(os.path.isfile(os.path.join(
            exp.brain.log_path, 'plots', '3D attention field step 500.png')))
        self.assertTrue(os.path.isfile(os.path.join(
            exp.brain.log_path, 'plots', 'Rewrds + attention 500.png')))
        # name_render = config['envs'][0]['name']
        # self.assertTrue(os.path.isdir(os.path.join('renders', name_render)))

    def test_finish(self):
        self.create_log_folder()
        config = self.get_config()
        exp = Experience(config=config, log_path='log')
        exp.finish()

    def test_video(self):
        self.create_log_folder()
        config = self.get_config()
        config['envs'][0]['render_mp4'] = {
            'active': True,
            'render_path': 'renders'
        }
        exp = Experience(config=config, log_path='log')
        result = exp.loop(max_iterations=100)
        name_render = config['envs'][0]['name']
        self.assertTrue(os.path.isfile(os.path.join('renders',
                                                    name_render,
                                                    'gym-episode-0.mp4')))
        shutil.rmtree(os.path.join('renders', name_render))


if __name__ == '__main__':
    unittest.main()
