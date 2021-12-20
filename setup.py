from setuptools import setup

setup(
    name="brainrl",
    description="Reinforcement learning framework with attention",
    version="0.1",
    author="Jose Andrés Millán, Carlos Perales",
    author_email="ja.millan@4i.ai",
    packages=['brainrl',
              'brainrl.brain',
              'brainrl.brain.util',
              'brainrl.brain.rl_agent',
              'brainrl.environment',
              'brainrl.management',
              ],
    zip_safe=False,
    install_requires=[
        # 'unityagents',
        # 'numpy==1.18.5',
        # 'torch',
        # 'seaborn',
        # 'torchvision',
        # 'gym==0.7.4',
        # 'box2d-py',
        # 'matplotlib'
    ],
    include_package_data=True,
    setup_requires=[],
    tests_require=['pytest']
)
