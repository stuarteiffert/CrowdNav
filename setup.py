from setuptools import setup


setup(
    name='crowdnav',
    version='0.0.1',
    packages=[
        'crowd_nav',
        'crowd_nav.configs',
        'crowd_nav.policy',
        'crowd_nav.utils',
        'crowd_sim',
        'crowd_sim.envs',
        'crowd_sim.envs.policy',
        'crowd_sim.envs.utils',
    ],
    
    install_requires=[
        'gitpython',
        'wheel==0.29.0',
        'gym==0.12.1',
        'matplotlib==3.0.2',
        'numpy==1.14.5',
        'scipy==1.1.0',
        'pykalman',
        'torch==0.4.0',
        'torchvision==0.2.1',
        'tensorflow==1.10.1',
        'pillow<7',
        'tk',
        'kiwisolver==1.0.1',
        'grpcio==1.8.6',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
