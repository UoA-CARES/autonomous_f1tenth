from setuptools import setup
import os
from glob import glob

package_name = 'reinforcement_learning'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='retinfai',
    maintainer_email='aferetipama@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sanity_check = reinforcement_learning.sanity_check:main',
            'train = reinforcement_learning.train:main',
            'test = reinforcement_learning.test:main',
            'greatest_gap = reinforcement_learning.greatest_gap:main',
            'recorder = reinforcement_learning.recorder:main', 
        ],
    },
)
