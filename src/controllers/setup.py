from setuptools import setup
import os
from glob import glob

package_name = 'controllers'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, f'{package_name}.path_planners', f'{package_name}.path_trackers'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='emilysteiner71',
    maintainer_email='emilysteiner71@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ftg_policy = controllers.ftg_policy:main',
            'rl_policy = controllers.rl_policy:main',
            'sim = controllers.sim:main',
            'track = controllers.track:main',
            'load_path = controllers.load_path:main',
            'planner = controllers.planner:main'
        ],
    },
)
