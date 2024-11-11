from setuptools import setup
import os
from glob import glob

package_name = 'environments'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 
              f'{package_name}.autoencoders'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'sdf'), glob('sdf/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
        (os.path.join('share', package_name, 'map_info'), glob('map_info/track_01/*')),
        (os.path.join('share', package_name, 'map_info'), glob('map_info/track_02/*')),
        (os.path.join('share', package_name, 'map_info'), glob('map_info/track_03/*')),
        (os.path.join('share', package_name, 'map_info'), glob('map_info/track_04/*')),
        (os.path.join('share', package_name, 'map_info'), glob('map_info/track_05/*')),
        (os.path.join('share', package_name, 'map_info'), glob('map_info/track_06/*')),
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
            'CarGoalReset = environments.CarGoalReset:main',
            'CarWallReset = environments.CarWallReset:main',
            'CarBlockReset = environments.CarBlockReset:main',
            'CarTrackReset = environments.CarTrackReset:main',
            'CarBeatReset = environments.CarBeatReset:main',
            'SteppingService = environments.SteppingService:main',
            'LidarLogger = environments.lidar_logger:main'
        ],
    },
)
