from setuptools import setup
import os
from glob import glob

package_name = 'environments'

folders = glob('map_info/*')
map_infos = []
for folder in folders:
    try:
        map_infos.append((os.path.join('share', package_name, 'map_info'), glob(f"{folder}/*")))
    except:
        print("Not working")

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
        (os.path.join('share', 'f1tenth', 'f1tenth_description', 'meshes'), glob('meshes/*')),
        *map_infos,
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
            'CarRaceReset = environments.CarRaceReset:main',
            'CarOvertakeReset = environments.CarOvertakeReset:main',
            'TwoCarReset = environments.TwoCarReset:main',
            'SteppingService = environments.SteppingService:main',
            'LidarLogger = environments.lidar_logger:main'
        ],
    },
)
