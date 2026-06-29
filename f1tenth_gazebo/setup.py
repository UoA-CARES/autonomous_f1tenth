from glob import glob
import os

from setuptools import find_packages, setup

package_name = "f1tenth_gazebo"

folders = glob("map_info/*")
map_infos = []
for folder in folders:
    folder_name = os.path.basename(folder)
    map_infos.append(
        (
            os.path.join("share", package_name, "map_info", folder_name),
            glob(f"{folder}/*"),
        )
    )

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*")),
        (os.path.join("share", package_name, "meshes"), glob("meshes/*")),
        (os.path.join("share", package_name, "sdf"), glob("sdf/*")),
        (os.path.join("share", package_name), glob("launch/*launch.[pxy][yma]*")),
        *map_infos,
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="anyone",
    maintainer_email="thenickys123@gmail.com",
    description="Gazebo simulation assets for autonomous_f1tenth",
    license="TODO: License declaration",
)
