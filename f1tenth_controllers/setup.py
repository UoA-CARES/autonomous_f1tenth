from setuptools import find_packages, setup
import os
from glob import glob

package_name = "f1tenth_controllers"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name),
            glob("launch/*launch.[pxy][yma]*"),
        ),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="emilysteiner71",
    maintainer_email="emilysteiner71@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    entry_points={
        "console_scripts": [
            "ftg_policy = f1tenth_controllers.ftg_policy:main",
            "rl_policy = f1tenth_controllers.rl_policy:main",
            "rl_deadman = f1tenth_controllers.rl_deadman:main",
            "sim = f1tenth_controllers.sim:main",
            "track = f1tenth_controllers.track:main",
            "load_path = f1tenth_controllers.load_path:main",
            "planner = f1tenth_controllers.planner:main",
            "state_machine = f1tenth_controllers.state_machine:main",
        ],
    },
)
