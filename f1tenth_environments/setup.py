from setuptools import find_packages, setup

package_name = "f1tenth_environments"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="retinfai",
    maintainer_email="aferetipama@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
)
