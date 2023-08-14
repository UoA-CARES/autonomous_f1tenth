FROM osrf/ros:humble-desktop

# Install gazebo
RUN apt-get update -y && \
    apt-get install apt-utils lsb-release wget gnupg pip -y && \
    wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    apt-get update -y && \
    apt-get install gz-garden -y

RUN sudo apt install python3-pip wget lsb-release gnupg curl -y && \
    sudo sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros2-latest.list' -y && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - && \
    sudo apt-get update -y && \
    sudo apt-get install python3-vcstool python3-colcon-common-extensions -y

WORKDIR /gz

RUN mkdir -p /gz/src && \ 
    cd /gz/src && \
    wget https://raw.githubusercontent.com/gazebo-tooling/gazebodistro/master/collection-garden.yaml && \
    vcs import < collection-garden.yaml && \
    sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    sudo apt-get update && \
    sudo apt -y install \
  $(sort -u $(find . -iname 'packages-'`lsb_release -cs`'.apt' -o -iname 'packages.apt' | grep -v '/\.git/') | sed '/gz\|sdf/d' | tr '\n' ' ')

RUN cd src && \
    rm -rdf gz-sim && \
    git clone https://github.com/UoA-CARES/gz-sim.git

RUN colcon build --merge-install

RUN echo ". /gz/install/setup.bash" >> ~/.bashrc

SHELL [ "/bin/bash", "-c" ]
WORKDIR /ws
COPY . .

# Install dep
RUN rosdep update -y && \
    rosdep install -r --from-paths src -i -y --rosdistro humble

# Build and install
RUN . /opt/ros/humble/setup.bash && \
    colcon build --symlink-install

RUN git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git

RUN cd cares_reinforcement_learning && \
    pip3 install -r requirements.txt && \
    pip3 install --editable .

RUN echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc && \
    echo 'source install/setup.bash' >> ~/.bashrc

WORKDIR /ws