FROM osrf/ros:humble-desktop

# Install gazebo
RUN apt-get update -y && \
    apt-get install apt-utils lsb-release wget gnupg pip -y && \
    wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    apt-get update -y && \
    apt-get install gz-garden -y

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