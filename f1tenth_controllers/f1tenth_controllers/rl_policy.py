import importlib
from pathlib import Path

import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from pydantic import BaseModel
from cares_reinforcement_learning.util.helpers import denormalize
from cares_reinforcement_learning.algorithm.algorithm_factory import AlgorithmFactory
from cares_reinforcement_learning.types.observation import SARLObservation
from f1tenth_environments.state_builder import ODOM_STATE_SIZES, StateBuilder

from .controller import Controller

if not hasattr(BaseModel, "model_dump"):
    BaseModel.model_dump = BaseModel.dict


def _resolve_path(path_value: str, package_share: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate

    search_roots = [
        Path.cwd(),
        package_share,
        package_share.parent,
        package_share.parent.parent,
        package_share.parent.parent.parent,
    ]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved

    return (package_share / candidate).resolve()


def _load_network_config(algorithm: str):
    configurations_module = importlib.import_module(
        "cares_reinforcement_learning.algorithm.configurations"
    )
    config_class_name = f"{algorithm}Config"
    try:
        config_class = getattr(configurations_module, config_class_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Unsupported RL algorithm '{algorithm}'. Expected a configuration class named {config_class_name}."
        ) from exc

    return config_class()

def _configure_actor_from_checkpoint(network_config, actor_state: dict):
    configurations_module = importlib.import_module(
        "cares_reinforcement_learning.algorithm.configurations"
    )
    linear_weights = [
        (name, tensor)
        for name, tensor in actor_state.items()
        if name.endswith("weight") and getattr(tensor, "ndim", 0) == 2
    ]
    if not linear_weights:
        raise ValueError("Actor checkpoint contains no linear weight tensors.")

    layers = []
    for index, (_, weight) in enumerate(linear_weights):
        in_features = int(weight.shape[1])
        out_features = int(weight.shape[0])
        is_output_layer = index == len(linear_weights) - 1

        layer_args = {"layer_type": "Linear"}
        if index > 0:
            layer_args["in_features"] = in_features
        if not is_output_layer:
            layer_args["out_features"] = out_features
        layers.append(configurations_module.TrainableLayer(**layer_args))
        layers.append(
            configurations_module.FunctionLayer(
                layer_type="Tanh" if is_output_layer else "ReLU"
            )
        )

    network_config.actor_config = configurations_module.MLPConfig(layers=layers)
    observation_size = int(linear_weights[0][1].shape[1])
    action_num = int(linear_weights[-1][1].shape[0])
    return observation_size, action_num



def main():
    rclpy.init()
    param_node = rclpy.create_node("rl_policy_params")

    controllers_share = Path(get_package_share_directory("f1tenth_controllers"))

    param_node.declare_parameters(
        "",
        [
            ("car_name", "f1tenth"),
            ("algorithm", "TD3"),
            ("checkpoint_path", ""),
            ("max_speed", 5.0),
            ("max_turn", 0.434),
            ("min_speed", 0.5),
            ("min_turn", -0.434),
            ("odom_mode", "velocity_only"),
            ("lidar_mode", "processed"),
            ("forward_half_angle", 45.0),
            ("n_forward", 5),
            ("wheelbase", 0.325),
        ],
    )
    params = {
        parameter.name: parameter.value
        for parameter in param_node.get_parameters(
            [
                "car_name",
                "algorithm",
                "checkpoint_path",
                "max_speed",
                "max_turn",
                "min_speed",
                "min_turn",
                "odom_mode",
                "lidar_mode",
                "forward_half_angle",
                "n_forward",
                "wheelbase",
            ]
        )
    }

    MAX_ACTIONS = np.asarray(
        [
            float(params["max_speed"]),
            float(params["max_turn"]),
        ]
    )
    MIN_ACTIONS = np.asarray(
        [
            float(params["min_speed"]),
            float(params["min_turn"]),
        ]
    )
    checkpoint_path = _resolve_path(params["checkpoint_path"], controllers_share)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Unable to find model checkpoint at '{checkpoint_path}'."
        )

    print(f"Reading saved model checkpoint from '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if not isinstance(checkpoint, dict) or "actor" not in checkpoint:
        raise ValueError(
            f"'{checkpoint_path}' is not a combined CARES RL checkpoint "
            "containing an 'actor' state dictionary."
        )

    network_config = _load_network_config(params["algorithm"])
    observation_size, action_num = _configure_actor_from_checkpoint(
        network_config, checkpoint["actor"]
    )
    if action_num != len(MAX_ACTIONS):
        raise ValueError(
            f"Checkpoint actor outputs {action_num} actions, but the controller expects "
            f"{len(MAX_ACTIONS)}."
        )

    odom_mode = params["odom_mode"]
    if odom_mode not in ODOM_STATE_SIZES:
        raise ValueError(
            f"Unsupported odom_mode '{odom_mode}'. "
            f"Expected one of {list(ODOM_STATE_SIZES)}."
        )

    lidar_points = observation_size - ODOM_STATE_SIZES[odom_mode]
    if lidar_points < 1:
        raise ValueError(
            f"Checkpoint observation size {observation_size} cannot represent "
            f"{ODOM_STATE_SIZES[odom_mode]} odometry values plus lidar data."
        )

    state_builder = StateBuilder(
        odom_mode=odom_mode,
        lidar_mode=params["lidar_mode"],
        lidar_state_size=lidar_points,
        min_speed=float(params["min_speed"]),
        max_speed=float(params["max_speed"]),
        max_turn=float(params["max_turn"]),
        wheelbase_m=float(params["wheelbase"]),
        forward_half_angle=float(params["forward_half_angle"]),
        n_forward=int(params["n_forward"]),
    )
    if state_builder.policy_state_size != observation_size:
        raise ValueError(
            f"Runtime state size {state_builder.policy_state_size} does not match "
            f"checkpoint actor input size {observation_size}."
        )

    controller = Controller(
        "rl_policy_",
        params["car_name"],
        step_sleep_time_ms=100,
        lidar_points=lidar_points,
        state_builder=state_builder,
    )
    policy_id = "rl"
    agent = AlgorithmFactory().create_network(
        {"vector": observation_size},
        action_num,
        config=network_config,
    )

    agent.actor_net.load_state_dict(checkpoint["actor"])
    if hasattr(agent, "target_actor_net") and "target_actor" in checkpoint:
        agent.target_actor_net.load_state_dict(checkpoint["target_actor"])
    print(
        f"Successfully loaded actor: observation_size={observation_size}, "
        f"lidar_points={lidar_points}, actions={action_num}"
    )

    state = controller.step([0, 0], policy_id)

    if len(state) != observation_size:
        raise ValueError(
            f"Initial runtime state has {len(state)} values; checkpoint expects "
            f"{observation_size}."
        )
    MAX_CONFIG_ACTIONS = MAX_ACTIONS
    MIN_CONFIG_ACTIONS = MIN_ACTIONS

    while True:
        observation = SARLObservation(
            vector_state=np.asarray(state, dtype=np.float32)
        )
        action = agent.act(observation, evaluation=True).action
        action = denormalize(action, MAX_CONFIG_ACTIONS, MIN_CONFIG_ACTIONS)
        action = np.clip(action, MIN_ACTIONS, MAX_ACTIONS)
        state = controller.step(action, policy_id)
        if len(state) != observation_size:
            raise ValueError(
                f"Runtime state has {len(state)} values; checkpoint expects "
                f"{observation_size}."
            )
