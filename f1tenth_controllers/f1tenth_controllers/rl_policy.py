import importlib
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cares_reinforcement_learning.util.helpers import denormalize
from cares_reinforcement_learning.util.network_factory import NetworkFactory

from .controller import Controller


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
        "cares_reinforcement_learning.util.configurations"
    )
    config_class_name = f"{algorithm}Config"
    try:
        config_class = getattr(configurations_module, config_class_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Unsupported RL algorithm '{algorithm}'. Expected a configuration class named {config_class_name}."
        ) from exc

    return config_class()


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
            ("max_speed", 2.0),
            ("max_turn", 0.45),
            ("min_speed", 0.0),
            ("min_turn", -0.45),
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
    OBSERVATION_SIZE = 12
    ACTION_NUM = 2

    controller = Controller("rl_policy_", params["car_name"], step_sleep_time_ms=100)
    policy_id = "rl"
    network_factory = NetworkFactory()
    network_config = _load_network_config(params["algorithm"])
    agent = network_factory.create_network(
        OBSERVATION_SIZE, ACTION_NUM, config=network_config
    )

    checkpoint_path = _resolve_path(params["checkpoint_path"], controllers_share)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Unable to find model checkpoint at '{checkpoint_path}'."
        )

    checkpoint_name = checkpoint_path.stem
    if checkpoint_name.endswith("_checkpoint"):
        checkpoint_name = checkpoint_name.removesuffix("_checkpoint")

    print(f"Reading saved model checkpoint from '{checkpoint_path}'")
    agent.load_models(str(checkpoint_path.parent), checkpoint_name)
    print("Successfully loaded model checkpoint")

    state = controller.step([0, 0], policy_id)
    state = state[6:]

    MAX_CONFIG_ACTIONS = MAX_ACTIONS
    MIN_CONFIG_ACTIONS = MIN_ACTIONS

    while True:
        action = agent.select_action_from_policy(state)
        action = denormalize(action, MAX_CONFIG_ACTIONS, MIN_CONFIG_ACTIONS)
        action = np.clip(action, MIN_ACTIONS, MAX_ACTIONS)
        state = controller.step(action, policy_id)
        state = state[6:]
