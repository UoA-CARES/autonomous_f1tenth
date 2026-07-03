import importlib
import sys
from pathlib import Path

import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from pydantic import BaseModel


def _has_cares_package(path: Path) -> bool:
    package_root = path / "cares_reinforcement_learning"
    return (package_root / "util").is_dir() and (package_root / "algorithm").is_dir()


def _clear_partial_cares_imports() -> None:
    for module_name in list(sys.modules):
        if module_name == "cares_reinforcement_learning" or module_name.startswith(
            "cares_reinforcement_learning."
        ):
            del sys.modules[module_name]


def _ensure_cares_import_path() -> None:
    roots = [Path.cwd(), Path(__file__).resolve()]
    candidates = []
    for root in roots:
        for parent in [root, *root.parents]:
            candidates.append(parent)
            candidates.append(parent / "cares")
            candidates.append(parent / "cares_reinforcement_learning")

    candidates.append(Path.home() / "workspace")
    candidates.append(Path.home() / "workspace" / "cares_reinforcement_learning")

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _has_cares_package(candidate):
            sys.path.insert(0, str(candidate))
            return


try:
    from cares_reinforcement_learning.util.helpers import denormalize
    from cares_reinforcement_learning.algorithm.algorithm_factory import AlgorithmFactory
    from cares_reinforcement_learning.types.observation import SARLObservation
except ModuleNotFoundError as exc:
    if not exc.name.startswith("cares_reinforcement_learning"):
        raise
    _clear_partial_cares_imports()
    _ensure_cares_import_path()
    try:
        from cares_reinforcement_learning.util.helpers import denormalize
        from cares_reinforcement_learning.algorithm.algorithm_factory import AlgorithmFactory
        from cares_reinforcement_learning.types.observation import SARLObservation
    except ModuleNotFoundError as retry_exc:
        raise ModuleNotFoundError(
            "Could not import cares_reinforcement_learning. Launch from the workspace root "
            "that contains the cares_reinforcement_learning package, or pass "
            "`cares_python_path:=/path/to/that/root`."
        ) from retry_exc

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

def _checkpoint_algorithm(algorithm: str, actor_state: dict) -> str:
    configured_algorithm = algorithm.upper()
    has_mean_head = "mean_linear.weight" in actor_state
    has_log_std_head = "log_std_linear.weight" in actor_state

    if has_mean_head != has_log_std_head:
        raise ValueError(
            "Actor checkpoint has only one SAC output head; expected both "
            "mean_linear.weight and log_std_linear.weight."
        )

    if has_mean_head:
        if configured_algorithm != "SAC":
            print(
                f"Checkpoint contains SAC actor heads; using SAC instead of "
                f"configured algorithm '{algorithm}'."
            )
        return "SAC"

    return configured_algorithm


def _configure_actor_from_checkpoint(
    algorithm: str, network_config, actor_state: dict
):
    configurations_module = importlib.import_module(
        "cares_reinforcement_learning.algorithm.configurations"
    )

    trunk_weights = [
        (name, tensor)
        for name, tensor in actor_state.items()
        if name.startswith("act_net.model.")
        and name.endswith("weight")
        and getattr(tensor, "ndim", 0) == 2
    ]
    trunk_weights.sort(key=lambda item: int(item[0].split(".")[2]))
    if not trunk_weights:
        raise ValueError("Actor checkpoint contains no act_net linear weights.")

    algorithm = algorithm.upper()
    is_sac = algorithm in {"SAC", "PERSAC", "LAPSAC", "LA3PSAC"}

    if is_sac:
        mean_weight = actor_state.get("mean_linear.weight")
        log_std_weight = actor_state.get("log_std_linear.weight")
        if mean_weight is None or log_std_weight is None:
            raise ValueError(
                f"{algorithm} checkpoint must contain mean_linear and "
                "log_std_linear actor heads."
            )
        if tuple(mean_weight.shape) != tuple(log_std_weight.shape):
            raise ValueError(
                "SAC mean and log-std actor head shapes do not match: "
                f"{tuple(mean_weight.shape)} vs {tuple(log_std_weight.shape)}."
            )
        action_num = int(mean_weight.shape[0])
    else:
        action_num = int(trunk_weights[-1][1].shape[0])

    layers = []
    for index, (_, weight) in enumerate(trunk_weights):
        in_features = int(weight.shape[1])
        out_features = int(weight.shape[0])
        is_output_layer = not is_sac and index == len(trunk_weights) - 1

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
    observation_size = int(trunk_weights[0][1].shape[1])
    return observation_size, action_num



MARL_ALGORITHMS = {
    "MADDPG",
    "M3DDPG",
    "MATD3",
    "MASAC",
    "MAPPO",
    "IDDPG",
    "ITD3",
    "ISAC",
    "IPPO",
}
INDEPENDENT_MARL_ALGORITHMS = {"IDDPG", "ITD3", "ISAC", "IPPO"}


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_marl_teams(value: str, agent_ids: list[str]) -> dict[str, list[str]]:
    if not str(value).strip():
        return {"team_0": agent_ids}

    teams = {}
    for team_spec in str(value).split(";"):
        if not team_spec.strip():
            continue
        if ":" not in team_spec:
            raise ValueError(
                "marl_teams entries must look like 'team_id:agent_0,agent_1'."
            )
        team_id, members = team_spec.split(":", 1)
        teams[team_id.strip()] = _parse_csv(members)

    assigned_agents = {agent_id for members in teams.values() for agent_id in members}
    missing = [agent_id for agent_id in agent_ids if agent_id not in assigned_agents]
    if missing:
        teams.setdefault("team_0", []).extend(missing)
    return teams


def _set_optional_config(config, name: str, value):
    if value in (None, "") or not hasattr(config, name):
        return
    current = getattr(config, name)
    if isinstance(current, bool):
        setattr(config, name, str(value).lower() in {"1", "true", "yes", "on"})
    elif isinstance(current, int) and not isinstance(current, bool):
        setattr(config, name, int(value))
    elif isinstance(current, float):
        setattr(config, name, float(value))
    else:
        setattr(config, name, str(value))


def _checkpoint_has_actor(path: Path) -> bool:
    if not path.is_file() or path.suffix not in {".pth", ".pht", ".pt"}:
        return False
    try:
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
    except Exception:
        return False
    return isinstance(checkpoint, dict) and isinstance(checkpoint.get("actor"), dict)


def _load_actor_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("actor"), dict):
        raise ValueError(f"'{path}' does not contain an actor checkpoint.")
    return checkpoint


def _find_actor_checkpoint(checkpoint_path: Path, learning_unit_id: str | None = None) -> Path:
    if checkpoint_path.is_file():
        if _checkpoint_has_actor(checkpoint_path):
            return checkpoint_path
        raise ValueError(f"'{checkpoint_path}' is not an actor checkpoint.")

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint path '{checkpoint_path}' does not exist.")

    search_roots = []
    if learning_unit_id is not None:
        unit_root = checkpoint_path / learning_unit_id
        if unit_root.is_dir():
            search_roots.append(unit_root)
    search_roots.append(checkpoint_path)

    seen = set()
    for root in search_roots:
        for candidate in sorted(root.rglob("*checkpoint*.pth")):
            if candidate in seen:
                continue
            seen.add(candidate)
            if learning_unit_id is not None and learning_unit_id not in str(candidate):
                continue
            if _checkpoint_has_actor(candidate):
                return candidate

    for candidate in sorted(checkpoint_path.rglob("*.pth")):
        if _checkpoint_has_actor(candidate):
            return candidate

    raise FileNotFoundError(
        f"No actor checkpoint found under '{checkpoint_path}'"
        + (f" for learning unit '{learning_unit_id}'." if learning_unit_id else ".")
    )


def _configure_actor_from_any_checkpoint(
    algorithm: str,
    network_config,
    checkpoint_path: Path,
) -> tuple[int, int, Path]:
    actor_checkpoint_path = _find_actor_checkpoint(checkpoint_path)
    actor_state = _load_actor_checkpoint(actor_checkpoint_path)["actor"]
    checkpoint_algorithm = _checkpoint_algorithm(algorithm, actor_state)
    observation_size, action_num = _configure_actor_from_checkpoint(
        checkpoint_algorithm,
        network_config,
        actor_state,
    )
    return observation_size, action_num, actor_checkpoint_path


def _identity_extra_size(config, agent_ids: list[str], teams: dict[str, list[str]]) -> int:
    if len(agent_ids) <= 1:
        return 0
    extra = 0
    if getattr(config, "parameter_sharing_scope", "") == "shared":
        if getattr(config, "use_team_id", 0):
            extra += len(teams)
        if getattr(config, "use_agent_id", 0):
            extra += len(agent_ids)
    return extra


def _build_marl_observation_size(
    algorithm: str,
    actor_observation_size: int,
    config,
    agent_ids: list[str],
    teams: dict[str, list[str]],
) -> tuple[dict, int]:
    raw_observation_size = actor_observation_size
    if algorithm in INDEPENDENT_MARL_ALGORITHMS:
        raw_observation_size -= _identity_extra_size(config, agent_ids, teams)
    if raw_observation_size < 1:
        raise ValueError(
            f"Invalid MARL raw observation size {raw_observation_size}; check agent ids, "
            "teams, and parameter-sharing settings against the checkpoint."
        )
    return {
        "obs": {agent_id: raw_observation_size for agent_id in agent_ids},
        "teams": teams,
        "state": raw_observation_size * len(agent_ids),
        "num_agents": len(agent_ids),
    }, raw_observation_size


def _learning_unit_for_agent(agent, agent_id: str):
    if hasattr(agent, "agent_id_to_actor_id"):
        unit_id = agent.agent_id_to_actor_id[agent_id]
    elif hasattr(agent, "agent_id_to_learning_unit_id"):
        unit_id = agent.agent_id_to_learning_unit_id[agent_id]
    else:
        raise TypeError("MARL agent does not expose a known agent-to-actor mapping.")
    return unit_id, agent.learning_units[unit_id]


def _load_actor_state_into_unit(learning_unit, actor_checkpoint_path: Path) -> None:
    checkpoint = _load_actor_checkpoint(actor_checkpoint_path)
    learning_unit.actor_net.load_state_dict(checkpoint["actor"])
    if hasattr(learning_unit, "target_actor_net") and "target_actor" in checkpoint:
        learning_unit.target_actor_net.load_state_dict(checkpoint["target_actor"])


def _load_marl_actor_weights(agent, checkpoint_path: Path, controlled_agent_id: str) -> None:
    controlled_unit_id, controlled_unit = _learning_unit_for_agent(agent, controlled_agent_id)
    loaded_units = set()

    for unit_id, learning_unit in getattr(agent, "learning_units", {}).items():
        try:
            actor_checkpoint_path = _find_actor_checkpoint(checkpoint_path, unit_id)
        except FileNotFoundError:
            continue
        _load_actor_state_into_unit(learning_unit, actor_checkpoint_path)
        loaded_units.add(unit_id)

    if controlled_unit_id not in loaded_units:
        actor_checkpoint_path = _find_actor_checkpoint(checkpoint_path)
        _load_actor_state_into_unit(controlled_unit, actor_checkpoint_path)
        loaded_units.add(controlled_unit_id)

    print(
        f"Loaded MARL actor weights for learning units: {sorted(loaded_units)}; "
        f"controlled_agent_id={controlled_agent_id}."
    )


def _marl_action(agent, agent_id: str, state: np.ndarray):
    _, learning_unit = _learning_unit_for_agent(agent, agent_id)
    obs = state.astype(np.float32, copy=False)
    if hasattr(agent, "augment_observation"):
        obs = agent.augment_observation(obs, agent_id)
    return learning_unit.act(SARLObservation(vector_state=obs), evaluation=True).action


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
            ("controlled_agent_id", ""),
            ("marl_agent_ids", ""),
            ("marl_teams", ""),
            ("marl_parameter_sharing_scope", ""),
            ("marl_use_agent_id", ""),
            ("marl_use_team_id", ""),
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
    parameter_names = [
        "car_name",
        "algorithm",
        "checkpoint_path",
        "controlled_agent_id",
        "marl_agent_ids",
        "marl_teams",
        "marl_parameter_sharing_scope",
        "marl_use_agent_id",
        "marl_use_team_id",
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
    params = {
        parameter.name: parameter.value
        for parameter in param_node.get_parameters(parameter_names)
    }

    algorithm = str(params["algorithm"]).upper()
    is_marl = algorithm in MARL_ALGORITHMS
    controlled_agent_id = str(params["controlled_agent_id"] or params["car_name"])

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

    print(f"Reading saved model checkpoint from '{checkpoint_path}'")

    network_config = _load_network_config(algorithm)
    _set_optional_config(
        network_config,
        "parameter_sharing_scope",
        params["marl_parameter_sharing_scope"],
    )
    _set_optional_config(network_config, "use_agent_id", params["marl_use_agent_id"])
    _set_optional_config(network_config, "use_team_id", params["marl_use_team_id"])

    if is_marl:
        agent_ids = _parse_csv(params["marl_agent_ids"])
        if not agent_ids:
            agent_ids = [controlled_agent_id]
        if controlled_agent_id not in agent_ids:
            agent_ids.insert(0, controlled_agent_id)
        teams = _parse_marl_teams(params["marl_teams"], agent_ids)

        actor_observation_size, action_num, actor_checkpoint_path = (
            _configure_actor_from_any_checkpoint(
                algorithm,
                network_config,
                checkpoint_path,
            )
        )
        observation_size, raw_observation_size = _build_marl_observation_size(
            algorithm,
            actor_observation_size,
            network_config,
            agent_ids,
            teams,
        )
        print(
            f"Configured MARL {algorithm}: agents={agent_ids}, teams={teams}, "
            f"controlled_agent_id={controlled_agent_id}, "
            f"raw_observation_size={raw_observation_size}, "
            f"actor_checkpoint='{actor_checkpoint_path}'."
        )
    else:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Unable to find model checkpoint at '{checkpoint_path}'."
            )
        checkpoint = _load_actor_checkpoint(checkpoint_path)
        checkpoint_algorithm = _checkpoint_algorithm(algorithm, checkpoint["actor"])
        network_config = _load_network_config(checkpoint_algorithm)
        actor_observation_size, action_num = _configure_actor_from_checkpoint(
            checkpoint_algorithm,
            network_config,
            checkpoint["actor"],
        )
        raw_observation_size = actor_observation_size
        observation_size = {"vector": raw_observation_size}

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

    lidar_points = raw_observation_size - ODOM_STATE_SIZES[odom_mode]
    if lidar_points < 1:
        raise ValueError(
            f"Checkpoint observation size {raw_observation_size} cannot represent "
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
    if state_builder.policy_state_size != raw_observation_size:
        raise ValueError(
            f"Runtime state size {state_builder.policy_state_size} does not match "
            f"checkpoint actor input size {raw_observation_size}."
        )

    controller = Controller(
        "rl_policy_",
        params["car_name"],
        step_sleep_time_ms=100,
        lidar_points=lidar_points,
        state_builder=state_builder,
        drive_topic=f"/{params['car_name']}/rl_drive",
    )
    policy_id = "rl"
    agent = AlgorithmFactory().create_network(
        observation_size,
        action_num,
        config=network_config,
    )

    if is_marl:
        _load_marl_actor_weights(agent, checkpoint_path, controlled_agent_id)
        print(
            f"Successfully loaded MARL actor: algorithm={algorithm}, "
            f"controlled_agent_id={controlled_agent_id}, "
            f"observation_size={raw_observation_size}, lidar_points={lidar_points}, "
            f"actions={action_num}"
        )
    else:
        checkpoint = _load_actor_checkpoint(checkpoint_path)
        agent.actor_net.load_state_dict(checkpoint["actor"])
        if hasattr(agent, "target_actor_net") and "target_actor" in checkpoint:
            agent.target_actor_net.load_state_dict(checkpoint["target_actor"])
        print(
            f"Successfully loaded actor: observation_size={raw_observation_size}, "
            f"lidar_points={lidar_points}, actions={action_num}"
        )

    state = controller.step([0, 0], policy_id)

    if len(state) != raw_observation_size:
        raise ValueError(
            f"Initial runtime state has {len(state)} values; checkpoint expects "
            f"{raw_observation_size}."
        )
    MAX_CONFIG_ACTIONS = MAX_ACTIONS
    MIN_CONFIG_ACTIONS = MIN_ACTIONS

    while True:
        if is_marl:
            action = _marl_action(
                agent,
                controlled_agent_id,
                np.asarray(state, dtype=np.float32),
            )
        else:
            observation = SARLObservation(
                vector_state=np.asarray(state, dtype=np.float32)
            )
            action = agent.act(observation, evaluation=True).action
        action = denormalize(action, MAX_CONFIG_ACTIONS, MIN_CONFIG_ACTIONS)
        action = np.clip(action, MIN_ACTIONS, MAX_ACTIONS)
        state = controller.step(action, policy_id)
        if len(state) != raw_observation_size:
            raise ValueError(
                f"Runtime state has {len(state)} values; checkpoint expects "
                f"{raw_observation_size}."
            )
