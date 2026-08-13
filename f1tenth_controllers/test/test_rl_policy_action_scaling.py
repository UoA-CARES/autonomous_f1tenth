import numpy as np
import torch
from cares_reinforcement_learning.algorithm.configurations import (
    MAPPOConfig,
    MATD3Config,
)
from f1tenth_controllers.rl_policy import (
    _configure_actor_from_checkpoint,
    _prepare_action_for_controller,
)


POLICY_MAX_ACTIONS = np.asarray([5.0, 0.434], dtype=np.float32)
DEPLOYMENT_MAX_ACTIONS = np.asarray([3.0, 0.434], dtype=np.float32)
MIN_ACTIONS = np.asarray([0.5, -0.434], dtype=np.float32)


def test_legacy_marl_actions_match_cares_direct_clip_semantics():
    action = np.asarray([0.8, 0.4], dtype=np.float32)

    command = _prepare_action_for_controller(
        action,
        is_marl=True,
        marl_action_scaling="legacy_direct",
        policy_max_actions=POLICY_MAX_ACTIONS,
        min_actions=MIN_ACTIONS,
        max_actions=DEPLOYMENT_MAX_ACTIONS,
    )

    np.testing.assert_allclose(command, [0.8, 0.4])


def test_normalized_marl_actions_are_denormalized_then_deployment_clipped():
    action = np.asarray([0.8, 0.4], dtype=np.float32)

    command = _prepare_action_for_controller(
        action,
        is_marl=True,
        marl_action_scaling="normalized",
        policy_max_actions=POLICY_MAX_ACTIONS,
        min_actions=MIN_ACTIONS,
        max_actions=DEPLOYMENT_MAX_ACTIONS,
    )

    np.testing.assert_allclose(command, [3.0, 0.1736], rtol=1e-5)


def test_single_agent_actions_remain_normalized():
    action = np.asarray([0.0, 0.0], dtype=np.float32)

    command = _prepare_action_for_controller(
        action,
        is_marl=False,
        marl_action_scaling="legacy_direct",
        policy_max_actions=POLICY_MAX_ACTIONS,
        min_actions=MIN_ACTIONS,
        max_actions=DEPLOYMENT_MAX_ACTIONS,
    )

    np.testing.assert_allclose(command, [2.75, 0.0])


def _actor_state():
    return {
        "act_net.model.0.weight": torch.zeros((4, 3)),
        "act_net.model.0.bias": torch.zeros(4),
        "act_net.model.2.weight": torch.zeros((2, 4)),
        "act_net.model.2.bias": torch.zeros(2),
    }


def _layer_types(config):
    return [layer.layer_type for layer in config.actor_config.layers]


def test_ppo_family_reconstruction_leaves_mean_output_unsquashed():
    config = MAPPOConfig()

    observation_size, action_num = _configure_actor_from_checkpoint(
        "MAPPO", config, _actor_state()
    )

    assert (observation_size, action_num) == (3, 2)
    assert _layer_types(config) == ["Linear", "ReLU", "Linear"]


def test_td3_family_reconstruction_keeps_actor_output_tanh():
    config = MATD3Config()

    observation_size, action_num = _configure_actor_from_checkpoint(
        "MATD3", config, _actor_state()
    )

    assert (observation_size, action_num) == (3, 2)
    assert _layer_types(config) == ["Linear", "ReLU", "Linear", "Tanh"]

