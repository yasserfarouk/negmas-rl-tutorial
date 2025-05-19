from functools import partial

import numpy as np
import pytest
from negmas.inout import Scenario, TableFun
from negmas.outcomes import make_issue, make_os
from negmas.preferences import AffineFun, LinearFun
from negmas.preferences import LinearUtilityAggregationFunction as LU
from negmas.sao.mechanism import SAOMechanism
from negmas_rl.env import DEFAULT_N_STEPS, NegoEnv, NegoMultiEnv
from negmas_rl.generators.mechanism import MechanismRepeater
from negmas_rl.generators.scenario import RandomScenarioGenerator, ScenarioRepeater

ENV_TYPES = [NegoMultiEnv, NegoEnv]
ENV_STEPS = int(5.5 * DEFAULT_N_STEPS)


def random_action(
    obs: np.ndarray | dict[str, np.ndarray] | dict[str, dict],
    env: NegoMultiEnv | NegoEnv,
) -> np.ndarray:
    """Samples a random action from the action space of the"""
    _ = obs
    return env.action_space.sample()


@pytest.mark.parametrize("env_type", ENV_TYPES)
def test_env_can_sample(env_type):
    env = env_type()
    for _ in range(10):
        obs = env.observation_space.sample()
        assert obs
        assert env.action_space.sample() is not None


@pytest.mark.parametrize("env_type", ENV_TYPES)
def test_reset_generates_a_valid_observation(env_type):
    env = env_type()
    obs1 = env.observation_space.sample()
    assert (
        obs1 in env.observation_space
    ), f"Sampled observation {obs1} is not in the observation space {env.observation_space}"
    obs, _ = env.reset()
    assert (
        obs in env.observation_space
    ), f"Reset output {obs} not in obs space but random sampled obs: {obs1} is in it"


def test_simple_env_runs():
    env = NegoEnv()

    obs, _ = env.reset()
    for _ in range(ENV_STEPS):
        action = partial(random_action, env=env)(obs)  # type: ignore
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float), f"Reward is not a float {reward}"
        assert not info, f"Unexpected info {info=}"
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


def test_multi_env_runs():
    env = NegoMultiEnv()

    obs, _ = env.reset()
    for _ in range(ENV_STEPS):
        action = partial(random_action, env=env)(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert isinstance(reward, dict) and all(
            isinstance(_, float) for _ in reward.values()
        ), f"Reward is not a float {reward}"
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


@pytest.mark.parametrize(
    ("n_negotiators", "n_learners", "one_per_step"),
    (
        (2, 2, False),
        (3, 1, False),
        (2, 1, False),
        (3, 2, False),
        (3, 3, False),
        (5, 1, False),
        (5, 2, False),
        (2, 2, True),
        (3, 1, True),
        (2, 1, True),
        (3, 2, True),
        (3, 3, True),
        (5, 1, True),
        (5, 2, True),
    ),
)
def test_multi_env_runs_multilateral(n_negotiators, n_learners, one_per_step):
    env = NegoMultiEnv(
        scenario_generator=RandomScenarioGenerator(
            n_ufuns=max(n_negotiators, n_learners)
        ),
        mechanism_generator=MechanismRepeater(
            SAOMechanism,
            params=dict(n_steps=DEFAULT_N_STEPS, one_offer_per_step=one_per_step),
        ),
    )

    obs, _ = env.reset()
    for _ in range(ENV_STEPS):
        action = partial(random_action, env=env)(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert isinstance(reward, dict) and all(
            isinstance(_, float) for _ in reward.values()
        ), f"Reward is not a float {reward}"
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


@pytest.mark.parametrize("env_type", ENV_TYPES)
def test_simple_trade(env_type):
    os = make_os(
        [
            make_issue((1.0, 5.0), "price"),
            make_issue((1, 20), "quantity"),
            make_issue(["high", "med", "low"], "carbon emission"),
        ],
        name="Trade",
    )
    seller_ufun = LU(
        {
            "price": LinearFun(1),
            "quantity": LinearFun(1),
            "carbon emission": TableFun(dict(high=1.0, med=0.5, low=0.0)),
        },
        outcome_space=os,
        reserved_value=0.01,
    )

    buyer_ufun = LU(
        {
            "price": LinearFun(-1),
            "quantity": AffineFun(0.5, 10),
            "carbon emission": TableFun(dict(high=1.0, med=0.5, low=0.0)),
        },
        outcome_space=os,
        reserved_value=0.1,
    )
    scenario = Scenario(os, (seller_ufun, buyer_ufun))
    env = env_type(scenario_generator=ScenarioRepeater(scenario))

    obs, _ = env.reset()
    for _ in range(5000):
        action = partial(random_action, env=env)(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
