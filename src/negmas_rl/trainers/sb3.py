from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
from negmas.inout import Scenario
from negmas.mechanisms import Mechanism
from negmas.negotiators import Negotiator
from negmas.sao.mechanism import SAOMechanism

from ..generators.assigner import Assigner, PositionBasedNegotiatorAssigner
from ..generators.negotiator import NegotiatorGenerator
from ..negotiator import SAORLNegotiator

try:
    import stable_baselines3

    SB3_AVAILABLE = stable_baselines3 is not None
except:
    SB3_AVAILABLE = False
if SB3_AVAILABLE:
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.policies import BaseModel, BasePolicy
    from stable_baselines3.common.type_aliases import (
        GymEnv,
        Schedule,
    )
    from stable_baselines3.sac import SAC

    from ..env import NegoMultiEnv

    __all__ = []
    __all__ = [
        "sb3policy",
        "train",
        "test",
        "make_negotiation",
        "default_policy_generator",
        "PolicyGenerator",
        "SB3_MODELS_BASE",
        "SB3_DEFAULT_ALGORITHM",
    ]

    PolicyGenerator = (
        Callable[[gym.Env], type[BasePolicy] | str] | type[BasePolicy] | str
    )

    SB3_MODELS_BASE = Path.cwd() / "sb3models"

    SB3_DEFAULT_ALGORITHM = SAC

    def sb3policy(obs, model: BaseModel | BaseAlgorithm):
        """Wraps a trained stable-baselines3 model as a negotiation policy"""
        return model.predict(obs)[0]

    def default_policy_generator(env: GymEnv) -> type[BasePolicy] | str:
        if isinstance(env.observation_space, gym.spaces.Dict):
            return "MultiInputPolicy"
        return "MlpPolicy"

    def train(
        env: NegoMultiEnv,
        steps: int,
        algorithm: type[BaseAlgorithm],
        learning_rate: float | Schedule,
        policy: PolicyGenerator = default_policy_generator,
        alg_params: dict[str, Any] | None = None,
        learn_params: dict[str, Any] | None = None,
        progress_bar: bool = True,
    ) -> BaseAlgorithm:
        """Trains a model."""
        if alg_params is None:
            alg_params = dict()
        if learn_params is None:
            learn_params = dict()
        if isinstance(policy, Callable):
            p = default_policy_generator(env)
        else:
            p = policy
        alg = algorithm(policy=p, env=env, learning_rate=learning_rate, **alg_params)
        alg = alg.learn(steps, progress_bar=progress_bar, **learn_params)
        return alg

    def test(
        env: NegoMultiEnv,
        model: BaseAlgorithm | BaseModel,
        n_steps: int = 10_000,
    ) -> float:
        """Tests a model."""
        policy = partial(sb3policy, model=model)
        obs, _ = env.reset()
        reward = 0.0
        for _ in range(n_steps):
            obs, r, terminated, truncated, _ = env.step(policy(obs))
            reward += r  # type: ignore
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()
        return reward

    def make_negotiation(
        scenario: Scenario,
        models: list[BaseAlgorithm | BaseModel],
        env: NegoMultiEnv | None = None,
        background_negotiators: list[Negotiator] | NegotiatorGenerator | None = None,
        assigner: Assigner | None = None,
        mechanism_type: type[Mechanism] = SAOMechanism,
        **kwargs,
    ) -> Mechanism:
        """Deploys a model and tests it in actual negotiations."""
        if env is None and background_negotiators is None:
            raise ValueError(
                "You must either pass an environment or a list of background_negotiators"
            )
        if assigner is None:
            assigner = (
                PositionBasedNegotiatorAssigner() if env is None else env._assigner
            )
        n_learners = len(models)
        learners = tuple(
            SAORLNegotiator(
                policy=partial(sb3policy, model=model),
                name=f"learner{i}",
                id=f"learner{i}",
            )
            for i, model in enumerate(models)
        )
        if isinstance(background_negotiators, Iterable):
            others = background_negotiators
        else:
            if background_negotiators is None and env is not None:
                background_negotiators = env._partner_generator
            assert (
                background_negotiators is not None
            ), "You must pass either an environment, a list of background_negotiators or a NegotiatorGenerator as background_negotiators"
            others = [
                background_negotiators(i)
                for i in range(len(scenario.ufuns) - n_learners)
            ]
        if isinstance(background_negotiators, NegotiatorGenerator):
            for i, neg in enumerate(others):
                neg.name = f"background{i}"
                neg.id = neg.name
        m = mechanism_type(outcome_space=scenario.outcome_space, **kwargs)
        m = assigner(scenario, m, others, learners)
        return m
