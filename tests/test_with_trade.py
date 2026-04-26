# This cell contains only imports.
from functools import partial

from negmas import (
    LambdaFun,
    LinearTBNegotiator,
    SAOMechanism,
)
from negmas.inout import Scenario
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearUtilityAggregationFunction as LU
from negmas.preferences.value_fun import AffineFun, TableFun
from rich import print
from stable_baselines3.sac import SAC

from negmas_rl.env import NegoEnv
from negmas_rl.generators.assigner import PositionBasedNegotiatorAssigner
from negmas_rl.generators.mechanism import MechanismRepeater
from negmas_rl.generators.negotiator import NegotiatorRepeater

NTRAIN, NTEST = 3000, 100


def TradeScenarioGenerator(indx: int | None = None) -> Scenario:
    """Defines the scenarios used for training/testing"""
    # 1. Define the outcome space
    os = make_os(
        [
            make_issue((1, 5), "price"),
            make_issue((1, 20), "quantity"),
            make_issue(["high", "med", "low"], "carbon emission"),
        ],
        name="Trade",
    )
    # 2. Define the sell-side utility function
    seller_ufun = LU(
        (
            AffineFun(1 / 4, -1 / 4),
            AffineFun(1 / 19, -1 / 19),
            TableFun(dict(high=1.0, med=0.5, low=0.0)),
        ),
        weights=(0.7, 0.2, 0.1),
        outcome_space=os,
        reserved_value=0.0,
    )

    # 3. Define  the buy-side utility function
    buyer_ufun = LU(
        (
            AffineFun(-1 / 4, 5 / 4),
            LambdaFun(
                lambda x: x / 4 - 1 / 4 if x <= 5 else (-1 / 15) * (x - 4) + 16 / 15
            ),
            TableFun(
                dict(
                    high=0.0,
                    med=0.8,
                    low=1.0,
                )
            ),
        ),
        weights=(0.4, 0.4, 0.2),
        outcome_space=os,
        reserved_value=0.0,
    )

    # 4. Return the constructed scenario
    return Scenario(os, (seller_ufun, buyer_ufun))  # type: ignore


def make_env():
    """Makes a trading environment."""
    return NegoEnv(
        scenario_generator=TradeScenarioGenerator,
        partner_generator=NegotiatorRepeater(LinearTBNegotiator),
        assigner=PositionBasedNegotiatorAssigner(always_starts=True),
        mechanism_generator=MechanismRepeater(SAOMechanism, dict(n_steps=100)),
    )


def make_trainer(policy="MultiInputPolicy"):
    """Trains a model."""
    env = make_env()
    trainer = SAC(policy=policy, env=env)
    return trainer


def train(trainer, steps=NTRAIN):
    """Trains a model."""
    return trainer.learn(steps, progress_bar=True)


def make_policy(obs, model):
    """Wrapps a stable-baselines3 model to create actions"""
    return model.predict(obs)[0]


def do_test(model):
    """Tests a model."""

    # 1. create the environment
    env = make_env()
    # 2. Wrap the model
    policy = partial(make_policy, model=model)
    # 3. Reset the environment
    obs, _ = env.reset()
    reward = 0
    # 4. Loop collecting rewards.
    for _ in range(NTEST):
        action = policy(obs)
        print(
            action,
            env._action_decoders[env.possible_agents[0]].decode(
                env._placeholders[0].nmi, action
            ),
        )
        obs, r, terminated, truncated, _ = env.step(action)
        env.render_frame()
        if terminated:
            print(env._mechanism.state)
        reward += r
    # 5. Close the environment
    env.close()
    return reward / env.n_negotiations


def test_train_test():
    untrained = make_trainer()
    trained = train(make_trainer())
    untrained_score = do_test(untrained)
    trained_score = do_test(trained)
    assert trained_score > untrained_score
    print(f"Untrained model got {untrained_score:3.2}")
    print(f"Trained model got {trained_score:3.2}")
    # assert False
