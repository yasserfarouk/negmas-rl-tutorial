from random import randint, seed
from random import random as rand

import numpy as np
from attr import define
from negmas import (
    BoulwareTBNegotiator,
    LambdaFun,
)
from negmas.genius.gnegotiators import Atlas3
from negmas.inout import Scenario
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearUtilityAggregationFunction as LU
from negmas.preferences.value_fun import AffineFun, TableFun

from negmas_rl.cli.experiments.common import (
    ALWAYS_STARTS,
    IGNORE_NEG_EXCEPTIONS,
    N_STEPS,
    TIME_LIMIT,
    VERBOSITY,
    make_conditions,
)
from negmas_rl.cli.experiments.scml_common import (
    CARBON,
    N_LEVELS,
    N_OUTCOMES,
    PRICE,
    QUANTITY,
)
from negmas_rl.common import DEFAULT_UTIL_LEVELS
from negmas_rl.experiment import BaselineCondition, Condition, ContextInfo, Experiment
from negmas_rl.helpers.negotiators import (
    AgentGG,
    AgentK,
    Atlas3,
    AverageTitForTat,
    CUHKAgent,
    HardHeaded,
)

__all__ = ["SCMLScenarioGenerator", "SCMLExperiment"]

TRAIN_NEGOTIATORS = (
    BoulwareTBNegotiator,
    AgentGG,
    AgentK,
    Atlas3,
    AverageTitForTat,
    CUHKAgent,
    HardHeaded,
)
TEST_NEGOTIATORS = TRAIN_NEGOTIATORS


CONDITIONS, BASELINE_CONDITIONS = make_conditions(
    n_levels=N_LEVELS,
    n_outcomes=N_OUTCOMES,
    n_training=800_000,
    sac=True,
    ppo=True,
    baselines=True,
    raw_outcome_methods=True,
    n_steps=100,
    n_utility_levels=DEFAULT_UTIL_LEVELS,
    train_negotiators=TRAIN_NEGOTIATORS,
    test_negotiators=TEST_NEGOTIATORS,
)


def SCMLScenarioGenerator(indx: int | None = None) -> Scenario:
    """Generates scenarios for training and testing."""
    if indx:
        seed(indx)
        np.random.seed(indx)

    # 0. Helper function to generate new carbon values and weights
    def mk_carbon_weights():
        carbon = [rand(), rand(), rand()]
        m = max(carbon)
        carbon = [_ / m for _ in carbon]
        weights = [rand(), rand(), rand()]
        s = sum(weights)
        weights = [_ / s for _ in weights]
        return carbon, weights

    # 1. Define the outcome space
    os = make_os(
        [
            make_issue(PRICE, "price"),
            make_issue(QUANTITY, "quantity"),
            make_issue(CARBON, "carbon emission"),
        ],
        name="Trade",
    )
    # 2. Define the sell-side utility function
    mxp = os.issues[0].max_value - 1
    mxq = os.issues[1].max_value - 1
    carbon, weights = mk_carbon_weights()

    seller_ufun = LU(
        (
            AffineFun(1 / mxp, -1 / mxp),
            AffineFun(1 / mxq, -1 / mxq),
            TableFun(dict(zip(os.issues[2].values, carbon))),
        ),
        weights=weights,
        outcome_space=os,
        reserved_value=rand() * 0.2,
        name="seller",
    )

    # 3. Define  the buy-side utility function
    sep = randint(2, mxq - 1)
    carbon, weights = mk_carbon_weights()

    buyer_ufun = LU(
        (
            AffineFun(-1 / mxp, (mxp + 1) / mxp),
            LambdaFun(
                lambda x: (
                    x / (sep - 1) - 1 / (sep - 1)
                    if x <= sep
                    else (
                        (-1 / (mxq + 2 - sep)) * (x - sep)
                        + (mxq + 3 - sep) / (mxq + 2 - sep)
                    )
                )
            ),
            TableFun(dict(zip(os.issues[2].values, carbon))),
        ),
        weights=weights,
        outcome_space=os,
        reserved_value=rand() * 0.1,
        name="buyer",
    )

    # 4. Return the constructed scenario
    return Scenario(os, (seller_ufun, buyer_ufun))  # type: ignore


@define
class SCMLExperiment(Experiment):
    name: str = "SCML"
    conditions: tuple[Condition, ...]
    training: ContextInfo
    testing: ContextInfo
    conditions: tuple[Condition, ...] = tuple(CONDITIONS.values())

    baseline_conditions: tuple[BaselineCondition, ...] = tuple(
        BASELINE_CONDITIONS.values()
    )

    training: ContextInfo = ContextInfo(
        n_neg_steps=N_STEPS,
        time_limit=TIME_LIMIT,
        partners=TRAIN_NEGOTIATORS,
        scenario_generator=SCMLScenarioGenerator,
    )
    testing: ContextInfo = ContextInfo(
        n_neg_steps=N_STEPS,
        time_limit=TIME_LIMIT,
        partners=TEST_NEGOTIATORS,
        scenario_generator=SCMLScenarioGenerator,
        mechanism_params=dict(ignore_negotiator_exceptions=IGNORE_NEG_EXCEPTIONS),
    )
    always_starts = ALWAYS_STARTS
    verbosity = VERBOSITY
