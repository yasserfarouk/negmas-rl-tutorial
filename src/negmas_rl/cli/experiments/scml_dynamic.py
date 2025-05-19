from random import randint
from random import random as rand

from attr import define
from negmas import (
    BoulwareTBNegotiator,
    ConcederTBNegotiator,
    LambdaFun,
    LinearTBNegotiator,
    SingleIssueFun,
)
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
from negmas_rl.cli.experiments.scml_common import CARBON, N_LEVELS, N_OUTCOMES
from negmas_rl.common import DEFAULT_UTIL_LEVELS
from negmas_rl.experiment import BaselineCondition, Condition, ContextInfo, Experiment
from negmas_rl.helpers.negotiators import (
    AgentGG,
    AgentK,
    Atlas3,
    CUHKAgent,
    HardHeaded,
)

__all__ = ["SCMLDynamicScenarioGenerator", "SCMLDynamicExperiment"]


PRICE = (5, 7)
QUANTITY = (10, 20)
TRAIN_NEGOTIATORS = (Atlas3, CUHKAgent, BoulwareTBNegotiator, ConcederTBNegotiator)
TEST_NEGOTIATORS = (
    AgentGG,
    HardHeaded,
    AgentK,
    LinearTBNegotiator,
    # NiceTitForTat,
)


CONDITIONS, BASELINE_CONDITIONS = make_conditions(
    n_levels=N_LEVELS,
    n_outcomes=N_OUTCOMES,
    n_training=800_000,
    sac=True,
    ppo=True,
    baselines=True,
    raw_outcome_methods=False,
    n_steps=500,
    n_utility_levels=DEFAULT_UTIL_LEVELS,
    train_negotiators=TRAIN_NEGOTIATORS,
    test_negotiators=TEST_NEGOTIATORS,
)


def SCMLDynamicScenarioGenerator(
    indx: int | None = None, variable_issue_number: bool | None = None
) -> Scenario:
    """Generates scenarios for training and testing."""

    # 0. Helper function to generate new carbon values and weights
    def mk_carbon_weights(has_carbon):
        carbon = [rand(), rand()]
        if has_carbon:
            carbon.append(rand())
        m = max(carbon)
        carbon = [_ / m for _ in carbon]
        weights = [rand(), rand()]
        if has_carbon:
            weights.append(rand())
        s = sum(weights)
        weights = [_ / s for _ in weights]
        return carbon, weights

    # 1. Define the outcome space
    issues = [
        make_issue((1, randint(*PRICE)), "price"),
        make_issue((1, randint(*QUANTITY)), "quantity"),
    ]
    has_carbon = (
        randint(0, 1) if variable_issue_number is None else int(variable_issue_number)
    )
    if has_carbon:
        issues.append(
            make_issue(CARBON, "carbon emission"),
        )
    os = make_os(name="Trade", issues=issues)
    # 2. Define the sell-side utility function
    mxp = os.issues[0].max_value - 1
    mxq = os.issues[1].max_value - 1
    carbon, weights = mk_carbon_weights(has_carbon)
    vals: list[SingleIssueFun] = [
        AffineFun(1 / mxp, -1 / mxp),
        AffineFun(1 / mxq, -1 / mxq),
    ]
    if has_carbon:
        vals.append(TableFun(dict(zip(os.issues[2].values, carbon))))

    seller_ufun = LU(
        tuple(vals),
        weights=weights,
        outcome_space=os,
        reserved_value=rand() * 0.2,
        name="seller",
    )

    # 3. Define  the buy-side utility function
    sep = randint(2, mxq - 1)
    carbon, weights = mk_carbon_weights(has_carbon)

    vals = [
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
    ]
    if has_carbon:
        vals.append(TableFun(dict(zip(os.issues[2].values, carbon))))

    buyer_ufun = LU(
        tuple(vals),
        weights=weights,
        outcome_space=os,
        reserved_value=rand() * 0.1,
        name="buyer",
    )

    # 4. Return the constructed scenario
    return Scenario(os, (seller_ufun, buyer_ufun))  # type: ignore


@define
class SCMLDynamicExperiment(Experiment):
    name: str = "SCMLDynamic"
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
        scenario_generator=SCMLDynamicScenarioGenerator,
    )
    testing: ContextInfo = ContextInfo(
        n_neg_steps=N_STEPS,
        time_limit=TIME_LIMIT,
        partners=TEST_NEGOTIATORS,
        scenario_generator=SCMLDynamicScenarioGenerator,
        mechanism_params=dict(ignore_negotiator_exceptions=IGNORE_NEG_EXCEPTIONS),
    )
    always_starts = ALWAYS_STARTS
    verbosity = VERBOSITY
