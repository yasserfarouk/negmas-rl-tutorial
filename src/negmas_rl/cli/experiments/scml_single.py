from attr import define
from negmas import (
    BoulwareTBNegotiator,
    LambdaFun,
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
from negmas_rl.cli.experiments.scml_common import (
    CARBON,
    N_LEVELS,
    N_OUTCOMES,
    PRICE,
    QUANTITY,
)
from negmas_rl.common import DEFAULT_UTIL_LEVELS
from negmas_rl.experiment import BaselineCondition, Condition, ContextInfo, Experiment

__all__ = ["SCMLSingleScenarioGenerator", "SCMLSingleScenarioExperiment"]

TRAIN_NEGOTIATORS = (BoulwareTBNegotiator,)
TEST_NEGOTIATORS = TRAIN_NEGOTIATORS


CONDITIONS, BASELINE_CONDITIONS = make_conditions(
    n_levels=N_LEVELS,
    n_outcomes=N_OUTCOMES,
    n_training=800_000,
    sac=True,
    ppo=True,
    baselines=True,
    raw_outcome_methods=True,
    n_steps=200,
    n_utility_levels=DEFAULT_UTIL_LEVELS,
    train_negotiators=TRAIN_NEGOTIATORS,
    test_negotiators=TEST_NEGOTIATORS,
)


class SCMLSingleScenarioGenerator:
    """Generates scenarios for training and testing."""

    def __call__(self, indx=None) -> Scenario:
        os = make_os(
            [
                make_issue(PRICE, "price"),
                make_issue(QUANTITY, "quantity"),
                make_issue(CARBON, "carbon emission"),
            ],
            name="Trade",
        )
        seller_ufun = LU(
            (
                AffineFun(1 / (PRICE[-1] - 1), -1 / (PRICE[-1] - 1)),
                AffineFun(1 / (QUANTITY[-1] - 1), -1 / (QUANTITY[-1] - 1)),
                TableFun(dict(high=1.0, med=0.5, low=0.0)),
            ),
            weights=(0.7, 0.2, 0.1),
            outcome_space=os,
            reserved_value=0.0,
        )
        mn, mx = seller_ufun.minmax()
        assert abs(mn) < 0.00001 and abs(mx - 1) < 0.00001

        # a, b => 0, 1       (x-a)/(b-a) = y = (1/(b-1)) x - a/(b-a)
        # a, b => 1, 0       1-(x-a)/(b-a) = y = -1/(b-a) x + (b) / (b-a)
        PMAX = PRICE[-1] - 1
        QMID = 4
        QMAX = QUANTITY[-1] - 1
        buyer_ufun = LU(
            (
                AffineFun(-1 / PMAX, PRICE[-1] / PMAX),
                LambdaFun(
                    lambda x: x / PMAX - 1 / PMAX
                    if x <= (QMID + 1)
                    else (-1 / (QMAX - QMID)) * (x - 4)
                    + (1 + QMAX - QMID) / (QMAX - QMID)
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
        mn, mx = buyer_ufun.minmax()
        assert abs(mn) < 0.00001 and abs(mx - 1) < 0.00001
        return Scenario(os, (seller_ufun, buyer_ufun))  # type: ignore


@define
class SCMLSingleScenarioExperiment(Experiment):
    name: str = "SCML-SingleScenario"
    conditions: tuple[Condition, ...] = tuple(CONDITIONS.values())

    baseline_conditions: tuple[BaselineCondition, ...] = tuple(
        BASELINE_CONDITIONS.values()
    )
    training: ContextInfo = ContextInfo(
        n_neg_steps=N_STEPS,
        time_limit=TIME_LIMIT,
        partners=TRAIN_NEGOTIATORS,
        scenario_generator=SCMLSingleScenarioGenerator(),
    )
    testing: ContextInfo = ContextInfo(
        n_neg_steps=N_STEPS,
        time_limit=TIME_LIMIT,
        partners=TEST_NEGOTIATORS,
        scenario_generator=SCMLSingleScenarioGenerator(),
        mechanism_params=dict(ignore_negotiator_exceptions=IGNORE_NEG_EXCEPTIONS),
    )
    always_starts = ALWAYS_STARTS
    verbosity = VERBOSITY
