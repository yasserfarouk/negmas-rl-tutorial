from pathlib import Path
from random import randint

from attr import define
from negmas import (
    BoulwareTBNegotiator,
    ConcederTBNegotiator,
    LinearTBNegotiator,
)
from negmas.inout import Scenario

from negmas_rl.cli.experiments.common import (
    ALWAYS_STARTS,
    IGNORE_NEG_EXCEPTIONS,
    N_STEPS,
    TIME_LIMIT,
    VERBOSITY,
    make_conditions,
)
from negmas_rl.common import DEFAULT_N_LEVELS, DEFAULT_UTIL_LEVELS
from negmas_rl.experiment import BaselineCondition, Condition, ContextInfo, Experiment
from negmas_rl.helpers.negotiators import (
    AgentGG,
    AgentK,
    Atlas3,
    CUHKAgent,
    HardHeaded,
)

__all__ = [
    "ANAC2024Experiment",
    "ANAC2024TrainingSG",
    "ANAC2024TestingSG",
    "load_scenarios",
]

TRAIN_NEGOTIATORS = (Atlas3, CUHKAgent, BoulwareTBNegotiator, ConcederTBNegotiator)
TEST_NEGOTIATORS = (
    AgentGG,
    HardHeaded,
    AgentK,
    LinearTBNegotiator,
)


CONDITIONS, BASELINE_CONDITIONS = make_conditions(
    n_levels=DEFAULT_N_LEVELS,
    n_outcomes=1500,
    n_training=1_000_000,
    sac=True,
    ppo=True,
    baselines=True,
    raw_outcome_methods=False,
    n_steps=500,
    n_utility_levels=DEFAULT_UTIL_LEVELS,
    train_negotiators=TRAIN_NEGOTIATORS,
    test_negotiators=TEST_NEGOTIATORS,
)


def load_scenarios() -> list[Scenario]:
    path = Path(__file__).parent.parent.parent.parent.parent / "scenarios" / "y2024"
    scenarios = []
    for f in path.glob("*"):
        if not f.is_dir():
            continue
        s = Scenario.load(f)
        if s is None:
            continue
        for u in s.ufuns:
            u.reserved_value = 0.0
        scenarios.append(s)
    return scenarios


class ANAC2024TrainingSG:
    def __init__(self) -> None:
        self.scenarios = load_scenarios()
        self.scenarios = [_ for _ in self.scenarios[0:50]] + [
            _ for _ in self.scenarios[-50:]
        ]

    def __call__(
        self, indx: int | None = None, variable_issue_number: bool | None = None
    ) -> Scenario:
        """Generates scenarios for training and testing."""
        if indx is not None:
            indx = indx % len(self.scenarios)
        else:
            indx = randint(0, len(self.scenarios) - 1)
        return self.scenarios[indx]


class ANAC2024TestingSG(ANAC2024TrainingSG):
    def __init__(self) -> None:
        self.scenarios = load_scenarios()
        self.scenarios = [_ for _ in self.scenarios[50:75]] + [
            _ for _ in self.scenarios[-75:-50]
        ]


@define
class ANAC2024Experiment(Experiment):
    name: str = "ANAC2024"
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
        scenario_generator=ANAC2024TrainingSG(),
    )
    testing: ContextInfo = ContextInfo(
        n_neg_steps=N_STEPS,
        time_limit=TIME_LIMIT,
        partners=TEST_NEGOTIATORS,
        scenario_generator=ANAC2024TestingSG(),
        mechanism_params=dict(ignore_negotiator_exceptions=IGNORE_NEG_EXCEPTIONS),
    )
    always_starts = ALWAYS_STARTS
    verbosity = VERBOSITY
