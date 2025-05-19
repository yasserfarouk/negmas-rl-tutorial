import itertools
import random
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from attr import define
from negmas.common import CartesianOutcomeSpace
from negmas.inout import Scenario
from negmas.preferences import UtilityFunction
from negmas.preferences.generators import ParetoGenerator, generate_multi_issue_ufuns
from negmas.preferences.ops import nash_points, pareto_frontier

DEFAULT_N_STEPS = 100
DEFAULT_VALUES_PER_OUTCOME = 10
DEFAULT_N_ISSUES = 3


ReservedRanges = tuple[tuple[float, ...], ...]
# ScenarioGenerator = Callable[[int], Scenario]

__all__ = ["ScenarioGenerator", "ScenarioRepeater", "ScenarioSampler", "ScenarioCycler"]


@runtime_checkable
class ScenarioGenerator(Protocol):
    def __call__(self, indx: int | None = None) -> Scenario: ...


def product(generator):
    """
    Calculates the product of all elements in a generator.

    Args:
        generator: A generator of numbers.

    Returns
    -------
        The product of all elements in the generator.
    """
    total = 1
    for num in generator:
        total *= num
    return total


def find_n_integers_for_number(x, n: int = 3, fraction=0.1) -> tuple[int, ...]:
    """Helper function to find n integers for a given number within some allowed error."""
    # Check for perfect cubes
    mx = x * (1 + fraction)
    mn = x * (1 - fraction)
    n_root = int(x ** (1 / n))
    if mx >= n_root**n >= mn:
        return (
            n_root,
            n_root + random.randint(-1, 1),
            n_root,
        )

    # Factor x into primes
    prime_factors = []
    divisor = 2
    while x > 1:
        while x % divisor == 0:
            prime_factors.append(divisor)
            x //= divisor
        divisor += 1

    # Try to group prime factors into N groups
    for combination in itertools.combinations(prime_factors, n):
        if mx >= product(combination) >= mn:
            return combination

    # Try combining factors to create N integers
    if len(prime_factors) > n:
        for i in range(1, len(prime_factors) - n + 1):
            prod = product(prime_factors[i + 1 :])
            if mx >= prime_factors[0] * prime_factors[i] * prod >= mn:
                return (prime_factors[0], prime_factors[i], prod)

    raise ValueError(
        f"Failed to generate {n} numbers that multiple to around {x} within {fraction:4.2%}"
    )


def sample_reserved_values(
    ufuns: tuple[UtilityFunction, ...],
    pareto: tuple[tuple[float, ...], ...] | None = None,
    reserved_ranges: ReservedRanges | None = None,
    eps: float = 1e-3,
) -> tuple[float, ...]:
    """
    Samples reserved values that are guaranteed to allow some rational outcomes for the given ufuns and sets the reserved values.

    Args:
        ufuns: tuple of utility functions to sample reserved values for
        pareto: The pareto frontier. If not given, it will be calculated
        reserved_ranges: the range to sample reserved values from. Notice that the upper limit of this range will be updated
                         to ensure some rational outcoms
        eps: A small number indicating the absolute guaranteed margin of the sampled reserved value from the Nash point.

    """
    n_funs = len(ufuns)
    if pareto is None:
        pareto = pareto_frontier(ufuns)[0]
    assert pareto is not None, "Cannot find the pareto frontier."
    nash = nash_points(ufuns, frontier=pareto, ranges=[(0, 1) for _ in range(n_funs)])
    if not nash:
        raise ValueError(
            "Cannot find the Nash point so we cannot find the appropriate reserved ranges"
        )
    nash_utils = nash[0][0]
    if not reserved_ranges:
        reserved_ranges = tuple((0, 0.999) for _ in range(n_funs))
    reserved_ranges = tuple(
        tuple(min(r[_], n) for _ in range(n_funs))
        for n, r in zip(nash_utils, reserved_ranges)
    )
    reserved = tuple(
        r[0] + (r[1] - eps - r[0]) * random.random() for r in reserved_ranges
    )
    for u, r in zip(ufuns, reserved):
        u.reserved_value = float(r)
    return reserved


@define
class RandomScenarioGenerator(ScenarioGenerator):
    n_issues: int = DEFAULT_N_ISSUES
    n_ufuns: int = 2
    n_values: int | tuple[int, int] = 0
    sizes: None = None
    pareto_generators: tuple[ParetoGenerator | str, ...] = ("piecewise_linear",)
    generator_params: tuple[dict[str, Any], ...] | None = None
    reserved_values: list[float] | tuple[float, float] | float = 0.0
    rational_fractions: list[float] | None = None
    reservation_selector: Callable[[float, float], float] = max
    issue_names: tuple[str, ...] | list[str] | None = None
    os_name: str | None = None
    ufun_names: tuple[str, ...] | None = None
    numeric: bool = False
    numeric_prob: float = -1
    linear: bool = True
    guarantee_rational: bool = False

    def __call__(self, indx: int | None = None) -> Scenario:
        """Generates a scenario."""
        assert self.n_ufuns > 0
        if self.sizes is None and not self.n_values:
            self.n_values = DEFAULT_VALUES_PER_OUTCOME
        ufuns = generate_multi_issue_ufuns(
            n_issues=self.n_issues,
            n_ufuns=self.n_ufuns,
            n_values=self.n_values,
            sizes=self.sizes,
            pareto_generators=self.pareto_generators,
            generator_params=self.generator_params,
            reserved_values=self.reserved_values,
            rational_fractions=self.rational_fractions,
            reservation_selector=self.reservation_selector,
            issue_names=self.issue_names,
            os_name=self.os_name,
            ufun_names=self.ufun_names,
            numeric=self.numeric,
            linear=self.linear,
            guarantee_rational=self.guarantee_rational,
            numeric_prob=self.numeric_prob,
        )
        os = ufuns[0].outcome_space
        assert os is not None and isinstance(os, CartesianOutcomeSpace)
        return Scenario(outcome_space=os, ufuns=ufuns)


class ScenarioCycler(ScenarioGenerator):
    def __init__(self, scenarios: list[Scenario] | Scenario):
        if isinstance(scenarios, Scenario):
            scenarios = [scenarios]
        self._scenarios = list(scenarios)
        self._next = 0

    def __call__(self, indx: int | None = None) -> Scenario:
        s = self._scenarios[self._next % len(self._scenarios)]
        self._next = (self._next + 1) % len(self._scenarios)
        return s


class ScenarioSampler(ScenarioGenerator):
    def __init__(self, scenarios: list[Scenario] | Scenario):
        if isinstance(scenarios, Scenario):
            scenarios = [scenarios]
        self._scenarios = list(scenarios)

    def __call__(self, indx: int | None = None) -> Scenario:
        return random.choice(self._scenarios)


class ScenarioRepeater(ScenarioGenerator):
    def __init__(self, scenario: Scenario):
        self._scenario = scenario

    def __call__(self, indx: int | None = None) -> Scenario:
        return self._scenario
