import random
from collections.abc import Callable
from typing import Any

from negmas.inout import Scenario
from negmas.mechanisms import Mechanism

__all__ = [
    "MechanismGenerator",
    "MechanismCycler",
    "MechanismSampler",
    "MechanismRepeater",
]
MechanismGenerator = Callable[[Scenario], Mechanism]


class MechanismCycler:
    def __init__(
        self,
        mechanisms: list[type[Mechanism]] | type[Mechanism],
        params: list[dict[str, Any]]
        | tuple[dict[str, Any], ...]
        | dict[str, Any]
        | None = None,
    ):
        if isinstance(mechanisms, type):
            mechanisms = [mechanisms]
        self._mechanisms = list(mechanisms)
        if params is None:
            params = [dict() for _ in self._mechanisms]
        elif isinstance(params, dict):
            params = [params for _ in self._mechanisms]
        elif len(params) > len(self._mechanisms):
            params = params[: len(mechanisms)]
        elif len(params) < len(self._mechanisms):
            raise ValueError(
                f"Got {len(self._mechanisms)} mechanism types and {len(params)} parameter dicts"
            )
        self._params = params
        self._next = 0

    def __call__(self, scenario: Scenario) -> Mechanism:
        i = self._next % len(self._mechanisms)
        self._next = (self._next + 1) % len(self._mechanisms)
        opt = self._params[i] | dict(outcome_space=scenario.outcome_space)
        return self._mechanisms[i](**opt)


class MechanismSampler:
    def __init__(
        self,
        mechanisms: list[type[Mechanism]] | type[Mechanism],
        params: list[dict[str, Any]]
        | tuple[dict[str, Any], ...]
        | dict[str, Any]
        | None = None,
    ):
        if isinstance(mechanisms, type):
            mechanisms = [mechanisms]
        self._mechanisms = list(mechanisms)
        if params is None:
            params = [dict() for _ in self._mechanisms]
        elif isinstance(params, dict):
            params = [params for _ in self._mechanisms]
        elif len(params) > len(self._mechanisms):
            params = params[: len(mechanisms)]
        elif len(params) < len(self._mechanisms):
            raise ValueError(
                f"Got {len(self._mechanisms)} mechanism types and {len(params)} parameter dicts"
            )
        self._params = params

    def __call__(self, scenario: Scenario) -> Mechanism:
        i = random.randint(0, len(self._mechanisms) - 1)
        opt = self._params[i] | dict(outcome_space=scenario.outcome_space)
        return self._mechanisms[i](**opt)


class MechanismRepeater:
    """Repeats the creation of the same mechanism type with the same parameters"""

    def __init__(
        self,
        mechanism: type[Mechanism],
        params: dict[str, Any] | None = None,
    ):
        self._mechanism = mechanism
        if params is None:
            params = dict()
        self._params = params

    def __call__(self, scenario: Scenario) -> Mechanism:
        opt = self._params | dict(outcome_space=scenario.outcome_space)
        return self._mechanism(**opt)
