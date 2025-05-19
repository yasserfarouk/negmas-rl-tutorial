"""Callables that can generate negotiators."""

import random
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from negmas.helpers import get_class
from negmas.negotiators import Negotiator

__all__ = [
    "NegotiatorGenerator",
    "NegotiatorCycler",
    "NegotiatorSampler",
    "NegotiatorRepeater",
]

NegType = str | type[Negotiator]


def type_(x: str | type):
    if isinstance(x, str):
        return get_class(x)
    return x


@runtime_checkable
class NegotiatorGenerator(Protocol):
    """A callable that takes nothing and returns a negotiator"""

    def __call__(self, indx: int | None = None) -> Negotiator: ...

    def set_negotiators(
        self,
        negotiators: list[NegType] | tuple[NegType, ...] | NegType,
        params: list[dict[str, Any]]
        | tuple[dict[str, Any], ...]
        | tuple[dict[str, Any], ...]
        | dict[str, Any]
        | None = None,
    ) -> None: ...


class NegotiatorCycler(NegotiatorGenerator):
    """A `NegotiatorGenerator` that cycles through a list of negotiator/param pairs"""

    def set_negotiators(
        self,
        negotiators: list[NegType] | tuple[NegType, ...] | NegType,
        params: list[dict[str, Any]]
        | tuple[dict[str, Any], ...]
        | tuple[dict[str, Any], ...]
        | dict[str, Any]
        | None = None,
    ) -> None:
        if isinstance(negotiators, type) or isinstance(negotiators, str):
            negotiators = [type_(negotiators)]
        self._negotiators = [type_(_) for _ in negotiators]
        if params is None:
            params = [dict() for _ in self._negotiators]
        elif isinstance(params, dict):
            params = [params for _ in self._negotiators]
        elif len(params) > len(self._negotiators):
            params = params[: len(negotiators)]
        elif len(params) < len(self._negotiators):
            raise ValueError(
                f"Got {len(self._negotiators)} negotiator types and {len(params)} parameter dicts"
            )
        self._params = params
        self._next = 0

    def __init__(
        self,
        negotiators: list[NegType] | tuple[NegType, ...] | NegType | None = None,
        params: list[dict[str, Any]]
        | tuple[dict[str, Any], ...]
        | tuple[dict[str, Any], ...]
        | dict[str, Any]
        | None = None,
    ):
        if negotiators:
            self.set_negotiators(negotiators, params)

    def __call__(self, indx: int | None = None) -> Negotiator:
        i = self._next % len(self._negotiators)
        self._next = (self._next + 1) % len(self._negotiators)
        return self._negotiators[i](**self._params[i])


class NegotiatorSampler(NegotiatorGenerator):
    """A `NegotiatorGenerator` that samples randomly from a list of negotiator/param pairs"""

    def set_negotiators(
        self,
        negotiators: list[NegType] | tuple[NegType, ...] | NegType,
        params: list[dict[str, Any]]
        | tuple[dict[str, Any], ...]
        | tuple[dict[str, Any], ...]
        | dict[str, Any]
        | None = None,
    ) -> None:
        if isinstance(negotiators, type) or isinstance(negotiators, str):
            negotiators = [type_(negotiators)]
        self._negotiators: list[Negotiator] = [type_(_) for _ in negotiators]  # type: ignore
        if params is None:
            params = [dict() for _ in self._negotiators]
        elif isinstance(params, dict):
            params = [params for _ in self._negotiators]
        elif len(params) > len(self._negotiators):
            params = params[: len(negotiators)]
        elif len(params) < len(self._negotiators):
            raise ValueError(
                f"Got {len(self._negotiators)} negotiator types and {len(params)} parameter dicts"
            )
        self._params = params

    def __init__(
        self,
        negotiators: list[NegType] | NegType | None = None,
        params: list[dict[str, Any]]
        | tuple[dict[str, Any], ...]
        | dict[str, Any]
        | None = None,
    ):
        if negotiators:
            self.set_negotiators(negotiators, params)

    def __call__(self, indx: int | None = None) -> Negotiator:
        if indx is not None:
            random.seed(indx)
        i = random.randint(0, len(self._negotiators) - 1)
        return self._negotiators[i](**self._params[i])  # type: ignore


class NegotiatorRepeater(NegotiatorGenerator):
    """A `NegotiatorGenerator` that repeats creating the same negotiator/param pair"""

    def set_negotiators(
        self,
        negotiators: list[NegType] | tuple[NegType, ...] | NegType,
        params: list[dict[str, Any]]
        | tuple[dict[str, Any], ...]
        | tuple[dict[str, Any], ...]
        | dict[str, Any]
        | None = None,
    ) -> None:
        negotiator = type_(
            negotiators[0] if isinstance(negotiators, Sequence) else negotiators
        )
        assert issubclass(negotiator, Negotiator)
        self._negotiator = negotiator
        if params is None:
            params = dict()
        if not isinstance(params, dict):
            params = params[0]
        self._params = params

    def __init__(
        self,
        negotiator: NegType | None = None,
        params: dict[str, Any] | None = None,
    ):
        if negotiator:
            self.set_negotiators([negotiator], params)

    def __call__(self, indx: int | None = None) -> Negotiator:
        return self._negotiator(**self._params)  # type: ignore
