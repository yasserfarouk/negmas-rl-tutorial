import random
import sys
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from negmas.inout import Scenario
from negmas.mechanisms import Mechanism
from negmas.negotiators import Negotiator

__all__ = ["Assigner", "PositionBasedNegotiatorAssigner"]


@runtime_checkable
class Assigner(Protocol):
    """Defines method for assigning negotiators to a negotiation"""

    def __call__(
        self,
        scenario: Scenario,
        mechanism: Mechanism,
        partners: list[Negotiator],
        placeholder: Negotiator
        | list[Negotiator]
        | tuple[Negotiator, ...]
        | Callable[[], Negotiator],
    ) -> Mechanism:
        """
        Assigns the given partners and placeholders to the mechanism for the given scenario.

        Args:
            scenario: The scenario used for the negotiation.
            mechanism: The mechanism used for the negotiation
            partners: The background negotiators (i.e. non-learning)
            placeholder: The foreground negotiators (i.e. learners) or their types
        """
        ...


class PositionBasedNegotiatorAssigner(Assigner):
    def __init__(
        self,
        always_starts: bool = False,
        always_ends: bool = False,
        never_starts: bool = False,
        never_ends: bool = False,
        min_pos: int = 0,
        max_pos: int = sys.maxsize,
        roles: list[str | None] | str | None = None,
        partner_roles: list[str | None] | str | None = None,
        random_partners: bool = False,
        random_placeholders: bool = False,
        consecutive_placeholders: bool = True,
    ) -> None:
        self._random_partners = random_partners
        self._always_starts = always_starts
        self._always_ends = always_ends
        self._never_starts = never_starts
        self._never_ends = never_ends
        self._min_pos = min_pos
        self._max_pos = max_pos
        self._roles = roles
        self._random_placeholders = random_placeholders
        self._consecutive_placeholders = consecutive_placeholders
        self._partner_roles = partner_roles

    def __call__(
        self,
        scenario: Scenario,
        mechanism: Mechanism,
        partners: list[Negotiator],
        placeholder: Negotiator
        | list[Negotiator]
        | tuple[Negotiator, ...]
        | Callable[[], Negotiator],
    ) -> Mechanism:
        if isinstance(placeholder, Callable):
            placeholder = [
                placeholder() for _ in range(len(scenario.ufuns) - len(partners))
            ]
        elif isinstance(placeholder, Negotiator):
            placeholder = [placeholder]
        placeholders = list(placeholder)
        partners = list(partners)
        if self._random_partners:
            random.shuffle(partners)
        if self._random_placeholders:
            random.shuffle(placeholders)
        if self._partner_roles is None or isinstance(self._partner_roles, str):
            partner_roles = [None] * len(partners)
        else:
            partner_roles = self._partner_roles
        if self._roles is None or isinstance(self._roles, str):
            roles = [None] * len(placeholders)
        else:
            roles = self._roles
        if self._always_ends:
            partners = partners + placeholders
            partner_roles = partner_roles + roles
        elif self._always_starts:
            partners = placeholders + partners
            partner_roles = roles + partner_roles
        elif self._consecutive_placeholders:
            beg = max(int(self._never_starts), self._min_pos)
            end = min(len(partners) - int(self._never_starts) - 1, self._max_pos)
            i = random.randint(beg, end) if end > beg else beg
            partners = partners[:i] + placeholders + partners[i:]
            partner_roles = partner_roles[:i] + roles + partner_roles[i:]
        else:
            for p, r in zip(placeholders, roles, strict=True):
                beg = max(int(self._never_starts), self._min_pos)
                end = min(len(partners) - int(self._never_starts) - 1, self._max_pos)
                i = random.randint(beg, end) if end > beg else beg
                partners = partners[:i] + [p] + partners[i:]
                partner_roles = partner_roles[:i] + [r] + partner_roles[i:]
        for i, (p, u, r) in enumerate(
            zip(partners, scenario.ufuns, partner_roles, strict=True)
        ):
            p.id = p.name = f"n{i}"
            if mechanism.add(p, ufun=u, role=r):
                continue
            raise ValueError(f"Failed to add {p} to {mechanism} on role {r}")
        return mechanism
