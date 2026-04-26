"""
Defines ways to decode and encode actions.
"""

import functools
from abc import abstractmethod
from collections.abc import Iterable
from enum import IntEnum
from itertools import product
from math import ceil, floor
from random import choice
from typing import Any, Protocol

import numpy as np
from attr import define
from gymnasium import Space, spaces
from negmas import (
    SAONMI,
    BaseUtilityFunction,
    CartesianOutcomeSpace,
    CategoricalIssue,
    Issue,
    ResponseType,
    UtilityFunction,
)
from negmas.common import NegotiatorMechanismInterface, PreferencesChange
from negmas.inout import field
from negmas.negotiators import Negotiator
from negmas.outcomes import Outcome
from negmas.preferences import InverseUFun
from negmas.sao.mechanism import SAOResponse

from .common import (
    DEFAULT_N_ISSUE_LEVELS,
    DEFAULT_N_OUTCOMES,
    DEFAULT_UTIL_LEVELS,
    FLOAT_TYPE,
    INT_TYPE,
    MAX_CARDINALITY,
    from_float,
    to_float,
)
from .component import Component
from .utils import enumerate_by_utility, enumerate_in_order

__all__ = [
    "SamplingMethod",
    "ActionDecoder",
    "SAOActionDecoder",
    "UtilityDecoder",
    "CUtilityDecoder",
    "CUtilityDecoderBox",
    "DUtilityDecoder",
    "CUtilityDecoder1D",
    "DUtilityDecoder1D",
    "RelativeUtilityDecoder",
    "DRelativeUtilityDecoder",
    "CRelativeUtilityDecoderBox",
    "CRelativeUtilityDecoder1D",
    "DRelativeUtilityDecoder1D",
    "OutcomeDecoder",
    "COutcomeDecoder",
    "COutcomeDecoder1D",
    "DOutcomeDecoder",
    "IssueDecoder",
    "CIssueDecoder",
    "DIssueDecoder",
    "RLBoaDecoder",
    "CRLBoaDecoder",
    "SenguptaDecoder",
    "DSenguptaDecoder",
    "MiPNDecoder",
    "CMiPNDecoder",
    "VeNASDecoder",
    "CVeNASDecoder",
]


REJECT = 0
ACCEPT = 1
END = 2
NORESPONSE = 3

DECISION = 0
ACTION = 1
DECISION_MARGIN = 0.05

ACTION_MAP = {
    ACCEPT: ResponseType.ACCEPT_OFFER,
    REJECT: ResponseType.REJECT_OFFER,
    END: ResponseType.END_NEGOTIATION,
    NORESPONSE: ResponseType.NO_RESPONSE,
}
ACTION_INV_MAP = {
    ResponseType.ACCEPT_OFFER: ACCEPT,
    ResponseType.REJECT_OFFER: REJECT,
    ResponseType.END_NEGOTIATION: END,
    ResponseType.NO_RESPONSE: NORESPONSE,
}

CURRENT_CODE = -1000
NONE_CODE = -2000

Numeric = np.ndarray | int | float


class AcceptancePolicy(Protocol):
    def __call__(
        self,
        last_sent: Outcome | None,
        last_received: Outcome | None,
        next: Outcome | None,
        ufun: BaseUtilityFunction,
        nmi: NegotiatorMechanismInterface,
    ) -> bool: ...

    def acceptance_code(self, nmi: SAONMI) -> SAOResponse:
        return SAOResponse(ResponseType.REJECT_OFFER, nmi.state.current_offer)


@define(frozen=True)
class DefaultAcceptancePolicy:
    alpha_last_sent: float = 0
    alpha_last_received: float = 0
    alpha_next: float = 1

    def __call__(
        self,
        last_sent: Outcome | None,
        last_received: Outcome | None,
        next: Outcome | None,
        ufun: BaseUtilityFunction,
        nmi: NegotiatorMechanismInterface,
    ) -> bool:
        unext = float(ufun(next))
        return (
            self.alpha_next * unext
            + (self.alpha_last_received if last_received else 0)
            * float(ufun(last_received))
            + (self.alpha_last_sent if last_sent else 0) * float(ufun(last_sent))
            >= unext
        )

    def acceptance_code(self, nmi: SAONMI) -> SAOResponse:
        return SAOResponse(ResponseType.REJECT_OFFER, nmi.state.current_offer)


def best_for(
    rng: tuple[float, float],
    inverter: InverseUFun,
    target_ufuns: list[UtilityFunction],
    normalized=False,
    **kwargs,
) -> Outcome | None:
    """Finds an outcome in the given range that is best in target-ufuns in order"""
    outcomes = inverter.some(rng, normalized=normalized, **kwargs)
    if not target_ufuns:
        return None
    utils = [
        (tuple([-float(u(o)) for u in target_ufuns]), o)
        for o in outcomes
        if o is not None
    ]
    if not utils:
        return None
    utils = sorted(utils)
    return utils[0][-1]


class SamplingMethod(IntEnum):
    Worst = 0
    Random = 1
    Best = 2
    BestForPartner = 3


def safemin(x: Iterable | int | float | str):
    """Finds the minimum if a non-string iterable else returns the same value"""
    if isinstance(x, Iterable) and not isinstance(x, str):
        return min(x)
    return x


@define
class ActionDecoder(Component):
    """
    Manges actions of an agent in an RL environment.
    """

    allow_no_response: bool = False
    allow_ending: bool = False
    prepared: bool = field(init=False, default=False)

    @abstractmethod
    def make_space(self) -> Space:
        """Creates the action space"""
        ...

    def parse(self, nmi: NegotiatorMechanismInterface, action: np.ndarray) -> Any:
        """Parses the encoded action from the model and applies final transformations to it."""
        return self.decode(nmi, action)

    @abstractmethod
    def decode(self, nmi: NegotiatorMechanismInterface, action: np.ndarray) -> Any:
        """Converts an action from the model/policy to a valid action for the mechanism it is intended to be used with."""
        ...

    def encode(self, nmi: NegotiatorMechanismInterface, action: Any) -> np.ndarray:
        """Converts an action to the original input used to decode it (not needed for the RL environments but useful for debugging).

        Remarks:
            - This is only used for testing so it is optional
            - The mapping of `decode` may not be invertible. If that is the case
              this function may not be possible to write.
            - Even if it is not possible to recover the actual action, it may be
              possible to recover parts of it. For example in SAOResponse, it may not
              be possible to recover the actual offer but acceptance, and ending can
              be recovered as well as some constraint on the utility of the endoded outcome.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `decode`."
        )

    @classmethod
    def is_outcome_invertible(cls) -> bool:
        """Whether or not this action decoder's encode exactly inverts decode"""
        return False

    @classmethod
    def is_utility_invertible(cls) -> bool:
        """Whether or not this action decoder's encode exactly inverts decode in terms of utility"""
        return True

    def prepare(self):
        """Prepare the manager here when the negotiation starts or preferences change"""
        self.prepared = True

    def on_negotiation_starts(
        self, owner: Negotiator, nmi: NegotiatorMechanismInterface
    ) -> None:
        super().on_negotiation_starts(owner, nmi)
        if not self.prepared:
            self.prepare()

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        if self.owner:
            self.prepare()


@define
class SAOActionDecoder(ActionDecoder):
    """
    An action decoder compatible with the SAOMechanism.

    Attributes
    ----------
        rational_offers: Only allow rational offers falling to offering best if `fallback_to_best` else ending
        rational_acceptance: Only accept rational offers falling to rejecting and offering best if `fallback_to_best` else ending
        cannot_accept_none: Whether to allow accepting None (the mechanism may treat this as ending the negotiation)
        accept_if_offering_last_received: If `True` and the decoder's `decode()` returned rejection with the same offer as the last
                                          received, it will be treated as acceptance. This is useful in cases where we do not explicitly
                                          model acceptance.
        acceptance_policy_overrides_decision: If given, the decision from the decoder will be ignored as long as the `acceptance_policy`
                                              returns `True`.
        acceptance_policy: The acceptance policy to use. If not given, the decoder should model acceptance itself. If given, it will
                           be used to decide whether or not to accept the offer from the partner(s).
        fallback_to_last: If given, any invalid action will be replaced with rejection and offering the last outcome
        fallback_to_offered: If given, any invalid action will be replaced with rejection and offering random previously offered outcome (if any)
        fallback_to_best: If given, any invalid action will be replaced with rejection and offering the best outcome (as long as it is rational)

        Remarks:
            - fallback options are processed in the following order: last, offered, best
    """

    rational_offers: bool = False
    rational_acceptance: bool = True
    cannot_accept_none: bool = True
    accept_if_offering_last_received: bool = False
    acceptance_policy_overrides_decision: bool = False
    acceptance_policy: AcceptancePolicy | None = None
    fallback_to_last: bool = True
    fallback_to_offered: bool = False
    fallback_to_best: bool = True
    explicit_accept: bool = field(init=False, default=False)
    _best: Outcome | None = field(init=False)
    _last_outcome: Outcome | None = field(init=False, default=None)
    _acceptance_action: np.ndarray | None = field(init=False, default=None)

    def __attrs_post_init__(self):
        self.explicit_accept = self.acceptance_policy is None

    def is_acceptance_action(
        self, nmi, action: np.ndarray | float | int | tuple | list
    ) -> bool:
        if not self.acceptance_policy:
            return False
        target = self.encode(nmi, self.acceptance_policy.acceptance_code(nmi))
        if isinstance(action, tuple) or isinstance(action, list):
            return all(
                a == b
                if not isinstance(a, np.ndarray) and not isinstance(b, np.ndarray)
                else np.all(a == b)
                for a, b in zip(action, target, strict=True)
            )
        return bool(np.all(np.asarray(action) == np.asarray(target)))

    def parse(self, nmi: SAONMI, action: np.ndarray) -> SAOResponse:  # type: ignore
        decoded = self.decode(nmi, action)
        decoded: SAOResponse
        assert self.owner.ufun
        # if the acceptance policy overrides the decision, it is checked before anything
        # and if it returns True, the current offer will be accepted otherwise processing
        # continues.
        if (
            self.acceptance_policy is not None
            and self.acceptance_policy_overrides_decision
        ):
            current = self.owner.nmi.state.current_offer
            if self.acceptance_policy(
                self._last_outcome, current, decoded.outcome, self.owner.ufun, nmi
            ):
                return SAOResponse(ResponseType.ACCEPT_OFFER, current)
        if decoded.response == ResponseType.END_NEGOTIATION:
            if not self.allow_ending:
                if self.fallback_to_last and self._last_outcome is not None:
                    return SAOResponse(ResponseType.REJECT_OFFER, self._last_outcome)
                if self.fallback_to_offered and (
                    offers := nmi.negotiator_offers(self.owner.id)
                ):
                    return SAOResponse(ResponseType.REJECT_OFFER, choice(offers))
                if self.fallback_to_best and self._best is not None:
                    return SAOResponse(ResponseType.REJECT_OFFER, self._best)
                raise ValueError(
                    f"Invalid response {decoded}. You are not allowing ending the negotiation"
                )
            return SAOResponse(decoded.response, None)
        if decoded.response == ResponseType.NO_RESPONSE:
            if not self.allow_no_response:
                if self.fallback_to_last and self._last_outcome is not None:
                    return SAOResponse(ResponseType.REJECT_OFFER, self._last_outcome)
                if self.fallback_to_offered and (
                    offers := nmi.negotiator_offers(self.owner.id)
                ):
                    return SAOResponse(ResponseType.REJECT_OFFER, choice(offers))
                if self.fallback_to_best and self._best is not None:
                    return SAOResponse(ResponseType.REJECT_OFFER, self._best)
                raise ValueError(
                    f"Invalid response {decoded}. You are not allowing no-response "
                )
            return SAOResponse(decoded.response, None)
        if (
            decoded.response == ResponseType.ACCEPT_OFFER
            and self.rational_acceptance
            and self.owner.ufun(decoded.outcome) < self.owner.ufun.reserved_value - 1e-5  # type: ignore
        ):
            if self.fallback_to_last and self._last_outcome is not None:
                return SAOResponse(ResponseType.REJECT_OFFER, self._last_outcome)
            if self.fallback_to_offered and (
                offers := nmi.negotiator_offers(self.owner.id)
            ):
                return SAOResponse(ResponseType.REJECT_OFFER, choice(offers))
            if (
                self.fallback_to_best
                and self._best
                and self.owner.ufun(self._best) >= self.owner.ufun.reserved_value
            ):
                return SAOResponse(ResponseType.REJECT_OFFER, self._best)
            if not self.allow_ending:
                raise ValueError(
                    f"Invalid response {decoded}. You are not allowing ending the negotiation"
                )
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if (
            decoded.response == ResponseType.ACCEPT_OFFER
            and self.cannot_accept_none
            and decoded.outcome is None
        ):
            if self.fallback_to_last and self._last_outcome is not None:
                return SAOResponse(ResponseType.REJECT_OFFER, self._last_outcome)
            if self.fallback_to_offered and (
                offers := nmi.negotiator_offers(self.owner.id)
            ):
                return SAOResponse(ResponseType.REJECT_OFFER, choice(offers))
            if self.fallback_to_best and self._best:
                return SAOResponse(ResponseType.REJECT_OFFER, self._best)
            if not self.allow_ending:
                raise ValueError(
                    f"Invalid response {decoded}. You are not allowing ending the negotiation"
                )
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if decoded.response == ResponseType.ACCEPT_OFFER:
            return SAOResponse(ResponseType.ACCEPT_OFFER, nmi.state.current_offer)
        # test rejections
        if decoded.response == ResponseType.REJECT_OFFER and decoded.outcome is None:
            if self.fallback_to_last and self._last_outcome is not None:
                return SAOResponse(ResponseType.REJECT_OFFER, self._last_outcome)
            if self.fallback_to_offered and (
                offers := nmi.negotiator_offers(self.owner.id)
            ):
                return SAOResponse(ResponseType.REJECT_OFFER, choice(offers))
            if self.fallback_to_best and self._best:
                return SAOResponse(ResponseType.REJECT_OFFER, self._best)
            if not self.allow_no_response:
                raise ValueError(
                    f"Invalid response {decoded}. You are not allowing rejecting with no offer given "
                )
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if (
            decoded.response == ResponseType.REJECT_OFFER
            and self.accept_if_offering_last_received
            and decoded.outcome == nmi.state.current_offer
        ):
            return SAOResponse(ResponseType.ACCEPT_OFFER, nmi.state.current_offer)
        if (
            decoded.response == ResponseType.REJECT_OFFER
            and self.rational_offers
            and self.owner.ufun(decoded.outcome) < self.owner.ufun.reserved_value - 1e-5  # type: ignore
        ):
            if self.fallback_to_last and self._last_outcome is not None:
                return SAOResponse(ResponseType.REJECT_OFFER, self._last_outcome)
            if self.fallback_to_offered and (
                offers := nmi.negotiator_offers(self.owner.id)
            ):
                return SAOResponse(ResponseType.REJECT_OFFER, choice(offers))
            if self.fallback_to_best and self._best:
                return SAOResponse(ResponseType.REJECT_OFFER, self._best)
            if not self.allow_ending:
                raise ValueError(
                    f"Invalid response {decoded}. You are not allowing ending the negotiation"
                )
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        if (
            decoded.response == ResponseType.REJECT_OFFER
            and self.acceptance_policy is not None
        ):
            current = self.owner.nmi.state.current_offer
            if self.acceptance_policy(
                self._last_outcome, current, decoded.outcome, self.owner.ufun, nmi
            ):
                return SAOResponse(ResponseType.ACCEPT_OFFER, current)
        if (
            decoded.response == ResponseType.REJECT_OFFER
            and decoded.outcome is not None
        ):
            self._last_outcome = decoded.outcome
        return decoded

    def prepare(self):
        """Prepare the manager here when the negotiation starts or preferences change"""
        super().prepare()
        nmi, ufun = self.owner.nmi, self.owner.ufun
        self._acceptance_action = (
            self.encode(nmi, self.acceptance_policy.acceptance_code(nmi))
            if self.acceptance_policy
            else None
        )
        self._last_outcome = None
        if self.fallback_to_best or self.fallback_to_last:
            self._best = ufun.invert(  # type: ignore
                rational_only=self.rational_acceptance or self.rational_offers
            ).best()
            if float(ufun(self._best)) < ufun.reserved_value:  # type: ignore
                self._best = None
            self._last_outcome = self._best


class _DiscreteSamplerMixin:
    def sample(self, action, sampler, sampler_params: dict[str, Any]) -> Outcome | None:
        u = float(action[ACTION])
        dbelow = max(0, u - floor(u * self.n_levels) / self.n_levels)  # type: ignore
        dabove = min(1, ceil(u * self.n_levels) / self.n_levels) - u  # type: ignore
        r, mx = self._lower_limit, self._max  # type: ignore
        d = mx - r
        u = r + u * d
        dbelow = r + dbelow * d
        dabove = r + dabove * d
        for below, above in [(dabove, dbelow)] + list(
            product(
                [_ for _ in self.delta_below if _ > dbelow],  # type: ignore
                [_ for _ in self.delta_above if _ > dabove],  # type: ignore
            )
        ):
            outcome = sampler(
                (
                    u - below if not self.rational_encoding else max(r, u - below),  # type: ignore
                    u + above if not self.rational_encoding else max(r, u + above),  # type: ignore
                ),
                normalized=True,
                **sampler_params,  # type: ignore
            )
            if outcome is not None:
                break
        else:
            if self.fallback_to_last and self._last_outcome is not None:  # type: ignore
                outcome = self._last_outcome  # type: ignore
            elif self.fallback_to_best and self._best is not None:  # type: ignore
                outcome = self._best  # type: ignore
            else:
                raise ValueError(
                    f"Cannot find any outcome for utility {u} with "
                    f"{self.delta_below=} and {self.delta_above=} and you are "  # type: ignore
                    f"indicating not to fallback-to-best."
                )

        return outcome

    def make_space(self) -> spaces.MultiDiscrete:
        """Creates the action space"""
        return spaces.MultiDiscrete(
            np.asarray(
                [
                    self._n_decisions,  # type: ignore
                    self.n_levels,  # type: ignore
                ]
            ),
            start=np.asarray([0, self.start_code], dtype=INT_TYPE),  # type: ignore
            dtype=INT_TYPE,  # type: ignore
        )

    def decode(self, nmi: SAONMI, action: tuple[int, int]) -> SAOResponse:  # type: ignore
        return super().decode(  # type: ignore
            nmi,
            (
                action[DECISION],
                to_float(int(action[ACTION]), self.n_levels, start=self.start_code),  # type: ignore
            ),
        )

    def encode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, action: SAOResponse
    ) -> np.ndarray:
        decoded = super().encode(nmi, action)  # type: ignore
        return np.asarray(
            [
                decoded[DECISION],
                from_float(
                    float(decoded[ACTION]),
                    self.n_levels,  # type: ignore
                    start=self.start_code,  # type: ignore
                    mn=self.min_value,  # type: ignore
                    mx=self.max_value,  # type: ignore
                ),
            ],
            dtype=INT_TYPE,
        )


@define
class _BoxMixin:
    def make_space(self) -> spaces.Box:
        """Creates the action space"""
        return spaces.Box(self.min_value, self.max_value, (2,), dtype=FLOAT_TYPE)  # type: ignore

    def encode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, action: SAOResponse
    ) -> np.ndarray:
        decoded = super().encode(nmi, action)  # type: ignore
        return np.asarray(
            [
                to_float(
                    decoded[DECISION],
                    self._n_decisions,  # type: ignore
                    mn=self.min_value,  # type: ignore
                    mx=self.max_value,  # type: ignore
                ),  # type: ignore
                float(decoded[ACTION]),
            ],
            dtype=FLOAT_TYPE,
        )

    def decode(self, nmi: SAONMI, action: np.ndarray) -> SAOResponse:  # type: ignore
        return super().decode(  # type: ignore
            nmi,
            (
                from_float(
                    action[DECISION],
                    self._n_decisions,  # type: ignore
                    mn=self.min_value,  # type: ignore
                    mx=self.max_value,  # type: ignore
                ),
                np.asarray([action[ACTION]], dtype=FLOAT_TYPE),
            ),
        )


class _D1DMixin:
    def set_limits(self):
        nxt = self.start_code  # type: ignore
        invalid = nxt - 1
        if self.allow_ending:  # type: ignore
            self._end_code = nxt
            nxt += 1
        else:
            self._end_code = invalid
        if self.allow_no_response:  # type: ignore
            self._no_response_code = nxt
            nxt += 1
        else:
            self._no_response_code = invalid
        if self.explicit_accept:  # type: ignore
            self._accept_code = nxt
            nxt += 1
        else:
            self._accept_code = invalid
        self._min_offer_code = nxt

    def make_space(self) -> spaces.Discrete:  # type: ignore
        """Creates the action space"""
        return spaces.Discrete(
            self.n_levels + self._n_decisions - 1,  # type: ignore
            start=self.start_code,  # type: ignore
        )

    def decode(self, nmi: SAONMI, action: int) -> SAOResponse:  # type: ignore
        action = (
            action[0] if isinstance(action, np.ndarray) and action.shape else action
        )
        if self.explicit_accept and action == self._accept_code:  # type: ignore
            return SAOResponse(ResponseType.ACCEPT_OFFER, nmi.state.current_offer)
        if self.is_acceptance_action(nmi, action):  # type: ignore
            return SAOResponse(ResponseType.ACCEPT_OFFER, nmi.state.current_offer)
        if self.allow_ending and action == self._end_code:  # type: ignore
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if self.allow_no_response and action == self._no_response_code:  # type: ignore
            return SAOResponse(ResponseType.NO_RESPONSE, None)
        return super().decode(nmi, (REJECT, action - self._min_offer_code))  # type: ignore

    def encode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, action: SAOResponse
    ) -> int:
        if action.response == ResponseType.END_NEGOTIATION:
            if not self.allow_ending:  # type: ignore
                raise ValueError(
                    f"Cannot encode action {action} because you are not specifying allow-ending"
                )
            return self._end_code  # type: ignore
        if action.response == ResponseType.NO_RESPONSE:
            if not self.allow_no_response:  # type: ignore
                raise ValueError(
                    f"Cannot encode action {action} because you are not specifying allow-no-response"
                )
            return self._no_response_code  # type: ignore
        if action.response == ResponseType.ACCEPT_OFFER:
            if self.explicit_accept:  # type: ignore
                # even if we do not have explicit accept, by returning 0 we are guaranteed to accept whatever current offer is
                return self._accept_code  # type: ignore
            else:
                # assert self._acceptance_action is not None
                return int(self._acceptance_action)  # type: ignore
            raise ValueError(
                f"Cannot encode action {action} because you are not allowing accept_if_better_than_current and also not encoding explicit_accept."
            )
        encoded = super().encode(nmi, action)[ACTION]  # type: ignore
        return max(encoded, 0) + self._min_offer_code  # type: ignore


@define
class UtilityDecoder(SAOActionDecoder):
    """
    converts actions as a tuple with two items: outcome utility and decision (accept/reject/end/no response)

    Args:

        allow_no_response: If given, the negotiator can return no-response instead of a counter offer.
        allow_ending: If given the negotiator can end the negotiation.
        fallback_to_best: Will fallback to best outcome if no outcome can be found in the given utility range
        sampling_method: Method used for sampling outcomes from the allowed range
        delta_above: Defines the lower limit of utility around the exact value returned by the policy. Values will be tried in order
        delta_below: Defines the upper limit of utility around the exact value returned by the policy. Values will be tried in order

    Remarks:
        - Using this action decoder, you will need to have a
          way to sample outcomes around a given utility value.
          Negmas provides this using the `invert()` method of
          `UtilityFunction` or you can use a dedicated `UtilityInverter` object.
    """

    rational_encoding: bool = False
    fallback_to_best: bool = True
    fallback_to_last: bool = True
    sampling_method: SamplingMethod = SamplingMethod.Random
    delta_below: tuple[float, ...] = (0, 1e-6)
    delta_above: tuple[float, ...] = (0, 1e-1)
    _inverter: InverseUFun = field(init=False, default=None)
    _min: float = field(init=False, default=0)
    _best: Outcome | None = field(init=False, default=None)
    _max: float = field(init=False, default=0)
    _r: float = field(init=False, default=0)
    _lower_limit: float = field(init=False, default=0)
    _n_decisions: int = field(init=False)
    min_value: float = field(init=False, default=0)
    max_value: float = field(init=False, default=1)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self._n_decisions = 2 + int(self.allow_ending) + int(self.allow_no_response)
        if self.allow_no_response and not self.allow_ending:
            raise ValueError(
                f"{self.__class__.__name__} does not support allowing no response while not allowing ending"
            )

    def make_space(
        self,
    ) -> spaces.Tuple | spaces.MultiDiscrete | spaces.Box | spaces.Discrete:
        return spaces.Tuple(
            (
                spaces.Discrete(self._n_decisions),
                spaces.Box(self.min_value, self.max_value, dtype=FLOAT_TYPE),
            )
        )

    def decode(  # type: ignore
        self, nmi: SAONMI, action: tuple[int, np.ndarray | float]
    ) -> SAOResponse:
        if not self._inverter:
            self.prepare()
        accept = action[DECISION] == ACCEPT
        end = action[DECISION] == END and self.allow_ending
        no_response = action[DECISION] == NORESPONSE and self.allow_no_response
        if accept:
            return SAOResponse(ResponseType.ACCEPT_OFFER, nmi.state.current_offer)
        if end:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if no_response:
            return SAOResponse(ResponseType.NO_RESPONSE, None)
        sampler_params = dict()
        if self.sampling_method == SamplingMethod.Random:
            sampler = self._inverter.one_in
        elif self.sampling_method == SamplingMethod.Worst:
            sampler = self._inverter.worst_in
            sampler_params = dict(cycle=False)
        elif self.sampling_method == SamplingMethod.Best:
            sampler = self._inverter.best_in
            sampler_params = dict(cycle=False)
        elif self.sampling_method == SamplingMethod.BestForPartner:
            sampler = functools.partial(
                best_for,
                inverter=self._inverter,
                target_ufuns=(self.owner.partner_ufun, self.owner.ufun),  # type: ignore
            )
        else:
            raise ValueError(f"Sampling method {self.sampling_method} is unknown")
        outcome = self.sample(action, sampler, sampler_params)
        return SAOResponse(ResponseType.REJECT_OFFER, outcome)

    def sample(self, action, sampler, sampler_params: dict[str, Any]) -> Outcome | None:
        u = float(action[ACTION])
        u = (u - self.min_value) / (self.max_value - self.min_value)
        r, mx = self._lower_limit, self._max
        u = u * (mx - r) + r
        outcome = None
        for below, above in product(self.delta_below, self.delta_above):
            outcome = sampler(
                (
                    float(
                        u - below if not self.rational_encoding else max(r, u - below)
                    ),
                    float(u + above)
                    if not self.rational_encoding
                    else max(r, u + above),
                ),
                normalized=True,
                **sampler_params,  # type: ignore
            )
            if outcome is not None:
                break
        else:
            if self.fallback_to_last and self._last_outcome:
                outcome = self._last_outcome
            elif self.fallback_to_best:
                outcome = self._best
            else:
                raise ValueError(
                    f"Cannot find any outcome for utility {u} with "
                    f"{self.delta_below=} and {self.delta_above=} and you are "
                    f"indicating not to fallback-to-best."
                )

        return outcome

    def encode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, action: SAOResponse
    ) -> tuple[int, np.ndarray]:
        _ = nmi
        if action.response == ResponseType.END_NEGOTIATION:
            return (END, np.asarray([0.0], dtype=FLOAT_TYPE))
        if action.response == ResponseType.ACCEPT_OFFER:
            return (ACCEPT, np.asarray([0.0], dtype=FLOAT_TYPE))
        if action.response == ResponseType.NO_RESPONSE:
            return (NORESPONSE, np.asarray([0.0], dtype=FLOAT_TYPE))
        assert self.owner.ufun is not None
        ufun = self.owner.ufun
        u = float(ufun(action.outcome))
        r, mx = self._lower_limit, self._max
        u = (u - r) / (mx - r)
        u = u * (self.max_value - self.min_value) + self.min_value
        return (REJECT, np.asarray([u], dtype=FLOAT_TYPE))

    def prepare(self):
        super().prepare()
        assert self.owner, (
            "Unknown owner (make sure that on_negotiation_starts is called for the action decoder)"
        )
        assert self.owner.ufun, f"Unknown ufun for owner {self.owner}"
        self._inverter = self.owner.ufun.invert(rational_only=self.rational_encoding)
        self._min, self._max = self.owner.ufun.minmax()
        self._r = float(self.owner.reserved_value) if self.owner.reserved_value else 0
        self._lower_limit = (
            max(self._min, self._r) if self.rational_encoding else self._min
        )
        self._best = self.owner.ufun.best()


@define
class RelativeUtilityDecoder(UtilityDecoder):
    """
    converts actions as a change in utility from previous offer (starting at max)

    Remarks:
        - initial_utility is assumed to be normalized between zero and one
          - if rational_encoding: 0 -> max(reserved_value, min_utility), 1 -> max_utility
          - if not rational_encoding: 0 -> min_utility, 1 -> max_utility
    """

    initial_utility: float = 1
    code_range: tuple[float, float] = field(init=False, default=(-1, 1))
    _current_utility_normalized: float = field(init=False, default=None)

    def decode(
        self, nmi: SAONMI, action: tuple[int, np.ndarray | float]
    ) -> SAOResponse:  # type: ignore
        if self._last_outcome is not None:
            self._current_utility_normalized = (
                float(self.owner.ufun(self._last_outcome)) - self._lower_limit  # type: ignore
            ) / (self._max - self._lower_limit)
        decision = action[DECISION]
        # convert to a value between 0 and 1 to be on the same range as _current_utility_normalized
        mn, mx = self.code_range
        u = (
            2.0 * (action[ACTION] - mn) / (mx - mn) - 1
        ) + self._current_utility_normalized
        # convert to a real utility between lower-limit and max to pass to UtilityDecoder
        util = np.asarray(
            [
                max(
                    self._lower_limit,
                    min(
                        self._max,
                        float(u * (self._max - self._lower_limit) + self._lower_limit),
                    ),
                )
            ],
            dtype=FLOAT_TYPE,
        )
        return super().decode(nmi, (decision, util))

    def encode(
        self, nmi: NegotiatorMechanismInterface, action: SAOResponse
    ) -> tuple[int, np.ndarray]:  # type: ignore
        # assumes that encode is called immediately after decode
        if self._last_outcome is not None:
            self._current_utility_normalized = (
                float(self.owner.ufun(self._last_outcome)) - self._lower_limit  # type: ignore
            ) / (self._max - self._lower_limit)
        encoded = super().encode(nmi, action)
        # util = (encoded[ACTION] - self._lower_limit) / (self._max - self._lower_limit)
        if action.response != ResponseType.REJECT_OFFER:
            return super().encode(nmi, action)
        # now diff ranges between zero and 1
        mn, mx = self.code_range
        diff = (float(encoded[ACTION] - self._current_utility_normalized) + 1) * (
            mx - mn
        ) / 2.0 + mn
        return (encoded[DECISION], np.asarray(diff, dtype=FLOAT_TYPE))

    def prepare(self):
        super().prepare()
        self._current_utility_normalized = self.initial_utility


@define(frozen=False)
class OutcomeDecoder(SAOActionDecoder):
    """Offers are encoded as outcomes not as utilities"""

    os: CartesianOutcomeSpace = field(default=None, init=False)
    rational_encoding: bool = False
    allow_none: bool = True
    order_by_utility: bool = True
    order_by_welfare: bool = False
    order_by_relative_welfare: bool = False
    order_by_similarity: bool = False
    _outcome_map: dict[Outcome | None, float] = field(init=False, factory=dict)
    _outcomes: list[Outcome | None] = field(init=False, factory=list)
    _last_outcome: Outcome | None = field(init=False, default=None)
    _n_decisions: int = field(init=False)
    _n_outcomes: int = field(init=False, default=None)
    min_value: float = field(init=False, default=0)
    max_value: float = field(init=False, default=1)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self._n_decisions = 2 + int(self.allow_ending) + int(self.allow_no_response)
        if self.allow_no_response and not self.allow_ending:
            raise ValueError(
                f"{self.__class__.__name__} does not support allowing no response while not allowing ending"
            )

    def make_space(
        self,
    ) -> spaces.Tuple | spaces.MultiDiscrete | spaces.Box | spaces.Discrete:
        """
        Creates the action space.

        The type of this action space is a tuple with one field
        """
        return spaces.Tuple(
            (
                spaces.Discrete(self._n_decisions),
                spaces.Box(self.min_value, self.max_value, dtype=FLOAT_TYPE),
            )
        )

    def decode(  # type: ignore
        self, nmi: SAONMI, action: tuple[int, np.ndarray | float]
    ) -> SAOResponse:
        return self._decode(nmi, action)

    def _decode(
        self, nmi: SAONMI, action: tuple[int, np.ndarray | float]
    ) -> SAOResponse:
        """Converts an action from the model/policy to a valid action for the mechanism it is intended to be used with."""
        outcome = action[ACTION]
        accept = action[DECISION] == ACCEPT
        end = action[DECISION] == END and self.allow_ending
        no_response = action[DECISION] == NORESPONSE and self.allow_no_response
        if end:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if no_response:
            return SAOResponse(ResponseType.NO_RESPONSE, None)
        if accept:
            return SAOResponse(ResponseType.ACCEPT_OFFER, nmi.state.current_offer)
        if self.os is None:
            self.prepare()
        return SAOResponse(
            ResponseType.REJECT_OFFER,
            self._outcomes[
                from_float(
                    float(outcome),
                    self._n_outcomes,
                    mn=self.min_value,
                    mx=self.max_value,
                )
            ],
        )

    def encode_outcome(self, outcome: Outcome | None) -> float:
        if self.os is None:
            self.prepare()
        return self._outcome_map[outcome]

    def encode(  # type: ignore
        self, nmi: SAONMI, action: SAOResponse
    ) -> tuple[int, float]:
        """Inverses decode()"""
        if action.response == ResponseType.ACCEPT_OFFER:
            return (ACCEPT, self.encode_outcome(nmi.state.current_offer))
        if action.response == ResponseType.END_NEGOTIATION:
            if not self.allow_ending:
                raise ValueError(
                    "Ending negotiation is not allowed in this action decoder"
                )
            return (END, 0)
        if action.response == ResponseType.NO_RESPONSE:
            if not self.allow_no_response:
                raise ValueError("No-response is not allowed in this action decoder")
            return (NORESPONSE, 0)

        return (REJECT, self.encode_outcome(action.outcome))

    def set_outcome_space(self, os: CartesianOutcomeSpace):
        self.os = os
        nlevels = DEFAULT_N_ISSUE_LEVELS
        if self.order_by_utility:
            enumerator = functools.partial(
                enumerate_by_utility,
                issues=os.issues,
                ufuns=(self.owner.ufun,),  # type: ignore
                n_levels=nlevels,
                max_cardinality=MAX_CARDINALITY,
                by_similarity_first=self.order_by_similarity,
                by_welfare=self.order_by_welfare,
                by_relative_welfare=self.order_by_relative_welfare,
            )
        elif self.order_by_similarity:
            enumerator = functools.partial(
                enumerate_in_order, issues=os.issues, n_levels=nlevels
            )
        elif not os.is_discrete():
            enumerator = functools.partial(
                os.enumerate_or_sample,
                levels=nlevels,
                max_cardinality=DEFAULT_N_OUTCOMES,
            )
        else:
            enumerator = os.enumerate  # type: ignore
        outcomes = [
            _
            for _ in enumerator()
            if not self.rational_encoding
            or (float(self.owner.ufun(_)) >= self.owner.ufun.reserved_value)  # type: ignore We assume that rational_only is only given if owner is set
        ]
        if self.allow_none:
            outcomes = [None] + outcomes
        self._outcomes = outcomes  # type: ignore
        self._n_outcomes = len(self._outcomes)
        self._outcome_map = dict(
            zip(
                outcomes,
                [
                    to_float(_, self._n_outcomes, mn=self.min_value, mx=self.max_value)
                    for _ in range(len(outcomes))
                ],
            )
        )

    def prepare(self):
        super().prepare()
        assert self.owner
        if self.owner.preferences:
            os = self.owner.preferences.outcome_space
        else:
            os = self.owner.nmi.outcome_space
        assert os is not None and isinstance(os, CartesianOutcomeSpace)
        self.set_outcome_space(os)

    @classmethod
    def is_outcome_invertible(cls) -> bool:
        return True


@define
class IssueDecoder(SAOActionDecoder):
    """Offers are encoded as outcomes not as utilities"""

    n_issues: int = 0
    allow_none: bool = True
    os: CartesianOutcomeSpace = field(default=None, init=False)
    min_encodable: float = field(init=False, default=0.05)
    _n_decisions: int = field(init=False, default=2)
    min_value: float = field(init=False, default=0)
    max_value: float = field(init=False, default=1)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self._n_decisions = 2 + int(self.allow_ending) + int(self.allow_no_response)
        if self.allow_none and not self.min_encodable:
            self.min_encodable = (self.max_value - self.min_value) / 20 + self.min_value
        else:
            self.min_encodable = self.min_value

    def make_space(
        self,
    ) -> spaces.Tuple | spaces.MultiDiscrete | spaces.Box | spaces.Discrete:
        """
        Creates the action space.

        The type of this action space is a tuple with one field
        """
        n_issues = self.n_issues
        return spaces.Tuple(
            (
                spaces.Discrete(self._n_decisions),
                spaces.Box(
                    np.ones(n_issues) * self.min_value,
                    np.ones(n_issues) * self.max_value,
                    dtype=FLOAT_TYPE,
                ),
            )
        )

    def decode(self, nmi: SAONMI, action: tuple[int, np.ndarray]) -> SAOResponse:  # type: ignore
        """Converts an action from the model/policy to a valid action for the mechanism it is intended to be used with."""
        model_offer = action[ACTION]
        accept = action[DECISION] == ACCEPT
        end = action[DECISION] == END and self.allow_ending
        no_response = action[DECISION] == NORESPONSE and self.allow_no_response
        if end:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if no_response:
            return SAOResponse(ResponseType.NO_RESPONSE, None)
        if accept:
            return SAOResponse(ResponseType.ACCEPT_OFFER, nmi.state.current_offer)

        if self.os is None:
            self.prepare()
        outcome = (
            tuple(
                self.decode_value(
                    ((v - self.min_encodable) / (1 - self.min_encodable))
                    if self.allow_none
                    else v,
                    i,
                )
                for v, i in zip(model_offer, self.os.issues)
            )
            if not self.allow_none or np.all(model_offer >= self.min_encodable)
            else None
        )
        return SAOResponse(ResponseType.REJECT_OFFER, outcome)

    def encode_outcome(self, outcome: Outcome | None) -> np.ndarray:
        if outcome is None:
            if self.allow_none:
                return np.zeros(self.n_issues, dtype=FLOAT_TYPE)
            raise ValueError(f"{self.__class__.__name__} Cannot encode None")
        if self.os is None:
            self.prepare()
        return np.asarray(
            [self.encode_value(v, i) for v, i in zip(outcome, self.os.issues)],
            dtype=FLOAT_TYPE,
        )

    def encode(  # type: ignore
        self, nmi: SAONMI, action: SAOResponse
    ) -> tuple[int, np.ndarray]:
        """Inverses decode()"""
        state = nmi.state
        n_issues = self.n_issues

        if action.response == ResponseType.ACCEPT_OFFER:
            return (ACCEPT, self.encode_outcome(nmi.state.current_offer))
        if action.response == ResponseType.END_NEGOTIATION:
            if not self.allow_ending:
                raise ValueError("No-response is not allowed in this action decoder")
            return (END, np.asarray([0.0] * n_issues, dtype=FLOAT_TYPE))
        if action.response == ResponseType.NO_RESPONSE:
            if not self.allow_no_response:
                raise ValueError("No-response is not allowed in this action decoder")
            return (NORESPONSE, np.asarray([0.0] * n_issues, dtype=FLOAT_TYPE))

        offer = action.outcome

        return (REJECT, self.encode_outcome(offer))

    def encode_value(self, v: int | float | str, i: Issue) -> float:
        if isinstance(i, CategoricalIssue):
            vals = list(i.values)
            return to_float(
                vals.index(v), int(i.cardinality), mn=self.min_value, mx=self.max_value
            )
        if i.is_continuous():
            return (v - i.min_value) / (i.max_value - i.min_value)
        return to_float(v, i.cardinality, mn=self.min_value, mx=self.max_value)  # type: ignore

    def decode_value(self, v: float, i: Issue) -> Any:
        if isinstance(i, CategoricalIssue):
            return i.value_at(from_float(v, int(i.cardinality)))
        if i.is_continuous():
            x = v * (i.max_value - i.min_value) + i.min_value
            return x * (self.max_value - self.min_value) + self.min_value
        x = from_float(v, i.cardinality, mn=self.min_value, mx=self.max_value)  # type: ignore
        if x in i:
            return x
        best, diff = None, float("inf")
        for z in i.all:
            d = abs(z - x)
            if d < diff:
                best, diff = z, d
        return best

    def set_outcome_space(self, os: CartesianOutcomeSpace):
        if self.n_issues <= 0:
            self.n_issues = len(os.issues)
        assert len(os.issues) == self.n_issues, (
            f"Incorrect number of issues: {os=}, {self.n_issues=}"
        )
        self.os = os
        if not self.prepared:
            self.prepared = True

    def prepare(self):
        super().prepare()
        assert self.owner
        if self.owner.preferences:
            os = self.owner.preferences.outcome_space
        else:
            os = self.owner.nmi.outcome_space
        assert os is not None and isinstance(os, CartesianOutcomeSpace)
        self.set_outcome_space(os)

    @classmethod
    def is_outcome_invertible(cls) -> bool:
        return True


class _C1DMixin:
    def set_limits(self):
        self.end_limit = self.no_response_limit = self.accept_limit = self.min_value  # type: ignore
        if self.allow_ending:  # type: ignore
            self.end_limit = self.end_interval  # type: ignore
        if self.allow_no_response:  # type: ignore
            self.no_response_limit = self.end_limit + self.no_response_interval  # type: ignore
        else:
            self.no_response_limit = self.end_limit  # type: ignore
        if self.explicit_accept:  # type: ignore
            self.accept_limit = self.no_response_limit + self.accept_interval  # type: ignore
        else:
            self.accept_limit = self.no_response_limit  # type: ignore
        self.min_encodable = self.accept_limit + self.decision_margin  # type: ignore
        assert self.end_limit <= self.no_response_limit <= self.accept_limit, f"{self}"  # type: ignore
        assert (
            self.min_encodable < self.max_value  # type: ignore
        ), f"{self.min_encodable=} but {self.max_value=}\n{self}"  # type: ignore

    def make_space(self) -> spaces.Box:
        """Creates the action space"""
        return spaces.Box(self.min_value, self.max_value, (1,), dtype=FLOAT_TYPE)  # type: ignore

    def decode(self, nmi: SAONMI, action: np.ndarray) -> SAOResponse:  # type: ignore
        action = (
            action[0] if isinstance(action, np.ndarray) and action.shape else action
        )
        if (
            self.explicit_accept  # type: ignore
            and self.no_response_limit < action <= self.accept_limit
        ) or self.is_acceptance_action(nmi, action):  # type: ignore
            return SAOResponse(ResponseType.ACCEPT_OFFER, nmi.state.current_offer)
        if self.allow_ending and action <= self.end_limit:  # type: ignore
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if self.allow_no_response and self.end_limit < action <= self.no_response_limit:  # type: ignore
            return SAOResponse(ResponseType.NO_RESPONSE, None)

        return super().decode(  # type: ignore
            nmi,
            (
                REJECT,
                max(
                    self.min_value,  # type: ignore
                    min(
                        self.max_value,  # type: ignore
                        self.min_value  # type: ignore
                        + (
                            (action - self.min_encodable)
                            * (self.max_value - self.min_value)  # type: ignore
                            / (self.max_value - self.min_encodable)  # type: ignore
                        ),
                    ),
                ),
            ),
        )

    def encode(  # type: ignore
        self, nmi: SAONMI, action: SAOResponse
    ) -> np.ndarray:
        if action.response == ResponseType.END_NEGOTIATION:
            if not self.allow_ending:  # type: ignore
                raise ValueError(f"{action} cannot be decoded without allow-ending")
            return np.asarray([self.end_limit / 2], dtype=FLOAT_TYPE)
        if action.response == ResponseType.NO_RESPONSE:
            if not self.allow_no_response:  # type: ignore
                raise ValueError(
                    f"{action} cannot be decoded without allow-no-response"
                )
            return np.asarray(
                [(self.no_response_limit + self.end_limit) / 2],
                dtype=FLOAT_TYPE,
            )
        if action.response == ResponseType.ACCEPT_OFFER:
            if self.explicit_accept:  # type: ignore
                return np.asarray(
                    [(self.accept_limit + self.no_response_limit) / 2],
                    dtype=FLOAT_TYPE,
                )
            else:
                # assert self._acceptance_code is not None
                return self._acceptance_action  # type: ignore
        decoded = super().encode(nmi, action)  # type: ignore
        return np.asarray(
            [
                max(
                    self.min_encodable,
                    min(
                        self.max_value,  # type: ignore
                        self.min_encodable
                        + (float(decoded[ACTION]) - self.min_value)  # type: ignore
                        * (self.max_value - self.min_encodable)  # type: ignore
                        / (self.max_value - self.min_value),  # type: ignore
                    ),
                )  # type: ignore
            ],
            dtype=FLOAT_TYPE,
        )


@define
class CUtilityDecoder(UtilityDecoder):
    """Actions are modeled as a tuple of a utility value and a decision (Accept, End, No Response or Reject)."""

    min_value: float = field(init=True, default=0)
    max_value: float = field(init=True, default=1)


@define
class CRelativeUtilityDecoder(RelativeUtilityDecoder):
    code_range: tuple[float, float] = field(init=True, default=(-1, 1))


@define
class CUtilityDecoderBox(_BoxMixin, UtilityDecoder):  # type: ignore
    """Actions are modeled as a tuple of a utility value and a decision (Accept, End, No Response or Reject) in a Box space."""


@define
class CRelativeUtilityDecoderBox(_BoxMixin, RelativeUtilityDecoder):  # type: ignore
    """Actions are modeled as a tuple of a utility value and a decision (Accept, End, No Response or Reject) in a Box space."""

    min_value: float = 0
    max_value: float = 1


@define
class DUtilityDecoder(_DiscreteSamplerMixin, UtilityDecoder):  # type: ignore
    """
    Actions are modeled as a tuple (a value between 0 and `n_levels` - 1 encoding the outcome, and the decision).
    """

    n_levels: int = DEFAULT_UTIL_LEVELS
    start_code: int = field(default=0)


@define
class DRelativeUtilityDecoder(_DiscreteSamplerMixin, RelativeUtilityDecoder):  # type: ignore
    """
    Actions are modeled as a tuple (a value between 0 and `n_levels` - 1 encoding the outcome, and the decision).
    """

    n_levels: int = field(default=DEFAULT_UTIL_LEVELS * 2 + 1)
    start_code: int = field(default=None)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert self.n_levels % 2 == 1, (
            f"n levels must be odd for {self.__class__.__name__}"
        )
        if self.start_code is None:
            self.start_code = -int(self.n_levels // 2) - self._n_decisions + 1


@define
class DUtilityDecoder1D(_D1DMixin, DUtilityDecoder):  # type: ignore
    _min_offer_code: int = field(init=False, default=1)
    _end_code: int = field(init=False, default=-1)
    _no_response_code: int = field(init=False, default=-1)
    _accept_code: int = field(init=False, default=0)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.set_limits()


@define
class DRelativeUtilityDecoder1D(_D1DMixin, DRelativeUtilityDecoder):  # type: ignore
    """
    Actions are changes in utilities in `delta` increments.

    Remarks:
        - `n_levels` must be an ODD number.
        - `n_levels // 2 + 1` means no change. Lower values mean reducing
          utility and higher values mean increasing utility
    """

    n_deltas: int = int(DEFAULT_UTIL_LEVELS // 2)
    delta: float = 0.1
    first_code: int = 0
    n_levels: int = field(init=False, default=None)
    _min_offer_code: int = field(init=False, default=1)
    _end_code: int = field(init=False, default=-1)
    _no_response_code: int = field(init=False, default=-1)
    _accept_code: int = field(init=False, default=0)

    def __attrs_post_init__(self):
        self.n_levels = self.n_deltas * 2 + 1
        super().__attrs_post_init__()

        self.set_limits()


@define
class CUtilityDecoder1D(_C1DMixin, CUtilityDecoder):  # type: ignore
    end_interval: float = 0.08
    no_response_interval: float = 0.1
    accept_interval: float = 0.2
    decision_margin: float = DECISION_MARGIN
    end_limit: float = field(init=False)
    no_response_limit: float = field(init=False)
    accept_limit: float = field(init=False)
    min_encodable: float = field(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.set_limits()


@define
class CRelativeUtilityDecoder1D(_C1DMixin, RelativeUtilityDecoder):  # type: ignore
    end_interval: float = 0.08
    no_response_interval: float = 0.1
    accept_interval: float = 0.2
    decision_margin: float = DECISION_MARGIN
    end_limit: float = field(init=False)
    no_response_limit: float = field(init=False)
    accept_limit: float = field(init=False)
    min_encodable: float = field(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.set_limits()


# type: ignore
class _COutcomeMixin:
    def init(self):
        self.end_limit = self.no_response_limit = self.accept_limit = self.min_value  # type: ignore
        if self.allow_ending:  # type: ignore
            self.end_limit = self.end_interval  # type: ignore
        if self.allow_no_response:  # type: ignore
            self.no_response_limit = self.end_limit + self.no_response_interval  # type: ignore # type: ignore
        else:
            self.no_response_limit = self.end_limit  # type: ignore
        self.accept_limit = self.no_response_limit + self.accept_interval  # type: ignore
        assert self.accept_limit < self.max_value  # type: ignore

    @property
    def accept_val(self) -> float:
        return 0.5 * (self.accept_limit + self.no_response_limit)

    @property
    def reject_val(self) -> float:
        return 0.5 * (1 + self.accept_limit)

    @property
    def end_val(self) -> float:
        return 0.5 * self.end_limit

    @property
    def no_response_val(self) -> float:
        if not self.allow_no_response:  # type: ignore
            return float("nan")
        return 0.5 * (self.no_response_limit + self.end_limit)

    def decode(self, nmi: SAONMI, action: np.ndarray) -> SAOResponse:  # type: ignore
        decision, action = action[DECISION], self.get_outcome(action)  # type: ignore
        if self.allow_ending and decision <= self.end_limit:  # type: ignore
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if (
            self.allow_no_response  # type: ignore
            and self.end_limit < decision <= self.no_response_limit
        ):
            return SAOResponse(ResponseType.NO_RESPONSE, None)
        if decision <= self.accept_limit or self.is_acceptance_action(nmi, action):  # type: ignore
            return SAOResponse(ResponseType.ACCEPT_OFFER, nmi.state.current_offer)
        return super().decode(nmi, (REJECT, action))  # type: ignore

    def encode(  # type: ignore
        self, nmi: SAONMI, action: SAOResponse
    ) -> np.ndarray:
        encoded = super().encode(nmi, action)  # type: ignore
        decision = encoded[DECISION]
        if decision == END:
            x = self.end_val
        elif decision == NORESPONSE:
            x = self.no_response_val
        elif decision == ACCEPT:
            x = self.accept_val
        else:
            x = self.reject_val

        out = self.get_outcome(encoded)  # type: ignore
        out = out * (self.max_value - self.min_value) + self.min_value  # type: ignore
        return np.asarray(
            (x.flatten().tolist() if isinstance(x, np.ndarray) else [x])
            + (out.tolist() if isinstance(out, np.ndarray) else [out]),
            dtype=FLOAT_TYPE,
        )


@define
class COutcomeDecoder(_COutcomeMixin, OutcomeDecoder):  # type: ignore
    """
    Remarks:
        - The decision is encoded in the following ranges:
          - (0, end_limit) => END
          - (end_limit, no_response_limit) => NO RESPONSE
          - (no_response_limit, accept_limit) => ACCEPT
          - (accept_limit, 1.0) => REJECT
    """

    end_interval: float = 0.15
    no_response_interval: float = 0.15
    accept_interval: float = 0.35
    end_limit: float = field(init=False)
    no_response_limit: float = field(init=False)
    accept_limit: float = field(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.init()

    def make_space(self) -> spaces.Box:
        """
        Creates the action space.

        The type of this action space is a tuple with one field
        """
        return spaces.Box(self.min_value, self.max_value, (2,), dtype=FLOAT_TYPE)

    def get_outcome(self, action) -> np.ndarray:
        return action[ACTION]


@define
class COutcomeDecoder1D(COutcomeDecoder):
    """
    Remarks:
        - The decision is encoded in the following ranges:
          - (0, end_limit) => END
          - (end_limit, no_response_limit) => NO RESPONSE
          - (no_response_limit, accept_limit) => ACCEPT
          - (accept_limit, 1.0) => REJECT
    """

    end_interval: float = 0.08
    no_response_interval: float = 0.1
    accept_interval: float = 0.2
    decision_margin: float = DECISION_MARGIN
    end_limit: float = field(init=False)
    no_response_limit: float = field(init=False)
    accept_limit: float = field(init=False)
    min_encodable: float = field(init=False, default=0)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.min_encodable = self.accept_limit + self.decision_margin

    def make_space(self) -> spaces.Box:
        """
        Creates the action space.

        The type of this action space is a tuple with one field
        """
        return spaces.Box(self.min_value, self.max_value, dtype=FLOAT_TYPE)

    def decode(self, nmi: SAONMI, action: np.ndarray) -> SAOResponse:  # type: ignore
        if self.allow_ending and action <= self.end_limit:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if self.allow_no_response and self.end_limit < action <= self.no_response_limit:
            return SAOResponse(ResponseType.NO_RESPONSE, None)
        if action <= self.accept_limit or self.is_acceptance_action(nmi, action):
            return SAOResponse(ResponseType.ACCEPT_OFFER, nmi.state.current_offer)
        # this will apply the un-scaling to all dimensions
        # if not self.separate_decision_dim:
        scaled = float(
            (action - self.min_encodable) / (self.max_value - self.min_encodable)
        )
        return super()._decode(nmi, (REJECT, scaled))

    def encode(  # type: ignore
        self, nmi: SAONMI, action: SAOResponse
    ) -> np.ndarray:
        decoded = super().encode(nmi, action)
        decision, outcome = decoded[DECISION], decoded[ACTION]

        if self.allow_ending and decision <= self.end_limit:
            return np.asarray(self.end_val, dtype=FLOAT_TYPE)
        if self.allow_no_response and decision <= self.no_response_limit:
            return np.asarray(self.no_response_val, dtype=FLOAT_TYPE)
        if (
            self.explicit_accept and decision <= self.accept_limit
        ) or self.is_acceptance_action(nmi, decoded):
            return np.asarray(self.accept_val, dtype=FLOAT_TYPE)

        return np.asarray(
            self.min_encodable
            + decoded[ACTION] * (self.max_value - self.min_encodable),
            dtype=FLOAT_TYPE,
        )


@define
class DOutcomeDecoder(OutcomeDecoder):
    n_outcomes: int = DEFAULT_N_OUTCOMES

    def make_space(self) -> spaces.MultiDiscrete:  # type: ignore
        """Creates the action space"""
        return spaces.MultiDiscrete(
            np.asarray([self._n_decisions, self.n_outcomes + 1]),
            dtype=INT_TYPE,
        )

    def decode(self, nmi: SAONMI, action: np.ndarray) -> SAOResponse:  # type: ignore
        n = self.n_outcomes + 1
        decision, outcome = action[DECISION], action[ACTION]
        return super().decode(nmi, (decision, to_float(int(outcome), n)))

    def encode(  # type: ignore
        self, nmi: SAONMI, action: SAOResponse
    ) -> np.ndarray:
        encoded = super().encode(nmi, action)
        n = self.n_outcomes + 1
        return np.hstack(
            (
                encoded[DECISION],
                np.asarray(from_float(encoded[ACTION], n), dtype=INT_TYPE),
            ),
            dtype=INT_TYPE,
        )


@define
class CIssueDecoder(_COutcomeMixin, IssueDecoder):  # type: ignore
    """

    Remarks:
        - The decision is encoded in the following ranges:
          - (0, end_limit) => END
          - (end_limit, no_response_limit) => NO RESPONSE
          - (no_response_limit, accept_limit) => ACCEPT
          - (accept_limit, 1.0) => REJECT
    """

    end_interval: float = 0.15
    no_response_interval: float = 0.15
    accept_interval: float = 0.35
    end_limit: float = field(init=False)
    no_response_limit: float = field(init=False)
    accept_limit: float = field(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.init()

    def make_space(self) -> spaces.Box:
        """
        Creates the action space.

        The type of this action space is a tuple with one field
        """
        n_dims = self.n_issues + 1
        return spaces.Box(self.min_value, self.max_value, (n_dims,), dtype=FLOAT_TYPE)

    def get_outcome(self, action):
        if isinstance(action, tuple):
            return action[ACTION].flatten()
        return action[ACTION:].flatten()


@define
class DIssueDecoder(IssueDecoder):
    n_levels: tuple[int, ...] = field(default=None)
    n_issues: int = field(init=False, default=0)
    _levels = field(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.n_levels:
            self.n_issues = len(self.n_levels)
            self._levels = np.asarray(self.n_levels, dtype=INT_TYPE)

    def set_outcome_space(self, os: CartesianOutcomeSpace):
        if not self.n_levels:
            self.n_levels = tuple(int(i.cardinality) for i in os.issues)
            self.n_issues = len(self.n_levels)
            self._levels = np.asarray(self.n_levels, dtype=INT_TYPE)
        super().set_outcome_space(os)
        for i, issue in enumerate(self.os.issues):
            if issue.is_continuous() and self._levels[i] == 0:
                raise ValueError(
                    f"Issue {i} ({issue=}) is continuous. You must set n_levels[{i}] to a non-zero value in the action decoder"
                )
            if self._levels[i] == 0:
                self._levels[i] = int(issue.cardinality) + int(self.allow_none)

    def make_space(self) -> spaces.MultiDiscrete:
        """Creates the action space"""
        return spaces.MultiDiscrete(
            np.asarray([self._n_decisions] + self._levels.tolist()),
            dtype=INT_TYPE,
        )

    def decode(self, nmi: SAONMI, action: np.ndarray) -> SAOResponse:  # type: ignore
        return super().decode(
            nmi,
            (
                action[DECISION],
                np.asarray(to_float(action[ACTION:], self._levels), dtype=FLOAT_TYPE),
            ),
        )

    def encode(  # type: ignore
        self, nmi: SAONMI, action: SAOResponse
    ) -> np.ndarray:
        encoded = super().encode(nmi, action)
        return np.asarray(
            ([encoded[DECISION]] + from_float(encoded[ACTION], self._levels).tolist()),
            dtype=INT_TYPE,
        )


@define
class RLBoaDecoder(DRelativeUtilityDecoder1D):
    """The action decoder for RLBOA"""


@define
class CRLBoaDecoder(CRelativeUtilityDecoder1D):
    """The action decoder for RLBOA"""


@define
class SenguptaDecoder(CUtilityDecoder1D):
    """The action decoder for RLBOA"""


@define
class DSenguptaDecoder(DUtilityDecoder1D):
    """The action decoder for Sengupta (discrete)"""


@define
class MiPNDecoder(DIssueDecoder):
    pass


@define
class CMiPNDecoder(CIssueDecoder):
    pass


@define
class VeNASDecoder(DOutcomeDecoder):
    pass


@define
class CVeNASDecoder(COutcomeDecoder):
    pass


DefaultActionDecoder = CUtilityDecoderBox
"""The default action decoder"""
