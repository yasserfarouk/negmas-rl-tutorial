"""
Defines ways to encode and decode observations.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Iterable
from numbers import Real
from typing import Any, TypeVar

import numpy as np
from attr import define, field
from gymnasium import spaces
from gymnasium.spaces.box import Box
from gymnasium.spaces.utils import FlatType, flatten, flatten_space
from negmas import (
    SAONMI,
    MechanismState,
    OutcomeSpace,
    PreferencesChange,
    sort_by_utility,
)
from negmas.common import (
    CartesianOutcomeSpace,
    NegotiatorMechanismInterface,
    PreferencesChange,
)
from negmas.helpers import unique_name
from negmas.negotiators import Negotiator
from negmas.outcomes import Issue, Outcome

from .common import (
    DEFAULT_MIN_ENCODABLE,
    DEFAULT_N_LEVELS,
    DEFAULT_N_OFFERS,
    DEFAULT_N_OUTCOMES,
    FLOAT_TYPE,
    INT_TYPE,
)
from .component import Component
from .utils import enumerate_in_order

__all__ = [
    "ObservationEncoder",
    "DiscreteEncoder",
    "OneDimDiscreteEncoder",
    "ContinuousEncoder",
    "CTimeEncoder",
    "DTimeEncoder",
    "CWindowedUtilityEncoder",
    "DWindowedUtilityEncoder",
    "CUtilityEncoder",
    "DUtilityEncoder",
    "CIssueEncoder",
    "DIssueEncoder",
    "DOutcomeEncoder",
    "DOutcomeEncoderND",
    "DRankEncoder",
    "COutcomeEncoder",
    "CRankEncoder",
    "DOutcomeEncoder1D",
    "DRankEncoder1D",
    "DOutcomeEncoder",
    "DRankEncoder",
    "CWindowedOutcomeEncoder",
    "DWindowedOutcomeEncoder",
    "CWindowedRankEncoder",
    "DWindowedRankEncoder",
    "WindowedIssueEncoder",
    "CWindowedIssueEncoder",
    "DWindowedIssueEncoder",
    "CompositeEncoder",
    "DictEncoder",
    "TupleEncoder",
    "FlatEncoder",
    "BoxEncoder",
    "CTimeUtilityFlatEncoder",
    "CTimeUtilityBoxEncoder",
    "DTimeUtilityFlatEncoder",
    "DTimeUtilityBoxEncoder",
    "CTimeUtilityTupleEncoder",
    "DTimeUtilityTupleEncoder",
    "CTimeUtilityDictEncoder",
    "DTimeUtilityDictEncoder",
    "RLBoaEncoder",
    "SenguptaEncoder",
    "MiPNEncoder",
    "VeNASEncoder",
]

FloatType = TypeVar("FloatType", float, np.ndarray)
MARGIN = 0.8
BEFOREMARGIN = 0.9
NONE_VALID_CODE = -0.9
NONE_NEGATIVE_CODE = -1000.0
# unscale sets values less than min_encodable * BEFOREMARGIN (originally should have been None) to -1. This is the value to test against to check for that


# Function to convert Tuple to Box
def tuple_to_box(tuple_space) -> Box:
    low = np.concatenate(
        [
            t.low if hasattr(t, "low") else np.zeros(t.nvec.shape)
            for t in tuple_space.spaces
        ]
    )
    high = np.concatenate(
        [t.high if hasattr(t, "high") else t.nvec - 1 for t in tuple_space.spaces]
    )
    return Box(low=low, high=high, dtype=np.float32)


# Sample from the combined Box space
def sample_combined_box(samples):
    return np.hstack(samples)


Ranges = tuple[tuple[int, int] | tuple[float, float], ...]


def astuple(x: Iterable | int | float | str) -> tuple:
    if not isinstance(x, Iterable):
        return (x, x)
    return tuple(x)


@define
class ObservationEncoder(Component):
    """Manages the observations of an agent in an RL environment"""

    @abstractmethod
    def make_space(self) -> spaces.Space:
        """Creates the observation space"""
        ...

    @abstractmethod
    def encode(
        self, nmi: NegotiatorMechanismInterface
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Encodes an observation from the negotiators NMI"""

    def decode(
        self,
        nmi: NegotiatorMechanismInterface,
        encoded: np.ndarray | dict[str, np.ndarray],
    ) -> Any:
        """Decoder an observation to whatever encode encodes in it. Only used for testing."""

    def make_first_observation(
        self, nmi: NegotiatorMechanismInterface
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Creates the initial observation (returned from gym's reset())"""
        return self.encode(nmi)


@define
class DiscreteEncoder(ObservationEncoder):
    """
    Base class of observation encoders that generate discrete outcome spaces.

    Args:
        n_levels: Number of levels to use. This is used to quantize continuous values and also to adjust the cardinality for numeric discrete issues

    """

    n_levels: int = DEFAULT_N_LEVELS

    @abstractmethod
    def make_space(self) -> spaces.MultiDiscrete:
        """Creates the observation space (Always a `MultiDiscrete`)"""
        ...


@define
class OneDimDiscreteEncoder(ObservationEncoder):
    """
    Base class of observation encoders that generate discrete outcome spaces.

    Args:
        n_levels: Number of levels to quantize discrete values with

    """

    n_levels: int = DEFAULT_N_LEVELS

    @abstractmethod
    def make_space(self) -> spaces.Discrete:  # type: ignore
        """Creates the observation space (Always a `Discrete`)"""
        ...


@define
class ContinuousEncoder(ObservationEncoder):
    """
    Base class of observation encoders that generate continuous outcome spaces
    """

    @abstractmethod
    def make_space(self) -> spaces.Box:
        """Creates the observation space (Always a `Box`)"""
        ...


@define
class _Scalable(ContinuousEncoder):
    """
    Scales and unscaled to a given range.

    Args:
        min_encodable: Minimum value for outputs. If `encode_none`, we will
                       always add none_margin to this value for "real"
                       values keeping this value for none
        max_encodable: Maximum value for all outputs.
        encode_none: If given, `min_encodable` will be used to model None and a
                     margin of none_margin will then be added making the
                     minimum encoded value for "real" values (that are not None)
                     min_encodable + none_margin
        none_margin: The margin to use for None value (only used if encode_none is True)
    """

    min_encodable: float = 0
    max_encodable: float = 1
    encode_none: bool = True
    none_margin: float = DEFAULT_MIN_ENCODABLE
    _none_value: float = field(init=False, default=0)

    def __attrs_post_init__(self):
        self._none_value = self.min_encodable
        if self.encode_none and abs(self.min_encodable) < 1e-3:
            self.min_encodable += self.none_margin
        assert (
            self.min_encodable < self.max_encodable
            and self._none_value <= self.min_encodable
        )

    def scale(self, x: FloatType | None) -> FloatType:
        """Scales a value of an issue to range between min and max encodable"""
        if x is None:
            assert self.encode_none
            return self._none_value  # type: ignore
        return np.asarray(
            self.min_encodable + (self.max_encodable - self.min_encodable) * x,
            dtype=FLOAT_TYPE,
        )  # type: ignore

    def unscale(self, x: FloatType) -> np.ndarray:
        """Inverts the scale() call. If non_encoded, assumes that x is already scaled to include None"""
        y = np.asarray(x)
        y[y < self.min_encodable * BEFOREMARGIN] = NONE_NEGATIVE_CODE
        y[y >= self.min_encodable * BEFOREMARGIN] = (
            y[y >= self.min_encodable * BEFOREMARGIN] - self.min_encodable
        ) / (self.max_encodable - self.min_encodable)
        return y.astype(dtype=FLOAT_TYPE)


@define
class CTimeEncoder(_Scalable):
    """Encodes relative time only."""

    encode_none: bool = False

    def make_space(self) -> spaces.Box:
        """Creates the observation space"""
        return spaces.Box(self._none_value, self.max_encodable, (1,), dtype=FLOAT_TYPE)

    def encode(self, nmi: NegotiatorMechanismInterface) -> np.ndarray:
        """Encodes an observation from the agent's nmi"""
        return np.asarray([self.scale(nmi.state.relative_time)], dtype=FLOAT_TYPE)

    def decode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, encoded: np.ndarray
    ) -> float | None:
        """Returns relative time"""
        x = self.unscale(encoded)
        return float(x) if x >= NONE_VALID_CODE else None


@define
class DTimeEncoder(DiscreteEncoder):
    def make_space(self) -> spaces.Discrete:  # type: ignore
        """Creates the observation space"""
        return spaces.Discrete(self.n_levels)

    def encode(self, nmi: NegotiatorMechanismInterface) -> np.ndarray:
        """Encodes an observation from the agent's nmi"""
        state = nmi.state
        t = min(self.n_levels - 1, int(state.relative_time * self.n_levels))
        return np.asarray(t, dtype=INT_TYPE)

    def decode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, encoded: np.ndarray
    ) -> float:
        return float(np.minimum(1.0, encoded / self.n_levels))


@define
class _ScaledUtilEncoder(_Scalable):
    _min_util: float = field(init=False, default=0)
    _max_util: float = field(init=False, default=1)

    def _set_ufunlimit(self):
        assert self.owner and self.owner.ufun
        self._min_util, self._max_util = self.owner.ufun.minmax()

    def on_negotiation_starts(
        self, owner: Negotiator, nmi: NegotiatorMechanismInterface
    ) -> None:
        super().on_negotiation_starts(owner, nmi)
        self._set_ufunlimit()

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        super().on_preferences_changed(changes)
        self._set_ufunlimit()

    def scale(self, x: FloatType | None) -> FloatType:
        """Scales a utility value to be between min_encodable and max_encodable"""
        if x is None:
            return super().scale(x)
        return super().scale((x - self._min_util) / (self._max_util - self._min_util))  # type: ignore

    def unscale(self, x: FloatType) -> np.ndarray:
        """unscale(scale(None)) will return -1"""
        if isinstance(x, Real):
            if x < self.min_encodable * BEFOREMARGIN:
                return np.asarray(NONE_NEGATIVE_CODE, dtype=FLOAT_TYPE)
            return np.asarray(
                super().unscale(x) * (self._max_util - self._min_util) + self._min_util,
                dtype=FLOAT_TYPE,
            )

        assert isinstance(x, np.ndarray), f"{x=}, {type(x)=} but we expect np.ndarray"
        x[x < BEFOREMARGIN * self.min_encodable] = NONE_NEGATIVE_CODE
        x[x >= BEFOREMARGIN * self.min_encodable] = (
            (
                (x[x >= BEFOREMARGIN * self.min_encodable] - self.min_encodable)
                / (self.max_encodable - self.min_encodable)
            )
            * (self._max_util - self._min_util)
            + self._min_util  # type: ignore
        )
        return x


@define
class CWindowedUtilityEncoder(_ScaledUtilEncoder):
    """Encodes a window of utility values.

    Args:
        n_offers: window length
        missing_as_none: If given the reserved value will be used for missing values otherwise 0 (which indicates None).
        partner_utility: Encoder partner utility? (requires owner.ufun.opponent_ufun to be set).
        min_encodable: The value corresponding to minimum utility
        max_encodable: The value corresponding to maximum utility
        ignore_own_offers: If true, only encoder partner offers
        encode_none: If true, reserve some range (0 -> min_encodable) to encode None (or missing values)

    Remarks:
        - If missing_as_none is given and no min_encodable is given, min_encodable will be set to none_margin
        - Last offer is encoded first (i.e. we reverse the history before encoding it)

    """

    n_offers: int = DEFAULT_N_OFFERS
    partner_utility: bool = False
    ignore_own_offers: bool = False
    missing_as_none: bool = False
    flat: bool = True

    def __attrs_post_init__(self):
        if self.missing_as_none:
            self.encode_none = True
        super().__attrs_post_init__()

    def make_space(self) -> spaces.Box:
        """Creates the observation space"""
        return spaces.Box(
            self._none_value,
            self.max_encodable,
            (self.n_offers * (1 + int(self.partner_utility)),)
            if self.flat
            else (
                self.n_offers,
                (1 + int(self.partner_utility)),
            ),
            dtype=FLOAT_TYPE,
        )

    def encode(
        self,
        nmi: NegotiatorMechanismInterface,  # type: ignore
    ) -> np.ndarray:
        """Encodes an observation from the agent's nmi"""
        if not hasattr(nmi, "trace"):
            raise ValueError("Only SAONMI is supported")
        nmi: SAONMI = nmi  # type: ignore
        trace = nmi.trace
        trace.reverse()
        assert self.owner
        assert self.owner.ufun
        myid = self.owner.id
        offers = []
        for sender, outcome in trace:
            if not self.ignore_own_offers or (sender != myid):
                offers.append(outcome)
            if len(offers) == self.n_offers:
                break

        n_window_dims = 1 + int(self.partner_utility)
        history = (
            (
                np.ones(n_window_dims * self.n_offers) * self._none_value
                if self.missing_as_none
                else (
                    np.ones(n_window_dims * self.n_offers)
                    * self.scale(float(self.owner.ufun.reserved_value))
                )
            )
            if not self.partner_utility
            else np.ones(n_window_dims * self.n_offers) * self._none_value
            if self.missing_as_none
            else np.hstack(
                [
                    np.asarray(  # type: ignore
                        [
                            self.scale(float(self.owner.ufun.reserved_value)),
                            self.scale(
                                float(self.owner.opponent_ufun.reserved_value)  # type: ignore
                            ),
                        ],
                        dtype=FLOAT_TYPE,
                    )
                ]
                * self.n_offers,
            ).flatten()
        )
        if self.partner_utility:
            assert self.owner.opponent_ufun is not None
            # make last offer appear first
            for i, offer in enumerate(offers):
                j = i * 2
                history[j] = self.scale(float(self.owner.ufun(offer)))
                history[j + 1] = self.scale(float(self.owner.opponent_ufun(offer)))
                # first_vals = history[j : j + 2]
        else:
            for i, offer in enumerate(offers):
                history[i] = self.scale(float(self.owner.ufun(offer)))
        if self.missing_as_none:
            history[np.isnan(history)] = self._none_value
            history[np.isinf(history)] = self._none_value
        return (
            history.astype(FLOAT_TYPE)
            if self.flat
            else history.astype(FLOAT_TYPE).reshape((self.n_offers, -1))
        )

    def decode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, encoded: np.ndarray
    ) -> list[float]:
        x = self.unscale(encoded)
        assert isinstance(x, np.ndarray)
        utils = []
        for v in x:
            if v < NONE_VALID_CODE:
                continue
            utils.append(v)
        return utils


@define
class DWindowedUtilityEncoder(CWindowedUtilityEncoder):
    n_levels: int = DEFAULT_N_LEVELS

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.min_encodable > 1e-5:
            self.min_encodable = max(self.min_encodable, 1.0 / self.n_levels)
            assert self.min_encodable < self.max_encodable

    def make_space(self) -> spaces.MultiDiscrete:  # type: ignore
        """Creates the observation space"""
        return spaces.MultiDiscrete(
            np.asarray(
                [self.n_levels + int(self.encode_none)]
                * (self.n_offers * (1 + int(self.partner_utility))),
                INT_TYPE,
            )
        )

    def encode(self, nmi: NegotiatorMechanismInterface) -> np.ndarray:
        """Encodes an observation from the agent's nmi"""
        x = super().encode(nmi).flatten()
        levels = self.n_levels - 1 if self.encode_none else self.n_levels
        if self.encode_none:
            x[x < MARGIN * self.min_encodable] = 0
            x[x >= BEFOREMARGIN * self.min_encodable] = np.minimum(
                levels,
                np.round(
                    1 + levels * self.unscale(x[x >= BEFOREMARGIN * self.min_encodable])
                ).astype(INT_TYPE),
            )
        else:
            x = np.minimum(levels, levels * self.unscale(x))
        return x.astype(INT_TYPE)

    def decode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, encoded: np.ndarray
    ) -> list[float]:
        r = float(self.owner.ufun.reserved_value)  # type: ignore
        assert isinstance(encoded, np.ndarray)
        y = self.scale(encoded / self.n_levels)
        assert isinstance(y, np.ndarray)
        if self.encode_none:
            y[encoded == 0] = self._none_value
        return super().decode(nmi, y)


@define
class CUtilityEncoder(_ScaledUtilEncoder):
    partner_utility: bool = False

    def encode(self, nmi: NegotiatorMechanismInterface) -> np.ndarray:
        """Encodes an observation from the agent's nmi"""
        assert self.owner and self.owner.ufun
        state = nmi.state
        offer = state.current_offer  # type: ignore
        if self.partner_utility:
            assert self.owner.opponent_ufun
            u = [
                self.scale(float(self.owner.ufun(offer))),
                self.scale(float(self.owner.opponent_ufun(offer))),
            ]
        else:
            u = [self.scale(float(self.owner.ufun(offer)))]
        return np.asarray(u, FLOAT_TYPE)

    def decode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, encoded: np.ndarray
    ) -> tuple[float] | tuple[float, float]:
        v = self.unscale(encoded)
        if isinstance(v, float) or isinstance(v, int):
            assert not self.partner_utility
            v = [float(v)]
        utils = []
        if v[0] > NONE_VALID_CODE:
            utils.append(v[0])
        if self.partner_utility:
            assert len(v) > 1
            if v[1] > NONE_VALID_CODE:
                utils.append(v[1])
        return tuple(utils)

    def make_space(self) -> spaces.Box:
        """Creates the observation space"""
        return spaces.Box(
            self._none_value,
            self.max_encodable,
            (1 + int(self.partner_utility),),
            dtype=FLOAT_TYPE,
        )


@define
class DUtilityEncoder(CUtilityEncoder):
    n_levels: int = DEFAULT_N_LEVELS

    def make_space(self) -> spaces.MultiDiscrete | spaces.Discrete:  # type: ignore
        """Creates the observation space"""
        if self.partner_utility:
            return spaces.MultiDiscrete(
                np.asarray([self.n_levels, self.n_levels], dtype=INT_TYPE)
            )
        else:
            return spaces.Discrete(self.n_levels)

    def encode(self, nmi: NegotiatorMechanismInterface) -> np.ndarray:
        """Encodes an observation from the agent's nmi"""
        encoded = super().encode(nmi)
        if np.all(encoded < MARGIN * self.min_encodable) and self.encode_none:
            return (
                np.ones(2, dtype=INT_TYPE) * self._none_value
                if self.partner_utility
                else np.asarray(self._none_value, dtype=INT_TYPE)
            )
        if not self.partner_utility:
            encoded = float(encoded)
        return np.asarray(
            np.minimum(
                self.n_levels - 1,
                np.round(self.n_levels * self.unscale(encoded)).astype(INT_TYPE),  # type: ignore
            ).astype(INT_TYPE)
        )

    def decode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, encoded: np.ndarray
    ) -> tuple[float] | tuple[float, float]:
        x = encoded / self.n_levels
        x[encoded == 0] = self._none_value
        x[encoded > 0] = self.scale(x[encoded > 0])
        return super().decode(nmi, x)


@define
class COutcomeEncoder(_Scalable):
    """Encodes each offer as its outcome index in the outcome-space (scaled as needed)

    Args:
        n_outcomes: number of outcomes to encode into (the outcome  space need not have this exact number)
        os: The outcome-space to use. If not given, the ufun will be used
        encode_unknown_as_none: Encode any unkonwn outcome as None (zero)
        sort_by_utility: Sort outcomes by utility before indexing them
        best_first: Make the best outcome have lowest index (used only if sort_by_utility is given)

    Remarks:
        - This will work for any outcome-space independent of the value of n_outcomes
    """

    os: OutcomeSpace | None = None
    sort_by_utility: bool = False
    best_first: bool = False
    order_by_similarity: bool = True
    _outcome_map: dict[Outcome | None, int] = field(init=False, factory=dict)
    _outcomes: list[Outcome | None] = field(init=False, factory=list)

    def make_space(self) -> spaces.Box:
        """Creates the observation space"""
        # 0 encodes None
        return spaces.Box(self._none_value, self.max_encodable, (1,))

    def encode(self, nmi: NegotiatorMechanismInterface) -> np.ndarray:
        """Encodes an observation from the agent's nmi"""
        offer = nmi.state.current_offer  # type: ignore
        if offer is None:
            return np.asarray([0.0], dtype=FLOAT_TYPE)

        if not self.os:
            self._update_os()
        assert self.os is not None
        n_real_outcomes = len(self._outcome_map) - 1
        return np.asarray(
            [self.scale((self._outcome_map[offer] - 1) / (n_real_outcomes - 1))],
            dtype=FLOAT_TYPE,
        )

    def decode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, encoded: np.ndarray
    ) -> Outcome | None:
        n = len(self._outcome_map) - 1
        indx = (
            min(n, 1 + int(self.unscale(encoded) * n))
            if encoded > BEFOREMARGIN * self.min_encodable
            else 0
        )
        return self._outcomes[indx]

    def on_negotiation_starts(
        self, owner: Negotiator, nmi: NegotiatorMechanismInterface
    ) -> None:
        super().on_negotiation_starts(owner, nmi)
        self._update_os()

    def _update_os(self):
        assert self.owner
        if self.owner.preferences:
            os = self.owner.preferences.outcome_space
        else:
            os = self.owner.nmi.outcome_space
        assert os is not None, "No outcome space"
        self.os = os
        if self.order_by_similarity:
            outcomes = enumerate_in_order(os.issues, n_levels=DEFAULT_N_LEVELS)  # type: ignore
        else:
            outcomes = (
                os.enumerate_or_sample(DEFAULT_N_LEVELS)
                if not os.is_discrete()
                else os.enumerate()  # type: ignore
            )
        if self.sort_by_utility:
            assert self.owner.ufun
            _, outcomes = sort_by_utility(  # type: ignore
                self.owner.ufun,
                outcomes=outcomes,  # type: ignore
                best_first=self.best_first,
            )
        outcomes: list[Outcome | None]
        self._outcome_map = dict(
            zip([None] + outcomes, list(range(len(outcomes) + 1)), strict=True)
        )
        self._outcomes = [None] + outcomes

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        if self.owner:
            self._update_os()


@define
class DOutcomeEncoder(COutcomeEncoder):
    """Encodes each offer as its outcome index in the outcome-space (scaled as needed)

    Args:
        n_outcomes: number of outcomes to encode into (the outcome  space need not have this exact number)
        os: The outcome-space to use. If not given, the ufun will be used
        encode_unknown_as_none: Encode any unkonwn outcome as None (zero)
        sort_by_utility: Sort outcomes by utility before indexing them
        best_first: Make the best outcome have lowest index (used only if sort_by_utility is given)

    Remarks:
        - This will work for any outcome-space independent of the value of n_outcomes
    """

    n_outcomes: int = DEFAULT_N_OUTCOMES

    def make_space(self) -> spaces.Discrete:  # type: ignore
        """Creates the observation space"""
        # 0 encodes None
        return spaces.Discrete(self.n_outcomes + 1)

    def encode(self, nmi: NegotiatorMechanismInterface) -> np.ndarray:
        """Encodes an observation from the agent's nmi"""
        encoded = super().encode(nmi)
        n_outcomes = self.n_outcomes

        if self.encode_none and encoded < BEFOREMARGIN * self.min_encodable:
            return np.asarray(self._none_value, dtype=INT_TYPE)
        return np.asarray(
            int(
                np.minimum(
                    n_outcomes,
                    np.round(1 + n_outcomes * self.unscale(encoded)).astype(INT_TYPE),
                )
            ),
            dtype=INT_TYPE,
        )

    def decode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, encoded: np.ndarray
    ) -> Outcome | None:
        if np.all(encoded == 0):
            return None
        n_real_outcoms = len(self._outcome_map) - 1
        indx = min(
            n_real_outcoms - 1,
            int(0.5 + encoded * (n_real_outcoms - 1) / (self.n_outcomes - 1)),
        )
        return self._outcomes[indx]


@define
class CRankEncoder(COutcomeEncoder):
    sort_by_utility: bool = True


@define
class DRankEncoder(DOutcomeEncoder):
    sort_by_utility: bool = True


@define
class DOutcomeEncoder1D(DOutcomeEncoder):
    pass


@define
class DRankEncoder1D(DOutcomeEncoder1D):
    sort_by_utility: bool = True


@define
class DOutcomeEncoderND(DOutcomeEncoder):
    def make_space(self) -> spaces.MultyDiscrete:  # type: ignore
        return spaces.MultiDiscrete([self.n_outcomes + 1])

    def encode(self, nmi: NegotiatorMechanismInterface) -> np.ndarray:
        encoded = super().encode(nmi)
        return encoded.reshape((1,))

    def decode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, encoded: np.ndarray
    ) -> Outcome | None:
        return super().decode(nmi, encoded[0])


@define
class CWindowedOutcomeEncoder(COutcomeEncoder):
    n_offers: int = 1
    ignore_own_offers: bool = False

    def make_space(self) -> spaces.Box:  # type: ignore
        """Creates the observation space"""
        # 0 encodes None
        return spaces.Box(self._none_value, self.max_encodable, (self.n_offers,))

    def encode(self, nmi: NegotiatorMechanismInterface) -> np.ndarray:
        """Encodes an observation from the agent's nmi"""
        if not hasattr(nmi, "trace"):
            raise ValueError("Only SAONMI is supported")
        trace = nmi.trace  # type: ignore
        trace.reverse()
        assert self.owner
        assert self.owner.ufun
        myid = self.owner.id
        offers = []
        for i, (sender, outcome) in enumerate(trace):
            if not self.ignore_own_offers or (sender != myid):
                offers.append(outcome)
            if len(offers) >= self.n_offers:
                break

        # [self.scale((self._outcome_map[offer] - 1) / (n_real_outcomes - 1))],
        history = np.ones(self.n_offers, dtype=FLOAT_TYPE) * self._none_value
        n_real = len(self._outcome_map)
        for i, offer in enumerate(offers):
            history[i] = self.scale((self._outcome_map[offer] - 1) / (n_real - 1))
        return history.astype(FLOAT_TYPE)

    def decode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, encoded: np.ndarray
    ) -> list[Outcome | None]:
        x = self.unscale(encoded)
        if not isinstance(x, Iterable):
            x = [x]
        n = len(self._outcome_map) - 1

        offers = []
        for v in x:
            offers.append(
                self._outcomes[min(n, 1 + int(v * n)) if v >= NONE_VALID_CODE else 0]
            )
        return offers


@define
class DWindowedOutcomeEncoder(CWindowedOutcomeEncoder):
    n_outcomes: int = DEFAULT_N_OUTCOMES

    def make_space(self) -> spaces.MultiDiscrete:  # type: ignore
        """Creates the observation space"""
        return spaces.MultiDiscrete(
            np.asarray([self.n_outcomes + 1] * self.n_offers, dtype=INT_TYPE)
        )

    def encode(self, nmi: NegotiatorMechanismInterface) -> np.ndarray:
        encoded = super().encode(nmi)
        x = np.asarray(self.unscale(encoded))
        x[x > NONE_VALID_CODE] = np.minimum(
            self.n_outcomes,
            np.round(1 + self.n_outcomes * x[x > NONE_VALID_CODE]).astype(INT_TYPE),
        )
        x[encoded <= NONE_VALID_CODE] = 0

        return x.astype(dtype=INT_TYPE)

    def decode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, encoded: np.ndarray
    ) -> list[Outcome | None]:
        x = np.asarray(self.scale((encoded - 1) / self.n_outcomes))
        x[encoded < BEFOREMARGIN * self.min_encodable] = self._none_value
        return super().decode(nmi, x)


@define
class CWindowedRankEncoder(CWindowedOutcomeEncoder):
    sort_by_utility: bool = True


@define
class DWindowedRankEncoder(DWindowedOutcomeEncoder):
    sort_by_utility: bool = True


@define
class CIssueEncoder(_Scalable):
    """
    Encodes the current offer as issue values (scalable raw-outcome encoding)

    Args:
        n_issues: Number of issues
        os: Outcome-space. If not given, it will be read directly from the ufun
        encode_none: Encoder None as zero
        min_encodable: Minimum non-None value
        max_encodable: Maximum value

    Remarks:
        - Only works if the number of issues in the outcome-space of the negotiation equals to n_issues
    """

    n_issues: int = 1
    os: CartesianOutcomeSpace | None = None
    _val_map: dict[str, dict[int | str | tuple, int]] = field(init=False, factory=dict)
    _none_value: float = field(init=False, default=0)

    def normalize(self, v, issue: Issue) -> float:
        """Normalizes a value of an issue to range between 0 and 1"""
        if not issue.is_numeric():
            return max(
                0.0, min(1.0, self._val_map[issue.name][v] / (issue.cardinality - 1))
            )
        return max(
            0.0,
            min(
                1.0,
                (v - issue.min_value) / (issue.max_value - issue.min_value),
            ),
        )

    def denormalize(
        self, v, issue: Issue, always_numeric: bool = False
    ) -> int | float | str:
        """Returns the value of the issue that is normalized as v (inverts normalize())"""
        if not issue.is_numeric():
            v = int(v * (issue.cardinality - 1) + 0.5)
            if always_numeric:
                return v
            return issue.values[v]
        x = max(
            issue.min_value,
            min(
                issue.max_value,
                v * (issue.max_value - issue.min_value) + issue.min_value,
            ),
        )
        if issue.is_continuous():
            return x
        return int(0.5 + x)

    def make_space(self) -> spaces.Box:
        """Creates the observation space"""
        n_issues = self.n_issues
        return spaces.Box(
            self._none_value, self.max_encodable, (self.n_issues,), dtype=FLOAT_TYPE
        )

    def encode(self, nmi: NegotiatorMechanismInterface) -> np.ndarray:
        """Encodes an observation from the agent's nmi"""
        n_issues = self.n_issues
        state = nmi.state
        offer = state.current_offer  # type: ignore
        if offer is None:
            return np.zeros(n_issues, dtype=FLOAT_TYPE)

        if not self.os:
            self._update_os()
        assert self.os is not None
        return np.asarray(
            [
                self.scale(self.normalize(v, issue))
                for v, issue in zip(offer, self.os.issues, strict=True)
            ],
            dtype=FLOAT_TYPE,
        )

    def decode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, encoded: np.ndarray
    ) -> Outcome | None:
        issues: list[Issue] = self.os.issues  # type: ignore
        if np.all(encoded < self.min_encodable * BEFOREMARGIN):
            return None
        return tuple(
            self.denormalize(self.unscale(v), issue)
            for v, issue in zip(encoded, issues, strict=True)
        )

    def on_negotiation_starts(
        self, owner: Negotiator, nmi: NegotiatorMechanismInterface
    ) -> None:
        super().on_negotiation_starts(owner, nmi)
        self._update_os()

    def _update_os(self):
        assert self.owner
        if self.owner.preferences:
            os = self.owner.preferences.outcome_space
        else:
            os = self.owner.nmi.outcome_space

        assert isinstance(os, CartesianOutcomeSpace), (
            f"{type(os)} is not CartesianOutcomeSpace"
        )

        assert os is not None and len(os.issues) == self.n_issues, (
            f"{self.n_issues=}, {len(os.issues)=}\n{os=} "
        )
        self.os = os
        for issue in self.os.issues:
            if issue.is_numeric():
                continue
            self._val_map[issue.name] = dict(zip(issue.all, range(issue.cardinality)))  # type: ignore

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        if self.owner:
            self._update_os()


@define
class DIssueEncoder(CIssueEncoder):
    """Encodes issue values as discrete numbers

    Args:
        n_levels: number of levels to use for all issues (must be passed in conjunction with n_issues)
        n_levels_per_issue: a separate number of levels for each issue (no need to pass n_issues)

    Remarks:
        - n_levels_per_issue need not match the number of values for each issue. If they do not, we will rescale
        - We always add one level to represent None so that all zeros is None
    """

    n_levels_per_issue: tuple[int, ...] = field(default=None)
    exact: bool = False
    n_levels: int | None = DEFAULT_N_LEVELS
    _levels: np.ndarray = field(init=False)
    _os_levels: np.ndarray = field(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.n_levels_per_issue is not None:
            self.n_issues = len(self.n_levels_per_issue)
        else:
            assert self.n_levels is not None, (
                "You are passing n_levels and n_levels_per_issue as None. One of them must be passed"
            )
            self.n_levels_per_issue = tuple([self.n_levels] * self.n_issues)
        assert self.n_levels_per_issue is not None
        self._levels = np.asarray(self.n_levels_per_issue, dtype=INT_TYPE)

    def make_space(self) -> spaces.MultiDiscrete:  # type: ignore
        """Creates the observation space"""
        # zero will encode None
        return spaces.MultiDiscrete(self._levels + 1)

    def encode(self, nmi: NegotiatorMechanismInterface) -> np.ndarray:
        encoded = super().encode(nmi).flatten()
        if np.all(encoded < MARGIN * self.min_encodable) and self.encode_none:
            return np.zeros(self.n_issues, dtype=INT_TYPE)
        levels = self._levels
        x = self.unscale(encoded)
        x = np.minimum(levels, np.floor(1 + levels * x).astype(INT_TYPE))
        x[x < NONE_VALID_CODE] = 0
        return x.astype(INT_TYPE)

    def decode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, encoded: np.ndarray
    ) -> Outcome | None:
        x = (encoded - 1) / (self._levels - 1)
        x = self.scale(x)
        x[encoded < BEFOREMARGIN * self.min_encodable] = 0
        return super().decode(nmi, x)

    def _update_os(self):
        super()._update_os()
        assert self.os is not None
        self._os_levels = np.asarray(
            [i.cardinality for i in self.os.issues], dtype=INT_TYPE
        )
        if self.exact and np.any(self._levels != self._os_levels):
            raise ValueError(
                f"You are using an exact issue encoder with levels {self._levels} but the outcome space has issues with cardinalities {self._os_levels}"
            )


@define
class CWindowedIssueEncoder(CIssueEncoder):
    n_offers: int = 1
    ignore_own_offers: bool = False
    flat: bool = True

    def make_space(self) -> spaces.Box:
        """Creates the observation space"""
        n_issues = self.n_issues
        return spaces.Box(
            self._none_value,
            self.max_encodable,
            (self.n_issues * self.n_offers,)
            if self.flat
            else (self.n_issues, self.n_offers),
            dtype=FLOAT_TYPE,
        )

    def encode(self, nmi: NegotiatorMechanismInterface) -> np.ndarray:
        """Encodes an observation from the agent's nmi"""
        if not hasattr(nmi, "trace"):
            raise ValueError("Only SAONMI is supported")
        trace = nmi.trace  # type: ignore
        trace.reverse()
        assert self.owner
        assert self.owner.ufun
        myid = self.owner.id
        offers = []
        for i, (sender, outcome) in enumerate(trace):
            if not self.ignore_own_offers or (sender != myid):
                offers.append(outcome)
            if len(offers) >= self.n_offers:
                break

        n_issues = self.n_issues
        issues = self.owner.ufun.outcome_space.issues  # type: ignore
        history = np.ones(n_issues * self.n_offers) * self._none_value
        for i, offer in enumerate(offers):
            j = i * n_issues
            history[j : j + n_issues] = np.asarray(
                [
                    self.scale(self.normalize(v, issue))
                    for v, issue in zip(offer, issues, strict=True)
                ]
            )
        return (
            history.astype(FLOAT_TYPE).flatten()
            if self.flat
            else history.astype(FLOAT_TYPE)
        )

    def decode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, encoded: np.ndarray
    ) -> list[Outcome | None]:
        encoded = encoded.flatten()
        issues: list[Issue] = self.os.issues  # type: ignore
        if np.all(encoded < self.min_encodable * BEFOREMARGIN):
            return [None] * self.n_offers
        n = self.n_issues
        return [
            tuple(
                self.denormalize(self.unscale(v), issue)
                for v, issue in zip(encoded[i * n : (i + 1) * n], issues, strict=True)
            )
            if not np.all(
                encoded[i * n : (i + 1) * n] < self.min_encodable * BEFOREMARGIN
            )
            else None
            for i in range(self.n_offers)
        ]


@define
class DWindowedIssueEncoder(CWindowedIssueEncoder):
    n_levels: tuple[int, ...] = (DEFAULT_N_LEVELS,)
    n_issues: int = field(init=False, default=1)
    _levels: np.ndarray = field(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.n_issues = len(self.n_levels)
        self._levels = np.asarray([list(self.n_levels)] * self.n_offers).flatten()

    def make_space(self) -> spaces.MultiDiscrete:  # type: ignore
        """Creates the observation space"""
        return spaces.MultiDiscrete(
            np.asarray(
                [[_ + 1 for _ in self.n_levels] for _ in range(self.n_offers)],
                dtype=INT_TYPE,
            ).flatten()
        )

    def indexofvalue(self, v, issue: Issue, issue_index: int) -> int:
        """J is the index of issue in the outcome space issues"""
        N = self.n_levels[issue_index] - 1  # we have one level for the decision
        if issue.is_numeric():
            return max(
                0,
                min(
                    N - 1,
                    int(
                        N * (v - issue.min_value) / (issue.max_value - issue.min_value)
                    ),
                ),
            )
        # assert (
        #     issue.cardinality == N + 1
        # ), f"Issue {issue_index} ({issue=}) has cardinality {issue.cardinality} but the number of values expected by the model is {N}"
        vals = list(issue.values)
        return max(
            0, min(N - 1, int(0.5 + vals.index(v) * (N - 1) / (issue.cardinality - 1)))
        )

    def valueofindex(
        self, indx: int, issue: Issue, issue_index: int
    ) -> int | float | str | None:
        """J is the index of issue in the outcome space issues"""
        if indx == 0:
            return None
        indx -= 1
        if issue.is_numeric():
            return indx * (issue.max_value - issue.min_value) + issue.min_value
        N = issue.cardinality - 1
        indx = int(
            max(
                0,
                min(
                    N - 1,
                    int(0.5 + indx * (N - 1) / (self.n_levels[issue_index] - 2)),
                ),
            )
        )

        return issue.values[indx]

    def encode(self, nmi: NegotiatorMechanismInterface) -> np.ndarray:
        encoded = super().encode(nmi).flatten()
        levels = self._levels
        x = np.asarray(self.unscale(encoded))
        x = np.minimum(levels, np.round(1 + levels * x).astype(INT_TYPE))
        x[x < NONE_VALID_CODE] = 0
        return x.astype(INT_TYPE)
        # offers = [
        #     encoded[i * self.n_issues : (i + 1) * self.n_issues]
        #     for i in range(self.n_offers)
        # ]
        # discretized = [
        #     [
        #         min(level - 2, int(0.5 + self.unscale(v) * (level - 1)))
        #         for level, v in zip(self.n_levels, offer)
        #     ]
        #     for offer in offers
        # ]
        #
        # return np.asarray(list(itertools.chain(*discretized)), dtype=INT_TYPE)

    def decode(  # type: ignore
        self, nmi: NegotiatorMechanismInterface, encoded: np.ndarray
    ) -> list[Outcome | None]:
        return super().decode(nmi, encoded / self._levels)


@define
class CompositeEncoder(ObservationEncoder):
    children: tuple[ObservationEncoder, ...] = tuple()
    names: tuple[str, ...] = tuple()

    def __attrs_post_init__(self):
        if not self.names:
            self.names = tuple(_.__class__.__name__ for _ in self.children)
            if len(set(self.names)) != len(self.names):
                self.names = tuple(
                    unique_name(_.__class__.__name__, add_time=False, sep="")
                    for _ in self.children
                )

    def on_negotiation_starts(
        self, owner: Negotiator, nmi: NegotiatorMechanismInterface
    ) -> None:
        super().on_negotiation_starts(owner, nmi)

        for child in self.children:
            child.on_negotiation_starts(owner, nmi)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        super().on_preferences_changed(changes)
        for child in self.children:
            child.on_preferences_changed(changes)

    def on_negotiation_ends(
        self, owner: Negotiator, nmi: NegotiatorMechanismInterface
    ) -> None:
        super().on_negotiation_ends(owner, nmi)

        for child in self.children:
            child.on_negotiation_ends(owner, nmi)

    def after_partner_action(
        self,
        partner_id: str,
        state: MechanismState,
        action: Any,
    ) -> None:
        for child in self.children:
            child.after_partner_action(partner_id, state, action)

    def after_learner_actions(
        self,
        states: dict[str, MechanismState],
        actions: dict[str, Any],
        encoded_action: dict[str, np.ndarray],
    ) -> None:
        for child in self.children:
            child.after_learner_actions(states, actions, encoded_action)


@define
class DictEncoder(CompositeEncoder):
    def make_space(self) -> spaces.Dict:
        """Creates the observation space"""
        assert len(self.children) > 0
        return spaces.Dict(
            dict(zip(self.names, [_.make_space() for _ in self.children], strict=True))
        )

    def encode(
        self, nmi: NegotiatorMechanismInterface
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Encodes an observation from the agent's nmi"""
        assert len(self.children) > 0
        return OrderedDict(
            zip(self.names, [_.encode(nmi) for _ in self.children], strict=True)
        )  # type: ignore

    def decode(
        self,
        nmi: NegotiatorMechanismInterface,
        encoded: np.ndarray | dict[str, np.ndarray],
    ) -> dict:
        return dict(
            zip(
                self.names,
                [
                    c.decode(nmi, encoded[n])
                    for n, c in zip(self.names, self.children, strict=True)
                ],
                strict=True,
            )
        )


@define
class TupleEncoder(CompositeEncoder):
    def make_space(self) -> spaces.Tuple:
        """Creates the observation space"""
        assert len(self.children) > 0
        return spaces.Tuple(tuple(_.make_space() for _ in self.children))

    def encode(self, nmi: NegotiatorMechanismInterface) -> tuple:  # type: ignore
        """Encodes an observation from the agent's nmi"""
        assert len(self.children) > 0
        return tuple(_.encode(nmi) for _ in self.children)

    def decode(
        self,
        nmi: NegotiatorMechanismInterface,
        encoded: np.ndarray | dict[str, np.ndarray],
    ) -> tuple:
        return tuple(
            c.decode(nmi, encoded[n])
            for n, c in zip(self.names, self.children, strict=True)
        )


@define
class FlatEncoder(TupleEncoder):
    """
    Returns a flattened version of the `children` encoders.

    Remarks:
        - continuous dimensions (i.e. Box) are kept as they are.
        - discrete dimensions are converted using one-hot encoding

    """

    def make_space(self) -> spaces.Box:  # type: ignore
        """Creates the observation space"""
        assert len(self.children) > 0
        space = flatten_space(super().make_space())
        assert isinstance(space, spaces.Box)
        return space

    def encode(self, nmi: NegotiatorMechanismInterface) -> FlatType:  # type: ignore
        """Encodes an observation from the agent's nmi"""
        assert len(self.children) > 0
        return flatten(super().make_space(), super().encode(nmi))

    def decode(
        self,
        nmi: NegotiatorMechanismInterface,
        encoded: np.ndarray | dict[str, np.ndarray],
    ) -> tuple:
        return tuple(
            c.decode(nmi, encoded[n])
            for n, c in zip(self.names, self.children, strict=True)
        )


@define
class BoxEncoder(TupleEncoder):
    """
    Returns a flattened version of the `children` encoders.

    Remarks:
        - continuous dimensions (i.e. Box) are kept as they are.
        - discrete dimensions are converted using one-hot encoding

    """

    rescale: bool = True
    _high: np.ndarray = field(default=None, init=False)
    _low: np.ndarray = field(default=None, init=False)

    def make_space(self) -> spaces.Box:  # type: ignore
        """Creates the observation space"""
        tuple_space = super().make_space()
        low = np.concatenate(
            [
                [t.start]
                if isinstance(t, spaces.Discrete)
                else t.low
                if isinstance(t, spaces.Box)
                else np.zeros_like(t.nvec)  # type: ignore
                for t in tuple_space.spaces
            ]
        )
        high = np.concatenate(
            [
                [t.start + t.n]
                if isinstance(t, spaces.Discrete)
                else t.high
                if isinstance(t, spaces.Box)
                else t.nvec - 1  # type: ignore
                for t in tuple_space.spaces
            ]
        )
        self._high = high
        self._low = low
        if self.rescale:
            low = np.zeros_like(low, dtype=FLOAT_TYPE)
            high = np.ones_like(high, dtype=FLOAT_TYPE)
        space = Box(low=low, high=high, dtype=FLOAT_TYPE)
        assert isinstance(space, Box), f"{space} is not of type Box"
        return space

    def encode(self, nmi: NegotiatorMechanismInterface) -> np.ndarray:  # type: ignore
        """Encodes an observation from the agent's nmi"""
        encoded = np.hstack(tuple(_.encode(nmi) for _ in self.children))  # type: ignore
        if not self.rescale:
            return encoded
        if self._high is None or self._low is None:
            self.make_space()
        return ((encoded - self._low) / (self._high - self._low)).astype(FLOAT_TYPE)


@define
class CTimeUtilityTupleEncoder(TupleEncoder):
    children: tuple[ObservationEncoder, ...] = (CTimeEncoder(), CUtilityEncoder())
    names: tuple[str, ...] = ("time", "utility")


@define
class CTimeUtilityDictEncoder(DictEncoder):
    children: tuple[ObservationEncoder, ...] = (
        CTimeEncoder(),
        CUtilityEncoder(),
    )
    names: tuple[str, ...] = ("time", "utility")


@define
class CTimeUtilityFlatEncoder(FlatEncoder):
    children: tuple[ObservationEncoder, ...] = (
        CTimeEncoder(),
        CUtilityEncoder(),
    )
    names: tuple[str, ...] = ("time", "utility")


@define
class CTimeUtilityBoxEncoder(BoxEncoder):
    children: tuple[ObservationEncoder, ...] = (
        CTimeEncoder(),
        CUtilityEncoder(),
    )
    names: tuple[str, ...] = ("time", "utility")


@define
class DTimeUtilityFlatEncoder(FlatEncoder):
    children: tuple[ObservationEncoder, ...] = (DTimeEncoder(), DUtilityEncoder())
    names: tuple[str, ...] = ("time", "utility")


@define
class DTimeUtilityBoxEncoder(BoxEncoder):
    children: tuple[ObservationEncoder, ...] = (DTimeEncoder(), DUtilityEncoder())
    names: tuple[str, ...] = ("time", "utility")


@define
class DTimeUtilityTupleEncoder(TupleEncoder):
    children: tuple[ObservationEncoder, ...] = (
        DTimeEncoder(),
        DUtilityEncoder(),
    )
    names: tuple[str, ...] = ("time", "utility")


@define
class DTimeUtilityDictEncoder(DictEncoder):
    children: tuple[ObservationEncoder, ...] = (
        DTimeEncoder(),
        DUtilityEncoder(),
    )
    names: tuple[str, ...] = ("time", "utility")


@define
class RLBoaEncoder(DictEncoder):
    """The observation encoder of RLBOA according to the paper."""

    n_offers: int = 4
    n_bins: int = 10
    n_time_bins: int = 5
    partner_utility: bool = False
    ignore_own_offers: bool = False
    missing_as_none: bool = False
    flat: bool = True
    min_encodable: float = 0
    max_encodable: float = 1
    encode_none: bool = True
    none_margin: float = DEFAULT_MIN_ENCODABLE
    children: tuple[ObservationEncoder, ...] = field(init=False, factory=tuple)
    names: tuple[str, ...] = field(init=False, factory=tuple)

    def __attrs_post_init__(self):
        self.names = ("time", "utility")
        self.children = (
            DTimeEncoder(n_levels=self.n_time_bins),
            DWindowedUtilityEncoder(
                n_offers=self.n_offers,
                n_levels=self.n_bins,
                partner_utility=self.partner_utility,
                ignore_own_offers=self.ignore_own_offers,
                missing_as_none=self.missing_as_none,
                flat=self.flat,
                min_encodable=self.min_encodable,
                max_encodable=self.max_encodable,
                encode_none=self.encode_none,
                none_margin=self.none_margin,
            ),
        )


@define
class SenguptaEncoder(DictEncoder):
    """The observation encoder of RLBOA according to the paper."""

    n_offers: int = 6
    partner_utility: bool = False
    ignore_own_offers: bool = False
    missing_as_none: bool = False
    flat: bool = True
    min_encodable: float = 0
    max_encodable: float = 1
    encode_none: bool = True
    none_margin: float = DEFAULT_MIN_ENCODABLE
    children: tuple[ObservationEncoder, ...] = field(init=False, factory=tuple)
    names: tuple[str, ...] = field(init=False, factory=tuple)

    def __attrs_post_init__(self):
        self.names = ("time", "utility")
        self.children = (
            CTimeEncoder(),
            CWindowedUtilityEncoder(
                n_offers=self.n_offers,
                partner_utility=self.partner_utility,
                ignore_own_offers=self.ignore_own_offers,
                missing_as_none=self.missing_as_none,
                flat=self.flat,
                min_encodable=self.min_encodable,
                max_encodable=self.max_encodable,
                encode_none=self.encode_none,
                none_margin=self.none_margin,
            ),
        )


@define
class VeNASEncoder(DictEncoder):
    n_outcomes: int = 1000
    n_time_levels: int = 100
    sort_by_utility: bool = False
    best_first: bool = False
    order_by_similarity: bool = False
    children: tuple[ObservationEncoder, ...] = field(init=False, factory=tuple)
    names: tuple[str, ...] = field(init=False, factory=tuple)

    def __attrs_post_init__(self):
        self.names = ("time", "outcome")
        self.children = (
            DTimeEncoder(n_levels=self.n_time_levels),
            DOutcomeEncoder1D(
                n_outcomes=self.n_outcomes,
                sort_by_utility=self.sort_by_utility,
                best_first=self.best_first,
                order_by_similarity=self.order_by_similarity,
            ),
        )


@define
class MiPNEncoder(DictEncoder):
    n_time_levels: int = 100
    n_issue_levels: tuple[int, ...] = (DEFAULT_N_LEVELS,)
    n_offers: int = 2
    ignore_own_offers: bool = False
    flat: bool = True
    children: tuple[ObservationEncoder, ...] = tuple()
    names: tuple[str, ...] = tuple()

    def __attrs_post_init__(self):
        self.names = ("time", "outcome")
        self.children = (
            DTimeEncoder(n_levels=self.n_time_levels),
            DWindowedIssueEncoder(
                n_levels=self.n_issue_levels,
                n_offers=self.n_offers,
                ignore_own_offers=self.ignore_own_offers,
                flat=self.flat,
            ),
        )


DefaultObsEncoder = CTimeUtilityDictEncoder
"""The default observation encoder"""
