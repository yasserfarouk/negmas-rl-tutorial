from math import isinf, isnan
from typing import Any, Protocol, runtime_checkable

from negmas.mechanisms import Mechanism
from negmas.negotiators import Negotiator

__all__ = ["RewardFunction", "DefaultRewardFunction"]


@runtime_checkable
class RewardFunction(Protocol):
    """
    Represents a reward function.

    Remarks:
        - `before_action` is called before the action is executed for initialization and should return info to be passed to the call
        - `__call__` is called with the awi (to get the state), action and info and should return the reward

    """

    def init(self, mechanism: Mechanism) -> Any:
        """
        Called before starting to evaluate the reward for each agent. This is called once per negotiation step

        Remarks:
            The returned value will be passed as `info` to `before_action()` when it is time to calculate
            the reward.
        """
        ...

    def before_action(self, neg: Negotiator, info: Any) -> Any:
        """
        Called before executing the action from the RL negotiator to save any required information for
        calculating the reward in its return

        Remarks:
            The returned value will be passed as `info` to `__call__()` when it is time to calculate
            the reward.
        """
        ...

    def __call__(self, neg: Negotiator, action: Any, info: Any) -> float:
        """
        Called to calculate the reward to be given to the agent at the end of a step.

        Args:
            awi: `OneShotAWI` to access the agent's state
            action: The action form the agent being rewarded. For SAOMechainsm, this will be an SAOResponse for example
            info: Information generated from `before_action()`. You an use this to store baselines for calculating the reward

        Returns
        -------
            The reward (a number) to be given to the agent at the end of the step.
        """
        ...

    def get_range(self) -> tuple[float, float]:
        """
        Returns the range of reward values. Use the tightest range possible
        """
        ...


class DefaultRewardFunction(RewardFunction):
    """
    The default reward function which simply uses utility

    Remarks:
        - The reward is the difference between the balance before the action and after it.

    """

    def __init__(self, normalize: bool = True, advantage: bool = True) -> None:
        super().__init__()
        self._advantage, self._normalize = advantage, normalize

    def init(self, mechanism: Mechanism) -> Any:
        return None

    def before_action(self, neg: Negotiator, info: Any) -> Any:
        return None

    def __call__(self, neg: Negotiator, action: Any, info: Any) -> float:
        nmi = neg.nmi
        if not nmi.state.done:
            return 0.0
        if not neg.ufun:
            return float("nan")
        u = float(neg.ufun(nmi.state.agreement))
        r = float(neg.ufun.reserved_value)
        mx = 1
        if self._advantage:
            if not (isinf(r) or isnan(r)):
                u -= r
        if self._normalize:
            mx = float(neg.ufun.max())
            if self._advantage:
                mx -= r
        return u / mx

    def get_range(self) -> tuple[float, float]:
        if self._normalize:
            return (0, 1)
        return (-float("inf"), float("inf"))


class EgaliterianRewardFunction(RewardFunction):
    """
    A reward function that uses the minimum advantage

    Remarks:
        - The reward is the difference between the balance before the action and after it.

    """

    def init(self, mechanism: Mechanism) -> Any:
        return mechanism.negotiators

    def before_action(self, neg: Negotiator, info: Any) -> Any:
        return info

    def __call__(self, neg: Negotiator, action: Any, info: list[Negotiator]) -> float:
        r = float("inf")
        if not info:
            return 0.0
        for neg in info:
            nmi = neg.nmi
            if not nmi.state.done:
                continue
            if not neg.ufun:
                continue
            res = float(neg.ufun.reserved_value)
            if isinf(res) or isnan(res):
                res = 0
            r = min(r, float(neg.ufun(nmi.state.agreement)) - res)
        if isinf(r) or isnan(r):
            return 0.0
        return r

    def get_range(self) -> tuple[float, float]:
        return (-float("inf"), float("inf"))


class TotalAdvantageRewardFunction(RewardFunction):
    """
    A reward function that uses the total advantage of all negotiators learning

    Remarks:
        - The reward is the difference between the balance before the action and after it.

    """

    def init(self, mechanism: Mechanism) -> Any:
        return mechanism.negotiators

    def before_action(self, neg: Negotiator, info: Any) -> Any:
        return info

    def __call__(self, neg: Negotiator, action: Any, info: list[Negotiator]) -> float:
        r = 0.0
        if not info:
            return 0.0
        for neg in info:
            nmi = neg.nmi
            if not nmi.state.done:
                continue
            if not neg.ufun:
                continue

            res = float(neg.ufun.reserved_value)
            if isinf(res) or isnan(res):
                res = 0
            r += float(neg.ufun(nmi.state.agreement)) - res
        return r / len(info)

    def get_range(self) -> tuple[float, float]:
        return (-float("inf"), float("inf"))


class WelfareRewardFunction(RewardFunction):
    """
    A reward function that uses the welfare of all negotiators learning

    Remarks:
        - The reward is the difference between the balance before the action and after it.

    """

    def init(self, mechanism: Mechanism) -> Any:
        return mechanism.negotiators

    def before_action(self, neg: Negotiator, info: Any) -> Any:
        return info

    def __call__(self, neg: Negotiator, action: Any, info: list[Negotiator]) -> float:
        r = 0.0
        if not info:
            return 0.0
        for neg in info:
            nmi = neg.nmi
            if not nmi.state.done:
                continue
            if not neg.ufun:
                continue
            r += float(neg.ufun(nmi.state.agreement)) - float(neg.ufun.reserved_value)
        return r / len(info)

    def get_range(self) -> tuple[float, float]:
        return (-float("inf"), float("inf"))
