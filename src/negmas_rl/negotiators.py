from typing import TYPE_CHECKING

import numpy as np

from negmas_rl.action import (
    CMiPNDecoder,
    CRLBoaDecoder,
    CVeNASDecoder,
    DSenguptaDecoder,
    MiPNDecoder,
    RLBoaDecoder,
    SenguptaDecoder,
    VeNASDecoder,
)
from negmas_rl.common import (
    DEFAULT_N_LEVELS,
    DEFAULT_N_OUTCOMES,
    DEFAULT_UTIL_LEVELS,
    INT_TYPE,
)
from negmas_rl.negotiator import SAORLNegotiator
from negmas_rl.obs import (
    MiPNEncoder,
    RLBoaEncoder,
    SenguptaEncoder,
    VeNASEncoder,
)

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm

__all__ = [
    "RLBoa",
    "RLBoaC",
    "Sengupta",
    "SenguptaD",
    "MiPN",
    "MiPNC",
    "VeNAS",
    "VeNASC",
]


class RLBoa(SAORLNegotiator):
    """RLBOA implementation"""

    @classmethod
    def default_trainer_type(cls) -> type["BaseAlgorithm"]:
        from stable_baselines3.ppo import PPO

        return PPO

    @classmethod
    def default_obs_encoder_type(cls) -> type[RLBoaEncoder]:
        return RLBoaEncoder

    @classmethod
    def default_action_decoder_type(cls) -> type[RLBoaDecoder]:
        return RLBoaDecoder


class RLBoaC(SAORLNegotiator):
    """Continuous action version of RLBOA implementation (continuous version)"""

    @classmethod
    def default_obs_encoder_type(cls) -> type[RLBoaEncoder]:
        return RLBoaEncoder

    @classmethod
    def default_action_decoder_type(cls) -> type[CRLBoaDecoder]:
        return CRLBoaDecoder


class Sengupta(SAORLNegotiator):
    """Sengupta et. al. implementation (continuous version)"""

    @classmethod
    def default_obs_encoder_type(cls) -> type[SenguptaEncoder]:
        return SenguptaEncoder

    @classmethod
    def default_action_decoder_type(cls) -> type[SenguptaDecoder]:
        return SenguptaDecoder


class SenguptaD(SAORLNegotiator):
    """Discrete version of Sengupta's algorithm implementation (continuous version)"""

    @classmethod
    def default_trainer_type(cls) -> type["BaseAlgorithm"]:
        from stable_baselines3.ppo import PPO

        return PPO

    @classmethod
    def default_action_decoder_params(cls) -> dict:
        return dict(n_levels=DEFAULT_UTIL_LEVELS)

    @classmethod
    def default_obs_encoder_type(cls) -> type[SenguptaEncoder]:
        return SenguptaEncoder

    @classmethod
    def default_action_decoder_type(cls) -> type[DSenguptaDecoder]:
        return DSenguptaDecoder

    def __init__(self, *args, n_levels: int = DEFAULT_N_LEVELS, **kwargs):
        kwargs["action_decoder"] = self.default_action_decoder_type()(
            **(self.default_action_decoder_params() | dict(n_levels=n_levels))
        )
        super().__init__(*args, **kwargs)


class MiPN(SAORLNegotiator):
    """MiPN (Higa et. al. 2023) implementation"""

    @classmethod
    def default_trainer_type(cls) -> type["BaseAlgorithm"]:
        from stable_baselines3.ppo import PPO

        return PPO

    @classmethod
    def default_obs_encoder_type(cls) -> type[MiPNEncoder]:
        return MiPNEncoder

    @classmethod
    def default_action_decoder_type(cls) -> type[MiPNDecoder]:
        return MiPNDecoder

    @classmethod
    def default_obs_encoder_params(cls) -> dict:
        return dict(n_issue_levels=(DEFAULT_N_LEVELS,))

    @classmethod
    def default_action_decoder_params(cls) -> dict:
        return dict(n_levels=np.asarray((DEFAULT_N_LEVELS,), dtype=INT_TYPE))

    def __init__(
        self,
        *args,
        n_issue_levels: tuple[int, ...] = (DEFAULT_N_LEVELS,),
        n_time_levels: int = DEFAULT_N_LEVELS,
        n_action_levels: tuple[int, ...] = (DEFAULT_N_LEVELS,),
        **kwargs,
    ):
        kwargs["obs_encoder"] = self.default_obs_encoder_type()(
            **(
                self.default_obs_encoder_params()
                | dict(n_issue_levels=n_issue_levels, n_time_levels=n_time_levels)
            )
        )
        kwargs["action_decoder"] = self.default_action_decoder_type()(
            **(self.default_action_decoder_params() | dict(n_levels=n_action_levels))
        )
        super().__init__(*args, **kwargs)


class MiPNC(SAORLNegotiator):
    """MiPN (Higa et. al. 2023) implementation"""

    @classmethod
    def default_obs_encoder_type(cls) -> type[MiPNEncoder]:
        return MiPNEncoder

    @classmethod
    def default_obs_encoder_params(cls) -> dict:
        return dict(n_issue_levels=(DEFAULT_N_LEVELS,), n_time_levels=DEFAULT_N_LEVELS)

    @classmethod
    def default_action_decoder_type(cls) -> type[CMiPNDecoder]:
        return CMiPNDecoder

    def __init__(
        self,
        *args,
        n_issue_levels: tuple[int, ...] = (DEFAULT_N_LEVELS,),
        n_time_levels: int = DEFAULT_N_LEVELS,
        **kwargs,
    ):
        kwargs["obs_encoder"] = self.default_obs_encoder_type()(
            **(
                self.default_obs_encoder_params()
                | dict(n_issue_levels=n_issue_levels, n_time_levels=n_time_levels)
            )
        )
        super().__init__(*args, **kwargs)


class VeNAS(SAORLNegotiator):
    """VeNAS (Higa et. al. 2023) implementation"""

    @classmethod
    def default_trainer_type(cls) -> type["BaseAlgorithm"]:
        from stable_baselines3.ppo import PPO

        return PPO

    @classmethod
    def default_obs_encoder_type(cls) -> type[VeNASEncoder]:
        return VeNASEncoder

    @classmethod
    def default_action_decoder_type(cls) -> type[VeNASDecoder]:
        return VeNASDecoder

    @classmethod
    def default_obs_encoder_params(cls) -> dict:
        return dict(n_outcomes=DEFAULT_N_OUTCOMES)

    @classmethod
    def default_action_decoder_params(cls) -> dict:
        return dict(n_outcomes=DEFAULT_N_OUTCOMES)

    def __init__(self, *args, n_outcomes: int = DEFAULT_N_OUTCOMES, **kwargs):
        kwargs["obs_encoder"] = self.default_obs_encoder_type()(
            **(self.default_obs_encoder_params() | dict(n_outcomes=n_outcomes))
        )
        kwargs["action_decoder"] = self.default_action_decoder_type()(
            **(self.default_action_decoder_params() | dict(n_outcomes=n_outcomes))
        )
        super().__init__(*args, **kwargs)


class VeNASC(SAORLNegotiator):
    """VeNAS (Higa et. al. 2023) implementation"""

    @classmethod
    def default_obs_encoder_type(cls) -> type[VeNASEncoder]:
        return VeNASEncoder

    @classmethod
    def default_action_decoder_type(cls) -> type[CVeNASDecoder]:
        return CVeNASDecoder

    def __init__(self, *args, n_outcomes: int = DEFAULT_N_OUTCOMES, **kwargs):
        kwargs["obs_encoder"] = self.default_obs_encoder_type()(
            **(self.default_obs_encoder_params() | dict(n_outcomes=n_outcomes))
        )
        super().__init__(*args, **kwargs)

    @classmethod
    def default_obs_encoder_params(cls) -> dict:
        return dict(n_outcomes=DEFAULT_N_OUTCOMES)
