from typing import TYPE_CHECKING

from negmas import PreferencesChange, ResponseType, SAONegotiator, SAOResponse, SAOState
from negmas.mechanisms import Mechanism
from negmas.negotiators import Negotiator
from negmas.outcomes import Outcome
from negmas.sao.mechanism import SAOMechanism

from .action import ActionDecoder, DefaultActionDecoder
from .common import Policy
from .obs import DefaultObsEncoder, ObservationEncoder

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.policies import BasePolicy


class Placeholder(Negotiator):
    """An agent that always raises an exception if called to negotiate. It is useful as a placeholder (for example for RL and MARL exposition)"""


class SAOPlaceholder(SAONegotiator):
    """An agent that always raises an exception if called to negotiate. It is useful as a placeholder (for example for RL and MARL exposition)"""

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        raise RuntimeError("This negotiator is not supposed to ever be called")


class SAORLNegotiator(SAONegotiator):
    @classmethod
    def default_trainer_type(cls) -> type["BaseAlgorithm"]:
        from stable_baselines3.sac import SAC

        return SAC

    @classmethod
    def default_trainer_params(cls) -> dict:
        return dict()

    @classmethod
    def default_policy_params(cls) -> dict:
        return dict()

    @classmethod
    def default_policy_type(cls) -> type["BasePolicy"] | str:
        return "MultiInputPolicy"

    @classmethod
    def default_obs_encoder_type(cls) -> type[ObservationEncoder]:
        return DefaultObsEncoder

    @classmethod
    def default_action_decoder_type(cls) -> type[ActionDecoder]:
        return DefaultActionDecoder

    @classmethod
    def default_obs_encoder_params(cls) -> dict:
        return dict()

    @classmethod
    def default_action_decoder_params(cls) -> dict:
        return dict()

    def __init__(
        self,
        *args,
        policy: Policy,
        obs_encoder: ObservationEncoder | None = None,
        action_decoder: ActionDecoder | None = None,
        **kwargs,
    ):
        if obs_encoder is None:
            self.obs_encoder = self.default_obs_encoder_type()(
                **self.default_obs_encoder_params()
            )
        else:
            self.obs_encoder = obs_encoder
        if action_decoder is None:
            self.action_decoder = self.default_action_decoder_type()(
                **self.default_action_decoder_params()
            )
        else:
            self.action_decoder = action_decoder
        self.obs_encoder.owner = self
        self.action_decoder.owner = self
        self.policy = policy
        super().__init__(*args, **kwargs)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        return self.action_decoder.decode(
            self.nmi,  # type: ignore
            self.policy(self.obs_encoder.encode(self.nmi)),
        )

    def on_partner_proposal(  # type: ignore
        self, state: SAOState, partner_id: str, offer: Outcome
    ) -> None:
        super().on_partner_proposal(state, partner_id, offer)
        self.obs_encoder.after_partner_action(
            partner_id, state, SAOResponse(ResponseType.REJECT_OFFER, offer)
        )
        self.action_decoder.after_partner_action(
            partner_id, state, SAOResponse(ResponseType.REJECT_OFFER, offer)
        )

    def on_negotiation_start(self, state: SAOState) -> None:  # type: ignore
        super().on_negotiation_start(state)
        self.obs_encoder.on_negotiation_starts(self, self.nmi)
        self.action_decoder.on_negotiation_starts(self, self.nmi)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        super().on_preferences_changed(changes)
        self.obs_encoder.on_preferences_changed(changes)
        self.action_decoder.on_preferences_changed(changes)

    def on_negotiation_end(self, state) -> None:
        super().on_negotiation_end(state)
        self.obs_encoder.on_negotiation_ends(self, self.nmi)
        self.action_decoder.on_negotiation_ends(self, self.nmi)


PLACEHOLDER_MAP = (
    (SAOMechanism, SAOPlaceholder),
    (Mechanism, Placeholder),
)
