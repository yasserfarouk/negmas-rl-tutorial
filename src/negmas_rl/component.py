from typing import Any

import numpy as np
from attr import define, field
from negmas import MechanismState, SAOResponse
from negmas.common import NegotiatorMechanismInterface, PreferencesChange
from negmas.negotiators import Negotiator
from negmas.sao import SAOState

__all__ = ["Component"]


@define
class Component:
    owner: Negotiator = field(default=None)

    def reset(self):
        self.owner = None  # type: ignore

    def on_negotiation_starts(
        self, owner: Negotiator, nmi: NegotiatorMechanismInterface
    ) -> None:
        self.owner = owner

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Called when preferences change"""

    def on_negotiation_ends(
        self, owner: Negotiator, nmi: NegotiatorMechanismInterface
    ) -> None:
        self.reset()

    def after_partner_action(
        self,
        partner_id: str,
        state: MechanismState,
        action: Any,
    ) -> None:
        """
        A callback called by the mechanism when a partner proposes something
        """

    def after_learner_actions(
        self,
        states: dict[str, MechanismState],
        actions: dict[str, Any],
        encoded_action: dict[str, np.ndarray],
    ) -> None:
        """
        A callback called by the mechanism when a partner proposes something
        """


@define
class SAOComponent(Component):
    def after_learner_actions(  # type: ignore
        self,
        states: dict[str, SAOState],
        actions: dict[str, SAOResponse],
        encoded_action: dict[str, np.ndarray],
    ) -> None:
        """
        A callback called by the mechanism when a partner proposes something
        """

    def after_partner_action(  # type: ignore
        self,
        partner_id: str,
        state: MechanismState,
        action: SAOResponse,
    ) -> None:
        """
        A callback called by the mechanism when a partner proposes something
        """
