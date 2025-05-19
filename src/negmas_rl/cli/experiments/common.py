from collections.abc import Sequence

from negmas import (
    BoulwareTBNegotiator,
    ConcederTBNegotiator,
    LinearTBNegotiator,
)

from negmas_rl.common import DEFAULT_UTIL_LEVELS
from negmas_rl.experiment import BaselineCondition, Condition
from negmas_rl.helpers.negotiators import (
    AgentGG,
    AgentK,
    Atlas3,
    HardHeaded,
)
from negmas_rl.negotiators import MiPN as MiPNAgent
from negmas_rl.negotiators import MiPNC as MiPNCAgent
from negmas_rl.negotiators import RLBoa as RLBOAAgent
from negmas_rl.negotiators import (
    Sengupta,
    SenguptaD,
)
from negmas_rl.negotiators import VeNAS as VeNASAgent
from negmas_rl.negotiators import VeNASC as VeNASCAgent

from ..constants import BaselineMethods, Methods

N_STEPS = 100
TIME_LIMIT = None
IGNORE_NEG_EXCEPTIONS = False
ALWAYS_STARTS = True
VERBOSITY = 1

BASELINE_CONDISIONS = {
    BaselineMethods.Atlas3: BaselineCondition(
        name=BaselineMethods.Atlas3.value,
        negotiator=Atlas3,
    ),
    BaselineMethods.Hardheaded: BaselineCondition(
        name=BaselineMethods.Hardheaded.value,
        negotiator=HardHeaded,
    ),
    BaselineMethods.AgentK: BaselineCondition(
        name=BaselineMethods.AgentK.value,
        negotiator=AgentK,
    ),
    BaselineMethods.AgentGG: BaselineCondition(
        name=BaselineMethods.AgentGG.value,
        negotiator=AgentGG,
    ),
    BaselineMethods.Boulware: BaselineCondition(
        name=BaselineMethods.Boulware.value,
        negotiator=BoulwareTBNegotiator,  # type: ignore
    ),
    BaselineMethods.Conceder: BaselineCondition(
        name=BaselineMethods.Conceder.value,
        negotiator=ConcederTBNegotiator,  # type: ignore
    ),
    BaselineMethods.Linear: BaselineCondition(
        name=BaselineMethods.Linear.value,
        negotiator=LinearTBNegotiator,  # type: ignore
    ),
}


def make_conditions(
    n_levels,
    n_outcomes,
    n_issues=None,
    n_training=800_000,
    sac=True,
    ppo=True,
    baselines=True,
    raw_outcome_methods=True,
    n_steps=500,
    n_utility_levels=DEFAULT_UTIL_LEVELS,
    train_negotiators=(BoulwareTBNegotiator,),
    test_negotiators=None,
):
    baseline_condisions = BASELINE_CONDISIONS if baselines else dict()
    if test_negotiators is None:
        test_negotiators = train_negotiators
    if n_issues is None:
        n_issues = len(n_levels) if isinstance(n_levels, Sequence) else 3

    sac_conditions = {}
    sac_conditions |= {
        Methods.Sengupta: Condition(
            name=Methods.Sengupta.value,
            negotiator=Sengupta,
            n_training_steps=n_training,
        ),
        Methods.Rlboac: Condition(
            name=Methods.Rlboac.value,
            negotiator=RLBOAAgent,
            n_training_steps=n_training,
        ),
    }

    if raw_outcome_methods:
        sac_conditions |= {
            Methods.Venasc: Condition(
                name=Methods.Venasc.value,
                negotiator=VeNASCAgent,
                negotiator_params=dict(n_outcomes=n_outcomes),
                obs_params=dict(n_outcomes=n_outcomes),
                action_params=dict(),
                n_training_steps=n_training,
            ),
            Methods.Mipnc: Condition(
                name=Methods.Mipnc.value,
                negotiator=MiPNCAgent,
                negotiator_params=dict(n_issue_levels=n_levels, n_time_levels=n_steps),
                obs_params=dict(n_issue_levels=n_levels, n_time_levels=n_steps),
                n_training_steps=n_training,
            ),
        }

    ppo_conditions = {
        Methods.Rlboa: Condition(
            name=Methods.Rlboa.value,
            negotiator=RLBOAAgent,
            n_training_steps=n_training,
        ),
        Methods.Senguptad: Condition(
            name=Methods.Senguptad.value,
            negotiator=SenguptaD,
            negotiator_params=dict(n_levels=n_utility_levels),
            action_params=dict(n_levels=n_utility_levels),
            n_training_steps=n_training,
        ),
    }
    if raw_outcome_methods:
        ppo_conditions |= {
            Methods.Mipn: Condition(
                name=Methods.Mipn.value,
                negotiator=MiPNAgent,
                negotiator_params=dict(
                    n_issue_levels=n_levels,
                    n_action_levels=n_levels,
                    n_time_levels=n_steps,
                ),
                obs_params=dict(n_issue_levels=n_levels, n_time_levels=n_steps),
                action_params=dict(n_levels=n_levels),
                n_training_steps=n_training,
            ),
            Methods.Venas: Condition(
                name=Methods.Venas.value,
                negotiator=VeNASAgent,
                negotiator_params=dict(n_outcomes=n_outcomes),
                obs_params=dict(n_outcomes=n_outcomes),
                action_params=dict(n_outcomes=n_outcomes),
                n_training_steps=n_training,
            ),
        }

    conditions = dict()
    if sac:
        conditions |= sac_conditions
    if ppo:
        conditions |= ppo_conditions
    return conditions, baseline_condisions
