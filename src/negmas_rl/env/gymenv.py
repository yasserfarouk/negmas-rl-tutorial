import random
from collections import defaultdict
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict
from negmas import ResponseType, SAOResponse
from negmas.gb.negotiators import AspirationNegotiator
from negmas.negotiators import Negotiator
from negmas.sao.mechanism import SAOMechanism
from rich import print
from rich.traceback import install

from ..action import ActionDecoder, DefaultActionDecoder
from ..generators.assigner import Assigner, PositionBasedNegotiatorAssigner
from ..generators.mechanism import MechanismGenerator, MechanismRepeater
from ..generators.negotiator import NegotiatorGenerator, NegotiatorRepeater
from ..generators.scenario import (
    RandomScenarioGenerator,
    ScenarioGenerator,
)
from ..negotiator import PLACEHOLDER_MAP
from ..obs import DefaultObsEncoder, ObservationEncoder
from ..reward import DefaultRewardFunction, RewardFunction

install(show_locals=True)

__all__ = [
    "NegoMultiEnv",
    "NegoEnv",
    "SingleNegViewEnv",
    "NegoMultiEnvWithPlaceholders",
    "DEFAULT_N_STEPS",
]

DEFAULT_N_STEPS = 100


class NegoMultiEnv(gym.Env):
    def __init__(
        self,
        *args,
        scenario_generator: ScenarioGenerator | None = None,
        mechanism_generator: MechanismGenerator | None = None,
        partner_generator: NegotiatorGenerator | None = None,
        assigner: Assigner | None = None,
        reward_function: RewardFunction | None = None,
        obs_encoder: ObservationEncoder
        | tuple[ObservationEncoder, ...]
        | type[ObservationEncoder]
        | None = None,
        action_decoder: ActionDecoder
        | tuple[ActionDecoder]
        | type[ActionDecoder]
        | None = None,
        n_learners: int = 2,
        extra_checks: bool = True,
        render_mode=None,
        must_step_all_negotiator: bool = True,
        must_step_any_learner: bool = True,
        must_step_all_learners: bool = False,
        step_at_most_one_learner: bool = False,
        repeating_received_is_acceptance: bool = True,
        allow_no_response: bool = False,
        allow_ending: bool = False,
        silent_ending_on_invalid_action: bool = True,
        silent_mapping_of_invalid_to_best: bool = False,
        debug=False,
        placeholder_params: dict[str, Any] | None = None,
        **kwargs,
    ):
        try:
            super().__init__(*args, **kwargs)
        except:
            pass
        self._placeholder_params = (
            placeholder_params if placeholder_params is not None else dict()
        )
        self.repeating_received_is_acceptance = repeating_received_is_acceptance
        self.silent_ending_on_invalid_action = silent_ending_on_invalid_action
        self.silent_mapping_of_invalid_to_best = silent_mapping_of_invalid_to_best
        self.allow_no_response = allow_no_response
        self.allow_ending = allow_ending
        if action_decoder is None:
            action_decoder = DefaultActionDecoder
        if obs_encoder is None:
            obs_encoder = DefaultObsEncoder
        if reward_function is None:
            reward_function = DefaultRewardFunction()
        if assigner is None:
            assigner = PositionBasedNegotiatorAssigner()
        if partner_generator is None:
            partner_generator = NegotiatorRepeater(AspirationNegotiator)
        if mechanism_generator is None:
            mechanism_generator = MechanismRepeater(
                SAOMechanism, params=dict(n_steps=DEFAULT_N_STEPS)
            )
        if scenario_generator is None:
            scenario_generator = RandomScenarioGenerator(n_ufuns=max(2, n_learners))
        self._scenario_generator: ScenarioGenerator = scenario_generator
        self._mechanism_generator: MechanismGenerator = mechanism_generator
        self._partner_generator: NegotiatorGenerator = partner_generator
        self._assigner: Assigner = assigner
        self._reward_function: RewardFunction = reward_function
        self._extra_checks = extra_checks
        self._render_mode = render_mode
        self._debug = debug
        self._current_step = 0

        self._n_learners = n_learners
        self.render_mode = render_mode

        self.possible_agents = [self._key(i) for i in range(n_learners)]

        def create(i: int, x: type | tuple | Any):
            if isinstance(x, tuple):
                return x[i]
            if isinstance(x, type):
                return x()
            return x

        self._obs_encoders = {
            self.possible_agents[i]: create(i, obs_encoder) for i in range(n_learners)
        }
        self._action_decoders = {
            self.possible_agents[i]: create(i, action_decoder)
            for i in range(n_learners)
        }
        self.action_space = Dict(
            dict(
                zip(
                    [self.possible_agents[i] for i in range(n_learners)],
                    [
                        self._action_decoders[self.possible_agents[i]].make_space()
                        for i in range(n_learners)
                    ],
                )
            )
        )
        self.observation_space = Dict(
            dict(
                zip(
                    [self.possible_agents[i] for i in range(n_learners)],
                    [
                        self._obs_encoders[self.possible_agents[i]].make_space()
                        for i in range(n_learners)
                    ],
                )
            )
        )
        self.reward_range = self._reward_function.get_range()
        self.set_stepping_behavior(
            must_step_all_negotiator=must_step_all_negotiator,
            must_step_any_learner=must_step_any_learner,
            must_step_all_learners=must_step_all_learners,
            step_at_most_one_learner=step_at_most_one_learner,
        )

        self.n_negotiations = 0
        # this is not used in this class but is used by SingleNegViewEnv() to broadcast information
        # to all views
        self._childrens = []
        self._last_rewards = dict()
        self._last_obs = dict()
        self._last_info = dict()

    def set_stepping_behavior(
        self,
        must_step_all_negotiator: bool = True,
        must_step_any_learner: bool = True,
        must_step_all_learners: bool = False,
        step_at_most_one_learner: bool = False,
    ):
        self._must_step_any_learner = must_step_any_learner
        self._must_step_all_learners = must_step_all_learners
        self._must_step_any_negotiator = must_step_all_negotiator
        self._step_at_most_one_learner = step_at_most_one_learner

    def _key(self, i: int) -> str:
        return f"{i}"

    def _get_obs(
        self,
    ) -> dict[str, np.ndarray] | np.ndarray | dict[str, dict[str, np.ndarray]]:
        return {
            a: self._obs_encoders[a].encode(n.nmi)
            for a, n in zip(self.possible_agents, self._placeholders)
        }

    def _get_first_obs(
        self,
    ) -> dict[str, np.ndarray] | np.ndarray | dict[str, dict[str, np.ndarray]]:
        return {
            a: self._obs_encoders[a].make_first_observation(n.nmi)
            for a, n in zip(self.possible_agents, self._placeholders)
        }

    def calc_info(self) -> dict:
        """Calculates info to be returned from `step()`."""
        return {key: dict() for key in self.possible_agents}

    def render_frame(self):
        """Used for rendering. Override with your rendering code"""
        if not self._mechanism:
            print("No mechanisms are running")
        print(
            f"{self._current_step} [magenta](#{self.n_negotiations}) {self._mechanism.id}[/magenta]: {self._mechanism.current_step} ({self._mechanism.time}): {self._mechanism_state_as_str()}"
        )

    def _mechanism_state_as_str(self):
        s = ""
        if isinstance(self._mechanism, SAOMechanism):
            s += f"{self._mechanism.state.current_proposer} > {self._mechanism.state.current_offer}"
        if self._mechanism.state.has_error:
            s += f"[red]E: {self._mechanism.state.erred_negotiator}: {self._mechanism.state.error_details}[/red]"
        if self._mechanism.state.done:
            if self._mechanism.agreement is not None:
                s += f"[green]\u2713 {self._mechanism.agreement}[/green]"
            elif self._mechanism.state.timedout:
                s += "[orange]x[/orange]"
            elif self._mechanism.state.ended:
                s += "[red]x[/red]"
        elif not self._mechanism.state.started:
            s += "[yellow]=[/yellow]"
        return s

    def close(self):
        pass

    def render(self):
        if self.render_mode == "human":
            return self.render_frame()

    def make_placeholder(self, indx: int) -> Negotiator:
        _ = indx
        for k, v in PLACEHOLDER_MAP:
            if isinstance(self._mechanism, k):
                return v(**self._placeholder_params)
        raise ValueError(
            "Cannot find an appropriate placeholder type for the given mechanism. See [blue]PLACEHOLDER_MAP[/blue] in the negotiator module"
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        _ = options

        random.seed(seed)
        self._scenario = self._scenario_generator(self.n_negotiations)
        self._mechanism = self._mechanism_generator(self._scenario)
        self._placeholders = tuple(
            self.make_placeholder(_) for _ in range(self._n_learners)
        )
        nlearners = len(self._placeholders)
        # self._negotiator = self.make_rl_negotiator(self._mechanism)
        n_partners = len(self._scenario.ufuns) - nlearners
        assert n_partners >= 0, (
            f"Got a scenarios with {len(self._scenario.ufuns)} ufuns "
            f"but we have {nlearners} learning agents. Make sure "
            f"the [blue]scenario_generator[/blue] can only generate "
            f"scenarios with at least {nlearners} learners."
        )
        self._partners = [
            self._partner_generator(self.n_negotiations + _) for _ in range(n_partners)
        ]
        self._mechanism = self._assigner(
            self._scenario, self._mechanism, self._partners, self._placeholders
        )

        self._placeholder_ids = [_.id for _ in self._placeholders]
        self._placeholder_indices = [
            v.nmi.negotiator_ids.index(k)
            for k, v in zip(self._placeholder_ids, self._placeholders)
        ]
        for i, _ in enumerate(self._placeholders):
            k = self.possible_agents[i]
            self._obs_encoders[k].on_negotiation_starts(_, _.nmi)
            self._action_decoders[k].on_negotiation_starts(_, _.nmi)
            self._obs_encoders[k].on_preferences_changed([])
            self._action_decoders[k].on_preferences_changed([])
        observation = self._get_obs()
        info = self.calc_info()

        if self.render_mode == "human":
            self.render_frame()

        # return OrderedDict(observation), OrderedDict(info)
        return observation, info

    def step(self, action):  # type: ignore
        try:
            atomic_step = self._mechanism.atomic_steps
            if self._step_at_most_one_learner:
                assert atomic_step, "The mechanism does not support atomic stepping (i.e. one negotiator every step) but you are requiring can_step_a_single_learner"
            rinfo = self._reward_function.init(self._mechanism)
            pre_reward_info = {
                v.id: self._reward_function.before_action(v, rinfo)
                for v in self._placeholders
            }

            def _adjust(nmi, resp: SAOResponse, action, agent):
                if resp.response == ResponseType.ACCEPT_OFFER:
                    return SAOResponse(resp.response, nmi.state.current_offer)
                if (
                    self.repeating_received_is_acceptance
                    and resp.response == ResponseType.REJECT_OFFER
                    and resp.outcome
                    and resp.outcome == nmi.state.current_offer
                ):
                    return SAOResponse(resp.response, nmi.state.current_offer)
                if (
                    not self.allow_ending
                    and resp.response == ResponseType.END_NEGOTIATION
                ):
                    raise ValueError(
                        f"Ending negotiation is not allowed!!\n{resp=}\n{action=}\n{agent=}"
                    )
                if (
                    not self.allow_no_response
                    and resp.response == ResponseType.NO_RESPONSE
                ):
                    if not self.silent_ending_on_invalid_action:
                        raise ValueError(
                            f"No response  is not allowed!!\n{resp=}\n{action=}\n{agent=}"
                        )
                    return SAOResponse(ResponseType.END_NEGOTIATION, None)
                if (
                    not self.allow_no_response
                    and resp.response == ResponseType.REJECT_OFFER
                    and not resp.outcome
                ):
                    if not self.silent_ending_on_invalid_action:
                        raise ValueError(
                            f"offering None is not allowed!!\n{resp=}\n{action=}\n{agent=}"
                        )
                    return SAOResponse(ResponseType.END_NEGOTIATION, None)
                return resp

            neg_action = {
                v.id: _adjust(
                    v.nmi,
                    self._action_decoders[self.possible_agents[i]].parse(
                        v.nmi, action[self.possible_agents[i]]
                    ),
                    action[self.possible_agents[i]],
                    self.possible_agents[i],
                )
                for i, v in enumerate(self._placeholders)
                if self.possible_agents[i] in action.keys()
            }

            received_actions = {
                v.id: action[self.possible_agents[i]]
                for i, v in enumerate(self._placeholders)
                if self.possible_agents[i] in action.keys()
            }
            states = {
                v.id: v.nmi.state
                for i, v in enumerate(self._placeholders)
                if self.possible_agents[i] in action.keys()
            }
            for i, _ in enumerate(self._placeholders):
                k = self.possible_agents[i]
                self._obs_encoders[k].after_learner_actions(
                    states, neg_action, received_actions
                )
                self._action_decoders[k].after_learner_actions(
                    states, neg_action, received_actions
                )
            learner_stepped, stepped = False, False
            if not atomic_step:
                self._mechanism.step(neg_action)
                learner_stepped, stepped = True, True
            else:
                negotiator_ids = self._mechanism.following_negotiators()  # type: ignore
                assert set(negotiator_ids) == set(self._mechanism.negotiator_ids)
                for current in negotiator_ids:
                    if self._mechanism.state.done:
                        break
                    active = current in self._placeholder_ids
                    if not active:
                        self._mechanism.step()
                        stepped = True
                        continue
                    if not self._must_step_all_learners and current not in neg_action:
                        continue
                    # By design, this will fail if one of the learners is not included in the action
                    self._mechanism.step({current: neg_action[current]})
                    learner_stepped, stepped = True, True
                    if self._step_at_most_one_learner:
                        break
            if not self._mechanism.state.done:
                if not learner_stepped and self._must_step_any_learner:
                    raise ValueError(
                        f"Did not step any learner with action {neg_action} in state: {self._mechanism.state}"
                    )
                if not stepped and self._must_step_any_negotiator:
                    raise ValueError(
                        f"Did not step any negotiators with action {neg_action} in state: {self._mechanism.state}"
                    )
            rewards = {
                self.possible_agents[i]: self._reward_function(
                    p, neg_action, pre_reward_info[p.id]
                )
                for i, p in enumerate(self._placeholders)
            }
            terminated = int(self._mechanism.ended)
            if terminated:
                self.n_negotiations += 1
            obs = self._get_obs()
            info = self.calc_info()

            if self.render_mode == "human":
                self.render_frame()
            self._current_step += 1

            return obs, rewards, terminated, False, info
        except Exception as e:
            # Handle the exception and potentially set a flag to stop the progress bar
            self.render_progress = False
            raise e


class NegoMultiEnvWithPlaceholders(NegoMultiEnv):
    def __init__(
        self,
        *args,
        placeholder_types: tuple[type[Negotiator] | None, ...]
        | type[Negotiator]
        | None = None,
        placeholder_params: tuple[dict, ...] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if placeholder_types is None:
            placeholder_types = tuple(None for _ in range(self._n_learners))
        elif isinstance(placeholder_types, type) and issubclass(
            placeholder_types, Negotiator
        ):
            placeholder_types = tuple(
                placeholder_types for _ in range(self._n_learners)
            )
        self._placeholder_types = placeholder_types
        if placeholder_params is None:
            self._placeholder_params = [dict() for _ in range(len(placeholder_types))]
        elif isinstance(placeholder_params, dict):
            self._placeholder_params = [
                deepcopy(placeholder_params) for _ in range(len(placeholder_types))
            ]
        else:
            self._placeholder_params = placeholder_params

    def set_placeholder_types(
        self,
        placeholder_types: tuple[type[Negotiator] | None, ...]
        | type[Negotiator]
        | None = None,
        placeholder_params: tuple[dict, ...] | None = None,
    ):
        if placeholder_types is None:
            placeholder_types = tuple(None for _ in range(self._n_learners))
        elif isinstance(placeholder_types, type) and issubclass(
            placeholder_types, Negotiator
        ):
            placeholder_types = tuple(
                placeholder_types for _ in range(self._n_learners)
            )
        self._placeholder_types = placeholder_types
        if placeholder_params is None:
            self._placeholder_params = [dict() for _ in range(len(placeholder_types))]
        elif isinstance(placeholder_params, dict):
            self._placeholder_params = [
                deepcopy(placeholder_params) for _ in range(len(placeholder_types))
            ]
        else:
            self._placeholder_params = placeholder_params

    def set_placeholders(self, placeholders: tuple[Negotiator, ...]):
        self._placeholders = placeholders
        self._placeholder_ids = [_.id for _ in placeholders]

    def make_placeholder(self, indx: int) -> Negotiator:
        t = self._placeholder_types[indx]
        if t is None:
            return super().make_placeholder(indx)
        return t(**self._placeholder_params[indx])  # type: ignore


class NegoEnv(NegoMultiEnv):
    def __init__(
        self,
        *args,
        scenario_generator: ScenarioGenerator | None = None,
        mechanism_generator: MechanismGenerator | None = None,
        partner_generator: NegotiatorGenerator | None = None,
        assigner: Assigner | None = None,
        reward_function: RewardFunction | None = None,
        obs_encoder: ObservationEncoder | None = None,
        action_decoder: ActionDecoder | None = None,
        extra_checks: bool = True,
        render_mode=None,
        debug=False,
        **kwargs,
    ):
        super().__init__(
            *args,
            scenario_generator=scenario_generator,
            mechanism_generator=mechanism_generator,
            partner_generator=partner_generator,
            assigner=assigner,
            reward_function=reward_function,
            obs_encoder=obs_encoder,
            action_decoder=action_decoder,
            extra_checks=extra_checks,
            render_mode=render_mode,
            debug=debug,
            n_learners=1,
            **kwargs,
        )
        key = self.possible_agents[0]

        self.action_space = self.action_space[key]  # type: ignore
        self.observation_space = self.observation_space[key]  # type: ignore

    def _get_obs(self) -> dict[str, np.ndarray] | np.ndarray:
        return super()._get_obs()[self.possible_agents[0]]

    def calc_info(self) -> dict:
        """Calculates info to be returned from `step()`."""
        return super().calc_info()[self.possible_agents[0]]

    def step(self, action):  # type: ignore
        obs, rewards, terminated, x, info = super().step(
            {self.possible_agents[0]: action}
        )
        key = self.possible_agents[0]
        return obs, rewards[key], terminated, x, info


class SingleNegViewEnv(gym.Env):
    def __init__(self, *args, env: NegoMultiEnv, indx: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        env.set_stepping_behavior(
            must_step_all_negotiator=False,
            must_step_any_learner=True,
            must_step_all_learners=False,
            step_at_most_one_learner=True,
        )
        self._env = env
        self._indx = indx
        self._key = self._env._key(self._indx)
        self._acc_reward = defaultdict(float)
        key = self._key

        self.action_space = self._env.action_space[key]  # type: ignore
        self.observation_space = self._env.observation_space[key]  # type: ignore
        self._env._childrens.append(self)

    def calc_info(self) -> dict:
        """Calculates info to be returned from `step()`."""
        return dict()

    def render_frame(self):
        """Used for rendering. Override with your rendering code"""
        self._env.render_frame()

    def close(self):
        pass

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        k = self._key
        return obs[k], info[k]

    def step(self, action):  # type: ignore
        neg_id = self._env._placeholder_ids[self._indx]
        assert self._env._mechanism.atomic_steps, (
            f"{self.__class__.__name__} must have atomic steps "
            f"(i.e. its atomic_steps property must return True).\n"
            f"For SAOMechanism, you can set that by passing "
            f"one_offer_per_step=True to the constructor"
        )
        old_rewards = dict()
        k = self._key
        if isinstance(self._env._mechanism, SAOMechanism):
            if self._env._mechanism.state.started:
                assert self._env._mechanism.next_negotitor_ids()[0] == neg_id, (
                    f"{self._env._mechanism.following_negotiators()=}\nbut {neg_id=}\n"
                    f"placeholders: {self._env._placeholder_ids}\n"
                    f"{self._env._mechanism.state}\n{self._env.n_negotiations}"
                )
            else:
                old_rewards = self._env._last_rewards
                obs = self._env._last_obs
                info = self._env._last_info
                if not self._env._mechanism.next_negotitor_ids()[0] == neg_id:
                    return (
                        obs[k],
                        old_rewards.pop(k, 0.0),
                        True,
                        False,
                        info[k] if info else info,
                    )
        obs, rewards, terminated, x, info = self._env.step({k: action})
        # broadcast termination info to all other views.
        if terminated:
            self._env._last_rewards = deepcopy(rewards)  # type: ignore
            self._env._last_obs = deepcopy(obs)  # type: ignore
            self._env._last_info = deepcopy(info)  # type: ignore
        for k, v in rewards.items():
            self._acc_reward[k] += v

        return (
            obs[k],
            rewards.pop(k, 0.0) + old_rewards.pop(k, 0.0),
            terminated,
            x,
            info[k] if info else info,
        )
