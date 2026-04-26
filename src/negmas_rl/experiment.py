import random
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from attr import asdict, define, field
from matplotlib import pyplot as plt
from negmas import SAOMechanism, SAONegotiator, calc_outcome_distances
from negmas.helpers import humanize_time, unique_name
from negmas.helpers.inout import add_records, pd
from negmas.inout import dump, get_full_type_name, serialize
from negmas.outcomes import Outcome
from negmas.preferences.ops import (
    OutcomeOptimality,
    ScenarioStats,
    calc_outcome_optimality,
    calc_scenario_stats,
    conflict_level,
    estimate_max_dist,
    opposition_level,
)
from numpy import mean, std
from rich import print
from rich.console import Console
from rich.progress import track
from rich.traceback import install
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BaseModel, BasePolicy

from negmas_rl.action import DECISION, ActionDecoder
from negmas_rl.env.gymenv import NegoEnv
from negmas_rl.generators.assigner import PositionBasedNegotiatorAssigner
from negmas_rl.generators.mechanism import MechanismRepeater
from negmas_rl.generators.negotiator import (
    NegotiatorGenerator,
    NegotiatorSampler,
    NegType,
)
from negmas_rl.generators.scenario import ScenarioGenerator
from negmas_rl.negotiator import SAORLNegotiator
from negmas_rl.obs import ObservationEncoder

install()
_ = calc_outcome_optimality, calc_scenario_stats, ScenarioStats, OutcomeOptimality
RESULTS_PATH = "results"
NEG_RESULTS_PATH = "neg_results"
MODELS_PATH = "models"
SHOW_LOCALS = False


def flatten_dict(dd: dict[str, Any], separator="_", prefix="") -> dict[str, Any]:
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


def myserialize(x, exclude: set[str] = set(), exclude_private: bool = True) -> dict:
    d = serialize(
        x,
        python_class_identifier="type",
        ignore_lambda=True,
        ignore_methods=True,
        keep_private=not exclude_private,
    )
    if isinstance(d, dict):
        return {
            k: v
            for k, v in d.items()
            if k not in exclude and not (exclude_private and k.startswith("_"))
        }
    else:
        return d


def wrapper(obs, model: BaseModel | BaseAlgorithm):
    """Wraps a trained stable-baselines3 model as a negotiation policy"""
    return model.predict(obs)[0]


def random_policy(obs, env):
    """Behaves randomly."""
    action = env.action_space.sample()
    if not isinstance(action, Iterable):
        return action
    was_tuple = isinstance(action, tuple)
    if was_tuple:
        action = list(action)
    if random.random() > 0.1:
        action[DECISION] = 0  # type: ignore
    if was_tuple:
        action = tuple(action)

    return action


@define
class NegotiationResult:
    advantage: float = 0.0
    partner_advantage: float = 0.0
    utility: float = 0.0
    partner_utility: float = 0.0
    time: float = 0.0
    steps: int = 0
    relative_time: float = 0.0
    pareto_optimality: float = float("nan")
    nash_optimality: float = float("nan")
    kalai_optimality: float = float("nan")
    modified_kalai_optimality: float = float("nan")
    max_welfare_optimality: float = float("nan")
    ks_optimality: float = float("nan")
    modified_ks_optimality: float = float("nan")
    opposition: float = float("nan")
    conflict: float = float("nan")
    agreement: Outcome | None = None
    scenario_stats: ScenarioStats | None = None


def accomulate_neg_results(
    neg_results: list[NegotiationResult],
) -> dict[str, float | int]:
    d = [asdict(_) for _ in neg_results]
    acc = dict()
    keys = asdict(NegotiationResult()).keys()
    for k in keys:
        if k in ("agreement", "scenario_stats"):
            continue
        x = np.asarray([_[k] for _ in d])
        if len(x) == 0:
            acc[f"{k}_mean"] = float("nan")
            acc[f"{k}_std"] = float("nan")
            acc[f"{k}_min"] = float("nan")
            acc[f"{k}_max"] = float("nan")
            continue
        acc[f"{k}_mean"] = mean(x)
        acc[f"{k}_std"] = std(x)
        acc[f"{k}_min"] = min(x)
        acc[f"{k}_max"] = max(x)
    return acc


@define
class TestResult:
    reward: float = 0.0
    n_negotiations: int = 0
    n_test_steps: int = 0
    pareto_optimality: float = float("nan")
    nash_optimality: float = float("nan")
    kalai_optimality: float = float("nan")
    modified_kalai_optimality: float = float("nan")
    max_welfare_optimality: float = float("nan")
    ks_optimality: float = float("nan")
    modified_ks_optimality: float = float("nan")
    total_time: float | None = None
    opposition_mean: float | None = None
    opposition_std: float | None = None
    conflict_mean: float | None = None
    conflict_std: float | None = None


@define
class Result:
    n_test_negotiations: int = 0
    test: TestResult = field(factory=TestResult)
    on_train: TestResult = field(factory=TestResult)
    before_training: TestResult = field(factory=TestResult)
    training_time: float | None = None
    # negotiations: list[NegotiationResult] = field(factory=list)


def type_representer(x: str | type) -> str:
    if isinstance(x, str):
        return x
    return get_full_type_name(x)


@define
class BaselineCondition:
    name: str
    negotiator: type[SAONegotiator]
    negotiator_params: dict[str, Any] = field(factory=dict)


@define
class Condition:
    name: str
    n_training_steps: int
    negotiator: type[SAORLNegotiator]
    policy: str | type[BasePolicy] = field(default=None, repr=type_representer)
    algorithm: type[BaseAlgorithm] = field(default=None, repr=type_representer)
    obs: type[ObservationEncoder] = field(default=None, repr=asdict)  # type: ignore
    action: type[ActionDecoder] = field(default=None, repr=asdict)  # type: ignore
    negotiator_params: dict[str, Any] = field(factory=dict)
    obs_params: dict[str, Any] = field(default=None)
    action_params: dict[str, Any] = field(default=None)
    algorithm_params: dict[str, Any] = field(default=None)
    policy_params: dict[str, Any] = field(default=None)
    learn_params: dict[str, Any] = field(factory=dict)

    def __attrs_post_init__(self):
        if self.policy is None:
            self.policy = self.negotiator.default_policy_type()
        if self.algorithm is None:
            self.algorithm = self.negotiator.default_trainer_type()
        if self.obs is None:
            self.obs = self.negotiator.default_obs_encoder_type()
        if self.action is None:
            self.action = self.negotiator.default_action_decoder_type()
        if self.obs_params is None:
            self.obs_params = self.negotiator.default_obs_encoder_params()
        else:
            self.obs_params = (
                self.negotiator.default_obs_encoder_params() | self.obs_params
            )
        if self.algorithm_params is None:
            self.algorithm_params = self.negotiator.default_trainer_params()
        else:
            self.algorithm_params = (
                self.negotiator.default_trainer_params() | self.algorithm_params
            )
        if self.policy_params is None:
            self.policy_params = self.negotiator.default_policy_params()
        else:
            self.policy_params = (
                self.negotiator.default_policy_params() | self.policy_params
            )
        if self.action_params is None:
            self.action_params = self.negotiator.default_action_decoder_params()
        else:
            self.action_params = (
                self.negotiator.default_action_decoder_params() | self.action_params
            )


@define
class ContextInfo:
    n_neg_steps: int | None
    time_limit: int | None
    partners: tuple[NegType, ...]
    scenario_generator: ScenarioGenerator
    partner_params: tuple[dict[str, Any], ...] | dict[str, Any] | None = None
    partner_generator: type[NegotiatorGenerator] = NegotiatorSampler
    mechanism_params: dict[str, Any] = field(factory=dict)

    def __attrs_post_init__(self):
        if self.partner_params is None:
            self.partner_params = dict()
        if isinstance(self.partner_params, dict):
            self.partner_params = tuple(
                deepcopy(self.partner_params) for _ in self.partners
            )
        assert (
            len(self.partners) == len(self.partner_params)
        ), f"Got {len(self.partners)} partners with {len(self.partner_params)} paramter dicts!!"


def _result_generator():
    return defaultdict(Result)


def _neg_result_generator():
    return defaultdict(list)


def _dict_repr(x: dict) -> str:
    return str({k: asdict(v) for k, v in x.items()})


@define
class Experiment:
    conditions: tuple[Condition, ...]
    training: ContextInfo
    testing: ContextInfo
    name: str = "experiment"
    always_starts: bool = True
    progress_bar: bool = True
    verbosity: int = 1
    path: Path | None = None
    baseline_conditions: tuple[BaselineCondition, ...] = tuple()
    results: dict[str, Result] = field(
        factory=_result_generator, init=False, repr=_dict_repr
    )
    neg_results: dict[str, dict[str, Any]] = field(factory=dict)
    models: dict[str, BaseAlgorithm | BaseModel] = field(
        init=False, factory=dict, repr=False
    )
    last_run: str = field(init=False)
    _run_paths: list[Path] = field(init=False, factory=list, repr=False)
    _condition_map: dict[str, Condition] = field(factory=dict, repr=False, init=False)
    _baselines_map: dict[str, BaselineCondition] = field(
        factory=dict, repr=False, init=False
    )

    @classmethod
    def from_condition_map(
        cls,
        conditions: dict[str, Condition],
        training: ContextInfo,
        testing: ContextInfo,
        **kwargs,
    ):
        return cls(
            conditions=tuple(conditions.values()),
            training=training,
            testing=testing,
            **kwargs,
        )

    def __attrs_post_init__(self):
        self._condition_map = dict(
            zip((_.name for _ in self.conditions), self.conditions)
        )
        assert (
            len(self._condition_map) == len(self.conditions)
        ), f"There are name repetitions in conditions:\n{[_.name for _ in self.conditions]}"

        self._baselines_map = dict(
            zip((_.name for _ in self.baseline_conditions), self.baseline_conditions)
        )
        assert (
            len(self._baselines_map) == len(self.baseline_conditions)
        ), f"There are name repetitions in baseline conditions:\n{[_.name for _ in self.baseline_conditions]}"

    def make_env(self, info: ContextInfo, condition_name: str):
        """Makes a trading environment."""
        condition = self._condition_map[condition_name]
        pgenerator = info.partner_generator()
        pgenerator.set_negotiators(info.partners, info.partner_params)
        return NegoEnv(
            scenario_generator=info.scenario_generator,
            partner_generator=pgenerator,
            mechanism_generator=MechanismRepeater(
                SAOMechanism,
                dict(n_steps=info.n_neg_steps, time_limit=info.time_limit)
                | info.mechanism_params,
            ),
            assigner=PositionBasedNegotiatorAssigner(always_starts=self.always_starts),
            obs_encoder=condition.obs(**condition.obs_params),
            action_decoder=condition.action(**condition.action_params),
        )

    def make_train_env(self, condition_name: str):
        return self.make_env(self.training, condition_name)

    def make_test_env(self, condition_name: str):
        """Makes a trading environment."""
        return self.make_env(self.testing, condition_name)

    def train(
        self,
        condition_name: str,
        path: Path | None = None,
        eval_freq: int | float = 0.001,
        save_freq: int | float | None = None,
        test_eval_freq: int | float | None = None,
        deterministic=True,
        save_replay_buffer=True,
        render=False,
    ) -> BaseAlgorithm:
        """Trains a model."""
        checkpoint_path = test_log_path = test_best_path = best_path = None
        env = self.make_env(self.training, condition_name)
        condition = self._condition_map[condition_name]
        if condition.algorithm is None:
            condition.algorithm = condition.negotiator.default_trainer_type()
        alg = condition.algorithm(
            policy=condition.policy, env=env, **condition.algorithm_params
        )
        condition.learn_params["tb_log_name"] = f"{self.name}:{self.last_run}"

        if 0 < eval_freq < 1:
            eval_freq = max(
                1,
                min(10, condition.n_training_steps // 2),
                int(condition.n_training_steps * eval_freq),
            )
        if save_freq is None:
            save_freq = int(eval_freq)
        if 0 < save_freq < 1:
            save_freq = max(
                1,
                min(10, condition.n_training_steps // 2),
                int(condition.n_training_steps * save_freq),
            )
        if test_eval_freq is None:
            test_eval_freq = int(eval_freq)
        if 0 < test_eval_freq < 1:
            test_eval_freq = max(
                1,
                min(10, condition.n_training_steps // 2),
                int(condition.n_training_steps * test_eval_freq),
            )
        base_path = path
        if base_path is not None:
            if eval_freq >= 1:
                path = base_path / "evals" / "train" / condition_name
                path.mkdir(parents=True, exist_ok=True)
                best_path = base_path / "best_train" / condition_name
                best_path.mkdir(parents=True, exist_ok=True)
            if test_eval_freq >= 1:
                test_log_path = base_path / "evals" / "test" / condition_name
                test_log_path.mkdir(parents=True, exist_ok=True)
                test_best_path = base_path / "best_test" / condition_name
                test_best_path.mkdir(parents=True, exist_ok=True)
            if save_freq:
                checkpoint_path = base_path / "checkpoints" / condition_name
                checkpoint_path.mkdir(parents=True, exist_ok=True)

        callbacks = []
        if base_path is not None:
            if save_freq >= 1:
                callbacks.append(
                    CheckpointCallback(
                        save_freq=int(save_freq),
                        save_path=str(checkpoint_path),
                        save_replay_buffer=save_replay_buffer,
                        name_prefix=f"{condition.name}-model",
                        verbose=self.verbosity > 2,
                    )
                )
            if eval_freq >= 1:
                callbacks.append(
                    EvalCallback(
                        eval_env=Monitor(self.make_env(self.training, condition_name)),
                        log_path=str(path),
                        best_model_save_path=str(best_path),
                        eval_freq=int(eval_freq),
                        deterministic=deterministic,
                        render=render,
                        verbose=self.verbosity > 2,
                    )
                )
            if test_eval_freq >= 1:
                callbacks.append(
                    EvalCallback(
                        eval_env=Monitor(self.make_env(self.testing, condition_name)),
                        log_path=str(test_log_path),
                        best_model_save_path=str(test_best_path),
                        eval_freq=int(eval_freq),
                        deterministic=deterministic,
                        render=render,
                        verbose=self.verbosity > 2,
                    )
                )
        if not callbacks:
            callbacks = None

        if base_path:
            formats = ["csv", "tensorboard"]
            if self.verbosity > 3:
                formats.append("stdout")
            log_path = base_path / "logs" / condition.name
            new_logger = configure(str(log_path), formats)
            alg.set_logger(new_logger)
        model = alg.learn(
            condition.n_training_steps,
            callback=callbacks,
            progress_bar=self.progress_bar,
            **condition.learn_params,
        )
        if self.verbosity:
            print(f"Training done with {env.n_negotiations} completed negotiations")
        self.models[condition_name] = model
        return model

    def test(
        self,
        condition_name: str,
        n_negs: int,
        on_train: bool = False,
        save_outcome_optimality: bool = True,
    ) -> TestResult:
        """Tests a model returning the total reward collected and number of complete negotiations finished."""
        model = self.models.get(condition_name, None)
        env = self.make_env(
            self.testing if not on_train else self.training, condition_name
        )
        if model is None:
            wrapped_model = partial(random_policy, env=env)
        else:
            wrapped_model = partial(wrapper, model=model)
        obs, _ = env.reset()
        reward = 0
        n_steps = 0
        oppositions, conflicts = [], []
        n_done = 0
        time_, start_ = None, perf_counter()
        pareto_optimality = 0.0 if save_outcome_optimality else float("nan")
        nash_optimality = 0.0 if save_outcome_optimality else float("nan")
        kalai_optimality = 0.0 if save_outcome_optimality else float("nan")
        modified_kalai_optimality = 0.0 if save_outcome_optimality else float("nan")
        max_welfare_optimality = 0.0 if save_outcome_optimality else float("nan")
        ks_optimality = 0.0 if save_outcome_optimality else float("nan")
        modified_ks_optimality = 0.0 if save_outcome_optimality else float("nan")
        while True:
            obs, r, terminated, truncated, _ = env.step(wrapped_model(obs))
            assert not isinstance(r, dict)
            reward += r  # type: ignore
            if terminated or truncated:
                s, m = env._scenario, env._mechanism
                utils = tuple(u(m.agreement) for u in s.ufuns)
                optimality = None
                outcomes = list(s.outcome_space.enumerate_or_sample())
                oppositions.append(
                    opposition_level(s.ufuns, issues=s.outcome_space.issues)
                )
                conflicts.append(conflict_level(s.ufuns[0], s.ufuns[1], outcomes))

                if save_outcome_optimality:
                    stats = calc_scenario_stats(s.ufuns)
                    optimality = calc_outcome_optimality(
                        calc_outcome_distances(utils, stats),
                        stats,
                        estimate_max_dist(s.ufuns),
                    )
                    pareto_optimality = (
                        pareto_optimality * n_done + optimality.pareto_optimality
                    ) / (n_done + 1)
                    nash_optimality = (
                        nash_optimality * n_done + optimality.nash_optimality
                    ) / (n_done + 1)
                    kalai_optimality = (
                        kalai_optimality * n_done + optimality.kalai_optimality
                    ) / (n_done + 1)
                    modified_kalai_optimality = (
                        modified_kalai_optimality * n_done
                        + optimality.modified_kalai_optimality
                    ) / (n_done + 1)
                    max_welfare_optimality = (
                        max_welfare_optimality * n_done
                        + optimality.max_welfare_optimality
                    ) / (n_done + 1)
                    ks_optimality = (
                        ks_optimality * n_done + optimality.ks_optimality
                    ) / (n_done + 1)
                    modified_ks_optimality = (
                        modified_ks_optimality * n_done
                        + optimality.modified_ks_optimality
                    ) / (n_done + 1)
                n_done += 1
                obs, _ = env.reset()
            if env.n_negotiations >= n_negs:
                break
            n_steps += 1
        time_ = perf_counter() - start_
        env.close()
        print(
            f"Testing {'[yellow]on-train[/yellow]' if on_train else ''} done with {env.n_negotiations} completed negotiations in {humanize_time(time_) if time_ is not None else 'unknown'}"
        )
        return TestResult(
            reward=reward,
            n_negotiations=n_done,
            n_test_steps=n_steps,
            pareto_optimality=pareto_optimality,
            nash_optimality=nash_optimality,
            kalai_optimality=kalai_optimality,
            modified_kalai_optimality=modified_kalai_optimality,
            max_welfare_optimality=max_welfare_optimality,
            ks_optimality=ks_optimality,
            modified_ks_optimality=modified_ks_optimality,
            total_time=time_,
            opposition_mean=float(mean(np.asarray(oppositions)))
            if oppositions
            else None,
            opposition_std=float(std(np.asarray(oppositions))) if oppositions else None,
            conflict_mean=float(mean(np.asarray(conflicts))) if conflicts else None,
            conflict_std=float(std(np.asarray(conflicts))) if conflicts else None,
        )

    def run_negotiations(
        self,
        condition_name: str,
        n: int,
        plot_every: int = 0,
        on_train: bool = False,
        raise_exceptions: bool = True,
        save_scenario_stats: bool = False,
        save_outcome_optimality: bool = True,
    ) -> list[NegotiationResult]:
        """Deploys a model and tests it in actual negotiations."""
        is_baseline = condition_name in self._baselines_map.keys()
        model = self.models.get(condition_name, None) if not is_baseline else None
        condition = (
            self._condition_map[condition_name]
            if not is_baseline
            else self._baselines_map[condition_name]
        )
        info = self.testing if not on_train else self.training
        pgenerator = info.partner_generator()
        pgenerator.set_negotiators(info.partners, info.partner_params)
        neg_results = []

        def mytrack(x, txt):
            print(txt)
            return x

        doprint = not self.progress_bar
        if not self.progress_bar:
            try:
                _ = [_ for _ in track(range(2))]
                self.progress_bar = False
                tracker = track
            except Exception:
                tracker, doprint = mytrack, True
        else:
            tracker = mytrack
        _strt = perf_counter()
        for _ in tracker(range(n), "Test Negotiations: "):
            if doprint:
                passed = perf_counter() - _strt
                print(f"{_+1} of {n} completed ETA: {passed * (n+1)/(_+1)}", end="\r")
            s = info.scenario_generator()
            if is_baseline:
                try:
                    learner = condition.negotiator(**condition.negotiator_params)
                except Exception as e:
                    if raise_exceptions:
                        raise e
                    print(
                        f"[red]ERROR[/red] Cannot create a baseline negotiator with exception {e}"
                    )
                    if self.verbosity > 1:
                        Console().print_exception(show_locals=SHOW_LOCALS)
                    continue
            else:
                try:
                    learner = condition.negotiator(
                        policy=partial(wrapper, model=model)  # type: ignore
                        if model is not None
                        else partial(
                            random_policy, env=self.make_env(info, condition_name)
                        ),
                        name="learner",
                        # name="seller",
                        **condition.negotiator_params,
                    )
                except Exception as e:
                    if raise_exceptions:
                        raise e
                    print(
                        f"[red]ERROR[/red] Cannot create a learner with exception {e}"
                    )
                    if self.verbosity > 1:
                        Console().print_exception(show_locals=SHOW_LOCALS)
                    continue
            m = SAOMechanism(
                outcome_space=s.outcome_space,
                n_steps=info.n_neg_steps,
                time_limit=info.time_limit,
                ignore_negotiator_exceptions=not raise_exceptions,
            )
            learner_loc = 0 if self.always_starts else int(random.random() > 0.5)
            m.add(learner if learner_loc == 0 else pgenerator(_), ufun=s.ufuns[0])  # type: ignore
            m.add(learner if learner_loc != 0 else pgenerator(_), ufun=s.ufuns[1])  # type: ignore
            # obs.on_negotiation_starts(seller, seller.nmi)
            # act.on_negotiation_starts(seller, seller.nmi)
            m.run()
            utility = s.ufuns[learner_loc](m.agreement)
            partner_utility = s.ufuns[1 - learner_loc](m.agreement)
            advantage = utility - s.ufuns[learner_loc].reserved_value
            partner_advantage = (
                partner_utility - s.ufuns[1 - learner_loc].reserved_value
            )
            stats = (
                calc_scenario_stats(s.ufuns)
                if save_scenario_stats or save_outcome_optimality
                else None
            )
            utils = tuple(u(m.agreement) for u in s.ufuns)
            outcomes = list(s.outcome_space.enumerate_or_sample())
            neg_results.append(
                NegotiationResult(
                    advantage=advantage,
                    partner_advantage=partner_advantage,
                    utility=utility,
                    partner_utility=partner_utility,
                    steps=m.current_step,
                    time=m.time,
                    relative_time=m.relative_time,
                    scenario_stats=stats if save_scenario_stats else None,
                    opposition=opposition_level(s.ufuns, issues=s.outcome_space.issues),
                    conflict=conflict_level(s.ufuns[0], s.ufuns[1], outcomes),
                    **(
                        asdict(
                            calc_outcome_optimality(
                                calc_outcome_distances(utils, stats),  # type: ignore
                                stats,  # type: ignore
                                estimate_max_dist(s.ufuns),
                            )
                        )
                    )
                    if save_outcome_optimality
                    else dict(),
                )
            )
            if self.verbosity > 2:
                print(
                    "\tExample ended with "
                    + (
                        f"[green]agreement[/green] {m.agreement}"
                        if m.agreement is not None
                        else "[red]disagreement[/red]"
                    )
                    + ": Learner gets "
                    + f"{s.ufuns[0](m.agreement):4.3}, Opponents get {[s.ufuns[_](m.agreement) for _ in range(1, len(s.ufuns))]}"
                )
            if plot_every and (_ % plot_every == 0):
                m.plot()
                plt.show()
        return neg_results

    def run(
        self,
        n_rand_negs: int = 0,
        n_test_negs: int = 0,
        plot_every: int = 0,
        on_train: bool = False,
        raise_exceptions: bool = True,
        raise_exceptions_in_depoyment: bool = False,
        override: bool = False,
        eval_freq: int | float = 0.001,
        test_eval_freq: int | float | None = None,
        save_freq: int | float | None = None,
        save_outcome_optimality: bool = True,
        save_scenario_stats_per_neg: bool = True,
        save_outcome_optimality_per_neg: bool = True,
    ) -> dict[str, Result]:
        if self.verbosity:
            print(f"[orange bold]Starting {self.name}[/orange bold]", flush=True)
        cond_names = [_.name for _ in self.conditions]
        baseline_cond_names = [_.name for _ in self.baseline_conditions]
        run_path, base_path = None, None
        rand_results, test_results, test_on_train_results = (
            TestResult(),
            TestResult(),
            TestResult(),
        )
        if self.path:
            base_path = self.path / self.name
            self.last_run = unique_name("", add_host=True, rand_digits=4, sep="")
            run_path = base_path / self.last_run
            run_path.mkdir(parents=True, exist_ok=True)
            self._run_paths.append(run_path)
            (run_path / MODELS_PATH).mkdir(parents=True, exist_ok=True)
            (run_path / RESULTS_PATH).mkdir(parents=True, exist_ok=True)
            dump(myserialize(self), run_path / f"{self.name}.json")

        combined_results = []
        combined_neg = []

        if base_path:
            combined_results_path = base_path / f"{RESULTS_PATH}.csv"
            combined_neg_path = base_path / f"{NEG_RESULTS_PATH}.csv"
        else:
            combined_results_path = None
            combined_neg_path = None
        if run_path:
            results_path = run_path / f"{RESULTS_PATH}.csv"
            neg_path = run_path / f"{NEG_RESULTS_PATH}.csv"
        else:
            results_path = None
            neg_path = None

        def process_results(
            condition: Condition | BaselineCondition,
            this_result: Result,
            neg_results: list[NegotiationResult],
        ):
            current_neg = dict(
                experiment=self.name,
                run_name=self.last_run,
                run_at=datetime.now(),
                condition=condition_name,
            ) | accomulate_neg_results(neg_results)
            current_results = flatten_dict(
                dict(
                    experiment=self.name,
                    run_name=self.last_run,
                    run_at=datetime.now(),
                    condition=condition_name,
                    test_reward=this_result.test.reward,
                    test_n_negs=this_result.test.n_negotiations,
                    test_n_test_steps=this_result.test.n_test_steps,
                    train_reward=this_result.on_train.reward,
                    train_n_negs=this_result.on_train.n_negotiations,
                    train_n_test_steps=this_result.on_train.n_test_steps,
                    n_test_negs=n_test_negs,
                    n_rand_negs=n_rand_negs,
                )
                | asdict(this_result, recurse=True)
            )
            if results_path:
                add_records(results_path, [current_results])
            if combined_results_path:
                combined_results.append(current_results)
                if self.verbosity:
                    print(
                        f"[green]{name}@{self.name}[/green]: Test Negotiation Advantage: {current_neg['advantage_mean']} ({current_neg['advantage_std']})"
                    )
            if neg_path:
                add_records(neg_path, [current_neg])
            if combined_neg_path:
                combined_neg.append(current_neg)
            self.results[name] = this_result
            self.neg_results[name] = current_neg
            if run_path is not None:
                dump(
                    asdict(this_result),
                    run_path / RESULTS_PATH / f"{condition.name}.json",
                )
                dump(
                    current_neg,
                    run_path / RESULTS_PATH / f"{condition.name}_negs.json",
                )
                dump(
                    myserialize(condition),
                    run_path / RESULTS_PATH / f"{condition.name}_info.json",
                )

        for name, condition in zip(cond_names, self.conditions):
            if self.verbosity:
                print(
                    f"[green bold]Working on {name}[/green bold] at {datetime.now()}",
                    flush=True,
                )
            condition_name = condition.name
            if n_rand_negs:
                rand_results = self.test(
                    condition_name,
                    n_rand_negs,
                    save_outcome_optimality=save_outcome_optimality,
                )
                if self.verbosity:
                    print(
                        f"{name}: Reward for random behavior: {rand_results.reward:4.3} in {rand_results.n_test_steps} test-steps ({rand_results.reward/rand_results.n_test_steps: 4.3})"
                    )
            training_time, start_ = None, perf_counter()
            try:
                model = self.train(
                    condition_name,
                    run_path,
                    eval_freq=eval_freq,
                    test_eval_freq=test_eval_freq,
                    save_freq=save_freq,
                )
                training_time = perf_counter() - start_
            except Exception as e:
                if raise_exceptions:
                    raise e
                print(
                    f"{name}: [red]ERROR Training[/red]: [bold red]{name}[/bold red]: Failed in training: {e}"
                )
                if self.verbosity > 1:
                    Console().print_exception(show_locals=SHOW_LOCALS)
                continue
            try:
                if run_path is not None:
                    model.save(run_path / MODELS_PATH / condition.name)
            except Exception as e:
                if raise_exceptions:
                    raise e
                print(
                    f"{name}: [red]ERROR Saving[/red]: [bold red]{name}[/bold red]: Failed in saving model: {e}"
                )
                if self.verbosity > 1:
                    Console().print_exception(show_locals=SHOW_LOCALS)

            train_reward, train_steps = 0.0, 0
            reward, test_steps = 0.0, 0
            try:
                test_results = self.test(
                    condition_name,
                    n_test_negs,
                    save_outcome_optimality=save_outcome_optimality,
                )
                testing_time = perf_counter() - start_
            except Exception as e:
                if raise_exceptions:
                    raise e
                print(
                    f"{name}: [red]ERROR Testing[/red]: [bold red]{name}[/bold red]: Failed in testing model: {e}"
                )
                if self.verbosity > 1:
                    Console().print_exception(show_locals=SHOW_LOCALS)
            if on_train:
                test_on_train_results = self.test(
                    condition_name,
                    n_test_negs,
                    on_train=True,
                    save_outcome_optimality=save_outcome_optimality,
                )
            if self.verbosity:
                if test_steps:
                    print(
                        f"{name}: Reward for trained model: {reward:4.3} in {test_steps} test-steps ({reward/test_steps: 4.3})"
                    )
                if train_steps:
                    print(
                        f"{name}: Reward for trained model [yellow](on-train)[/yellow]: {train_reward:4.3} in {train_steps} train-steps ({train_reward/train_steps: 4.3})"
                    )
            neg_results = self.run_negotiations(
                condition_name,
                n_test_negs,
                on_train=on_train,
                plot_every=plot_every,
                raise_exceptions=raise_exceptions_in_depoyment,
                save_scenario_stats=save_scenario_stats_per_neg,
                save_outcome_optimality=save_outcome_optimality_per_neg,
            )
            this_result = Result(
                training_time=training_time,
                n_test_negotiations=n_test_negs,
                test=test_results,
                on_train=test_on_train_results,
                before_training=rand_results,
            )
            process_results(condition, this_result, neg_results)

        for name, condition in zip(baseline_cond_names, self.baseline_conditions):
            if self.verbosity:
                print(
                    f"[yellow bold]Working on baseline {name}[/yellow bold] on {datetime.now()}",
                    flush=True,
                )
            condition_name = condition.name
            neg_results = self.run_negotiations(
                condition_name,
                n_test_negs,
                on_train=on_train,
                plot_every=plot_every,
                raise_exceptions=raise_exceptions_in_depoyment,
            )
            this_result = Result(
                n_test_negotiations=0,
                test=TestResult(),
                on_train=TestResult(),
                before_training=TestResult(),
            )
            process_results(condition, this_result, neg_results)

        if combined_results_path:
            if combined_results_path.exists():
                add_records(combined_results_path, combined_results)
            else:
                pd.DataFrame.from_records(combined_results).to_csv(
                    combined_results_path, index=False
                )

        if combined_neg_path:
            if combined_neg_path.exists():
                add_records(combined_neg_path, combined_neg)
            else:
                pd.DataFrame.from_records(combined_neg).to_csv(
                    combined_neg_path, index=False
                )
        if base_path is not None:
            with open(base_path / "last_run.txt", "w") as f:
                f.write(self.last_run)

            dump(
                {k: asdict(v) for k, v in self.results.items()},
                base_path / f"results_{self.last_run}.json",
            )
            # dump(myserialize(asdict(self)), base_path / f"{self.name}.json")
        return self.results

    def load_models(
        self, path: Path | None = None, on_train: bool = True
    ) -> dict[str, BaseAlgorithm | BaseModel]:
        if path is None and self.path is None:
            raise ValueError(
                "You must pass path to load_models or have path set in the Experiment."
            )
        if path is None and self.path is not None:
            with open(self.path / self.name / "last_run.txt") as f:
                last_run = f.read().replace("\n", "")
            path = self.path / self.name / last_run

        if path is None:
            raise ValueError(
                "You must pass path to load_models or have path set in the Experiment."
            )
        if path.name != MODELS_PATH and (path / MODELS_PATH).is_dir():
            path = path / MODELS_PATH
        for condition in self.conditions:
            model_path = path / f"{condition.name}.zip"
            if not model_path.is_file():
                continue

            env = (
                self.make_train_env(condition.name)
                if on_train
                else self.make_test_env(condition.name)
            )
            alg = condition.algorithm(
                policy=condition.policy, env=env, **condition.algorithm_params
            )
            self.models[condition.name] = alg.load(model_path, env)
        return self.models
