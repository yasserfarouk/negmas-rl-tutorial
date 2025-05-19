from typing import Any, TypeVar
from unittest.mock import Mock

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from negmas import (
    AffineFun,
    CartesianOutcomeSpace,
    GBNegotiator,
    Issue,
    LinearFun,
    LinearUtilityAggregationFunction,
    Outcome,
    RandomNegotiator,
    ResponseType,
    SAOMechanism,
    SAONegotiator,
    SAOResponse,
    TableFun,
    make_os,
)
from negmas.genius.gnegotiators import PNegotiator
from negmas.inout import Scenario
from negmas.outcomes import make_issue

from negmas_rl.common import INT_TYPE
from negmas_rl.env.gymenv import NegoEnv
from negmas_rl.generators.assigner import PositionBasedNegotiatorAssigner
from negmas_rl.generators.mechanism import MechanismRepeater
from negmas_rl.generators.negotiator import NegotiatorRepeater
from negmas_rl.generators.scenario import ScenarioRepeater
from negmas_rl.negotiator import SAOPlaceholder
from negmas_rl.obs import (
    CIssueEncoder,
    COutcomeEncoder,
    CRankEncoder,
    CTimeEncoder,
    CTimeUtilityBoxEncoder,
    CTimeUtilityDictEncoder,
    CTimeUtilityFlatEncoder,
    CTimeUtilityTupleEncoder,
    CUtilityEncoder,
    CWindowedIssueEncoder,
    CWindowedOutcomeEncoder,
    CWindowedRankEncoder,
    CWindowedUtilityEncoder,
    DIssueEncoder,
    DOutcomeEncoder,
    DOutcomeEncoder1D,
    DOutcomeEncoderND,
    DRankEncoder,
    DRankEncoder1D,
    DTimeEncoder,
    DTimeUtilityBoxEncoder,
    DTimeUtilityDictEncoder,
    DTimeUtilityFlatEncoder,
    DTimeUtilityTupleEncoder,
    DUtilityEncoder,
    DWindowedIssueEncoder,
    DWindowedOutcomeEncoder,
    DWindowedRankEncoder,
    DWindowedUtilityEncoder,
    MiPNEncoder,
    ObservationEncoder,
    RLBoaEncoder,
    SenguptaEncoder,
    VeNASEncoder,
)

ENCODERS = [
    CIssueEncoder,
    COutcomeEncoder,
    CRankEncoder,
    CTimeEncoder,
    CTimeUtilityBoxEncoder,
    CTimeUtilityDictEncoder,
    CTimeUtilityFlatEncoder,
    CTimeUtilityTupleEncoder,
    CUtilityEncoder,
    CWindowedIssueEncoder,
    CWindowedOutcomeEncoder,
    CWindowedRankEncoder,
    CWindowedUtilityEncoder,
    DIssueEncoder,
    DOutcomeEncoder,
    DOutcomeEncoder,
    DOutcomeEncoder1D,
    DOutcomeEncoderND,
    DRankEncoder,
    DRankEncoder,
    DRankEncoder1D,
    DTimeEncoder,
    DTimeUtilityBoxEncoder,
    DTimeUtilityDictEncoder,
    DTimeUtilityFlatEncoder,
    DTimeUtilityTupleEncoder,
    DUtilityEncoder,
    DWindowedIssueEncoder,
    DWindowedOutcomeEncoder,
    DWindowedRankEncoder,
    DWindowedUtilityEncoder,
    MiPNEncoder,
    RLBoaEncoder,
    SenguptaEncoder,
    VeNASEncoder,
]
NTRIALS = 300
NFEWTRIALS = 80
N, R, T = 10, 2.0, 100
DEADLINE_PER_TEST = 5000
LONG_DEADLINE_PER_TEST = 2 * DEADLINE_PER_TEST
NOUTCOMES = [0, T, 12, 40, 50]
NOUTCOMES = [0]
NLEVELS = [3, N, int(N * 1.34)]
NLEVELS = [0]
NOFFERS = (1, 10)
# with negatives
MIN_ENCODABLE = [-1, 0, 0.5]
ENCODABLE_LENGTH = [0.5, 1, 2]
# no negatives
# MIN_ENCODABLE = [0, 0.5]
# ENCODABLE_LENGTH = [0.5, 1, 2]
# range 0 - 1
MIN_ENCODABLE = [0]
ENCODABLE_LENGTH = [1]


def make_offer(i, first_categorical: bool, n_issues: int, all_equal: bool = True, N=N):
    i = i % N
    if all_equal:
        if first_categorical:
            return SAOResponse(
                ResponseType.REJECT_OFFER, tuple([f"{i}"] + ([i] * (n_issues - 1)))
            )
        return SAOResponse(ResponseType.REJECT_OFFER, tuple([i] * n_issues))
    else:
        if first_categorical:
            return SAOResponse(
                ResponseType.REJECT_OFFER,
                tuple([f"{i}"] + ([_ for _ in range(n_issues - 1)])),
            )
        return SAOResponse(
            ResponseType.REJECT_OFFER, tuple([_ for _ in range(n_issues)])
        )


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_env_making(
    n_issues,
    always_starts,
    first_categorical,
):
    encoder, env = make_encoder_and_env(
        CTimeEncoder,
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert env is not None
    assert encoder is not None
    env.reset()
    issues = env._mechanism.issues
    assert len(issues) == n_issues, f"{len(issues)=} but {n_issues=}, {issues=}"
    if always_starts:
        learner = env._mechanism.negotiators[0]
        assert isinstance(learner, SAOPlaceholder), (
            "Always starts but first negotiator is not a learner"
        )
    else:
        learner = env._mechanism.negotiators[-1]
        assert isinstance(learner, SAOPlaceholder), (
            "Never starts but last negotiator is not a learner"
        )
    if first_categorical:
        assert not issues[0].is_numeric(), f"First is not categorical {issues=}"
        for i, v in enumerate(issues[0].all):
            assert isinstance(v, str)
            assert int(v) == i
    else:
        assert issues[0].is_numeric(), f"First is not numeric {issues=}"
    if n_issues > 1:
        for i in range(1, n_issues):
            assert issues[i].is_numeric(), f"Issue {i} is not numeric {issues=}"


@pytest.mark.parametrize("type_", ENCODERS)
def test_ob_manger_can_sample(type_: type[ObservationEncoder]):
    t = type_()
    space = t.make_space()
    assert space is not None
    for _ in range(20):
        sample = space.sample()
        assert sample in space


def test_ob_manger_can_sample_example():
    t = DWindowedOutcomeEncoder()
    space = t.make_space()
    for _ in range(10):
        sample = space.sample()
        assert sample in space


EncType = TypeVar("EncType", bound=ObservationEncoder)


def make_encoder_and_env(
    type_: type[EncType],
    partner_type: type[SAONegotiator]
    | type[PNegotiator]
    | type[GBNegotiator] = RandomNegotiator,
    partner_params: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    n_issues: int = 1,
    always_starts: bool = True,
    first_categorical: bool = False,
    partner_utility: bool = False,
) -> tuple[EncType, NegoEnv]:
    if issubclass(partner_type, RandomNegotiator) and partner_params is None:
        partner_params = dict(p_acceptance=0.0, p_rejection=1.0, p_ending=0.0)
    elif partner_params is None:
        partner_params = dict()
    if params is None:
        params = dict()
    encoder = type_(**params)
    if first_categorical:
        os = make_os(
            [make_issue([f"{i}" for i in range(N)], "cat")]
            + [make_issue(N, f"price{_}") for _ in range(n_issues - 1)]
        )
        ufun = LinearUtilityAggregationFunction(
            values=tuple(
                [
                    TableFun(
                        dict(
                            zip(
                                os.issues[0].all,
                                [float(_) / N for _ in os.issues[0].values],
                            )
                        )
                    )
                ]
                + ([LinearFun(1.0 / N)] * (n_issues - 1))
            ),
            weights=tuple([1.0 / n_issues] + ([1.0 / n_issues] * (n_issues - 1))),
            outcome_space=os,
            reserved_value=R / N,
        )
        opp_ufun = LinearUtilityAggregationFunction(
            values=tuple(
                [
                    TableFun(
                        dict(
                            zip(
                                os.issues[0].values,
                                [1.0 - float(_) / N for _ in os.issues[0].values],
                            )
                        )
                    )
                ]
                + ([AffineFun(slope=-1.0 / N, bias=1.0)] * (n_issues - 1))
            ),
            weights=tuple([1.0 / n_issues] + ([1.0 / n_issues] * (n_issues - 1))),
            outcome_space=os,
            reserved_value=R / N,
        )
    else:
        os = make_os([make_issue(N, f"price{_}") for _ in range(n_issues)])
        ufun = LinearUtilityAggregationFunction(
            values=tuple([LinearFun(1.0 / N)] * n_issues),
            weights=tuple([1.0 / n_issues] * n_issues),
            outcome_space=os,
            reserved_value=R / N,
        )
        opp_ufun = LinearUtilityAggregationFunction(
            values=tuple([AffineFun(slope=-1.0 / N, bias=1.0)] * n_issues),
            weights=tuple([1.0 / n_issues] * n_issues),
            outcome_space=os,
            reserved_value=R / N,
        )
    owner = Mock()
    owner.ufun = ufun
    encoder.owner = owner

    env = NegoEnv(
        scenario_generator=ScenarioRepeater(
            Scenario(outcome_space=os, ufuns=(ufun, opp_ufun))
        ),
        mechanism_generator=MechanismRepeater(
            SAOMechanism, dict(n_steps=T, time_limit=None)
        ),
        assigner=PositionBasedNegotiatorAssigner(
            always_starts=always_starts, always_ends=not always_starts
        ),
        partner_generator=NegotiatorRepeater(partner_type, partner_params),
        obs_encoder=encoder,
        placeholder_params=dict(
            private_info=dict(opponent_ufun=opp_ufun if always_starts else ufun)
        )
        if partner_utility
        else dict(),
    )
    return encoder, env


@pytest.mark.parametrize("type_", ENCODERS)
def test_basic_obs_encoder_basic(type_: type[ObservationEncoder]):
    encoder, env = make_encoder_and_env(type_)
    env.reset()

    space = encoder.make_space()
    x = space.sample()
    assert x in space

    learner = env._mechanism.negotiators[0]
    for i in range(20):
        obs, r, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi, make_offer(i, first_categorical=False, n_issues=1)
            )
        )
        assert obs in space
        assert not isinstance(r, dict)
        # if _ > 3:
        #     assert False, f"{obs=}, {r=}, {terminated=},{truncated=}, {info=}, {obs_obs.children[1]._first_offer=},{obs_obs=}"
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


def test_basic_obs_encoder_example():
    type_ = MiPNEncoder
    encoder, env = make_encoder_and_env(type_)
    env.reset()

    space = encoder.make_space()
    x = space.sample()
    assert x in space

    learner = env._mechanism.negotiators[0]
    for i in range(20):
        obs, r, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi, make_offer(i, first_categorical=False, n_issues=1)
            )
        )
        assert obs in space
        assert not isinstance(r, dict)
        # if _ > 3:
        #     assert False, f"{obs=}, {r=}, {terminated=},{truncated=}, {info=}, {obs_obs.children[1]._first_offer=},{obs_obs=}"
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
    min_encodable=st.sampled_from(MIN_ENCODABLE),
    encodable_length=st.sampled_from(ENCODABLE_LENGTH),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_basic_obs_encoder_ctime(
    n_issues, always_starts, first_categorical, min_encodable, encodable_length
):
    encoder, env = make_encoder_and_env(
        CTimeEncoder,
        params=dict(
            min_encodable=min_encodable, max_encodable=min_encodable + encodable_length
        ),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    env.reset()

    space = encoder.make_space()
    learner = env._mechanism.negotiators[int(not always_starts)]
    for j in range(NTRIALS):
        obs, r, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        assert not isinstance(r, dict)
        assert obs in space, f"{obs=} not in {space=}\n({space.sample()=})"
        assert (
            abs(encoder.decode(learner.nmi, obs) - env._mechanism.relative_time) < 1e-3
        )
        # if _ > 3:
        #     assert False, f"{obs=}, {r=}, {terminated=},{truncated=}, {info=}, {obs_obs.children[1]._first_offer=},{obs_obs=}"
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
)
@settings(deadline=DEADLINE_PER_TEST)
@example(n_issues=1, always_starts=False, first_categorical=False)
def test_basic_obs_encoder_dtime(n_issues, always_starts, first_categorical):
    encoder, env = make_encoder_and_env(
        DTimeEncoder,
        params=dict(n_levels=T),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    env.reset()

    space = encoder.make_space()
    x = space.sample()
    assert x in space
    learner = env._mechanism.negotiators[int(not always_starts)]
    for j in range(NTRIALS):
        obs = encoder.encode(learner.nmi)
        assert obs in space, f"{obs=} not in {space=} ({space.sample()=})"
        assert obs == (env._mechanism.current_step) or (
            obs == encoder.n_levels - 1
            and env._mechanism.current_step >= env._mechanism.n_steps  # type: ignore
        ), (
            f"{obs=} but step={env._mechanism.current_step}, relative {env._mechanism.relative_time}, nsteps {env._mechanism.n_steps}\n{env._mechanism.state}\n{encoder}"
        )
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        # if _ > 3:
        #     assert False, f"{obs=}, {r=}, {terminated=},{truncated=}, {info=}, {obs_obs.children[1]._first_offer=},{obs_obs=}"
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_levels=st.sampled_from([50, T, 150]),
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_basic_obs_encoder_dtime_variable(
    n_levels, n_issues, always_starts, first_categorical
):
    encoder, env = make_encoder_and_env(
        DTimeEncoder,
        params=dict(n_levels=n_levels),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    env.reset()

    space = encoder.make_space()
    x = space.sample()
    assert x in space
    prev = 0
    learner = env._mechanism.negotiators[int(not always_starts)]
    for j in range(NTRIALS):
        obs = encoder.encode(learner.nmi)
        assert not isinstance(obs, dict)
        # observations are monotonically increasing
        assert obs >= prev, f"{obs=}, {prev=} for negotiation with {n_levels} levels"
        prev = max(obs, prev)  # type: ignore
        # observations start at 0 and end at n-1
        if env._mechanism.current_step == 0:
            assert obs == 0
        assert env._mechanism.n_steps is not None
        assert abs(
            encoder.decode(learner.nmi, obs) - env._mechanism.state.relative_time
        ) <= (1 / encoder.n_levels + 1e-2)
        if env._mechanism.current_step > env._mechanism.n_steps - 1:
            assert obs == n_levels - 1
        elif env._mechanism.current_step + 1 >= env._mechanism.n_steps:
            assert obs == int(env._mechanism.state.relative_time * encoder.n_levels)
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        # if _ > 3:
        #     assert False, f"{obs=}, {r=}, {terminated=},{truncated=}, {info=}, {obs_obs.children[1]._first_offer=},{obs_obs=}"
        if terminated or truncated:
            prev = 0
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    ignore=st.booleans(),
    missing_as_none=st.booleans(),
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
    min_encodable=st.sampled_from(MIN_ENCODABLE),
    encodable_length=st.sampled_from(ENCODABLE_LENGTH),
    n_offers=st.integers(*NOFFERS),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_basic_obs_encoder_cwindowed_util(
    ignore,
    missing_as_none,
    n_issues,
    always_starts,
    first_categorical,
    min_encodable,
    encodable_length,
    n_offers,
):
    encoder, env = make_encoder_and_env(
        CWindowedUtilityEncoder,
        params=dict(
            n_offers=n_offers,
            ignore_own_offers=ignore,
            missing_as_none=missing_as_none,
            min_encodable=min_encodable,
            max_encodable=min_encodable + encodable_length,
        ),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, CWindowedUtilityEncoder)
    env.reset()

    space = encoder.make_space()
    for j in range(NTRIALS):
        assert isinstance(env._mechanism, SAOMechanism)
        learner = env._mechanism.negotiators[int(not always_starts)]
        assert learner.ufun
        if not ignore:
            offers = env._mechanism.offers
        else:
            trace = env._mechanism.trace
            offers = [_[-1] for _ in trace if _[0] != learner.id]
        offers.reverse()
        offers = offers[: encoder.n_offers]
        if encoder.missing_as_none:
            utils = [float(learner.ufun(_)) for _ in offers]
        else:
            utils = [float(learner.ufun(_)) for _ in offers] + [
                float(learner.ufun.reserved_value)
            ] * (encoder.n_offers - len(offers))

        obs = encoder.encode(learner.nmi)
        assert obs in space, (
            f"{obs=} not in {space=} (example sample {space.sample()=})"
        )
        assert np.all(min_encodable <= obs) and np.all(
            obs <= min_encodable + encodable_length
        ), f"{obs=}"
        assert len(obs) == encoder.n_offers
        assert isinstance(obs, np.ndarray)
        assert len(obs) == encoder.n_offers

        unscaled = encoder.decode(learner.nmi, obs)
        np.testing.assert_almost_equal(
            unscaled,
            utils,
            err_msg=(
                f"obs not expected:\n{obs=}\n{unscaled=}\n{utils=}\n{encoder}"
                f"\n{env._mechanism.state=}\n{learner.ufun.minmax()=}\n{offers=}"
            ),
        )
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        # if _ > 3:
        #     assert False, f"{obs=}, {r=}, {terminated=},{truncated=}, {info=}, {obs_obs.children[1]._first_offer=},{obs_obs=}"
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
    missing_as_none=st.booleans(),
    ignore=st.booleans(),
    n_offers=st.integers(*NOFFERS),
)
@settings(deadline=DEADLINE_PER_TEST)
@example(
    n_issues=1,
    always_starts=False,
    first_categorical=False,
    missing_as_none=True,
    ignore=False,
    n_offers=1,
)
def test_basic_obs_encoder_dwindowed_util(
    ignore, missing_as_none, n_issues, always_starts, first_categorical, n_offers
):
    encoder, env = make_encoder_and_env(
        DWindowedUtilityEncoder,
        params=dict(
            n_offers=n_offers,
            ignore_own_offers=ignore,
            missing_as_none=missing_as_none,
        ),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, DWindowedUtilityEncoder)
    env.reset()

    space = encoder.make_space()
    x = space.sample()
    assert x in space
    for j in range(NTRIALS):
        assert isinstance(env._mechanism, SAOMechanism)
        learner = env._mechanism.negotiators[int(not always_starts)]
        assert learner.ufun
        if not ignore:
            offers = env._mechanism.offers
        else:
            trace = env._mechanism.trace
            offers = [_[-1] for _ in trace if _[0] != learner.id]
        offers.reverse()
        offers = offers[: encoder.n_offers]
        if encoder.missing_as_none:
            utils = [float(learner.ufun(_)) for _ in offers]
        else:
            utils = [float(learner.ufun(_)) for _ in offers] + [
                float(learner.ufun.reserved_value)
            ] * (encoder.n_offers - len(offers))
        obs = encoder.encode(learner.nmi)
        assert obs in space, f"{obs=} not in {space=} (example: {space.sample()=})"
        assert len(obs) == encoder.n_offers
        assert isinstance(obs, np.ndarray)
        assert len(obs) == encoder.n_offers
        unscaled = encoder.decode(learner.nmi, obs)
        assert np.all(
            np.abs(np.asarray(unscaled) - np.asarray(utils)) < (2 / encoder.n_levels)
        ), (
            f"obs not expected:\n{obs=}\n{unscaled=}\n{utils=}\n{encoder}"
            f"\n{env._mechanism.state=}\n{learner.ufun.minmax()=}\n{offers=}"
        )
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
    min_encodable=st.sampled_from(MIN_ENCODABLE),
    encodable_length=st.sampled_from(ENCODABLE_LENGTH),
)
@settings(deadline=DEADLINE_PER_TEST)
@example(
    n_issues=1,  # or any other generated value
    always_starts=True,
    first_categorical=False,  # or any other generated value
    min_encodable=0,
    encodable_length=1,
)
def test_basic_obs_encoder_cutility(
    n_issues, always_starts, first_categorical, min_encodable, encodable_length
):
    encoder, env = make_encoder_and_env(
        CUtilityEncoder,
        params=dict(
            min_encodable=min_encodable, max_encodable=encodable_length + min_encodable
        ),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )

    env.reset()

    space = encoder.make_space()
    learner = env._mechanism.negotiators[int(not always_starts)]
    for j in range(NTRIALS):
        obs, r, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi, make_offer(j, first_categorical, n_issues)
            )
        )
        offer = env._mechanism.nmi.state.current_offer
        util = float(learner.ufun(offer))
        obs = encoder.encode(learner.nmi)
        assert not isinstance(r, dict)
        assert obs in space, f"{obs=} not in {space=}\n({space.sample()=})"
        decoded = encoder.decode(learner.nmi, obs)
        assert np.abs(decoded[0] - util) < 1e-3, (
            f"{obs=}, {decoded=}, {util=}, {offer=}"
        )
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
)
@settings(deadline=DEADLINE_PER_TEST)
@example(n_issues=1, always_starts=False, first_categorical=False)
def test_basic_obs_encoder_dutility(n_issues, always_starts, first_categorical):
    encoder, env = make_encoder_and_env(
        DUtilityEncoder,
        params=dict(n_levels=T),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    env.reset()

    space = encoder.make_space()
    x = space.sample()
    assert x in space
    learner = env._mechanism.negotiators[int(not always_starts)]
    for j in range(NTRIALS):
        obs = encoder.encode(learner.nmi)
        assert obs in space, f"{obs=} not in {space=} ({space.sample()=})"
        obs_real = np.minimum(encoder.n_levels - 1, (obs / encoder.n_levels))
        u = learner.ufun(env._mechanism.nmi.state.current_offer)
        expected = np.minimum(
            encoder.n_levels - 1,
            np.asarray(u * encoder.n_levels).astype(INT_TYPE),
        )
        assert np.all(np.abs(obs - expected) <= 1), (
            f"{obs=} {obs_real=}, {expected=}, offer={env._mechanism.nmi.state.current_offer}, util={u}, "
            f"scaled-util={expected} (min-encodable={encoder.min_encodable})\n{encoder=}\n{env._mechanism.nmi.state}"
        )
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        # if _ > 3:
        #     assert False, f"{obs=}, {r=}, {terminated=},{truncated=}, {info=}, {obs_obs.children[1]._first_offer=},{obs_obs=}"
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
    min_encodable=st.sampled_from(MIN_ENCODABLE),
    encodable_length=st.sampled_from(ENCODABLE_LENGTH),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_basic_obs_encoder_cissue(
    n_issues, always_starts, first_categorical, min_encodable, encodable_length
):
    encoder, env = make_encoder_and_env(
        CIssueEncoder,
        params=dict(
            n_issues=n_issues,
            min_encodable=min_encodable,
            max_encodable=min_encodable + encodable_length,
        ),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    env.reset()

    space = encoder.make_space()
    x = space.sample()
    assert x in space
    learner = env._mechanism.negotiators[int(not always_starts)]
    for j in range(NTRIALS):
        obs = encoder.encode(learner.nmi)
        assert obs in space, f"{obs=} not in {space=} ({space.sample()=})"
        offer = learner.nmi.state.current_offer  # type: ignore
        decoded = encoder.decode(learner.nmi, obs)
        assert offer == decoded, f"{obs=}, {decoded=}, {offer=}\n{encoder=}"
        if env._mechanism.nmi.state.current_offer is None:
            assert np.all(obs == 0)
        else:
            for i, (val, issue) in enumerate(
                zip(
                    env._mechanism.nmi.state.current_offer,
                    learner.ufun.outcome_space.issues,
                )
            ):
                if not issue.is_numeric():
                    x = (float(val) - float(issue.values[0])) / (
                        float(issue.values[-1]) - float(issue.values[0])
                    )
                else:
                    x = (val - issue.min_value) / (issue.max_value - issue.min_value)
                assert abs(obs[i] - encoder.scale(x)) < 1e-3, (
                    f"{obs=}, {obs[i]=}, {val=}, {issue.values=}"
                )
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        # if _ > 3:
        #     assert False, f"{obs=}, {r=}, {terminated=},{truncated=}, {info=}, {obs_obs.children[1]._first_offer=},{obs_obs=}"
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_basic_obs_encoder_dissue(n_issues, always_starts, first_categorical):
    encoder, env = make_encoder_and_env(
        DIssueEncoder,
        params=dict(n_issues=n_issues, n_levels=N),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    env.reset()

    space = encoder.make_space()
    x = space.sample()
    assert x in space
    learner = env._mechanism.negotiators[int(not always_starts)]
    issues = learner.ufun.outcome_space.issues
    for j in range(NTRIALS):
        outcome = env._mechanism.nmi.state.current_offer
        obs = encoder.encode(learner.nmi)
        assert obs in space, f"{obs=} not in {space=} ({space.sample()=})"
        decoded = encoder.decode(learner.nmi, obs)
        assert decoded == outcome, (
            f"{obs=}, {decoded=}, {outcome=}, {outcome=}\n{encoder=}"
        )
        # if env._mechanism.nmi.state.current_offer is None:
        #     assert np.all(obs == 0)
        # else:
        #     for i, (val, issue) in enumerate(zip(outcome, issues)):
        #         # levels = np.asarray(encoder.n_levels_per_issue)
        #         if issue.is_numeric():
        #             assert (
        #                 obs[i] - 1 == val
        #             ), f"{obs=}, {obs[i]=}, {val=}, {issue.values=}, offer: {outcome}"
        #         else:
        #             assert obs[i] - 1 == int(
        #                 val
        #             ), f"{obs=}, {obs[i]=}, {val=}, {issue.values=}, offer: {outcome}"
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        # if _ > 3:
        #     assert False, f"{obs=}, {r=}, {terminated=},{truncated=}, {info=}, {obs_obs.children[1]._first_offer=},{obs_obs=}"
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    n_outcomes=st.sampled_from(NOUTCOMES),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
    order_by_similarity=st.booleans(),
    sort_by_utility=st.booleans(),
)
@settings(deadline=LONG_DEADLINE_PER_TEST)
@example(
    n_issues=2,
    n_outcomes=0,
    always_starts=False,  # or any other generated value
    first_categorical=False,  # or any other generated value
    order_by_similarity=False,  # or any other generated value
    sort_by_utility=False,
)
def test_basic_obs_encoder_doutcome(
    n_issues,
    always_starts,
    first_categorical,
    n_outcomes,
    order_by_similarity,
    sort_by_utility,
):
    if n_outcomes == 0:
        n_outcomes = 1
        for _ in range(n_issues):
            n_outcomes *= N
    encoder, env = make_encoder_and_env(
        DOutcomeEncoder,
        params=dict(
            n_outcomes=n_outcomes,
            order_by_similarity=order_by_similarity,
            sort_by_utility=sort_by_utility,
        ),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    env.reset()

    space = encoder.make_space()
    x = space.sample()
    assert x in space
    learner = env._mechanism.negotiators[int(not always_starts)]
    for j in range(NFEWTRIALS):
        outcome = env._mechanism.nmi.state.current_offer
        obs = encoder.encode(learner.nmi)
        assert obs in space, f"{obs=} not in {space=} ({space.sample()=})"
        if outcome is None:
            assert np.all(obs == 0)
        decoded = encoder.decode(learner.nmi, obs)
        assert (
            decoded == outcome
            or abs(encoder._outcome_map[decoded] - encoder._outcome_map[outcome]) < 2
        ), (
            f"{obs=}, {decoded=}, {outcome=}, {encoder._outcome_map[outcome]=}, {outcome=}\n{encoder=}"
        )
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        # if _ > 3:
        #     assert False, f"{obs=}, {r=}, {terminated=},{truncated=}, {info=}, {obs_obs.children[1]._first_offer=},{obs_obs=}"
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    n_outcomes=st.sampled_from(NOUTCOMES),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
    order_by_similarity=st.booleans(),
    sort_by_utility=st.booleans(),
)
@settings(deadline=LONG_DEADLINE_PER_TEST)
@example(
    n_issues=2,
    n_outcomes=0,
    always_starts=False,  # or any other generated value
    first_categorical=False,  # or any other generated value
    order_by_similarity=False,  # or any other generated value
    sort_by_utility=False,
)
def test_basic_obs_encoder_doutcome_nd(
    n_issues,
    always_starts,
    first_categorical,
    n_outcomes,
    order_by_similarity,
    sort_by_utility,
):
    if n_outcomes == 0:
        n_outcomes = 1
        for _ in range(n_issues):
            n_outcomes *= N
    encoder, env = make_encoder_and_env(
        DOutcomeEncoderND,
        params=dict(
            n_outcomes=n_outcomes,
            order_by_similarity=order_by_similarity,
            sort_by_utility=sort_by_utility,
        ),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    env.reset()

    space = encoder.make_space()
    x = space.sample()
    assert x in space
    learner = env._mechanism.negotiators[int(not always_starts)]
    for j in range(NFEWTRIALS):
        outcome = env._mechanism.nmi.state.current_offer
        obs = encoder.encode(learner.nmi)
        assert obs in space, f"{obs=} not in {space=} ({space.sample()=})"
        if outcome is None:
            assert np.all(obs == 0)
        decoded = encoder.decode(learner.nmi, obs)
        assert (
            decoded == outcome
            or abs(encoder._outcome_map[decoded] - encoder._outcome_map[outcome]) < 2
        ), (
            f"{obs=}, {decoded=}, {outcome=}, {encoder._outcome_map[outcome]=}, {outcome=}\n{encoder=}"
        )
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
    min_encodable=st.sampled_from(MIN_ENCODABLE),
    encodable_length=st.sampled_from(ENCODABLE_LENGTH),
    order_by_similarity=st.booleans(),
    sort_by_utility=st.booleans(),
)
@settings(deadline=LONG_DEADLINE_PER_TEST)
def test_basic_obs_encoder_coutcome(
    n_issues,
    always_starts,
    first_categorical,
    min_encodable,
    encodable_length,
    order_by_similarity,
    sort_by_utility,
):
    encoder, env = make_encoder_and_env(
        COutcomeEncoder,
        params=dict(
            min_encodable=min_encodable,
            max_encodable=min_encodable + encodable_length,
            order_by_similarity=order_by_similarity,
            sort_by_utility=sort_by_utility,
        ),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    env.reset()

    space = encoder.make_space()
    x = space.sample()
    assert x in space
    learner = env._mechanism.negotiators[int(not always_starts)]
    outcomes = list(learner.ufun.outcome_space.enumerate())
    cardinality = learner.ufun.outcome_space.cardinality
    for j in range(NFEWTRIALS):
        obs = encoder.encode(learner.nmi)
        assert obs in space, f"{obs=} not in {space=} ({space.sample()=})"
        outcome = env._mechanism.nmi.state.current_offer
        decoded = encoder.decode(learner.nmi, obs)
        assert decoded == outcome, (
            f"{obs=}, {decoded=}, {outcome=}, {encoder._outcome_map[outcome]=}, {outcome=}\n{encoder=}"
        )
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        # if _ > 3:
        #     assert False, f"{obs=}, {r=}, {terminated=},{truncated=}, {info=}, {obs_obs.children[1]._first_offer=},{obs_obs=}"
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    ignore=st.booleans(),
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
    min_encodable=st.sampled_from(MIN_ENCODABLE),
    encodable_length=st.sampled_from(ENCODABLE_LENGTH),
    n_offers=st.integers(*NOFFERS),
    order_by_similarity=st.booleans(),
)
@settings(deadline=LONG_DEADLINE_PER_TEST)
@example(
    ignore=False,  # or any other generated value
    n_issues=1,  # or any other generated value
    always_starts=False,  # or any other generated value
    first_categorical=False,  # or any other generated value
    min_encodable=0,
    encodable_length=1,
    n_offers=1,  # or any other generated value
    order_by_similarity=False,  # or any other generated value
)
def test_basic_obs_encoder_cwindowed_outcome(
    ignore,
    n_issues,
    always_starts,
    first_categorical,
    min_encodable,
    encodable_length,
    n_offers,
    order_by_similarity,
):
    encoder, env = make_encoder_and_env(
        CWindowedOutcomeEncoder,
        params=dict(
            n_offers=n_offers,
            ignore_own_offers=ignore,
            min_encodable=min_encodable,
            max_encodable=min_encodable + encodable_length,
            order_by_similarity=order_by_similarity,
        ),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, CWindowedOutcomeEncoder)
    env.reset()

    def is_ok(offers, decoded):
        return all(
            [
                a == b or abs(encoder._outcome_map[a] - encoder._outcome_map[b]) < 2
                for a, b in zip(offers, decoded, strict=True)
            ]
        )

    space = encoder.make_space()
    for j in range(NFEWTRIALS):
        assert isinstance(env._mechanism, SAOMechanism)
        learner = env._mechanism.negotiators[int(not always_starts)]
        assert learner.ufun
        if not ignore:
            offers = env._mechanism.offers
        else:
            trace = env._mechanism.trace
            offers = [_[-1] for _ in trace if _[0] != learner.id]
        offers.reverse()
        offers = offers[: encoder.n_offers]
        if len(offers) < encoder.n_offers:
            offers = offers + [None] * (encoder.n_offers - len(offers))
        obs = encoder.encode(learner.nmi)
        assert obs in space, (
            f"{obs=} not in {space=} (example sample {space.sample()=})"
        )
        assert np.all(min_encodable <= obs) and np.all(
            obs <= min_encodable + encodable_length
        ), f"{obs=}"
        assert len(obs) == encoder.n_offers
        assert isinstance(obs, np.ndarray)
        assert len(obs) == encoder.n_offers
        decoded = encoder.decode(learner.nmi, obs)

        assert is_ok(offers, decoded), (
            f"obs not expected:\n{obs=}\n{decoded=}\n{offers=}\n{encoder}"
            f"\n{env._mechanism.state=}\n{learner.ufun.minmax()=}"
        )
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    ignore=st.booleans(),
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
    min_encodable=st.sampled_from(MIN_ENCODABLE),
    encodable_length=st.sampled_from(ENCODABLE_LENGTH),
    n_offers=st.integers(*NOFFERS),
    n_outcomes=st.sampled_from(NOUTCOMES),
)
@settings(deadline=LONG_DEADLINE_PER_TEST)
@example(
    ignore=False,  # or any other generated value
    n_issues=2,
    always_starts=False,  # or any other generated value
    first_categorical=False,
    min_encodable=0,
    encodable_length=1,
    n_offers=1,  # or any other generated value
    n_outcomes=0,
)
@example(
    ignore=False,  # or any other generated value
    n_issues=1,  # or any other generated value
    always_starts=False,  # or any other generated value
    first_categorical=False,  # or any other generated value
    min_encodable=0,
    encodable_length=1,
    n_offers=2,
    n_outcomes=0,
)
def test_basic_obs_encoder_dwindowed_outcome(
    ignore,
    n_issues,
    always_starts,
    first_categorical,
    min_encodable,
    encodable_length,
    n_offers,
    n_outcomes,
):
    if n_outcomes == 0:
        n_outcomes = 1
        for _ in range(n_issues):
            n_outcomes *= N
    encoder, env = make_encoder_and_env(
        DWindowedOutcomeEncoder,
        params=dict(
            n_offers=n_offers,
            ignore_own_offers=ignore,
            min_encodable=min_encodable,
            max_encodable=min_encodable + encodable_length,
            n_outcomes=n_outcomes,
        ),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, DWindowedOutcomeEncoder)
    env.reset()

    space = encoder.make_space()

    def is_ok(offers, decoded):
        return all(
            [
                a == b or abs(encoder._outcome_map[a] - encoder._outcome_map[b]) < 2
                for a, b in zip(offers, decoded, strict=True)
            ]
        )

    for j in range(NFEWTRIALS):
        assert isinstance(env._mechanism, SAOMechanism)
        learner = env._mechanism.negotiators[int(not always_starts)]
        assert learner.ufun
        cartinality = learner.ufun.outcome_space.cardinality  # type: ignore
        if not ignore:
            offers = env._mechanism.offers
        else:
            trace = env._mechanism.trace
            offers = [_[-1] for _ in trace if _[0] != learner.id]
        offers.reverse()
        offers = offers[: encoder.n_offers]
        expected: list[Outcome | None] = [None] * encoder.n_offers
        for i, o in enumerate(offers):
            expected[i] = o
        offers = expected
        outcomes = list(encoder._outcome_map.keys())
        obs = encoder.encode(learner.nmi)
        assert obs in space, (
            f"{obs=} not in {space=} (example sample {space.sample()=})"
        )
        assert len(obs) == encoder.n_offers
        assert isinstance(obs, np.ndarray)
        assert len(obs) == encoder.n_offers

        decoded = encoder.decode(learner.nmi, obs)

        assert is_ok(offers, decoded), (
            f"obs not expected:\n{obs=}\n{decoded=}\n{offers=}\n{encoder}"
            f"\n{env._mechanism.state=}\n{learner.ufun.minmax()=}"
        )
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    ignore=st.booleans(),
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
    min_encodable=st.sampled_from(MIN_ENCODABLE),
    encodable_length=st.sampled_from(ENCODABLE_LENGTH),
    n_offers=st.integers(*NOFFERS),
)
@settings(deadline=LONG_DEADLINE_PER_TEST)
@example(
    ignore=False,  # or any other generated value
    n_issues=1,  # or any other generated value
    always_starts=False,  # or any other generated value
    first_categorical=False,  # or any other generated value
    min_encodable=0,
    encodable_length=1,
    n_offers=3,
)
def test_basic_obs_encoder_cwindowed_issue(
    ignore,
    n_issues,
    always_starts,
    first_categorical,
    min_encodable,
    encodable_length,
    n_offers,
):
    encoder, env = make_encoder_and_env(
        CWindowedIssueEncoder,
        params=dict(
            n_offers=n_offers,
            ignore_own_offers=ignore,
            min_encodable=min_encodable,
            max_encodable=min_encodable + encodable_length,
            n_issues=n_issues,
        ),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, CWindowedIssueEncoder)
    env.reset()

    space = encoder.make_space()

    def similar_issue_values(a, b, issues, tol=1e-3, itol=1):
        if a is None and b is None:
            return True
        if a is None and b is not None or a is None and b is not None:
            return False

        def loc(x, issue: Issue):
            if issue.is_numeric():
                return (x - issue.min_value) / (issue.max_value - issue.min_value)
            return issue.values.index(x)

        for x, y, issue in zip(a, b, issues, strict=True):
            if x != y and (
                issue.is_continuous()
                and abs(loc(x, issue) - loc(y, issue)) > tol
                or (
                    not issue.is_continuous()
                    and abs(loc(x, issue) - loc(y, issue)) > itol
                )
            ):
                return False
        return True

    def is_ok(offers, decoded, issues) -> bool:
        return all(
            [
                a == b or similar_issue_values(a, b, issues)
                for a, b in zip(offers, decoded, strict=True)
            ]
        )

    for j in range(NFEWTRIALS):
        assert isinstance(env._mechanism, SAOMechanism)
        learner = env._mechanism.negotiators[int(not always_starts)]
        assert learner.ufun
        assert isinstance(learner.ufun.outcome_space, CartesianOutcomeSpace)
        assert learner.ufun.outcome_space.issues
        issues = learner.ufun.outcome_space.issues
        if not ignore:
            offers = env._mechanism.offers
        else:
            trace = env._mechanism.trace
            offers = [_[-1] for _ in trace if _[0] != learner.id]
        offers.reverse()
        offers = offers[: encoder.n_offers]
        if len(offers) < encoder.n_offers:
            offers += [None] * (encoder.n_offers - len(offers))
        obs = encoder.encode(learner.nmi)
        assert obs in space, (
            f"{obs=} not in {space=} (example sample {space.sample()=})"
        )
        assert np.all(min_encodable <= obs) and np.all(
            obs <= min_encodable + encodable_length
        ), f"{obs=}"
        assert len(obs) == encoder.n_offers * encoder.n_issues
        assert isinstance(obs, np.ndarray)
        decoded = encoder.decode(learner.nmi, obs)

        assert is_ok(offers, decoded, issues), (
            f"obs not expected:\n{obs=}\n{decoded=}\n{offers=}\n{encoder}"
            f"\n{env._mechanism.state=}\n{learner.ufun.minmax()=}"
        )
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    ignore=st.booleans(),
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
    n_offers=st.integers(*NOFFERS),
    n_levels=st.sampled_from(NLEVELS),
)
@settings(deadline=DEADLINE_PER_TEST)
@example(
    ignore=False,
    n_issues=1,
    always_starts=False,
    first_categorical=False,
    n_offers=1,
    n_levels=3,
)
def test_basic_obs_encoder_dwindowed_issue(
    ignore,
    n_issues,
    always_starts,
    first_categorical,
    n_offers,
    n_levels,
):
    if n_levels == 0:
        n_levels = N
    levels = [n_levels] * n_issues
    encoder, env = make_encoder_and_env(
        DWindowedIssueEncoder,
        params=dict(
            n_offers=n_offers,
            ignore_own_offers=ignore,
            n_levels=levels,
        ),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, CWindowedIssueEncoder)
    env.reset()

    space = encoder.make_space()

    def similar_issue_values(a, b, issues, tol=1e-3, itol=1):
        if a is None and b is None:
            return True
        if a is None and b is not None or a is None and b is not None:
            return False

        def loc(x, issue: Issue):
            if issue.is_numeric():
                return (x - issue.min_value) / (issue.max_value - issue.min_value)
            return issue.values.index(x)

        for x, y, issue in zip(a, b, issues, strict=True):
            if x != y and (
                issue.is_continuous()
                and abs(loc(x, issue) - loc(y, issue)) > tol
                or (
                    not issue.is_continuous()
                    and abs(loc(x, issue) - loc(y, issue)) > itol
                )
            ):
                return False
        return True

    def is_ok(offers, decoded, issues) -> bool:
        return all(
            [
                a == b or similar_issue_values(a, b, issues)
                for a, b in zip(offers, decoded, strict=True)
            ]
        )

    for j in range(NFEWTRIALS):
        assert isinstance(env._mechanism, SAOMechanism)
        learner = env._mechanism.negotiators[int(not always_starts)]

        assert learner.ufun
        assert isinstance(learner.ufun.outcome_space, CartesianOutcomeSpace)
        assert learner.ufun.outcome_space.issues
        issues = learner.ufun.outcome_space.issues
        if not ignore:
            offers = env._mechanism.offers
        else:
            trace = env._mechanism.trace
            offers = [_[-1] for _ in trace if _[0] != learner.id]
        offers.reverse()
        offers = offers[: encoder.n_offers]
        expected: list[Outcome | None] = [None] * encoder.n_offers
        for i, o in enumerate(offers):
            expected[i] = o
        offers = expected
        obs = encoder.encode(learner.nmi)
        assert obs in space, (
            f"{obs=} not in {space=} (example sample {space.sample()=})"
        )
        assert np.all(0 <= obs) and np.all(obs <= n_levels), f"{obs=}"
        assert len(obs) == encoder.n_offers * encoder.n_issues
        assert isinstance(obs, np.ndarray)
        decoded = encoder.decode(learner.nmi, obs)

        assert is_ok(offers, decoded, issues), (
            f"obs not expected:\n{obs=}\n{decoded=}\n{offers=}\n{encoder}"
            f"\n{env._mechanism.state=}\n{learner.ufun.minmax()=}"
        )
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        # if _ > 3:
        #     assert False, f"{obs=}, {r=}, {terminated=},{truncated=}, {info=}, {obs_obs.children[1]._first_offer=},{obs_obs=}"
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_basic_obs_encoder_cutility_time_dict(
    n_issues,
    always_starts,
    first_categorical,
):
    encoder, env = make_encoder_and_env(
        CTimeUtilityDictEncoder,
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, CTimeUtilityDictEncoder)
    assert isinstance(encoder.children[0], CTimeEncoder)
    assert isinstance(encoder.children[1], CUtilityEncoder)

    env.reset()

    space = encoder.make_space()
    learner = env._mechanism.negotiators[int(not always_starts)]

    for j in range(NFEWTRIALS):
        obs, r, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi, make_offer(j, first_categorical, n_issues)
            )
        )
        assert not isinstance(r, dict)
        assert obs in space, f"{obs=} not in {space=}\n({space.sample()=})"
        assert isinstance(obs, dict)
        assert "time" in obs and "utility" in obs
        assert len(encoder.children) == len(encoder.names) == 2
        assert (
            abs(encoder.children[0].unscale(obs["time"]) - env._mechanism.relative_time)
            < 1e-3
        )
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_basic_obs_encoder_cutility_time_tuple(
    n_issues,
    always_starts,
    first_categorical,
):
    encoder, env = make_encoder_and_env(
        CTimeUtilityTupleEncoder,
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, CTimeUtilityTupleEncoder)

    env.reset()

    space = encoder.make_space()
    learner = env._mechanism.negotiators[int(not always_starts)]
    for j in range(NFEWTRIALS):
        obs, r, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi, make_offer(j, first_categorical, n_issues)
            )
        )
        assert not isinstance(r, dict)
        assert obs in space, f"{obs=} not in {space=}\n({space.sample()=})"
        assert isinstance(obs, tuple)
        assert len(encoder.children) == len(encoder.names) == 2
        assert isinstance(encoder.children[0], CTimeEncoder)
        assert isinstance(encoder.children[1], CUtilityEncoder)

        assert (
            abs(encoder.children[0].unscale(obs[0]) - env._mechanism.relative_time)
            < 1e-3
        )
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_basic_obs_encoder_cutility_time_flat(
    n_issues,
    always_starts,
    first_categorical,
):
    encoder, env = make_encoder_and_env(
        CTimeUtilityFlatEncoder,
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, CTimeUtilityFlatEncoder)

    env.reset()

    space = encoder.make_space()
    learner = env._mechanism.negotiators[int(not always_starts)]
    for j in range(NFEWTRIALS):
        obs, r, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi, make_offer(j, first_categorical, n_issues)
            )
        )
        assert not isinstance(r, dict)
        assert obs in space, f"{obs=} not in {space=}\n({space.sample()=})"
        assert isinstance(obs, np.ndarray)
        assert len(obs) == 2
        assert len(encoder.children) == len(encoder.names) == 2
        assert isinstance(encoder.children[0], CTimeEncoder)
        assert isinstance(encoder.children[1], CUtilityEncoder)

        assert (
            abs(encoder.children[0].unscale(obs[0]) - env._mechanism.relative_time)
            < 1e-3
        )
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_basic_obs_encoder_cutility_time_box(
    n_issues,
    always_starts,
    first_categorical,
):
    encoder, env = make_encoder_and_env(
        CTimeUtilityBoxEncoder,
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, CTimeUtilityBoxEncoder)

    env.reset()

    space = encoder.make_space()
    learner = env._mechanism.negotiators[int(not always_starts)]
    for j in range(NFEWTRIALS):
        obs, r, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi, make_offer(j, first_categorical, n_issues)
            )
        )
        assert not isinstance(r, dict)
        assert obs in space, f"{obs=} not in {space=}\n({space.sample()=})"
        assert isinstance(obs, np.ndarray)
        assert len(obs) == 2
        assert len(encoder.children) == len(encoder.names) == 2
        assert isinstance(encoder.children[0], CTimeEncoder)
        assert isinstance(encoder.children[1], CUtilityEncoder)

        assert (
            abs(encoder.children[0].unscale(obs[0]) - env._mechanism.relative_time)
            < 1e-3
        )
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_basic_obs_encoder_dutility_time_dict(
    n_issues,
    always_starts,
    first_categorical,
):
    encoder, env = make_encoder_and_env(
        DTimeUtilityDictEncoder,
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, DTimeUtilityDictEncoder)
    assert isinstance(encoder.children[0], DTimeEncoder)
    assert isinstance(encoder.children[1], DUtilityEncoder)

    env.reset()

    space = encoder.make_space()
    learner = env._mechanism.negotiators[int(not always_starts)]
    prev = -1
    for j in range(NFEWTRIALS):
        obs, r, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi, make_offer(j, first_categorical, n_issues)
            )
        )
        assert not isinstance(r, dict)
        assert obs in space, f"{obs=} not in {space=}\n({space.sample()=})"
        assert isinstance(obs, dict)
        assert "time" in obs and "utility" in obs
        assert len(encoder.children) == len(encoder.names) == 2
        assert not isinstance(r, dict)
        assert obs in space, f"{obs=} not in {space=}\n({space.sample()=})"
        assert obs["time"] >= prev, f"{obs=}, {prev=}"
        prev = max(obs["time"], prev)  # type: ignore
        if terminated or truncated:
            prev = -1
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_basic_obs_encoder_dutility_time_tuple(
    n_issues,
    always_starts,
    first_categorical,
):
    encoder, env = make_encoder_and_env(
        DTimeUtilityTupleEncoder,
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, DTimeUtilityTupleEncoder)
    assert isinstance(encoder.children[0], DTimeEncoder)
    assert isinstance(encoder.children[1], DUtilityEncoder)

    env.reset()

    space = encoder.make_space()
    learner = env._mechanism.negotiators[int(not always_starts)]
    prev = -1
    for j in range(NFEWTRIALS):
        obs, r, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi, make_offer(j, first_categorical, n_issues)
            )
        )
        assert not isinstance(r, dict)
        assert obs in space, f"{obs=} not in {space=}\n({space.sample()=})"
        assert isinstance(obs, tuple)
        assert len(obs) == 2
        assert len(encoder.children) == len(encoder.names) == 2
        assert obs[0] >= prev, f"{obs=}, {prev=}"
        prev = max(obs[0], prev)  # type: ignore
        if terminated or truncated:
            prev = -1
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_basic_obs_encoder_dutility_time_flat(
    n_issues,
    always_starts,
    first_categorical,
):
    encoder, env = make_encoder_and_env(
        DTimeUtilityFlatEncoder,
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, DTimeUtilityFlatEncoder)
    assert isinstance(encoder.children[0], DTimeEncoder)
    assert isinstance(encoder.children[1], DUtilityEncoder)

    env.reset()

    space = encoder.make_space()
    learner = env._mechanism.negotiators[int(not always_starts)]
    for j in range(NFEWTRIALS):
        obs, r, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi, make_offer(j, first_categorical, n_issues)
            )
        )
        assert not isinstance(r, dict)
        assert obs in space, f"{obs=} not in {space=}\n({space.sample()=})"
        assert isinstance(obs, np.ndarray)
        assert len(encoder.children) == len(encoder.names) == 2
        if terminated or truncated:
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_basic_obs_encoder_dutility_time_box(
    n_issues,
    always_starts,
    first_categorical,
):
    encoder, env = make_encoder_and_env(
        DTimeUtilityBoxEncoder,
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, DTimeUtilityBoxEncoder)
    assert isinstance(encoder.children[0], DTimeEncoder)
    assert isinstance(encoder.children[1], DUtilityEncoder)

    env.reset()

    space = encoder.make_space()
    learner = env._mechanism.negotiators[int(not always_starts)]
    prev = -1
    for j in range(NFEWTRIALS):
        obs, r, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi, make_offer(j, first_categorical, n_issues)
            )
        )
        assert not isinstance(r, dict)
        assert obs in space, f"{obs=} not in {space=}\n({space.sample()=})"
        assert isinstance(obs, np.ndarray)
        assert len(obs) == 2
        assert len(encoder.children) == len(encoder.names) == 2

        assert obs[0] >= prev, f"{obs=}, {prev=}"
        prev = max(obs[0], prev)  # type: ignore
        if terminated or truncated:
            prev = -1
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    n_outcomes=st.sampled_from(NOUTCOMES),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
    order_by_similarity=st.booleans(),
)
@settings(deadline=LONG_DEADLINE_PER_TEST)
@example(
    n_issues=1,
    n_outcomes=50,
    always_starts=False,
    first_categorical=False,
    order_by_similarity=False,
)
def test_basic_obs_encoder_venas(
    n_issues, always_starts, first_categorical, n_outcomes, order_by_similarity
):
    encoder, env = make_encoder_and_env(
        VeNASEncoder,
        params=dict(n_outcomes=n_outcomes, order_by_similarity=order_by_similarity),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    env.reset()
    assert isinstance(encoder, VeNASEncoder)

    space = encoder.make_space()
    x = space.sample()
    assert x in space
    learner = env._mechanism.negotiators[int(not always_starts)]
    prev = -1
    assert isinstance(encoder.children[0], DTimeEncoder)
    assert isinstance(encoder.children[1], DOutcomeEncoder1D)
    for j in range(NFEWTRIALS):
        outcome = env._mechanism.nmi.state.current_offer
        obs = encoder.encode(learner.nmi)
        assert obs in space, f"{obs=} not in {space=} ({space.sample()=})"
        assert "time" in obs and "outcome" in obs

        assert obs["time"] >= prev, f"{obs=}, {prev=}"
        prev = max(obs["time"], prev)  # type: ignore
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        # if _ > 3:
        #     assert False, f"{obs=}, {r=}, {terminated=},{truncated=}, {info=}, {obs_obs.children[1]._first_offer=},{obs_obs=}"
        if terminated or truncated:
            prev = -1
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    ignore=st.booleans(),
    missing_as_none=st.booleans(),
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
    min_encodable=st.sampled_from(MIN_ENCODABLE),
    encodable_length=st.sampled_from(ENCODABLE_LENGTH),
    n_offers=st.integers(*NOFFERS),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_basic_obs_encoder_sengupta(
    ignore,
    missing_as_none,
    n_issues,
    always_starts,
    first_categorical,
    min_encodable,
    encodable_length,
    n_offers,
):
    encoder, env = make_encoder_and_env(
        SenguptaEncoder,
        params=dict(
            n_offers=n_offers,
            ignore_own_offers=ignore,
            missing_as_none=missing_as_none,
            min_encodable=min_encodable,
            max_encodable=min_encodable + encodable_length,
        ),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, SenguptaEncoder)
    assert isinstance(encoder.children[1], CWindowedUtilityEncoder)
    assert isinstance(encoder.children[0], CTimeEncoder)
    env.reset()

    space = encoder.make_space()
    prev = -1
    for j in range(NFEWTRIALS):
        assert isinstance(env._mechanism, SAOMechanism)
        learner = env._mechanism.negotiators[int(not always_starts)]
        assert learner.ufun
        if not ignore:
            offers = env._mechanism.offers
        else:
            trace = env._mechanism.trace
            offers = [_[-1] for _ in trace if _[0] != learner.id]
        offers.reverse()
        offers = offers[: encoder.n_offers]
        if encoder.missing_as_none:
            utils = np.zeros(encoder.n_offers)
        else:
            utils = np.ones(encoder.n_offers) * float(learner.ufun.reserved_value)

        utils[: len(offers)] = [float(learner.ufun(_)) for _ in offers]
        if encoder.missing_as_none:
            utils[len(offers) :] = 0.0
        else:
            utils[len(offers) :] = float(learner.ufun.reserved_value)
        obs = encoder.encode(learner.nmi)
        assert obs in space, (
            f"{obs=} not in {space=} (example sample {space.sample()=})"
        )
        assert np.all(min_encodable <= obs["utility"]) and np.all(
            obs["utility"] <= min_encodable + encodable_length
        ), f"{obs=}"
        assert "time" in obs and "utility" in obs
        assert len(obs["utility"]) == encoder.n_offers
        assert isinstance(obs["utility"], np.ndarray)
        assert len(obs["utility"]) == encoder.n_offers
        assert obs["time"] >= prev, f"{obs=}, {prev=}"
        prev = max(obs["time"], prev)  # type: ignore
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        # if _ > 3:
        #     assert False, f"{obs=}, {r=}, {terminated=},{truncated=}, {info=}, {obs_obs.children[1]._first_offer=},{obs_obs=}"
        if terminated or truncated:
            prev = -1
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    n_issues=st.integers(1, 3),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
    missing_as_none=st.booleans(),
    ignore=st.booleans(),
    n_offers=st.integers(*NOFFERS),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_basic_obs_encoder_rlboa(
    ignore, missing_as_none, n_issues, always_starts, first_categorical, n_offers
):
    encoder, env = make_encoder_and_env(
        RLBoaEncoder,
        params=dict(
            n_offers=n_offers,
            ignore_own_offers=ignore,
            missing_as_none=missing_as_none,
        ),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, RLBoaEncoder)
    assert isinstance(encoder.children[1], DWindowedUtilityEncoder)
    assert isinstance(encoder.children[0], DTimeEncoder)
    env.reset()

    space = encoder.make_space()
    x = space.sample()
    assert x in space
    prev = -1
    for j in range(NFEWTRIALS):
        assert isinstance(env._mechanism, SAOMechanism)
        learner = env._mechanism.negotiators[int(not always_starts)]
        assert learner.ufun
        if not ignore:
            offers = env._mechanism.offers
        else:
            trace = env._mechanism.trace
            offers = [_[-1] for _ in trace if _[0] != learner.id]
        offers.reverse()
        offers = offers[: encoder.n_offers]
        if missing_as_none:
            utils = np.zeros(encoder.n_offers)
        else:
            utils = np.ones(encoder.n_offers) * float(learner.ufun.reserved_value)

        utils[: len(offers)] = [float(learner.ufun(_)) for _ in offers]
        utils[len(offers) :] = (
            float(learner.ufun.reserved_value) if not missing_as_none else 0
        )
        obs = encoder.encode(learner.nmi)
        assert obs in space, f"{obs=} not in {space=} (example: {space.sample()=})"
        assert "time" in obs and "utility" in obs
        assert len(obs["utility"]) == encoder.n_offers
        assert isinstance(obs["utility"], np.ndarray)
        assert len(obs["utility"]) == encoder.n_offers
        assert obs["time"] >= prev, f"{obs=}, {prev=}"
        prev = max(obs["time"], prev)  # type: ignore
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        if terminated or truncated:
            prev = -1
            obs, _ = env.reset()
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()


@given(
    ignore=st.booleans(),
    always_starts=st.booleans(),
    first_categorical=st.booleans(),
    n_offers=st.integers(*NOFFERS),
    n_levels=st.sampled_from(NLEVELS),
)
@settings(deadline=DEADLINE_PER_TEST)
def test_basic_obs_encoder_mipn(
    ignore,
    always_starts,
    first_categorical,
    n_offers,
    n_levels,
):
    n_issues = 2
    encoder, env = make_encoder_and_env(
        MiPNEncoder,
        params=dict(
            n_offers=n_offers,
            ignore_own_offers=ignore,
            n_issue_levels=[n_levels] * n_issues,
        ),
        n_issues=n_issues,
        always_starts=always_starts,
        first_categorical=first_categorical,
    )
    assert isinstance(encoder, MiPNEncoder)
    assert isinstance(encoder.children[1], DWindowedIssueEncoder)
    assert isinstance(encoder.children[0], DTimeEncoder)
    env.reset()

    space = encoder.make_space()
    prev = -1
    for j in range(NFEWTRIALS):
        assert isinstance(env._mechanism, SAOMechanism)
        learner = env._mechanism.negotiators[int(not always_starts)]
        assert learner.ufun
        if not ignore:
            offers = env._mechanism.offers
        else:
            trace = env._mechanism.trace
            offers = [_[-1] for _ in trace if _[0] != learner.id]
        offers.reverse()
        offers = offers[: encoder.n_offers]
        expected: list[Outcome | None] = [None] * encoder.n_offers
        for i, o in enumerate(offers):
            expected[i] = o
        obs = encoder.encode(learner.nmi)
        assert obs in space, (
            f"{obs=} not in {space=} (example sample {space.sample()=})"
        )
        assert np.all(0 <= obs["outcome"]) and np.all(obs["outcome"] < n_levels + 1), (
            f"{obs=}"
        )
        assert "time" in obs and "outcome" in obs
        assert (
            len(obs["outcome"])
            == encoder.children[1].n_offers * encoder.children[1].n_issues
        )
        assert isinstance(obs["outcome"], np.ndarray)
        assert isinstance(learner.ufun.outcome_space, CartesianOutcomeSpace)
        assert learner.ufun.outcome_space.issues
        assert obs["time"] >= prev, f"{obs=}, {prev=}"
        prev = max(obs["time"], prev)  # type: ignore
        obs, _, terminated, truncated, _ = env.step(
            env._action_decoders[env._key(0)].encode(
                learner.nmi,
                make_offer(j, first_categorical=first_categorical, n_issues=n_issues),
            )
        )
        # if _ > 3:
        #     assert False, f"{obs=}, {r=}, {terminated=},{truncated=}, {info=}, {obs_obs.children[1]._first_offer=},{obs_obs=}"
        if terminated or truncated:
            obs, _ = env.reset()
            prev = -1
            learner = env._mechanism.negotiators[int(not always_starts)]
    env.close()
