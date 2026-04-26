from itertools import product

import pytest
from negmas import (
    Issue,
    ResponseType,
    SAOResponse,
)
from negmas.outcomes import Outcome

from negmas_rl.action import (
    CIssueDecoder,
    CMiPNDecoder,
    COutcomeDecoder,
    COutcomeDecoder1D,
    CRelativeUtilityDecoder1D,
    CRelativeUtilityDecoderBox,
    CRLBoaDecoder,
    CUtilityDecoder,
    CUtilityDecoder1D,
    CUtilityDecoderBox,
    CVeNASDecoder,
    DIssueDecoder,
    DOutcomeDecoder,
    DRelativeUtilityDecoder,
    DRelativeUtilityDecoder1D,
    DSenguptaDecoder,
    DUtilityDecoder,
    DUtilityDecoder1D,
    IssueDecoder,
    MiPNDecoder,
    OutcomeDecoder,
    RelativeUtilityDecoder,
    RLBoaDecoder,
    SamplingMethod,
    SAOActionDecoder,
    SenguptaDecoder,
    UtilityDecoder,
    VeNASDecoder,
)
from negmas_rl.common import DEFAULT_N_LEVELS
from negmas_rl.env.gymenv import NegoEnv
from negmas_rl.generators.assigner import PositionBasedNegotiatorAssigner
from negmas_rl.generators.scenario import DEFAULT_N_ISSUES, RandomScenarioGenerator

DEBUG = True
# RATIONAL = (True, False)
RATIONAL = (None,)
EXPLICIT = (True, False)
NO_RESPONSE = (True, False)
ENDING = (True, False)
RESERVED = (0.0, 0.75)
STRICT_OUTCOME_MATCHING = True
IGNORE_NOT_ENDING_AND_NO_RESPONSE = True
DEF_UPPER_ERROR = 0.01
DEF_LOWER_ERROR = 0.01

N_RAND_ACTIONS = 200

DECODERS = [
    CIssueDecoder,
    CMiPNDecoder,
    COutcomeDecoder,
    COutcomeDecoder1D,
    CRLBoaDecoder,
    CRelativeUtilityDecoder1D,
    CUtilityDecoder,
    CUtilityDecoder1D,
    CRelativeUtilityDecoderBox,
    CVeNASDecoder,
    DIssueDecoder,
    DOutcomeDecoder,
    DRelativeUtilityDecoder,
    DRelativeUtilityDecoder1D,
    DSenguptaDecoder,
    DUtilityDecoder1D,
    DUtilityDecoder,
    IssueDecoder,
    MiPNDecoder,
    OutcomeDecoder,
    RLBoaDecoder,
    RelativeUtilityDecoder,
    SenguptaDecoder,
    UtilityDecoder,
    CUtilityDecoderBox,
    VeNASDecoder,
]

UTILITY_DECODERS = [
    UtilityDecoder,
    DUtilityDecoder,
    CUtilityDecoder,
    CUtilityDecoderBox,
    CUtilityDecoder1D,
    DUtilityDecoder1D,
    RelativeUtilityDecoder,
    DRelativeUtilityDecoder,
    CRelativeUtilityDecoder1D,
    DRelativeUtilityDecoder1D,
]


OUTCOME_DECODERS = list(set(DECODERS).difference(UTILITY_DECODERS))


def run_tests(
    typ_: type[SAOActionDecoder],
    allow_ending: bool | None = None,
    allow_no_response: bool | None = None,
    explicit_accept: bool | None = None,
    rational_encoding: bool | None = None,
    always_starts: bool | None = None,
    sampling_method: SamplingMethod | None = None,
    n_issues: int = DEFAULT_N_ISSUES,
    n_levels: tuple[int, ...] | list[int] | int | None = None,
    allowed_lower_error: float | None = None,
    allowed_upper_error: float | None = None,
    same_outcome=None,
    similar_outcome=None,
    similar_issues=None,
    similar_utility=True,
    allowed_differnt_issues=1,
    allowed_continuous_issue_diff=1e-3,
    allowed_discrete_issue_diff=0,
    allowed_str_issue_diff=0,
    allowed_outcome_index_diff=1,
    reserved_range=RESERVED,
):
    if same_outcome is None:
        same_outcome = typ_.is_outcome_invertible() and STRICT_OUTCOME_MATCHING
    if similar_issues is None:
        similar_issues = typ_.is_outcome_invertible()
    if similar_outcome is None:
        similar_outcome = typ_.is_outcome_invertible()
    if IGNORE_NOT_ENDING_AND_NO_RESPONSE and not allow_ending and allow_no_response:
        return
    epsilon = 1e-5
    params = dict()

    if allow_ending is not None:
        params["allow_ending"] = allow_ending
    if allow_no_response is not None:
        params["allow_no_response"] = allow_no_response
    if explicit_accept is not None:
        params["explicit_accept"] = explicit_accept
    if sampling_method is not None:
        params["sampling_method"] = SamplingMethod.Best
    if rational_encoding is not None:
        params["rational_encoding"] = rational_encoding
    if n_levels is not None:
        params["n_levels"] = n_levels
    try:
        decoder = typ_(
            **params,
            fallback_to_best=False,
            fallback_to_last=False,
            fallback_to_offers=False,
        )
    except TypeError:
        if "explicit_accept" in params:
            del params["explicit_accept"]
        decoder = typ_(**params)
    except Exception as e:
        raise e
    assert decoder.is_utility_invertible()

    assigner = None
    if always_starts is not None:
        assigner = PositionBasedNegotiatorAssigner(
            always_starts=always_starts, always_ends=not always_starts
        )

    if allowed_upper_error is None:
        allowed_upper_error = (
            max(decoder.delta_above)  # type: ignore
            if hasattr(decoder, "delta_above")
            else DEF_LOWER_ERROR  # type: ignore
        ) + epsilon
    if allowed_lower_error is None:
        allowed_lower_error = (
            max(decoder.delta_below)  # type: ignore
            if hasattr(decoder, "delta_below")
            else DEF_UPPER_ERROR  # type: ignore
        ) + epsilon
    assert allowed_lower_error + allowed_upper_error < 1
    fallback = (
        decoder.fallback_to_best if hasattr(decoder, "fallback_to_best") else False  # type: ignore
    )
    last = decoder.fallback_to_last if hasattr(decoder, "fallback_to_last") else False  # type: ignore
    env = NegoEnv(
        action_decoder=decoder,
        assigner=assigner,
        scenario_generator=RandomScenarioGenerator(
            numeric_prob=0.5, n_issues=n_issues, reserved_values=reserved_range
        ),
    )
    env.reset()
    agent = env._placeholders[0]
    assert agent is not None
    assert agent.ufun is not None
    agent.nmi.state.current_offer = agent.nmi.random_outcome()

    if allow_ending:
        decoded = SAOResponse(ResponseType.END_NEGOTIATION, None)
        encoded = decoder.encode(agent.nmi, decoded)
        decoded = decoder.parse(agent.nmi, encoded)
        assert decoded.response == decoded.response, (
            f"Decoding does not recover encoded ending:\n{decoded=}\n{decoded=}\n{encoded=}"
        )
    if allow_no_response:
        decoded = SAOResponse(ResponseType.NO_RESPONSE, None)
        encoded = decoder.encode(agent.nmi, decoded)
        decoded = decoder.parse(agent.nmi, encoded)
        assert agent is not None and agent.ufun is not None

        assert decoded.response == decoded.response, (
            f"Decoding does not recover encoded no-response:\n{decoded=}\n{decoded=}\n{encoded=}"
        )

    outcomes = list(env._mechanism.outcome_space.enumerate_or_sample())
    outcomes_map = dict(zip(outcomes, range(len(outcomes))))

    def has_few_diffs(
        x: Outcome | None,
        y: Outcome | None,
        allowed_dissimilar_issues=allowed_differnt_issues,
        seps=allowed_str_issue_diff,
        ieps=allowed_discrete_issue_diff,
        eps=allowed_continuous_issue_diff,
    ) -> bool:
        issues = agent.nmi.outcome_space.issues
        assert all(isinstance(i, Issue) for i in issues)
        if x is None and y is None:
            return True
        if x is None or y is None:
            return False
        return (
            sum(
                abs(x.index(a) - y.index(b)) < seps
                if not i.is_numeric()
                else abs(a - b) < ieps
                if not i.is_continuous()
                else abs(a - b) < eps
                for a, b, i in zip(x, y, issues, strict=True)
            )
            <= allowed_dissimilar_issues
        )

    def is_similar_outcome(x, y, max_diff=allowed_outcome_index_diff):
        if x is None and y is None:
            return True
        if x is None or y is None:
            return False
        return all(
            [
                a == b or abs(outcomes_map[a] - outcomes_map[b]) <= max_diff
                for a, b in zip(x, y, strict=True)
            ]
        )

    def assert_utility_ok(
        decoded,
        encoded,
        recovered,
        fallback,
        sample_action,
        decoder=decoder,
        agent=agent,
    ):
        if not similar_utility:
            return
        assert agent.ufun is not None
        ufun = agent.ufun
        utrue = float(agent.ufun(decoded.outcome))
        if rational_encoding:
            utarget = utrue
            r = float(ufun.reserved_value)
            nearest, dist = r, float("inf")
            for outcome in outcomes:
                u = float(ufun(outcome))
                if u < r:
                    continue
                d = abs(u - utarget)
                if d <= dist:
                    nearest, dist = u, d
            utrue = nearest
        ufound = float(agent.ufun(recovered.outcome))
        has_similar_utility = recovered.response == decoded.response and (
            recovered.response != ResponseType.REJECT_OFFER
            or (
                (utrue - allowed_lower_error <= ufound <= utrue + allowed_upper_error)
                or (fallback and decoder._best == recovered.outcome)  # type: ignore
                or (
                    last
                    and (
                        (
                            decoder._last_outcome  # type: ignore
                            and decoder._last_outcome == recovered.outcome  # type: ignore
                        )
                        or (fallback and decoder._best == recovered.outcome)  # type: ignore
                    )
                )
            )
        )
        if DEBUG and not has_similar_utility:
            if sample_action is not None:
                decoded2 = decoder.parse(agent.nmi, sample_action)
            else:
                decoded2 = decoded
            encoded2 = decoder.encode(agent.nmi, decoded2)
            recovered2 = decoder.parse(agent.nmi, encoded2)

        assert has_similar_utility, (
            f"{_}: Decode-Encode does not recover encoded action {utrue=}, {ufound=}:\n{decoded=}\n{recovered=}\n{sample_action=}\n{encoded=}"
        )

    def assert_outcome_ok(
        decoded,
        encoded,
        recovered,
        sample_action=None,
        decoder=decoder,
        agent=agent,
        assert_same_outcome=same_outcome,
        assert_similar_outcome=similar_outcome,
        assert_similar_issues=similar_issues,
    ):
        assert agent.ufun
        ufun = agent.ufun
        target = decoded.outcome
        if rational_encoding:
            utarget = float(ufun(target))
            r = float(ufun.reserved_value)
            nearest, dist = None, float("inf")
            for outcome in outcomes:
                u = float(ufun(outcome))
                if u < r:
                    continue
                d = abs(u - utarget)
                if d <= dist:
                    nearest, dist = outcome, d
            target = nearest
        if assert_same_outcome:
            same_outcome = recovered.response == decoded.response and (
                recovered.response != ResponseType.REJECT_OFFER
                or recovered.outcome == target
            )
            if DEBUG and not same_outcome:
                encoded2 = decoder.encode(agent.nmi, decoded)
                recovered2 = decoder.parse(agent.nmi, encoded2)

            assert same_outcome, (
                f"{_}: Decode-Encode does not recover encoded action:\n{decoded=}\n{recovered=}\n{sample_action=}\n{encoded=}"
            )

        if not assert_same_outcome and assert_similar_outcome:
            similar_outcome = recovered.response == decoded.response and (
                recovered.response != ResponseType.REJECT_OFFER
                or is_similar_outcome(recovered.outcome, target)
            )
            if DEBUG and not similar_outcome:
                if sample_action is not None:
                    decoded = decoder.parse(agent.nmi, sample_action)
                encoded = decoder.encode(agent.nmi, decoded)
                recovered = decoder.parse(agent.nmi, encoded)

            assert similar_outcome, (
                f"{_}: Decode-Encode does not recover encoded action:\n{decoded=}\n{recovered=}\n{sample_action=}\n{encoded=}"
            )

        if not assert_same_outcome and assert_similar_issues:
            similar_issues = recovered.response == decoded.response and (
                recovered.response != ResponseType.REJECT_OFFER
                or has_few_diffs(recovered.outcome, target)
            )
            if DEBUG and not similar_issues:
                if sample_action is not None:
                    decoded = decoder.parse(agent.nmi, sample_action)
                encoded = decoder.encode(agent.nmi, decoded)
                recovered = decoder.parse(agent.nmi, encoded)

            assert similar_issues, (
                f"{_}: Decode-Encode does not recover encoded action:\n{decoded=}\n{recovered=}\n{sample_action=}\n{encoded=}"
            )

    space = decoder.make_space()
    for _, outcome in enumerate(outcomes):
        # if isinstance(decoder, RelativeUtilityDecoder):
        #     continue
        decoded = SAOResponse(ResponseType.REJECT_OFFER, outcome)
        agent = env._placeholders[0]
        assert agent is not None
        encoded = decoder.encode(agent.nmi, decoded)
        recovered = decoder.parse(agent.nmi, encoded)
        assert agent.ufun is not None
        assert_utility_ok(decoded, encoded, recovered, fallback, None)
        assert_outcome_ok(decoded, encoded, recovered)
    for _ in range(N_RAND_ACTIONS):
        original_action = space.sample()
        assert original_action in space
        agent = env._placeholders[0]
        assert agent is not None and agent.ufun is not None
        decoded = decoder.parse(agent.nmi, original_action)
        assert allow_ending or decoded.response != ResponseType.END_NEGOTIATION, (
            f"{original_action} was encoded as {decoded} which is not allowed"
        )
        assert allow_no_response or decoded.response != ResponseType.NO_RESPONSE, (
            f"{original_action} was encoded as {decoded} which is not allowed"
        )
        if not allow_ending and decoded.response == ResponseType.END_NEGOTIATION:
            continue
        if not allow_no_response and decoded.response == ResponseType.NO_RESPONSE:
            continue
        assert decoded.response != ResponseType.WAIT
        encoded = decoder.encode(agent.nmi, decoded)
        recovered = decoder.parse(agent.nmi, encoded)

        assert_utility_ok(decoded, encoded, recovered, fallback, original_action)


@pytest.mark.parametrize(
    "allow_ending,allow_no_response,explicit_accept,rational_encoding",
    list(product(ENDING, NO_RESPONSE, EXPLICIT, RATIONAL)) + [[None] * 4],
)
def test_action_decoders_utility_encoder(
    allow_ending,
    allow_no_response,
    explicit_accept,
    rational_encoding,
):
    run_tests(
        UtilityDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        explicit_accept=explicit_accept,
        rational_encoding=rational_encoding,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response,explicit_accept,rational_encoding",
    [[None] * 4] + list(product(ENDING, NO_RESPONSE, EXPLICIT, RATIONAL)),
)
def test_action_decoders_cutilitybox(
    allow_ending,
    allow_no_response,
    explicit_accept,
    rational_encoding,
):
    run_tests(
        CUtilityDecoderBox,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        explicit_accept=explicit_accept,
        rational_encoding=rational_encoding,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response,explicit_accept,rational_encoding",
    list(product(ENDING, NO_RESPONSE, EXPLICIT, RATIONAL)) + [[None] * 4],
)
def test_action_decoders_dutility(
    allow_ending,
    allow_no_response,
    explicit_accept,
    rational_encoding,
):
    run_tests(
        DUtilityDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        explicit_accept=explicit_accept,
        rational_encoding=rational_encoding,
        allowed_lower_error=1 / DEFAULT_N_LEVELS,
        allowed_upper_error=1 / DEFAULT_N_LEVELS,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response,explicit_accept,rational_encoding",
    list(product([True, False], [True, False], EXPLICIT, RATIONAL)) + [[None] * 4],
)
def test_action_decoders_dutility1d_encoder(
    allow_ending,
    allow_no_response,
    explicit_accept,
    rational_encoding,
):
    run_tests(
        DUtilityDecoder1D,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        explicit_accept=explicit_accept,
        rational_encoding=rational_encoding,
        allowed_lower_error=1 / DEFAULT_N_LEVELS,
        allowed_upper_error=1 / DEFAULT_N_LEVELS,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response,explicit_accept,rational_encoding",
    [[None] * 4] + list(product(ENDING, NO_RESPONSE, EXPLICIT, RATIONAL)),
)
def test_action_decoders_cutility1d_encoder(
    allow_ending,
    allow_no_response,
    explicit_accept,
    rational_encoding,
):
    run_tests(
        CUtilityDecoder1D,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        explicit_accept=explicit_accept,
        rational_encoding=rational_encoding,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response,explicit_accept,rational_encoding",
    [[None] * 4] + list(product(ENDING, NO_RESPONSE, EXPLICIT, RATIONAL)),
)
def test_action_decoders_cutility_encoder(
    allow_ending,
    allow_no_response,
    explicit_accept,
    rational_encoding,
):
    run_tests(
        CUtilityDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        explicit_accept=explicit_accept,
        rational_encoding=rational_encoding,
        same_outcome=False,
        similar_outcome=False,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response,explicit_accept,rational_encoding",
    [[None] * 4] + list(product(ENDING, NO_RESPONSE, EXPLICIT, RATIONAL)),
)
def test_action_decoders_drelative_utility(
    allow_ending,
    allow_no_response,
    explicit_accept,
    rational_encoding,
):
    run_tests(
        DRelativeUtilityDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        explicit_accept=explicit_accept,
        rational_encoding=rational_encoding,
        allowed_lower_error=0.15,
        allowed_upper_error=0.15,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response,explicit_accept,rational_encoding",
    [[None] * 4] + list(product(ENDING, NO_RESPONSE, (None,), RATIONAL)),
)
def test_action_decoders_drelative_utility_1d(
    allow_ending,
    allow_no_response,
    explicit_accept,
    rational_encoding,
):
    run_tests(
        DRelativeUtilityDecoder1D,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        explicit_accept=explicit_accept,
        rational_encoding=rational_encoding,
        allowed_lower_error=0.15,
        allowed_upper_error=0.15,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response,explicit_accept,rational_encoding",
    list(product(ENDING, NO_RESPONSE, EXPLICIT, RATIONAL)) + [[None] * 4],
)
def test_action_decoders_crelative_utility_box(
    allow_ending,
    allow_no_response,
    explicit_accept,
    rational_encoding,
):
    run_tests(
        CRelativeUtilityDecoderBox,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        explicit_accept=explicit_accept,
        rational_encoding=rational_encoding,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response,explicit_accept,rational_encoding",
    list(product(ENDING, NO_RESPONSE, EXPLICIT, RATIONAL)) + [[None] * 4],
)
def test_action_decoders_crelative_utility_1d(
    allow_ending,
    allow_no_response,
    explicit_accept,
    rational_encoding,
):
    run_tests(
        CRelativeUtilityDecoder1D,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        explicit_accept=explicit_accept,
        rational_encoding=rational_encoding,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response,rational_encoding",
    [[None] * 3] + list(product(ENDING, NO_RESPONSE, RATIONAL)),
)
def test_outcome_decoder(
    allow_ending,
    allow_no_response,
    rational_encoding,
):
    run_tests(
        OutcomeDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        rational_encoding=rational_encoding,
        sampling_method=None,
        always_starts=False,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response,rational_encoding",
    [[None] * 3] + list(product(ENDING, NO_RESPONSE, RATIONAL)),
)
def test_action_decoder_COutcomeDecoder(
    allow_ending,
    allow_no_response,
    rational_encoding,
):
    run_tests(
        COutcomeDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        rational_encoding=rational_encoding,
        sampling_method=None,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response,rational_encoding",
    [[None] * 3] + list(product(ENDING, NO_RESPONSE, RATIONAL)),
)
def test_action_decoder_COutcomeDecoder1D(
    allow_ending,
    allow_no_response,
    rational_encoding,
):
    run_tests(
        COutcomeDecoder1D,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        rational_encoding=rational_encoding,
        sampling_method=None,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response",
    [[None] * 2] + list(product(ENDING, NO_RESPONSE)),
)
def test_action_decoder_DOutcomeDecoder(
    allow_ending,
    allow_no_response,
):
    run_tests(
        DOutcomeDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        sampling_method=None,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response",
    [[None] * 2] + list(product(ENDING, NO_RESPONSE)),
)
def test_action_decoder_IssueDecoder(
    allow_ending,
    allow_no_response,
):
    run_tests(
        IssueDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        sampling_method=None,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response",
    [[None] * 2] + list(product(ENDING, NO_RESPONSE)),
)
def test_action_decoder_CIssueDecoder(
    allow_ending,
    allow_no_response,
):
    run_tests(
        CIssueDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        sampling_method=None,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response",
    [[None] * 2] + list(product(ENDING, NO_RESPONSE)),
)
def test_action_decoder_DIssueDecoder(
    allow_ending,
    allow_no_response,
):
    run_tests(
        DIssueDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        n_levels=[20] * DEFAULT_N_ISSUES,
        sampling_method=None,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response",
    [[None] * 2] + list(product(ENDING, NO_RESPONSE)),
)
def test_action_decoder_RLBoaDecoder(
    allow_ending,
    allow_no_response,
):
    run_tests(
        RLBoaDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        sampling_method=None,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response",
    [[None] * 2] + list(product(ENDING, NO_RESPONSE)),
)
def test_action_decoder_CRLBoaDecoder(
    allow_ending,
    allow_no_response,
):
    run_tests(
        CRLBoaDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        sampling_method=None,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response",
    [[None] * 2] + list(product(ENDING, NO_RESPONSE)),
)
def test_action_decoder_SenguptaDecoder(
    allow_ending,
    allow_no_response,
):
    run_tests(
        SenguptaDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        sampling_method=None,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response",
    [[None] * 2] + list(product(ENDING, NO_RESPONSE)),
)
def test_action_decoder_DSenguptaDecoder(
    allow_ending,
    allow_no_response,
):
    run_tests(
        DSenguptaDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        sampling_method=None,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response",
    [[None] * 2] + list(product(ENDING, NO_RESPONSE)),
)
def test_action_decoder_MiPNDecoder(
    allow_ending,
    allow_no_response,
):
    run_tests(
        MiPNDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        n_levels=[20] * DEFAULT_N_ISSUES,
        sampling_method=None,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response",
    [[None] * 2] + list(product(ENDING, NO_RESPONSE)),
)
def test_action_decoder_CMiPNDecoder(
    allow_ending,
    allow_no_response,
):
    run_tests(
        CMiPNDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        sampling_method=None,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response",
    [[None] * 2] + list(product(ENDING, NO_RESPONSE)),
)
def test_action_decoder_VeNASDecoder(
    allow_ending,
    allow_no_response,
):
    run_tests(
        VeNASDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        sampling_method=None,
    )


@pytest.mark.parametrize(
    "allow_ending,allow_no_response",
    [[None] * 2] + list(product(ENDING, NO_RESPONSE)),
)
def test_action_decoder_CVeNASDecoder(
    allow_ending,
    allow_no_response,
):
    run_tests(
        CVeNASDecoder,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        sampling_method=None,
    )


@pytest.mark.parametrize(
    "typ_,allow_ending,allow_no_response,explicit_accept",
    product(UTILITY_DECODERS, ENDING, NO_RESPONSE, EXPLICIT),
)
def test_action_decoders_utility(
    typ_: type[UtilityDecoder],
    allow_ending,
    allow_no_response,
    explicit_accept,
):
    if issubclass(typ_, RelativeUtilityDecoder):
        explicit_accept = None
    run_tests(
        typ_,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        explicit_accept=explicit_accept,
    )


@pytest.mark.parametrize("typ_", UTILITY_DECODERS)
def test_action_decoders_utility_default(typ_):
    run_tests(typ_)


@pytest.mark.parametrize("typ_", OUTCOME_DECODERS)
def test_action_decoders_outcome_default(typ_):
    n_levels = None
    if issubclass(typ_, DIssueDecoder):
        n_levels = [30] * DEFAULT_N_ISSUES
    run_tests(typ_, n_levels=n_levels)


@pytest.mark.parametrize(
    "typ_,allow_ending,allow_no_response",
    product(OUTCOME_DECODERS, ENDING, NO_RESPONSE),
)
def test_action_decoders_outcome(
    typ_,
    allow_ending,
    allow_no_response,
):
    n_levels = None
    if issubclass(typ_, DIssueDecoder):
        n_levels = [30] * 3
    run_tests(
        typ_,
        allow_ending=allow_ending,
        allow_no_response=allow_no_response,
        n_levels=n_levels,
    )
