from typing import Any

from negmas import Issue, Outcome, UtilityFunction, make_os

__all__ = ["enumerate_in_order"]


def enumerate_similar(lst: list[list[Any]]) -> tuple:
    from itertools import product

    # Generate all possible tuples using the Cartesian product
    all_tuples = list(product(*lst))
    n = len(all_tuples)

    # Function to check if two tuples differ in exactly one position
    def differs_by_one(tuple1, tuple2):
        diff_count = sum(1 for a, b in zip(tuple1, tuple2) if a != b)
        return diff_count == 1

    # Backtracking to find a valid sequence
    def backtrack(path):
        if len(path) == n:
            return path  # Found a valid sequence
        last_tuple = path[-1]
        for tuple_ in all_tuples:
            if tuple_ not in path and differs_by_one(last_tuple, tuple_):
                path.append(tuple_)
                result = backtrack(path)
                if result:
                    return result
                path.pop()  # Backtrack
        return None

    # Start with the first tuple
    start_tuple = all_tuples[0]
    sequence = backtrack([start_tuple])
    return sequence  # type: ignore


def enumerate_in_order(
    issues: list[Issue] | tuple[Issue, ...], n_levels: int = 10
) -> list[Outcome]:
    """Enumerates the given issues making sure that adjacent outcomes have a single difference

    Args:
        issues: list of Issues to sample
        n_levels: Number of levels to discritize continuous issues with

    Returns
    -------
        [TODO:return]
    """
    return enumerate_similar(  # type: ignore
        [
            list(i.ordered_value_generator())
            if i.is_discrete()
            else list(
                i.ordered_value_generator(
                    n_levels, grid=True, compact=True, endpoints=True
                )
            )
            for i in issues
        ]
    )


def enumerate_by_utility(
    issues: list[Issue] | tuple[Issue, ...],
    ufuns: tuple[UtilityFunction, ...] | UtilityFunction,
    best_first=True,
    n_levels: int = 10,
    max_cardinality: float = float("inf"),
    by_welfare: bool = False,
    by_relative_welfare: bool = False,
    by_similarity_first: bool = False,
) -> list[Outcome]:
    """Enumerates the given issues in the order of utility given by the utility function

    Args:
        issues: list of Issues to sample
        n_levels: Number of levels to discritize continuous issues with

    Returns
    -------
        [TODO:return]
    """
    if by_relative_welfare:
        by_welfare = True
    if isinstance(ufuns, UtilityFunction):
        ufuns = (ufuns,)
    if by_similarity_first:
        outcomes = enumerate_in_order(issues, n_levels)
    else:
        outcomes = make_os(issues).enumerate_or_sample(
            levels=n_levels, max_cardinality=max_cardinality
        )
    m = -1 if best_first else 1
    if by_welfare:
        x = [
            (
                sum(
                    -(u(_) if not by_relative_welfare else u(_) - u.reserved_value)
                    for u in ufuns
                ),
                _,
            )
            for _ in outcomes
        ]
    else:
        x = [(tuple(m * u(_) for u in ufuns), _) for _ in outcomes]
    x = sorted(x)
    return [_[-1] for _ in x]


# from negmas_rl.action import ActionDecoder
# from negmas_rl.negotiator import SAORLNegotiator
# from negmas_rl.obs import ObservationEncoder
# def recommended_trainer(
#     obs: ObservationEncoder,
#     action: ActionDecoder,
#     negotiator: SAORLNegotiator | None = None,
#     outcome_space: OutcomeSpace | None = None,
# ): ...
