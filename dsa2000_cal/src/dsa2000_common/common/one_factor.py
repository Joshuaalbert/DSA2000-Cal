from typing import List, Tuple

import numpy as np


def get_one_factors(N, ordering: str = 'random') -> List[List[Tuple[int, int]]]:
    """
    Get the one-factorization of N vertices.

    Args:
        N: the number of vertices (must be even)

    Returns:
        a list of list of tuples (i, j) with i < j, each representing a round.
    """
    if N % 2 != 0:
        raise ValueError("Only even N is supported.")

    # Create an ordering of players
    ordering = reorder_vertices_for_locality(N, ordering=ordering)

    return [one_factorization_round(N, round_idx, ordering=ordering) for round_idx in range(N - 1)]


def reorder_vertices_for_locality(N: int, ordering: str) -> List[int]:
    if ordering == 'random':
        return np.random.permutation(N).tolist()
    elif ordering == 'natural':
        return list(range(N))
    elif ordering == 'odd-even':
        # A simple heuristic: put all even indices first, then all odd indices.
        evens = [i for i in range(N) if i % 2 == 0]
        odds = [i for i in range(N) if i % 2 == 1]
        return evens + odds
    else:
        raise ValueError('Unknown ordering')


def one_factorization_round(N: int, round_idx: int, ordering: List[int]) -> List[Tuple[int, int]]:
    """
    Compute the one factor (perfect matching) for a complete graph K_N
    for a given round index using a round-robin (circle method) algorithm.

    Args:
      N: int - number of vertices (must be even)
      round_idx: int - index of the round, 0 <= round_idx < N-1

    Returns:
      A list of tuples (i, j) with i < j, representing the matching for that round,
      sorted lexicographically.
    """
    if not (0 <= round_idx < N - 1):
        raise ValueError(f"round_idx {round_idx} must be 0 <= round_idx < N - 1.")

    # Apply the standard round-robin approach on the reordered vertices.
    fixed = ordering[0]
    rotated = ordering[1:]
    rotated = rotated[round_idx:] + rotated[:round_idx]
    order = [fixed] + rotated

    # Generate the pairs: match the first half with the second half in reverse.
    pairs = []
    half = N // 2
    for i in range(half):
        a = order[i]
        b = order[N - 1 - i]
        # Ensure each pair is ordered lexicographically (i.e. smaller first)
        if a > b:
            a, b = b, a
        pairs.append((a, b))

    # Return the pairs sorted lexicographically.
    return sorted(pairs)
