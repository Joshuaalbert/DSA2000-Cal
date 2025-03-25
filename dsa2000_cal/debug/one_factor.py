import pytest


def one_factorization_round(N: int, round_idx: int):
    """
    Compute the one factor (perfect matching) for a complete graph K_N
    for a given round index using a round-robin (circle method) algorithm.

    Parameters:
      N         : int - number of vertices (must be even)
      round_idx : int - index of the round, 0 <= round_idx < N-1

    Returns:
      A list of tuples (i, j) with i < j, representing the matching for that round,
      sorted lexicographically.
    """
    assert N % 2 == 0, "Only even N is supported."
    assert 0 <= round_idx < N - 1, "round_idx must be between 0 and N-2 (inclusive)."

    # Create a list of players labeled 0 to N-1.
    players = list(range(N))

    # Fix the first player and rotate the others by round_idx.
    fixed = players[0]
    rotated = players[1:]
    rotated = rotated[round_idx:] + rotated[:round_idx]

    # The order for this round: fixed player, then the rotated players.
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


# ================== Pytest Test Cases ==================
@pytest.mark.parametrize('N', [4, 6, 2048])
def test_one_factorization_round_correctness(N):
    # For N=4, the complete graph K_4 has edges:
    # {(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)}
    N = 4
    all_edges = {(i, j) for i in range(N) for j in range(i + 1, N)}
    rounds = []

    # There should be N-1 = 3 rounds.
    for r in range(N - 1):
        factor = one_factorization_round(N, r)
        # Check that each factor forms a perfect matching (each vertex appears once)
        vertices = set()
        for i, j in factor:
            vertices.add(i)
            vertices.add(j)
            # Each pair should be in order (i < j)
            assert i < j, f"Pair {(i, j)} is not in lex order."
        assert vertices == set(range(N)), f"Round {r} does not cover all vertices."
        rounds.append(set(factor))

        # Check that the pairs in this round are lex sorted.
        assert factor == sorted(factor), f"Round {r} pairs are not lex sorted."

    # The union of all rounds should equal the full set of edges.
    union_edges = set.union(*rounds)
    assert union_edges == all_edges, "Union of all rounds does not equal complete graph edges."
