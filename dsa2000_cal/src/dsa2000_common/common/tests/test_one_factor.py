import pytest

from dsa2000_common.common.one_factor import one_factorization_round


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
