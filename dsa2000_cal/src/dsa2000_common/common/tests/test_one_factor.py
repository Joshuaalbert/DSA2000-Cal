import numpy as np
import pytest

from dsa2000_common.common.one_factor import get_one_factors


@pytest.mark.parametrize('N', [4, 6, 2048])
def test_one_factorization_round_correctness(N):
    # For N=4, the complete graph K_4 has edges:
    # {(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)}
    all_edges = {(i, j) for i in range(N) for j in range(i + 1, N)}
    rounds = []

    # There should be N-1 = 3 rounds.
    factors = get_one_factors(N)
    for r in range(N - 1):
        factor = factors[r]
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
        antenna1, antenna2 = zip(*factor)
        assert antenna1 == tuple(sorted(antenna1))


    # The union of all rounds should equal the full set of edges.
    union_edges = set.union(*rounds)
    assert union_edges == all_edges, "Union of all rounds does not equal complete graph edges."


@pytest.mark.parametrize("N", [(2048)])
def test_one_factorisation_compare(N):
    import pylab as plt
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    def calc_concentration(rows):
        return np.max(np.bincount(np.asarray(rows).flatten(), minlength=N)) / len(rows)

    def calc_cache_locality(rows):
        rows = np.asarray(rows)
        return np.mean(rows[:, 1] - rows[:, 0])

    blocks = get_one_factors(N)
    concentration = []
    for block in blocks:
        concentration.append(calc_concentration(block))
    axs[0].plot(concentration, c='purple', label=f"{len(blocks)} Blocks (one-factor)")

    for c, ordering in zip(['purple', 'orange', 'grey'], ['random', 'natural', 'odd-even']):
        blocks = get_one_factors(N, ordering=ordering)
        locality = []
        for block in blocks:
            locality.append(calc_cache_locality(block))

        axs[1].plot(locality, c=c, label=f"{len(blocks)} Blocks ({ordering} one-factor )")

    rows = []
    for i in range(N):
        for j in range(i + 1, N):
            rows.append((i, j))

    for c, M in zip(['purple', 'orange', 'grey', 'green', 'cyan'], [32, 64, 128, 512, 1024]):
        B = len(rows) // M
        concentration = []
        locality = []
        for start_idx in range(0, len(rows), B):
            end_idx = min(start_idx + B, len(rows))
            concentration.append(calc_concentration(rows[start_idx:end_idx]))
            locality.append(calc_cache_locality(rows[start_idx:end_idx]))

        axs[0].plot(concentration, c=c, ls='dashed', label=f"{M} Blocks (row-factor)")
        axs[1].plot(locality, c=c, ls='dashed', label=f"{M} Blocks (row-factor)")

    axs[1].set_xlabel("Block")
    axs[1].set_xscale('log')
    axs[0].set_ylabel("Concentration")
    axs[1].set_ylabel("Cache-locality")
    axs[0].legend()
    axs[1].legend()
    plt.show()
