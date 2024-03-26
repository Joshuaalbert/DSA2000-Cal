from dsa2000_cal.gain_models.gain_model import get_interp_indices_and_weights


def test_get_interp_indices_and_weights():
    xp = [0, 1, 2, 3]
    x = 1.5
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 1
    assert alpha0 == 0.5
    assert i1 == 2
    assert alpha1 == 0.5

    x = 0
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 1
    assert i1 == 1
    assert alpha1 == 0

    x = 3
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 2
    assert alpha0 == 0
    assert i1 == 3
    assert alpha1 == 1

    x = -1
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert xp[i0] * alpha0 + xp[i1] * alpha1 == -1

    x = 4
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert xp[i0] * alpha0 + xp[i1] * alpha1 == 4

    x = 5
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert xp[i0] * alpha0 + xp[i1] * alpha1 == 5
