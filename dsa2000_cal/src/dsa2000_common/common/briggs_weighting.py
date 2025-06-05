import numpy as np

C_LIGHT = 299_792_458.0  # [m s-1]


def briggs_weighting(
        robust: float,
        uvw: np.ndarray,  # [T, B, 3]  – metres
        weights: np.ndarray,  # [T, B, C]  – natural vis-weights (1/σ²)
        freqs: np.ndarray,  # [C]        – Hz
        dl: float,  # rad / pixel   (image cell size in l-dir)
        dm: float,  # rad / pixel   (image cell size in m-dir)
        num_l: int,  # Npix in l
        num_m: int  # Npix in m
) -> np.ndarray:
    """
    Compute Briggs/robust imaging weights.

    Parameters
    ----------
    robust : float
        Briggs “robust” parameter (≈ −2 ↔ uniform, 0 ↔ balanced, +2 ↔ natural).
    uvw : array_like, shape (T, B, 3)
        Baseline UVW coordinates in **metres** (per timestamp T and baseline B).
    weights : array_like, shape (T, B, C)
        Natural visibility weights ω_i (usually 1/σ_i²) per freq-channel C.
    freqs : array_like, shape (C,)
        Channel sky-frequencies in **Hz**.
    dl, dm : float
        Image cell size in the (l,m) directions [radians].
    num_l, num_m : int
        Image dimensions (# pixels) along (l,m).

    Returns
    -------
    briggs_weights : ndarray, shape (T, B, C)
        Imaging weights w_i suitable for gridding.
    """
    # --- 1. Pre-compute helpers ------------------------------------------------
    # λ for each channel; broadcast to [T, B, C]
    lam = C_LIGHT / freqs  # [C]
    lam = lam.reshape((1, 1, -1))  # -> [1,1,C] for broadcasting

    # uv in wavelengths for every (T,B,C)
    u_m, v_m = uvw[..., 0:1], uvw[..., 1:2]  # [T,B,1]
    u = u_m / lam  # [T,B,C]
    v = v_m / lam

    # uv-cell size (Fourier of image sampling)
    du = 1.0 / (num_l * dl)
    dv = 1.0 / (num_m * dm)

    # grid indices (origin at centre of array)
    u_pix = np.floor(u / du + num_l / 2).astype(np.int64)
    v_pix = np.floor(v / dv + num_m / 2).astype(np.int64)

    # Mask out points that fall outside the requested uv-grid
    valid = (
            (u_pix >= 0) & (u_pix < num_l) &
            (v_pix >= 0) & (v_pix < num_m)
    )

    # --- 2. Build the natural weight-density map  W_k --------------------------
    #   W_k = Σ_cell=k ω_i                                         ──── CASAdocs eqn. (11.73):contentReference[oaicite:0]{index=0}
    W_grid = np.zeros((num_l, num_m, freqs.size), dtype=np.float64)

    for ch in range(freqs.size):
        sel = valid[..., ch]
        np.add.at(
            W_grid[..., ch],
            (u_pix[..., ch][sel], v_pix[..., ch][sel]),
            weights[..., ch][sel]
        )

    # --- 3. Robust scaling factor  f² -----------------------------------------
    #   f² = (5·10^(−R))² / ( Σ_k W_k² / Σ_i ω_i )                ──── CASAdocs eqn. (11.74):contentReference[oaicite:1]{index=1}
    sum_w = weights.sum(dtype=np.float64)
    sum_Wk2 = np.square(W_grid).sum(dtype=np.float64)
    f2 = (5.0 * 10.0 ** (-robust)) ** 2 / (sum_Wk2 / sum_w)

    # --- 4. Per-visibility Briggs weights  w_i ------------------------------
    #   w_i = ω_i / (1 + W_k · f²)                                 ──── CASAdocs eqn. (11.73):contentReference[oaicite:2]{index=2}
    briggs = np.zeros_like(weights, dtype=np.float64)

    for ch in range(freqs.size):
        sel = valid[..., ch]
        # Gather W_k for each visibility
        Wk_vis = np.zeros_like(weights[..., ch], dtype=np.float64)
        Wk_vis[sel] = W_grid[u_pix[..., ch][sel], v_pix[..., ch][sel], ch]
        briggs[..., ch] = weights[..., ch] / (1.0 + Wk_vis * f2)

        # Set weights to zero for visibilities that fell outside the grid
        briggs[..., ch][~sel] = 0.0

    return briggs


def test_briggs_two_visibilities_same_cell():
    """
    Both visibilities fall into the very same uv–cell, so we can compute the
    expected Briggs weight analytically and compare.
    """
    # ---------- inputs -------------------------------------------------------
    robust = 0.0  # balanced (CASA default)
    C_LIGHT = 299_792_458.0  # m s⁻¹

    freqs = np.array([C_LIGHT])  # λ = 1 m  → uvw (m) == uvw (λ)
    uvw = np.zeros((1, 2, 3))  # T=1, B=2  – both baselines at u=v=0
    weights = np.ones((1, 2, 1))  # natural weights ω_i = 1 for both vis.

    dl = dm = 1.0  # rad pix⁻¹  → du = dv = 1/(N Δl) = 0.25
    num_l = num_m = 4  # 4×4 uv-grid is enough for the test

    # ---------- run code under test -----------------------------------------
    briggs = briggs_weighting(
        robust, uvw, weights, freqs, dl, dm, num_l, num_m
    )

    # ---------- expected answer ---------------------------------------------
    # All visibilities land in one cell  ⇒  W_k = Σ ω_i = 2
    sum_w = 2.0
    sum_Wk2 = 2.0 ** 2  # only one populated cell
    f2 = (5.0 * 10 ** (-robust)) ** 2 / (sum_Wk2 / sum_w)  # eqn. (11.74)

    expected = 1.0 / (1.0 + 2.0 * f2)  # ω_i / (1 + W_k f²)

    # ---------- assertions ---------------------------------------------------
    assert briggs.shape == (1, 2, 1)
    assert np.allclose(briggs, expected, rtol=1e-12, atol=0.0)
    assert np.all(briggs <= weights)  # Briggs never increases weight

    print(briggs)
