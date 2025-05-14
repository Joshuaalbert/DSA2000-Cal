import time

import numpy as np
from numba import njit, prange


# Original implementation
@njit(parallel=True)
def subtract_psf2d_orig(residuals, psf, gain, peak_val, li0, mi0, li0_psf, mi0_psf):
    num_m_psf, num_l_psf = psf.shape
    num_m, num_l = residuals.shape
    for li_psf in prange(num_l_psf):
        li = li0 + (li_psf - li0_psf)
        if li < 0 or li >= num_l:
            continue
        for mi_psf in range(num_m_psf):
            mi = mi0 + (mi_psf - mi0_psf)
            if mi < 0 or mi >= num_m:
                continue
            residuals[mi, li] -= gain * peak_val * psf[mi_psf, li_psf]


# Fast Numba-parallel flattened loop
@njit(parallel=True)
def subtract_psf2d_fast(residuals, psf, gain, peak_val, li0, mi0, li0_psf, mi0_psf):
    num_m, num_l = residuals.shape
    num_m_psf, num_l_psf = psf.shape

    li_min = li0 - li0_psf
    li_max = li_min + num_l_psf
    mi_min = mi0 - mi0_psf
    mi_max = mi_min + num_m_psf

    l0 = max(0, li_min)
    l1 = min(num_l, li_max)
    m0 = max(0, mi_min)
    m1 = min(num_m, mi_max)

    psf_l0 = l0 - li_min
    psf_m0 = m0 - mi_min

    factor = gain * peak_val
    scaled_psf = psf * factor

    nl = l1 - l0
    nm = m1 - m0
    total = nl * nm

    for idx in prange(total):
        di = idx // nm
        dj = idx % nm
        li = l0 + di
        mi = m0 + dj
        li_psf = psf_l0 + di
        mi_psf = psf_m0 + dj
        residuals[mi, li] -= scaled_psf[mi_psf, li_psf]


# NumPy vectorized slice
def subtract_psf2d_slice(residuals, psf, gain, peak_val, li0, mi0, li0_psf, mi0_psf):
    num_m, num_l = residuals.shape
    num_m_psf, num_l_psf = psf.shape

    li_min = li0 - li0_psf
    li_max = li_min + num_l_psf
    mi_min = mi0 - mi0_psf
    mi_max = mi_min + num_m_psf

    l0 = max(0, li_min)
    l1 = min(num_l, li_max)
    m0 = max(0, mi_min)
    m1 = min(num_m, mi_max)

    psf_l0 = l0 - li_min
    psf_l1 = psf_l0 + (l1 - l0)
    psf_m0 = m0 - mi_min
    psf_m1 = psf_m0 + (m1 - m0)

    residuals[m0:m1, l0:l1] -= gain * peak_val * psf[psf_m0:psf_m1, psf_l0:psf_l1]


def benchmark():
    # problem size
    num_l = num_m = 4096*2
    psf_size = num_l

    # center indices
    li0 = mi0 = num_l // 2
    li0_psf = mi0_psf = psf_size // 2

    gain, peak_val = 0.1, 1.0

    # Prepare data
    residuals_base = np.random.rand(num_m, num_l).astype(np.float32)

    # PSF: all zeros except a single 1 at center
    psf = 0.001 * np.ones((psf_size, psf_size), dtype=np.float32)
    psf[li0_psf, li0_psf] = 1.0

    # Warm up Numba (compile)
    subtract_psf2d_orig(residuals_base.copy(), psf, gain, peak_val, li0, mi0, li0_psf, mi0_psf)
    subtract_psf2d_fast(residuals_base.copy(), psf, gain, peak_val, li0, mi0, li0_psf, mi0_psf)
    subtract_psf2d_slice(residuals_base.copy(), psf, gain, peak_val, li0, mi0, li0_psf, mi0_psf)

    methods = {
        'original': subtract_psf2d_orig,
        'fast': subtract_psf2d_fast,
        'slice': subtract_psf2d_slice
    }

    reps = 50
    times = {}

    # First, verify correctness against original
    orig_out = None
    for name, func in methods.items():
        # apply to fresh copy
        res = residuals_base.copy()
        func(res, psf, gain, peak_val, li0, mi0, li0_psf, mi0_psf)
        if name == 'original':
            orig_out = res
        else:
            assert np.allclose(res, orig_out), f"{name} output differs from original!"

    print("All methods produce identical output ✔️")

    # Benchmark timings
    for name, func in methods.items():
        # extra warm‐ups
        for _ in range(2):
            func(residuals_base.copy(), psf, gain, peak_val, li0, mi0, li0_psf, mi0_psf)
        t0 = time.perf_counter()
        for _ in range(reps):
            func(residuals_base.copy(), psf, gain, peak_val, li0, mi0, li0_psf, mi0_psf)
        t1 = time.perf_counter()
        times[name] = (t1 - t0) / reps

    orig_time = times['original']
    print(f"\n{'Method':<10}{'Time (s)':>12}{'Speedup':>12}")
    for name, t in times.items():
        print(f"{name:<10}{t:12.6f}{orig_time / t:12.2f}×")


if __name__ == "__main__":
    benchmark()
