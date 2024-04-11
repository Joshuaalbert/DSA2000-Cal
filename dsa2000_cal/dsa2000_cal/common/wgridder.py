import jax
import jax.numpy as jnp
import numpy as np
from ducc0 import wgridder

__all__ = [
    'dirty2vis'
]


def dirty2vis(uvw: jax.Array, freqs: jax.Array, dirty: jax.Array,
              pixsize_x: float | jax.Array, pixsize_y: float | jax.Array,
              center_x: float | jax.Array, center_y: float | jax.Array,
              epsilon: float, do_wgridding: bool = True,
              wgt: jax.Array | None = None, mask: jax.Array | None = None,
              flip_v: bool = False, divide_by_n: bool = True,
              sigma_min: float = 1.1, sigma_max: float = 2.6,
              nthreads: int = 1, verbosity: int = 0) -> jax.Array:
    """
    Compute the visibilities from the dirty image.

    Args:
        uvw: [num_rows, 3] array of uvw coordinates.
        freqs: [num_freqs] array of frequencies.
        dirty: [num_x, num_y] array of dirty image.
        pixsize_x: scalar, pixel size in x direction.
        pixsize_y: scalar, pixel size in y direction.
        center_x: scalar, center of image in x direction.
        center_y: scalar, center of image in y direction.
        epsilon: scalar, gridding kernel width.
        do_wgridding: scalar, whether to do w-gridding.
        wgt: [num_rows, num_freqs] array of weights, multiplied with output visibilities.
        mask: [num_rows, num_freqs] array of mask, only predict where mask!=0.
        flip_v: scalar, whether to flip the v axis.
        divide_by_n: whether to divide by n.
        sigma_min: scalar, minimum sigma for gridding.
        sigma_max: scalar, maximum sigma for gridding.
        nthreads: number of threads to use.
        verbosity: verbosity level, 0, 1.

    Returns:
        [num_rows, num_freqs] array of visibilities.
    """

    if len(np.shape(uvw)) != 2:
        raise ValueError(f"Expected uvw to be shape (num_rows, 3), got {np.shape(uvw)}")
    if len(np.shape(freqs)) != 1:
        raise ValueError(f"Expected freqs to be shape (num_freqs,), got {np.shape(freqs)}")
    if len(np.shape(dirty)) != 2:
        raise ValueError(f"Expected dirty to be shape (num_x, num_y), got {np.shape(dirty)}")
    if wgt is not None and len(np.shape(wgt)) != 2:
        raise ValueError(f"Expected wgt to be shape (num_rows, num_freqs), got {np.shape(wgt)}")
    if mask is not None and len(np.shape(mask)) != 2:
        raise ValueError(f"Expected mask to be shape (num_rows, num_freqs), got {np.shape(mask)}")

    num_rows = np.shape(uvw)[0]
    num_freqs = np.shape(freqs)[0]

    # Upgrade dirty to complex dtype.
    output_dtype = (1j * jnp.ones(1, dtype=dirty.dtype)).dtype

    # Define the expected shape & dtype of output.
    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=(num_rows, num_freqs),
        dtype=output_dtype
    )

    args = (
        uvw, freqs, dirty, wgt, mask, pixsize_x, pixsize_y, center_x, center_y,
        epsilon, do_wgridding, flip_v, divide_by_n, sigma_min, sigma_max,
        nthreads, verbosity
    )

    # We use vectorize=True because scipy.special.jv handles broadcasted inputs.
    return jax.pure_callback(_host_dirty2vis, result_shape_dtype, *args, vectorized=False)


def _host_dirty2vis(uvw: np.ndarray, freqs: np.ndarray,
                    dirty: np.ndarray, wgt: np.ndarray | None,
                    mask: np.ndarray | None,
                    pixsize_x: float, pixsize_y: float,
                    center_x: float, center_y: float,
                    epsilon: float, do_wgridding: bool,
                    flip_v: bool, divide_by_n: bool,
                    sigma_min: float, sigma_max: float,
                    nthreads: int, verbosity: int):
    """
    Compute the visibilities from the dirty image.

    Args:
        uvw: [num_rows, 3] array of uvw coordinates.
        freqs: [num_freqs] array of frequencies.
        dirty: [num_x, num_y] array of dirty image.
        wgt: [num_rows, num_freqs] array of weights, multiplied with output visibilities.
        mask: [num_rows, num_freqs] array of mask, only predict where mask!=0.
        pixsize_x: scalar, pixel size in x direction.
        pixsize_y: scalar, pixel size in y direction.
        center_x: scalar, center of image in x direction.
        center_y: scalar, center of image in y direction.
        epsilon: scalar, gridding kernel width.
        do_wgridding: scalar, whether to do w-gridding.
        flip_v: scalar, whether to flip the v axis.
        divide_by_n: whether to divide by n.
        sigma_min: scalar, minimum sigma for gridding.
        sigma_max: scalar, maximum sigma for gridding.
        nthreads: number of threads to use.
        verbosity: verbosity level, 0, 1.

    Returns:
        [num_rows, num_freqs] array of visibilities.
    """

    uvw = np.asarray(uvw, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    num_rows = np.shape(uvw)[0]
    num_freqs = np.shape(freqs)[0]

    if wgt is not None:
        wgt = wgt.astype(dirty.dtype)

    output_dtype = (1j * np.ones(1, dtype=dirty.dtype)).dtype

    output_vis = np.zeros((num_rows, num_freqs), dtype=output_dtype)
    return wgridder.dirty2vis(
        uvw=uvw,
        freq=freqs,
        dirty=dirty,
        wgt=wgt,
        mask=mask,
        pixsize_x=float(pixsize_x),
        pixsize_y=float(pixsize_y),
        center_x=float(center_x),
        center_y=float(center_y),
        epsilon=float(epsilon),
        do_wgridding=do_wgridding,
        flip_v=flip_v,
        divide_by_n=divide_by_n,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        nthreads=nthreads,
        verbosity=verbosity,
        vis=output_vis
    )


def test_dirty2vis():
    uvw = jnp.ones((100, 3))
    freqs = jnp.ones((4,))
    dirty = jnp.ones((100, 100))
    wgt = jnp.ones((100, 4))
    pixsize_x = 0.1
    pixsize_y = 0.1
    center_x = 0.0
    center_y = 0.0
    epsilon = 1e-4
    do_wgridding = True
    flip_v = False
    divide_by_n = True
    sigma_min = 1.1
    sigma_max = 2.6
    nthreads = 1
    verbosity = 0
    visibilities = dirty2vis(
        uvw=uvw, freqs=freqs, dirty=dirty, wgt=wgt,
        pixsize_x=pixsize_x, pixsize_y=pixsize_y,
        center_x=center_x, center_y=center_y,
        epsilon=epsilon, do_wgridding=do_wgridding,
        flip_v=flip_v, divide_by_n=divide_by_n,
        sigma_min=sigma_min, sigma_max=sigma_max,
        nthreads=nthreads, verbosity=verbosity
    )
    assert np.all(np.isfinite(visibilities))

    # wgt works as expected
    visibilities2 = dirty2vis(
        uvw=uvw, freqs=freqs, dirty=dirty, wgt=2 * wgt,
        pixsize_x=pixsize_x, pixsize_y=pixsize_y,
        center_x=center_x, center_y=center_y,
        epsilon=epsilon, do_wgridding=do_wgridding,
        flip_v=flip_v, divide_by_n=divide_by_n,
        sigma_min=sigma_min, sigma_max=sigma_max,
        nthreads=nthreads, verbosity=verbosity
    )

    np.testing.assert_allclose(visibilities2, visibilities * 2)

    # JIT works
    fn = jax.jit(lambda uvw:
                 dirty2vis(
                     uvw=uvw, freqs=freqs, dirty=dirty, wgt=wgt,
                     pixsize_x=pixsize_x, pixsize_y=pixsize_y,
                     center_x=center_x, center_y=center_y,
                     epsilon=epsilon, do_wgridding=do_wgridding,
                     flip_v=flip_v, divide_by_n=divide_by_n,
                     sigma_min=sigma_min, sigma_max=sigma_max,
                     nthreads=nthreads, verbosity=verbosity
                 ))
    assert np.all(np.isfinite(fn(uvw)))
