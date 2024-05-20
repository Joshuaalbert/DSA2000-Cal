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
        dirty: [num_x, num_y] array of dirty image, in units of JY/PIXEL.
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
        dirty: [num_x, num_y] array of dirty image, in units of JY/PIXEL.
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

    if mask is not None:
        mask = mask.astype(np.uint8)

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


def vis2dirty(uvw: jax.Array, freqs: jax.Array, vis: jax.Array,
              npix_x: int, npix_y: int,
              pixsize_x: float | jax.Array, pixsize_y: float | jax.Array,
              center_x: float | jax.Array, center_y: float | jax.Array,
              epsilon: float, do_wgridding: bool = True,
              wgt: jax.Array | None = None, mask: jax.Array | None = None,
              flip_v: bool = False, divide_by_n: bool = True,
              sigma_min: float = 1.1, sigma_max: float = 2.6,
              nthreads: int = 1, verbosity: int = 0,
              double_precision_accumulation: bool = False) -> jax.Array:
    """
    Compute the dirty image from the visibilities.

    Args:
        uvw: [num_rows, 3] array of uvw coordinates.
        freqs: [num_freqs] array of frequencies.
        vis: [num_rows, num_freqs] array of visibilities.
        wgt: [num_rows, num_freqs] array of weights, multiplied with input visibilities.
        mask: [num_rows, num_freqs] array of mask, only predict where mask!=0.
        npix_x: number of pixels in x direction.
        npix_y: number of pixels in y direction.
        pixsize_x: scalar, pixel size in x direction in projected radians (l-units)
        pixsize_y: scalar, pixel size in y direction in projected radians (l-units)
        center_x: scalar, center of image in x direction in projected radians (l-units)
        center_y: scalar, center of image in y direction in projected radians (l-units)
        epsilon: scalar, gridding kernel width.
        do_wgridding: scalar, whether to do w-gridding.
        flip_v: scalar, whether to flip the v axis.
        divide_by_n: whether to divide by n.
        sigma_min: scalar, minimum sigma for gridding.
        sigma_max: scalar, maximum sigma for gridding.
        nthreads: number of threads to use.
        verbosity: verbosity level, 0, 1.
        double_precision_accumulation: whether to use double precision for accumulation, which reduces numerical
            errors for special cases.

    Returns:
        [npix_x, npix_y] array of dirty image, in units of JY/PIXEL.
    """

    if len(np.shape(uvw)) != 2:
        raise ValueError(f"Expected uvw to be shape (num_rows, 3), got {np.shape(uvw)}")
    if len(np.shape(freqs)) != 1:
        raise ValueError(f"Expected freqs to be shape (num_freqs,), got {np.shape(freqs)}")
    if len(np.shape(vis)) != 2:
        raise ValueError(f"Expected vis to be shape (num_rows, num_freqs), got {np.shape(vis)}")
    if wgt is not None and np.shape(wgt) != np.shape(vis):
        raise ValueError(f"Expected wgt to be shape (num_rows, num_freqs), got {np.shape(wgt)}")
    if mask is not None and np.shape(mask) != np.shape(vis):
        raise ValueError(f"Expected mask to be shape (num_rows, num_freqs), got {np.shape(mask)}")

    if not jnp.iscomplexobj(vis):
        raise ValueError("vis must be complex.")

    output_dtype = (jnp.ones(1, dtype=vis.dtype).real).dtype

    # Define the expected shape & dtype of output.
    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=(npix_x, npix_y),
        dtype=output_dtype
    )

    args = (
        uvw, freqs, vis, wgt, mask, npix_x, npix_y, pixsize_x, pixsize_y,
        center_x, center_y, epsilon, do_wgridding, flip_v, divide_by_n,
        sigma_min, sigma_max, nthreads, verbosity, double_precision_accumulation
    )

    return jax.pure_callback(_host_vis2dirty, result_shape_dtype, *args, vectorized=False)


def _host_vis2dirty(uvw: np.ndarray, freqs: np.ndarray,
                    vis: np.ndarray, wgt: np.ndarray | None,
                    mask: np.ndarray | None,
                    npix_x: int, npix_y: int,
                    pixsize_x: float, pixsize_y: float,
                    center_x: float, center_y: float,
                    epsilon: float, do_wgridding: bool,
                    flip_v: bool, divide_by_n: bool,
                    sigma_min: float, sigma_max: float,
                    nthreads: int, verbosity: int,
                    double_precision_accumulation: bool):
    """
    Compute the dirty image from the visibilities.

    Args:
        uvw: [num_rows, 3] array of uvw coordinates.
        freqs: [num_freqs] array of frequencies.
        vis: [num_rows, num_freqs] array of visibilities.
        wgt: [num_rows, num_freqs] array of weights, multiplied with input visibilities.
        mask: [num_rows, num_freqs] array of mask, only predict where mask!=0.
        npix_x: number of pixels in x direction.
        npix_y: number of pixels in y direction.
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
        double_precision_accumulation: whether to use double precision for accumulation, which reduces numerical
            errors for special cases.

    Returns:
        [npix_x, npix_y] array of dirty image, in units of JY/PIXEL.
    """
    uvw = np.asarray(uvw, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)

    float_dtype = (np.ones(1, dtype=vis.dtype).real).dtype
    dirty = np.zeros((npix_x, npix_y), dtype=float_dtype)

    if wgt is not None:
        wgt = wgt.astype(float_dtype)

    if mask is not None:
        mask = mask.astype(np.uint8)

    if npix_x % 2 != 0 or npix_y % 2 != 0:
        raise ValueError("npix_x and npix_y must both be even.")

    if npix_x < 32 or npix_y < 32:
        raise ValueError("npix_x and npix_y must be at least 32.")

    return wgridder.vis2dirty(
        uvw=uvw,
        freq=freqs,
        vis=vis,
        wgt=wgt,
        mask=mask,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        center_x=center_x,
        center_y=center_y,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        flip_v=flip_v,
        divide_by_n=divide_by_n,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        nthreads=nthreads,
        verbosity=verbosity,
        dirty=dirty,
        double_precision_accumulation=double_precision_accumulation
    )
