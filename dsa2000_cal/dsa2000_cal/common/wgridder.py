import jax
import jax.numpy as jnp
import numpy as np
from ducc0 import wgridder

__all__ = [
    'dirty2vis',
    'vis2dirty'
]


def dirty2vis(uvw: jax.Array, freqs: jax.Array, dirty: jax.Array,
              pixsize_m: float | jax.Array, pixsize_l: float | jax.Array,
              center_m: float | jax.Array, center_l: float | jax.Array,
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
        dirty: [num_l, num_m] array of dirty image, in units of JY/PIXEL.
        pixsize_m: scalar, pixel size in x direction.
        pixsize_l: scalar, pixel size in y direction.
        center_m: scalar, center of image in x direction.
        center_l: scalar, center of image in y direction.
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
        raise ValueError(f"Expected dirty to be shape (num_m, num_l), got {np.shape(dirty)}")
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
        uvw, freqs, dirty, wgt, mask, pixsize_m, pixsize_l, center_m, center_l,
        epsilon, do_wgridding, flip_v, divide_by_n, sigma_min, sigma_max,
        nthreads, verbosity
    )

    return jax.pure_callback(_host_dirty2vis, result_shape_dtype, *args, vectorized=False)


def _host_dirty2vis(uvw: np.ndarray, freqs: np.ndarray,
                    dirty: np.ndarray, wgt: np.ndarray | None,
                    mask: np.ndarray | None,
                    pixsize_m: float, pixsize_l: float,
                    center_m: float, center_l: float,
                    epsilon: float, do_wgridding: bool,
                    flip_v: bool, divide_by_n: bool,
                    sigma_min: float, sigma_max: float,
                    nthreads: int, verbosity: int):
    """
    Compute the visibilities from the dirty image.

    Args:
        uvw: [num_rows, 3] array of uvw coordinates.
        freqs: [num_freqs] array of frequencies.
        dirty: [num_l, num_m] array of dirty image, in units of JY/PIXEL.
        wgt: [num_rows, num_freqs] array of weights, multiplied with output visibilities.
        mask: [num_rows, num_freqs] array of mask, only predict where mask!=0.
        pixsize_m: scalar, pixel size in x direction.
        pixsize_l: scalar, pixel size in y direction.
        center_m: scalar, center of image in x direction.
        center_l: scalar, center of image in y direction.
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
    _ = wgridder.dirty2vis(
        uvw=uvw,
        freq=freqs,
        dirty=dirty,
        wgt=wgt,
        mask=mask,
        pixsize_x=float(pixsize_l),
        pixsize_y=float(pixsize_m),
        center_x=float(center_l),
        center_y=float(center_m),
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
    return output_vis


def vis2dirty(uvw: jax.Array, freqs: jax.Array, vis: jax.Array,
              npix_m: int, npix_l: int,
              pixsize_m: float | jax.Array, pixsize_l: float | jax.Array,
              center_m: float | jax.Array, center_l: float | jax.Array,
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
        npix_m: number of pixels in y direction.
        npix_l: number of pixels in x direction.
        pixsize_m: scalar, pixel size in y direction in projected radians (l-units)
        pixsize_l: scalar, pixel size in x direction in projected radians (l-units)
        center_m: scalar, center of image in y direction in projected radians (l-units)
        center_l: scalar, center of image in x direction in projected radians (l-units)
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
        [npix_l, npix_m] array of dirty image, in units of JY/PIXEL.
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
        shape=(npix_m, npix_l),
        dtype=output_dtype
    )

    args = (
        uvw, freqs, vis, wgt, mask, npix_m, npix_l, pixsize_m, pixsize_l,
        center_m, center_l, epsilon, do_wgridding, flip_v, divide_by_n,
        sigma_min, sigma_max, nthreads, verbosity, double_precision_accumulation
    )

    return jax.pure_callback(_host_vis2dirty, result_shape_dtype, *args, vectorized=False)


def _host_vis2dirty(uvw: np.ndarray, freqs: np.ndarray,
                    vis: np.ndarray, wgt: np.ndarray | None,
                    mask: np.ndarray | None,
                    npix_m: int, npix_l: int,
                    pixsize_m: float, pixsize_l: float,
                    center_m: float, center_l: float,
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
        npix_m: number of pixels in y direction.
        npix_l: number of pixels in x direction.
        pixsize_m: scalar, pixel size in y direction.
        pixsize_l: scalar, pixel size in x direction.
        center_m: scalar, center of image in x direction.
        center_l: scalar, center of image in y direction.
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
        [npix_l, npix_m] array of dirty image, in units of JY/PIXEL.
    """
    uvw = np.asarray(uvw, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)

    float_dtype = (np.ones(1, dtype=vis.dtype).real).dtype
    dirty = np.zeros((npix_m, npix_l), dtype=float_dtype)

    if wgt is not None:
        wgt = wgt.astype(float_dtype)

    if mask is not None:
        mask = mask.astype(np.uint8)

    if npix_m % 2 != 0 or npix_l % 2 != 0:
        raise ValueError("npix_m and npix_l must both be even.")

    if npix_m < 32 or npix_l < 32:
        raise ValueError("npix_x and npix_y must be at least 32.")

    # Make sure the output is in JY/PIXEL
    num_rows = np.shape(uvw)[0]
    num_freqs = np.shape(freqs)[0]
    # Factor to convert adjoint gridding and degridding into inverses in the limit of total uvw coverage.
    adjoint_factor = np.reciprocal((4. * num_rows - np.sqrt(8. * num_rows + 1.) - 1)) * 4. / num_freqs
    if wgt is not None:
        adjoint_factor /= np.mean(wgt)

    return adjoint_factor * wgridder.vis2dirty(
        uvw=uvw,
        freq=freqs,
        vis=vis,
        wgt=wgt,
        mask=mask,
        npix_x=npix_l,
        npix_y=npix_m,
        pixsize_x=pixsize_l,
        pixsize_y=pixsize_m,
        center_x=center_l,
        center_y=center_m,
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
