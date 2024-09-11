import os
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
from ducc0 import wgridder

__all__ = [
    'image_to_vis',
    'vis_to_image'
]

from dsa2000_cal.common.types import FloatArray, ComplexArray, mp_policy


# TODO: set JVP for these, which is just the operator itself.
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
            If num_freqs is present, the visibilities in each channel are computed with respective image slice.
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

    # Negate w to handle ducc#34
    uvw = uvw.at[:, 2].multiply(-1., indices_are_sorted=True, unique_indices=True)

    args = (
        uvw, freqs, dirty, wgt, mask, pixsize_m, pixsize_l, center_m, center_l,
        epsilon, do_wgridding, flip_v, divide_by_n, sigma_min, sigma_max,
        nthreads, verbosity
    )

    return jax.pure_callback(_host_dirty2vis, result_shape_dtype, *args, vectorized=True)


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
        dirty: [[num_freqs,]num_l, num_m] array of dirty image, in units of JY/PIXEL.
            If num_freqs is present, the visibilities in each channel are computed with respective image slice.
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

    uvw = np.asarray(uvw, order='F', dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    dirty = np.asarray(dirty, order='F')
    if len(np.shape(dirty)) == 3:
        if np.shape(dirty)[0] != np.shape(freqs)[0]:
            raise ValueError(f"Expected dirty to have shape (num_freqs, num_l, num_m), got {np.shape(dirty)}")
        dirty = np.asarray(dirty, order='c')  # [num_freqs, num_l, num_m]
        per_chan_image = True
    else:
        per_chan_image = False
    num_rows, _ = np.shape(uvw)

    if wgt is not None:
        wgt = np.asarray(wgt, order='F').astype(dirty.dtype)

    if mask is not None:
        mask = np.asarray(mask, order='F').astype(np.uint8)

    output_dtype = (1j * np.ones(1, dtype=dirty.dtype)).dtype

    def compute_vis_for_channel(chan_idx):
        chan_slice = slice(chan_idx, chan_idx + 1)
        dirty_slice = dirty[chan_idx, :, :]
        freqs_slice = freqs[chan_idx, :]
        wgt_slice = wgt[:, chan_slice] if wgt is not None else None
        mask_slice = mask[:, chan_slice] if mask is not None else None
        output_vis_slice = output_vis[chan_slice, :, :]
        wgridder.dirty2vis(
            uvw=uvw,
            freq=freqs_slice,
            dirty=dirty_slice,
            wgt=wgt_slice,
            mask=mask_slice,
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
            nthreads=1,  # Each thread will handle one channel
            verbosity=verbosity,
            vis=output_vis_slice
        )

    if per_chan_image:
        num_freqs, _ = np.shape(freqs)
        output_vis = np.zeros((num_freqs, num_rows) + np.shape(freqs)[1:], order='F', dtype=output_dtype)
        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            executor.map(compute_vis_for_channel, range(num_freqs))
    else:
        num_freqs, = np.shape(freqs)
        output_vis = np.zeros((num_rows, num_freqs), order='F', dtype=output_dtype)
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
              double_precision_accumulation: bool = False,
              spectral_cube: bool = False) -> jax.Array:
    """
    Compute the dirty image from the visibilities, scaled such that the PSF has unit peak flux.

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
        spectral_cube: if True, an image per channel is produced.

    Returns:
        if spectral_cube=False an [npix_l, npix_m] array of dirty image, in units of JY/PIXEL,
        else an [npix_l, npix_m, num_freqs] array of dirty images.
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

    output_dtype = vis.real.dtype

    # Define the expected shape & dtype of output.
    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=(npix_m, npix_l),
        dtype=output_dtype
    )

    # Negate w to handle ducc#34
    uvw = uvw.at[:, 2].multiply(-1., indices_are_sorted=True, unique_indices=True)

    args = (
        uvw, freqs, vis, wgt, mask, npix_m, npix_l, pixsize_m, pixsize_l,
        center_m, center_l, epsilon, do_wgridding, flip_v, divide_by_n,
        sigma_min, sigma_max, nthreads, verbosity, double_precision_accumulation,
        spectral_cube
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
                    double_precision_accumulation: bool,
                    spectral_cube: bool):
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
        spectral_cube: if True, an image per channel is produced.

    Returns:
        if spectral_cube=False an [npix_l, npix_m] array of dirty image, in units of JY/PIXEL,
        else an [npix_l, npix_m, num_freqs] array of dirty images.
    """
    uvw = np.asarray(uvw, order='F', dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    vis = np.asarray(vis, order='F')  # Fortran order for better cache locality

    output_type = vis.real.dtype

    if wgt is not None:
        wgt = np.asarray(wgt, order='F').astype(output_type)

    if mask is not None:
        mask = np.asarray(mask, order='F').astype(np.uint8)

    if npix_m % 2 != 0 or npix_l % 2 != 0:
        raise ValueError("npix_m and npix_l must both be even.")

    if npix_m < 32 or npix_l < 32:
        raise ValueError("npix_l and npix_m must be at least 32.")

    spectral_cube = bool(spectral_cube)

    # Make sure the output is in JY/PIXEL
    if spectral_cube:
        dirty = np.zeros((npix_l, npix_m, len(freqs)), order='F', dtype=output_type)

        # Threaded computation for each frequency slice
        def compute_dirty_for_channel(chan_idx):
            chan_slice = slice(chan_idx, chan_idx + 1)
            vis_slice = vis[:, chan_slice]
            wgt_slice = wgt[:, chan_slice] if wgt is not None else None
            mask_slice = mask[:, chan_slice] if mask is not None else None
            freqs_slice = freqs[chan_slice]
            dirty_slice = dirty[:, :, chan_idx]
            wgridder.vis2dirty(
                uvw=uvw,
                freq=freqs_slice,
                vis=vis_slice,
                wgt=wgt_slice,
                mask=mask_slice,
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
                nthreads=1,  # Each thread handles one channel
                verbosity=verbosity,
                dirty=dirty_slice,
                double_precision_accumulation=double_precision_accumulation
            )

        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            executor.map(compute_dirty_for_channel, range(len(freqs)))

    else:
        dirty = np.zeros((npix_l, npix_m), order='F', dtype=output_type)
        _ = wgridder.vis2dirty(
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

    return dirty


def vis_to_image(uvw: FloatArray, freqs: FloatArray,
                 vis: ComplexArray,
                 pixsize_m: FloatArray, pixsize_l: FloatArray,
                 center_m: FloatArray, center_l: FloatArray,
                 npix_m: int, npix_l: int,
                 wgt: FloatArray | None = None,
                 mask: FloatArray | None = None,
                 epsilon: float = 1e-6,
                 nthreads: int | None = None, verbosity: int = 0,
                 double_precision_accumulation: bool = False,
                 scale_by_n: bool = True,
                 normalise: bool = True,
                 spectral_cube: bool = False) -> jax.Array:
    """
    Compute the image from the visibilities.

    Args:
        uvw: [num_rows, 3] array of uvw coordinates.
        freqs: [num_freqs] array of frequencies.
        vis: [num_rows, num_freqs] array of visibilities.
        wgt: [num_rows, num_freqs] array of weights, multiplied with input visibilities.
        mask: [num_rows, num_freqs] array of mask, only image vis[mask!=0]
        npix_m: number of pixels in m direction.
        npix_l: number of pixels in l direction.
        pixsize_m: scalar, pixel size in m direction.
        pixsize_l: scalar, pixel size in l direction.
        center_m: scalar, m at center of image.
        center_l: scalar, l at center of image.
        epsilon: scalar, gridding accuracy
        nthreads: number of threads to use.
        verbosity: verbosity level, 0, 1.
        double_precision_accumulation: whether to use double precision for accumulation, which reduces numerical errors.
        scale_by_n: whether to scale the image by n(l,m).
        normalise: whether to normalise the image by the zero-term of the DFT.

    Returns:
        [npix_l, npix_m] array of image.
    """
    if nthreads is None:
        nthreads = os.cpu_count()
    # Make scaled image, I'(l,m)=I(l,m)/n(l,m) such that PSF(l=0,m=0)=1
    image = vis2dirty(
        uvw=uvw,
        freqs=freqs,
        vis=vis,
        wgt=wgt,
        mask=mask,
        npix_m=npix_m,
        npix_l=npix_l,
        pixsize_m=pixsize_m,
        pixsize_l=pixsize_l,
        center_m=center_m,
        center_l=center_l,
        epsilon=epsilon,
        do_wgridding=True,
        flip_v=False,
        divide_by_n=False,
        nthreads=nthreads,
        double_precision_accumulation=double_precision_accumulation,
        verbosity=verbosity,
        spectral_cube=spectral_cube
    )
    if scale_by_n:
        l = (-0.5 * npix_l + jnp.arange(npix_l)) * pixsize_l + center_l
        m = (-0.5 * npix_m + jnp.arange(npix_m)) * pixsize_m + center_m
        l, m = jnp.meshgrid(l, m, indexing='ij')
        n = jnp.sqrt(1. - (jnp.square(l) + jnp.square(m)))
        n = jnp.where(jnp.isnan(n), 0., n)
        image = image * n
    if normalise:
        # Adjoint normalising factor is the DFT zero-term i.e. sum_{u,v,nu} S(u,v,nu)
        sampling_function = jnp.ones(np.shape(vis), image.dtype)
        if wgt is not None:
            sampling_function *= mp_policy.cast_to_image(wgt)
        if mask is not None:
            sampling_function *= mp_policy.cast_to_image(mask, quiet=True)
        adjoint_normalising_factor = jnp.reciprocal(jnp.sum(sampling_function))
        image *= adjoint_normalising_factor
    return mp_policy.cast_to_image(image)


def image_to_vis(uvw: jax.Array, freqs: jax.Array, dirty: jax.Array,
                 pixsize_m: float | jax.Array, pixsize_l: float | jax.Array,
                 center_m: float | jax.Array, center_l: float | jax.Array,
                 mask: jax.Array | None = None,
                 epsilon: float = 1e-6,
                 nthreads: int | None = None, verbosity: int = 0):
    """
    Compute the visibilities from the dirty image.

    Args:
        uvw: [num_rows, 3] array of uvw coordinates.
        freqs: [num_freqs] array of frequencies.
        dirty: [num_l, num_m] array of dirty image, in units of JY/PIXEL.
        pixsize_m: scalar, pixel size in m direction.
        pixsize_l: scalar, pixel size in l direction.
        center_m: scalar, m at center of image.
        center_l: scalar, l at center of image.
        mask: [num_rows, num_freqs] array of mask, only predict where mask!=0.
        epsilon: scalar, gridding accuracy
        nthreads: number of threads to use.
        verbosity: verbosity level, 0, 1.

    Returns:
        [num_rows, num_freqs] array of visibilities.
    """
    if nthreads is None:
        nthreads = os.cpu_count()
    # Divides I(l,m) by n(l,m) then applies gridding with w-term taken into account.
    # Pixels should be in Jy/pixel.
    return mp_policy.cast_to_vis(dirty2vis(
        uvw=uvw,
        freqs=freqs,
        dirty=dirty,
        pixsize_m=pixsize_m,
        pixsize_l=pixsize_l,
        center_m=center_m,
        center_l=center_l,
        epsilon=epsilon,
        do_wgridding=True,
        wgt=None,
        mask=mask,
        flip_v=False,
        divide_by_n=True,
        nthreads=nthreads,
        verbosity=verbosity
    ))
