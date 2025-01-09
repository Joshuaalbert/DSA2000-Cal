import jax
import jax.numpy as jnp
import numpy as np
from ducc0 import wgridder

from dsa2000_cal.common.array_types import ComplexArray, FloatArray, IntArray
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.pure_callback_utils import construct_threaded_pure_callback

__all__ = [
    'image_to_vis',
    'vis_to_image'
]


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

    num_rows, _ = np.shape(uvw)
    num_freqs, = np.shape(freqs)

    output_dtype = (1j * jnp.ones(1, dtype=dirty.dtype)).dtype

    # Define the expected shape & dtype of output.
    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=(num_rows, num_freqs),
        dtype=output_dtype
    )

    def cb_kernel(uvw, freqs, dirty, wgt, mask, pixsize_m, pixsize_l, center_m, center_l):
        return _host_dirty2vis(
            uvw=uvw, freqs=freqs, dirty=dirty, wgt=wgt, mask=mask,
            pixsize_m=pixsize_m, pixsize_l=pixsize_l, center_m=center_m, center_l=center_l,
            epsilon=epsilon, do_wgridding=do_wgridding, flip_v=flip_v, divide_by_n=divide_by_n,
            sigma_min=sigma_min, sigma_max=sigma_max, verbosity=verbosity
        )

    cb = construct_threaded_pure_callback(
        cb_kernel,
        result_shape_dtype,
        2, 1, 2, 2, 2, 0, 0, 0, 0, num_threads=nthreads
    )

    # Negate w to handle ducc#34
    uvw = uvw.at[..., 2].multiply(-1., indices_are_sorted=True, unique_indices=True)

    args = (
        uvw, freqs, dirty, wgt, mask, pixsize_m, pixsize_l, center_m, center_l
    )

    return cb(*args)


def _host_dirty2vis(uvw: FloatArray, freqs: FloatArray,
                    dirty: FloatArray, wgt: FloatArray | None,
                    mask: FloatArray | None,
                    pixsize_m: IntArray, pixsize_l: IntArray,
                    center_m: FloatArray, center_l: FloatArray,
                    epsilon: float, do_wgridding: bool,
                    flip_v: bool, divide_by_n: bool,
                    sigma_min: float, sigma_max: float,
                    verbosity: int):
    """
    Compute the visibilities from the dirty image.

    Args:
        uvw: [num_rows, 3] array of uvw coordinates.
        freqs: [num_freqs[,1]] array of frequencies.
        dirty: [[num_freqs], ..., num_l, num_m] array of dirty image, in units of JY/PIXEL.
        wgt: [num_rows, num_freqs] array of weights, multiplied with output visibilities.
        mask: [num_rows, num_freqs] array of mask, only predict where mask!=0.
        pixsize_m: [[num_freqs]], pixel size in x direction.
        pixsize_l: [[num_freqs]], pixel size in y direction.
        center_m: [[num_freqs]], center of image in x direction.
        center_l: [[num_freqs]], center of image in y direction.
        epsilon: scalar, gridding kernel width.
        do_wgridding: scalar, whether to do w-gridding.
        flip_v: scalar, whether to flip the v axis.
        divide_by_n: whether to divide by n.
        sigma_min: scalar, minimum sigma for gridding.
        sigma_max: scalar, maximum sigma for gridding.
        verbosity: verbosity level, 0, 1.

    Returns:
        [num_rows, num_freqs] array of visibilities.
    """

    uvw = np.asarray(uvw, order='C', dtype=np.float64)  # [num_rows, 3]
    freqs = np.asarray(freqs, order='C', dtype=np.float64)  # [num_freqs]
    dirty = np.asarray(dirty, order='C')  # [num_l, num_m]

    num_rows, _ = np.shape(uvw)
    num_freq, = np.shape(freqs)

    if not len({np.shape(pixsize_l), np.shape(pixsize_m), np.shape(center_l), np.shape(center_m)}) == 1:
        raise ValueError("pixsize_l, pixsize_m, center_l, center_m must have the same shape.")

    if wgt is not None:
        wgt = np.asarray(wgt, order='F').astype(dirty.dtype)  # [num_rows, num_freqs]

    if mask is not None:
        mask = np.asarray(mask, order='F').astype(np.uint8)  # [num_rows, num_freqs]
    if dirty.dtype == np.float32:
        output_vis = np.zeros((num_rows, num_freq), order='F', dtype=np.complex64)
    elif dirty.dtype == np.float64:
        output_vis = np.zeros((num_rows, num_freq), order='F', dtype=np.complex128)
    else:
        raise ValueError(f"Expected dirty to be float32 or float64, got {dirty.dtype}")

    wgridder.dirty2vis(
        uvw=uvw,
        freq=freqs,
        dirty=dirty,
        wgt=wgt,
        mask=mask,
        pixsize_x=pixsize_l,
        pixsize_y=pixsize_m,
        center_x=center_l,
        center_y=center_m,
        epsilon=float(epsilon),
        do_wgridding=bool(do_wgridding),
        flip_v=bool(flip_v),
        divide_by_n=bool(divide_by_n),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        nthreads=1,
        verbosity=int(verbosity),
        vis=output_vis
    )

    return output_vis


def vis2dirty(uvw: jax.Array, freqs: jax.Array, vis: jax.Array,
              npix_m: int, npix_l: int,
              pixsize_m: float | jax.Array, pixsize_l: float | jax.Array,
              center_m: float | jax.Array, center_l: float | jax.Array,
              wgt: jax.Array | None, mask: jax.Array | None,
              epsilon: float, do_wgridding: bool,
              flip_v: bool = False, divide_by_n: bool = True,
              sigma_min: float = 1.1, sigma_max: float = 2.6,
              nthreads: int = 1, verbosity: int = 0,
              double_precision_accumulation: bool = False,
              ) -> jax.Array:
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

    Returns:
        an [npix_l, npix_m] array of dirty image, in units of JY/PIXEL,
    """

    if not jnp.issubdtype(vis, jnp.complexfloating):
        raise ValueError(f"Expected vis to be complex, got {vis.dtype}")

    output_dtype = vis.real.dtype

    shape = (npix_l, npix_m)

    # Define the expected shape & dtype of output.
    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=shape,
        dtype=output_dtype
    )

    # Negate w to handle ducc#34
    uvw = uvw.at[:, 2].multiply(-1., indices_are_sorted=True, unique_indices=True)

    def cb_kernel(
            uvw: FloatArray, freqs: FloatArray, vis: ComplexArray, wgt: FloatArray | None, mask: FloatArray | None,
            pixsize_m: FloatArray, pixsize_l: FloatArray, center_m: FloatArray, center_l: FloatArray
    ):
        return _host_vis2dirty(
            uvw=uvw, freqs=freqs, vis=vis, wgt=wgt, mask=mask,
            npix_m=npix_m, npix_l=npix_l, pixsize_m=pixsize_m, pixsize_l=pixsize_l,
            center_m=center_m, center_l=center_l, epsilon=epsilon, do_wgridding=do_wgridding,
            flip_v=flip_v, divide_by_n=divide_by_n, sigma_min=sigma_min, sigma_max=sigma_max,
            verbosity=verbosity, double_precision_accumulation=double_precision_accumulation
        )

    args = (
        uvw, freqs, vis, wgt, mask, pixsize_m, pixsize_l, center_m, center_l
    )

    cb = construct_threaded_pure_callback(
        cb_kernel,
        result_shape_dtype,
        2, 1, 2, 2, 2, 0, 0, 0, 0, num_threads=nthreads
    )
    return cb(*args)


def _host_vis2dirty(
        uvw: FloatArray, freqs: FloatArray,
        vis: FloatArray, wgt: FloatArray | None,
        mask: FloatArray | None,
        npix_m: IntArray, npix_l: IntArray,
        pixsize_m: FloatArray, pixsize_l: FloatArray,
        center_m: FloatArray, center_l: FloatArray,
        epsilon: float, do_wgridding: bool,
        flip_v: bool, divide_by_n: bool,
        sigma_min: float, sigma_max: float,
        verbosity: int,
        double_precision_accumulation: bool,
        output_buffer: np.ndarray | None = None,
        num_threads: int = 1
):
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
        verbosity: verbosity level, 0, 1.
        double_precision_accumulation: whether to use double precision for accumulation, which reduces numerical
            errors for special cases.
        output_buffer: optional [npix_l, npix_m] array of dirty image, in units of JY/PIXEL

    Returns:
        [npix_l, npix_m] array of dirty image, in units of JY/PIXEL.
    """
    uvw = np.asarray(uvw, order='C', dtype=np.float64)  # [num_rows, 3]
    freqs = np.asarray(freqs, order='C', dtype=np.float64)  # [num_freqs]
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

    # Make sure the output is in JY/PIXEL
    if output_buffer is not None:
        if np.shape(output_buffer) != (npix_l, npix_m):
            raise ValueError(f"Expected output_buffer to have shape {(npix_l, npix_m)}, got {np.shape(output_buffer)}")
        if not np.issubdtype(np.result_type(output_buffer), output_type):
            raise ValueError(f"Expected output_buffer to have dtype {output_type}, got {np.result_type(output_buffer)}")
        dirty = output_buffer
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
        nthreads=num_threads,
        verbosity=verbosity,
        dirty=dirty,
        double_precision_accumulation=double_precision_accumulation
    )
    return dirty


def vis_to_image(uvw: FloatArray, freqs: FloatArray, vis: ComplexArray, pixsize_m: FloatArray, pixsize_l: FloatArray,
                 center_m: FloatArray, center_l: FloatArray, npix_m: int, npix_l: int, wgt: FloatArray | None = None,
                 mask: FloatArray | None = None, epsilon: float = 1e-6, nthreads: int | None = None, verbosity: int = 0,
                 double_precision_accumulation: bool = False, scale_by_n: bool = True,
                 normalise: bool = True) -> jax.Array:
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
        verbosity: verbosity level, 0, 1.
        double_precision_accumulation: whether to use double precision for accumulation, which reduces numerical errors.
        scale_by_n: whether to scale the image by n(l,m).
        normalise: whether to normalise the image by the zero-term of the DFT.

    Returns:
        [npix_l, npix_m] array of image.
    """

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
        verbosity=verbosity
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


def vis_to_image_np(uvw: FloatArray, freqs: FloatArray, vis: ComplexArray, pixsize_m: FloatArray, pixsize_l: FloatArray,
                    center_m: FloatArray, center_l: FloatArray, npix_m: int, npix_l: int, wgt: FloatArray | None = None,
                    mask: FloatArray | None = None, epsilon: float = 1e-6,
                    verbosity: int = 0,
                    double_precision_accumulation: bool = False, scale_by_n: bool = True,
                    normalise: bool = True,
                    output_buffer: np.ndarray | None = None,
                    num_threads: int= 1) -> jax.Array:
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
        verbosity: verbosity level, 0, 1.
        double_precision_accumulation: whether to use double precision for accumulation, which reduces numerical errors.
        scale_by_n: whether to scale the image by n(l,m).
        normalise: whether to normalise the image by the zero-term of the DFT.
        output_buffer: optional [npix_l, npix_m] array of dirty image, in units of JY/PIXEL
            Should be same precision as vis.
        num_threads: number of threads to use.

    Returns:
        [npix_l, npix_m] array of image.
    """

    # Make scaled image, I'(l,m)=I(l,m)/n(l,m) such that PSF(l=0,m=0)=1
    uvw[:, 2] *= -1

    image = _host_vis2dirty(
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
        sigma_min=1.1,
        sigma_max=2.6,
        double_precision_accumulation=double_precision_accumulation,
        verbosity=verbosity,
        output_buffer=output_buffer,
        num_threads=num_threads
    )
    if scale_by_n:
        l = (-0.5 * npix_l + np.arange(npix_l)) * pixsize_l + center_l
        m = (-0.5 * npix_m + np.arange(npix_m)) * pixsize_m + center_m
        l, m = np.meshgrid(l, m, indexing='ij')
        n = np.sqrt(1. - (np.square(l) + np.square(m)))
        del l
        del m
        image *= np.logical_not(np.isnan(n)).astype(image.dtype)
    if normalise:
        # Adjoint normalising factor is the DFT zero-term i.e. sum_{u,v,nu} S(u,v,nu)
        sampling_function = np.ones(np.shape(vis), image.dtype)
        if wgt is not None:
            sampling_function *= mp_policy.cast_to_image(wgt)
        if mask is not None:
            sampling_function *= mp_policy.cast_to_image(mask, quiet=True)
        adjoint_normalising_factor = np.reciprocal(np.sum(sampling_function))
        image *= adjoint_normalising_factor
    return mp_policy.cast_to_image(image)


def image_to_vis(uvw: jax.Array, freqs: jax.Array, dirty: jax.Array,
                 pixsize_m: float | jax.Array, pixsize_l: float | jax.Array,
                 center_m: float | jax.Array, center_l: float | jax.Array,
                 mask: jax.Array | None = None,
                 epsilon: float = 1e-6,
                 nthreads: int | None = None,
                 verbosity: int = 0):
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
