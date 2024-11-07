import dataclasses
import os
import time as time_mod
from typing import List

import astropy.units as au
import jax.numpy as jnp
import numpy as np
import pylab as plt

from dsa2000_cal.common.fits_utils import ImageModel
from dsa2000_cal.common.jax_utils import block_until_ready
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.imaging.base_imagor import BaseImagor, fit_beam
from dsa2000_cal.measurement_sets.measurement_set import  MeasurementSet


@dataclasses.dataclass(eq=False)
class Imagor:
    """
    Performs imaging (without deconvolution) of visibilties using W-gridder.

    Args:
        plot_folder: the folder to save the images
        field_of_view: the field of view in degrees
        baseline_min: the minimum baseline length in meters, shorter baselines are flagged
        oversample_factor: the oversampling factor, higher is more accurate but bigger image
        nthreads: the number of threads to use, None for all
        epsilon: the epsilon value of wgridder
        convention: the convention to use
        verbose: whether to print verbose output
        weighting: the weighting scheme to use
        coherencies: the coherencies to image, None for all
        spectral_cube: whether to image as a spectral cube
        seed: the random seed
    """

    # Imaging parameters

    plot_folder: str

    field_of_view: au.Quantity | None = None
    baseline_min: au.Quantity = 1. * au.m
    oversample_factor: float = 5.
    nthreads: int | None = None
    epsilon: float = 1e-4
    convention: str = 'physical'
    verbose: bool = False
    weighting: str = 'natural'
    coherencies: List[str] | None = None
    spectral_cube: bool = False
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)
        if self.field_of_view is not None and not self.field_of_view.unit.is_equivalent(au.deg):
            raise ValueError(f"Expected field_of_view to be in degrees, got {self.field_of_view.unit}")

    def image(self, image_name: str, ms: MeasurementSet, psf: bool = False, overwrite: bool = False) -> ImageModel:
        print(f"Imaging {ms}")
        # Metrics
        t0 = time_mod.time()
        gen = ms.create_block_generator(vis=True, weights=True, flags=True, corrs=self.coherencies)
        gen_response = None
        uvw = []
        vis = []
        weights = []
        flags = []

        while True:
            try:
                time, visibility_coords, data = gen.send(gen_response)
            except StopIteration:
                break
            uvw.append(visibility_coords.uvw)
            vis.append(data.vis)
            weights.append(data.weights)
            flags.append(data.flags)

        uvw = jnp.concatenate(uvw, axis=0)  # [num_rows, 3]
        vis = jnp.concatenate(vis, axis=0)  # [num_rows, chan, 4/1]
        weights = jnp.concatenate(weights, axis=0)  # [num_rows, chan, 4/1]
        flags = jnp.concatenate(flags, axis=0)  # [num_rows, chan, 4/1]
        freqs = quantity_to_jnp(ms.meta.freqs)  # [num_chan]

        base_imagor = BaseImagor(
            baseline_min=0.,
            nthreads=self.nthreads,
            epsilon=self.epsilon,
            convention=self.convention,
            verbose=self.verbose,
            weighting=self.weighting
        )

        num_pixel, dl, dm, center_l, center_m = base_imagor.get_image_parameters(
            ms=ms,
            field_of_view=self.field_of_view,
            oversample_factor=self.oversample_factor
        )

        if psf:
            dirty_image = block_until_ready(
                base_imagor.image_psf(
                    uvw=uvw,
                    weights=weights,
                    flags=flags,
                    freqs=freqs,
                    num_pixel=num_pixel,
                    dl=dl,
                    dm=dm,
                    center_l=center_l,
                    center_m=center_m
                )
            )  # [Nl, Nm, coh]
        else:
            dirty_image = block_until_ready(
                base_imagor.image_visibilties(
                    uvw=uvw,
                    vis=vis,
                    weights=weights,
                    flags=flags,
                    freqs=freqs,
                    num_pixel=num_pixel,
                    dl=dl,
                    dm=dm,
                    center_l=center_l,
                    center_m=center_m
                )
            )  # [Nl, Nm, coh]

        # Get beam
        inner_psf_image = block_until_ready(
            base_imagor.image_psf(
                uvw=uvw,
                weights=weights,
                flags=flags,
                freqs=freqs,
                num_pixel=32,
                dl=dl,
                dm=dm,
                center_l=center_l,
                center_m=center_m
            )
        )  # [Nl, Nm, coh]

        major, minor, posang = fit_beam(
            psf=inner_psf_image[:, :, 0], # only the first coherency used.
            dl=dl,
            dm=dm
        )

        dirty_image = dirty_image[:, :, None, :]  # [nl, nm, 1, coh]
        t1 = time_mod.time()
        print(f"Completed imaging in {t1 - t0:.2f} seconds.")
        for coh in range(dirty_image.shape[-1]):
            # plot to png
            plt.imshow(
                np.log10(np.abs(dirty_image[..., 0, coh].T)),
                origin='lower',
                extent=(
                    -num_pixel / 2 * dl + center_l,
                    num_pixel / 2 * dl + center_l,
                    -num_pixel / 2 * dm + center_m,
                    num_pixel / 2 * dm + center_m
                )
            )
            plt.xlabel('l [rad]')
            plt.ylabel('m [rad]')
            plt.colorbar()
            plt.title(f"{image_name} {ms.meta.coherencies[coh]}")
            plt.savefig(f"{self.plot_folder}/{image_name}_{ms.meta.coherencies[coh]}.png")
            plt.show()
        image_model = ImageModel(
            phase_tracking=ms.meta.phase_tracking,
            obs_time=ms.ref_time,
            dl=dl*au.dimensionless_unscaled,
            dm=dm*au.dimensionless_unscaled,
            freqs=np.mean(ms.meta.freqs, keepdims=True),
            bandwidth=ms.meta.channel_width * len(ms.meta.freqs),
            coherencies=ms.meta.coherencies,
            beam_major=np.asarray(major) * au.rad,
            beam_minor=np.asarray(minor) * au.rad,
            beam_pa=np.asarray(posang) * au.rad,
            unit='JY/PIXEL',
            object_name='forward_model',
            image=au.Quantity(np.asarray(dirty_image), 'Jy')
        )
        # with open(f"{image_name}.json", 'w') as fp:
        #     fp.write(image_model.json(indent=2))
        image_model.save_image_to_fits(f"{image_name}.fits", overwrite=overwrite)
        print(f"Saved FITS image to {image_name}.fits")

        return image_model
