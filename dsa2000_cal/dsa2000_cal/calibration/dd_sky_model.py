from typing import List

import numpy as np
from astropy import units as au, coordinates as ac
from pydantic import Field

from dsa2000_cal.common.serialise_utils import SerialisableBaseModel


class PointSource(SerialisableBaseModel):
    name: str | None = Field(
        default=None,
        description="The name of the point source, does not need to be unique.",
    )
    icrs: ac.ICRS = Field(
        description="The ICRS coordinates of the point source.",
    )
    stokes: au.Quantity = Field(
        description="The Stokes IQUV parameters of the point source. If only I is given, the rest are assumed to be 0."
                    "Otherwise, the shape should be (4,).",
    )
    freq: au.Quantity | None = Field(
        default=None,
        description="The frequency of the point source. If not given, the source is assumed to be "
                    "constant all frequencies. Note, linear interpolation is used to interpolate the brightness "
                    "to the requested frequencies.",
    )


class SourceList(SerialisableBaseModel):
    """
    A list of point sources that make up a direction in a direction dependent source model.
    """
    name: str | None = Field(
        default=None,
        description="The name of this source, which makes up a single direction.",
    )
    point_sources: List[PointSource] = Field(
        description="A list of point sources that make up this direction. All these points will receive the same "
                    "antenna-based gain. The directional dispersion should be low enough that the gain can be "
                    "approximated as a single value. Precisely, the relationship between visibilities and image should "
                    "be represented by a planar convolution for all these source, so the directional dispersion should "
                    "be less than the isoplanatic patch size.",
    )


class DDSkyModel(SerialisableBaseModel):
    sources: List[SourceList] = Field(
        description="A list of sources, each of which is a list of point sources. "
                    "Each source gets its a single antenna-based gain, with an implicitly defined direction."
    )

    def __init__(self, **data) -> None:
        # Call the superclass __init__ to perform the standard validation
        super(DDSkyModel, self).__init__(**data)
        # Use _check_measurement_set_meta_v0 as instance-wise validator
        _check_dd_sky_model(self)


def _check_dd_sky_model(sky_model: DDSkyModel):
    for source_idx, source in enumerate(sky_model.sources):
        if source.name is None:
            source.name = f"S{source_idx:02d}"

        for point_idx, point in enumerate(source.point_sources):
            if point.name is None:
                point.name = f"{source.name}_P{point_idx:02d}"

            # Check Freq
            if point.freq is None:
                point.freq = au.Quantity(0, 'Hz')
            if not point.freq.unit.is_equivalent(au.Hz):
                raise ValueError(f"Expected freq to be in Hz, got {point.freq.unit}")
            if not point.freq.isscalar and point.freq.size == 1:
                point.freq = point.freq.reshape(())
            if not point.freq.isscalar:
                raise ValueError("Expected freq to be a scalar")

            # Check Stokes
            if not point.stokes.unit.is_equivalent(au.Jy):
                raise ValueError(f"Expected stokes to be in Jy, got {point.stokes.unit}")
            if point.stokes.isscalar or point.stokes.size == 1:
                # Assume it was I given and expand to IQUV
                point.stokes = np.array([point.stokes.value, 0, 0, 0]) * point.stokes.unit
            if point.stokes.size != 4:
                raise ValueError(f"Expected stokes to have 4 coherencies, got {point.stokes.size}")

            # Check ICRS
            if not point.icrs.isscalar:
                raise ValueError("Expected icrs to be a scalar")


def test_dd_sky_model():
    sky_model = DDSkyModel(
        sources=[
            SourceList(
                name="source_1",
                point_sources=[
                    PointSource(
                        name="point_1",
                        icrs=ac.ICRS(0 * au.deg, 0 * au.deg),
                        stokes=au.Quantity([1, 0, 0, 0], 'Jy'),
                        freq=au.Quantity(1e9, 'Hz')
                    ),
                    PointSource(
                        name="point_2",
                        icrs=ac.ICRS(0 * au.deg, 0 * au.deg),
                        stokes=au.Quantity(1, 'Jy')
                    ),
                ]
            )
        ]
    )
    assert sky_model.sources[0].name == "source_1"
    assert sky_model.sources[0].point_sources[0].name == "point_1"
    assert sky_model.sources[0].point_sources[0].icrs == ac.ICRS(0 * au.deg, 0 * au.deg)
    np.testing.assert_allclose(sky_model.sources[0].point_sources[0].stokes, [1, 0, 0, 0] * au.Jy)
    assert sky_model.sources[0].point_sources[0].freq == au.Quantity(1e9, 'Hz')
    assert sky_model.sources[0].point_sources[1].name == "point_2"
    assert sky_model.sources[0].point_sources[1].icrs == ac.ICRS(0 * au.deg, 0 * au.deg)
    np.testing.assert_allclose(sky_model.sources[0].point_sources[1].stokes, [1, 0, 0, 0] * au.Jy)
    assert sky_model.sources[0].point_sources[1].freq == au.Quantity(0, 'Hz')

    print(sky_model.json(indent=2))