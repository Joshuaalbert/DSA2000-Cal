import jax
import numpy as np
from astropy import units as au, time as at, coordinates as ac
from jax import numpy as jnp

from dsa2000_cal.common.cache_utils import check_cache
from dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel


def test_check_cache_file():

    class InnerCache(SerialisableBaseModel):
        a: au.Quantity
        b: at.Time
    class MockCache(SerialisableBaseModel):
        a: au.Quantity
        b: at.Time
        c: ac.EarthLocation
        d: ac.ICRS
        e: InterpolatedArray
        f: np.ndarray
        g: jax.Array
        h: InnerCache

    c = MockCache(
        a=au.Quantity(700, 'MHz'),
        b=at.Time('2021-01-01T00:00:00', scale='utc'),
        c=ac.EarthLocation(0, 0, 0),
        d=ac.ICRS(ra=0 * au.deg, dec=0 * au.deg),
        e=InterpolatedArray(x=jnp.asarray([0, 1, 2]), values=jnp.asarray([0, 1, 2]), regular_grid=True),
        f=np.array([0, 1, 2]),
        g=jnp.asarray([0, 1, 2]),
        h=InnerCache(a=au.Quantity(700e6, 'Hz'), b=at.Time('2021-01-01T00:00:00', scale='utc'))
    )

    check_cache(
        c,
        a=au.Quantity(700e6, 'Hz'),
        b=at.Time('2021-01-01T00:00:00.00001', scale='utc'),
        c=ac.EarthLocation(0, 0, 0),
        d=ac.ICRS(ra=0 * au.deg, dec=1 * au.arcsec),
        e=InterpolatedArray(x=jnp.asarray([0, 1, 2]), values=jnp.asarray([0, 1, 2]), regular_grid=True),
        f=np.array([0, 1, 2]),
        g=jnp.asarray([0, 1, 2]),
        h=InnerCache(a=au.Quantity(700e6, 'Hz'), b=at.Time('2021-01-01T00:00:00', scale='utc'))
    )
