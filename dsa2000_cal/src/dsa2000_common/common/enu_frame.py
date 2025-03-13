from __future__ import (absolute_import, unicode_literals, division, print_function)

from astropy.coordinates import AltAz
from astropy.coordinates.attributes import (TimeAttribute, EarthLocationAttribute)
from astropy.coordinates.baseframe import (BaseCoordinateFrame, RepresentationMapping, frame_transform_graph)
from astropy.coordinates.representation import (CartesianRepresentation)
from astropy.coordinates.transformations import FunctionTransform


class ENU(BaseCoordinateFrame):
    """
    Written by Joshua G. Albert - albert@strw.leidenuniv.nl
    A coordinate or frame in the East-North-Up (ENU) system.
    This frame has the following frame attributes, which are necessary for
    transforming from ENU to some other system:
    * ``obstime``
        The time at which the observation is taken.  Used for determining the
        position and orientation of the Earth.
    * ``location``
        The location on the Earth.  This can be specified either as an
        `~astropy.coordinates.EarthLocation` object or as anything that can be
        transformed to an `~astropy.coordinates.ITRS` frame.

    Parameters
    ----------
    representation : `BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    east : :class:`~astropy.units.Quantity`, optional, must be keyword
        The east coordinate for this object (``north`` and ``up`` must also be given and
        ``representation`` must be None).
    north : :class:`~astropy.units.Quantity`, optional, must be keyword
        The north coordinate for this object (``east`` and ``up`` must also be given and
        ``representation`` must be None).
    up : :class:`~astropy.units.Quantity`, optional, must be keyword
        The up coordinate for this object (``north`` and ``east`` must also be given and
        ``representation`` must be None).
    """

    frame_specific_representation_info = {
        'cartesian': [RepresentationMapping('x', 'east'),
                      RepresentationMapping('y', 'north'),
                      RepresentationMapping('z', 'up')],
    }

    default_representation = CartesianRepresentation

    obstime = TimeAttribute(default=None)
    location = EarthLocationAttribute(default=None)


@frame_transform_graph.transform(FunctionTransform, AltAz, ENU)
def altaz_to_enu(altaz_coo: AltAz, enu_frame: ENU):
    '''Defines the transformation between AltAz and the ENU frame.
    AltAz usually has units attached but ENU does not require units
    if it specifies a direction.'''
    intermediate_altaz_frame = AltAz(location=enu_frame.location, obstime=enu_frame.obstime)
    intermediate_altaz_coo = altaz_coo.transform_to(intermediate_altaz_frame)
    rep = CartesianRepresentation(x=intermediate_altaz_coo.cartesian.y,
                                  y=intermediate_altaz_coo.cartesian.x,
                                  z=intermediate_altaz_coo.cartesian.z,
                                  copy=False)

    return enu_frame.realize_frame(rep)


@frame_transform_graph.transform(FunctionTransform, ENU, AltAz)
def enu_to_altaz(enu_coo: ENU, altaz_frame: AltAz):
    intermediate_altaz_frame = AltAz(location=enu_coo.location, obstime=enu_coo.obstime)
    rep = CartesianRepresentation(x=enu_coo.north,
                                  y=enu_coo.east,
                                  z=enu_coo.up,
                                  copy=False)
    intermediate_altaz_coo = intermediate_altaz_frame.realize_frame(rep)
    return intermediate_altaz_coo.transform_to(altaz_frame)


@frame_transform_graph.transform(FunctionTransform, ENU, ENU)
def enu_to_enu(from_coo: ENU, to_frame: ENU):
    # To convert from ENU at one location and time to ENU at another location and time, we go through AltAz
    return from_coo.transform_to(AltAz(location=from_coo.location, obstime=from_coo.obstime)).transform_to(to_frame)
