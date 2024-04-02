from __future__ import (absolute_import, unicode_literals, division, print_function)

import astropy.units as u
from astropy.coordinates import AltAz
from astropy.coordinates.attributes import (TimeAttribute, EarthLocationAttribute, CoordinateAttribute)
from astropy.coordinates.baseframe import (BaseCoordinateFrame, RepresentationMapping, frame_transform_graph)
from astropy.coordinates.representation import (UnitSphericalRepresentation,
                                                CartesianRepresentation)
from astropy.coordinates.transformations import FunctionTransform


class UVW(BaseCoordinateFrame):

    frame_specific_representation_info = {
        'cartesian': [RepresentationMapping('x', 'u'),
                      RepresentationMapping('y', 'v'),
                      RepresentationMapping('z', 'w')],
    }

    default_representation = CartesianRepresentation

    obstime = TimeAttribute(default=None)
    location = EarthLocationAttribute(default=None)
    phase_tracking = CoordinateAttribute(default=None)

    def __init__(self, *args, **kwargs):
        super(UVW, self).__init__(*args, **kwargs)


@frame_transform_graph.transform(FunctionTransform, AltAz, ENU)
def altaz_to_enu(altaz_coo, enu_frame):
    '''Defines the transformation between AltAz and the ENU frame.
    AltAz usually has units attached but ENU does not require units
    if it specifies a direction.'''
    is_directional = (isinstance(altaz_coo.data, UnitSphericalRepresentation) or
                      altaz_coo.cartesian.x.unit == u.one)

    if is_directional:
        rep = CartesianRepresentation(x=altaz_coo.cartesian.y,
                                      y=altaz_coo.cartesian.x,
                                      z=altaz_coo.cartesian.z,
                                      copy=False)
    else:
        # location_altaz = ITRS(*enu_frame.location.to_geocentric()).transform_to(
        #     AltAz(location=enu_frame.location, obstime=enu_frame.obstime))
        rep = CartesianRepresentation(x=altaz_coo.cartesian.y,  # - location_altaz.cartesian.y,
                                      y=altaz_coo.cartesian.x,  # - location_altaz.cartesian.x,
                                      z=altaz_coo.cartesian.z,  # - location_altaz.cartesian.z,
                                      copy=False)
    return enu_frame.realize_frame(rep)


@frame_transform_graph.transform(FunctionTransform, ENU, AltAz)
def enu_to_altaz(enu_coo, altaz_frame):
    is_directional = (isinstance(enu_coo.data, UnitSphericalRepresentation) or
                      enu_coo.cartesian.x.unit == u.one)

    if is_directional:
        rep = CartesianRepresentation(x=enu_coo.north,
                                      y=enu_coo.east,
                                      z=enu_coo.up,
                                      copy=False)
    else:
        # location_altaz = ITRS(*enu_coo.location.to_geocentric()).transform_to(
        #     AltAz(location=enu_coo.location, obstime=enu_coo.obstime))
        rep = CartesianRepresentation(x=enu_coo.north,  # + location_altaz.cartesian.x,
                                      y=enu_coo.east,  # + location_altaz.cartesian.y,
                                      z=enu_coo.up,  # + location_altaz.cartesian.z,
                                      copy=False)
    return altaz_frame.realize_frame(rep)


@frame_transform_graph.transform(FunctionTransform, ENU, ENU)
def enu_to_enu(from_coo, to_frame):
    # for now we just implement this through AltAz to make sure we get everything
    # covered
    return from_coo.transform_to(AltAz(location=from_coo.location, obstime=from_coo.obstime)).transform_to(to_frame)
