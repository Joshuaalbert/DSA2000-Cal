import pickle

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
import ujson

from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.interp_utils import InterpolatedArray
from dsa2000_common.common.serialise_utils import SerialisableBaseModel
from dsa2000_common.visibility_model.source_models.rfi.parametric_rfi_emitter import ParametricDelayACF


class MockModelInt(SerialisableBaseModel):
    value: int


def test_serialise_deserialise_model():
    model = MockModelInt(value=10)
    serialized_data = pickle.dumps(model)
    deserialized_model = pickle.loads(serialized_data)

    assert isinstance(deserialized_model, MockModelInt)
    assert deserialized_model.value == model.value


def test_config_values():
    assert MockModelInt.Config.validate_assignment is True
    assert MockModelInt.Config.arbitrary_types_allowed is True
    assert MockModelInt.Config.json_loads == ujson.loads


class MockModelNp(SerialisableBaseModel):
    array: np.ndarray


def test_numpy_array_json_serialization():
    model = MockModelNp(array=np.array([1, 2, 3]))
    serialized_data = model.json()

    # Deserialize from the serialized data
    # deserialized_model = TestModelNp.model_validate_json(serialized_data)
    deserialized_model = MockModelNp.parse_raw(serialized_data)

    # Assert that the reconstructed numpy array is correct
    np.testing.assert_array_equal(deserialized_model.array, model.array)


def test_icrs_serialisation():
    class ICRSModel(SerialisableBaseModel):
        coord: ac.ICRS

    original = ICRSModel(coord=ac.ICRS(ra=10 * au.degree, dec=45 * au.degree))

    # Serialise the object to JSON
    serialised = original.json()
    print(serialised)

    # Deserialise the object from JSON
    deserialised = ICRSModel.parse_raw(serialised)

    # Validate the deserialisation
    assert np.isclose(deserialised.coord.ra.deg, original.coord.ra.deg)
    assert np.isclose(deserialised.coord.dec.deg, original.coord.dec.deg)

    original = ICRSModel(coord=ac.ICRS(ra=[0, 10] * au.degree, dec=[30, 45] * au.degree))

    # Serialise the object to JSON
    serialised = original.json()
    print(serialised)

    # Deserialise the object from JSON
    deserialised = ICRSModel.parse_raw(serialised)

    # Validate the deserialisation
    np.testing.assert_allclose(deserialised.coord.ra.deg, original.coord.ra.deg, atol=1e-10)
    np.testing.assert_allclose(deserialised.coord.dec.deg, original.coord.dec.deg, atol=1e-10)


def test_itrs_serialisation():
    class ITRSModel(SerialisableBaseModel):
        coord: ac.ITRS

    original = ITRSModel(coord=ac.ITRS(x=1 * au.m, y=2 * au.m, z=3 * au.m))

    # Serialise the object to JSON
    serialised = original.json()

    # Deserialise the object from JSON
    deserialised = ITRSModel.parse_raw(serialised)

    # Validate the deserialisation
    assert np.isclose(deserialised.coord.x, original.coord.x)
    assert np.isclose(deserialised.coord.y, original.coord.y)
    assert np.isclose(deserialised.coord.z, original.coord.z)

    original = ITRSModel(coord=ac.ITRS(x=[1, 1] * au.m, y=[2, 2] * au.m, z=[3, 3] * au.m))

    # Serialise the object to JSON
    serialised = original.json()

    # Deserialise the object from JSON
    deserialised = ITRSModel.parse_raw(serialised)

    # Validate the deserialisation
    np.testing.assert_array_equal(deserialised.coord.x, original.coord.x)
    np.testing.assert_array_equal(deserialised.coord.y, original.coord.y)
    np.testing.assert_array_equal(deserialised.coord.z, original.coord.z)


def test_at_time_serialisation():
    class TimeModel(SerialisableBaseModel):
        time: at.Time

    original = TimeModel(time=at.Time('2021-01-01T00:00:00', scale='utc'))

    # Serialise the object to JSON
    serialised = original.json()
    print(serialised)

    # Deserialise the object from JSON
    deserialised = TimeModel.parse_raw(serialised)

    # Validate the deserialisation
    assert deserialised.time == original.time

    original = TimeModel(time=at.Time(['2021-01-01T00:00:00', '2021-01-01T00:00:00'], scale='utc'))

    # Serialise the object to JSON
    serialised = original.json()
    print(serialised)

    # Deserialise the object from JSON
    deserialised = TimeModel.parse_raw(serialised)

    # Validate the deserialisation
    np.testing.assert_array_equal(deserialised.time.jd, original.time.jd)


def test_au_quantity_serialisation():
    class QuantityModel(SerialisableBaseModel):
        quantity: au.Quantity

    original = QuantityModel(quantity=au.Quantity(10, unit=au.m))

    # Serialise the object to JSON
    serialised = original.json()
    print(serialised)

    # Deserialise the object from JSON
    deserialised = QuantityModel.parse_raw(serialised)

    # Validate the deserialisation
    assert deserialised.quantity == original.quantity

    original = QuantityModel(quantity=au.Quantity([10, 20], unit=au.m))

    # Serialise the object to JSON
    serialised = original.json()

    # Deserialise the object from JSON
    deserialised = QuantityModel.parse_raw(serialised)

    # Validate the deserialisation
    np.testing.assert_array_equal(deserialised.quantity, original.quantity)


def test_earth_location_serialisation():
    class EarthLocationModel(SerialisableBaseModel):
        location: ac.EarthLocation

    original = EarthLocationModel(location=ac.EarthLocation(lat=10 * au.deg, lon=20 * au.deg, height=30 * au.m))

    # Serialise the object to JSON
    serialised = original.json()
    print(serialised)

    # Deserialise the object from JSON
    deserialised = EarthLocationModel.parse_raw(serialised)

    # Validate the deserialisation
    assert np.isclose(deserialised.location.lat.deg, original.location.lat.deg)
    assert np.isclose(deserialised.location.lon.deg, original.location.lon.deg)
    assert np.isclose(deserialised.location.height.value, original.location.height.value)

    original = EarthLocationModel(
        location=ac.EarthLocation(lat=[10, 20] * au.deg, lon=[20, 30] * au.deg, height=[30, 40] * au.m))

    # Serialise the object to JSON
    serialised = original.json()

    # Deserialise the object from JSON
    deserialised = EarthLocationModel.parse_raw(serialised)

    # Validate the deserialisation
    np.testing.assert_array_equal(deserialised.location.lat.deg, original.location.lat.deg)
    np.testing.assert_array_equal(deserialised.location.lon.deg, original.location.lon.deg)
    np.testing.assert_array_equal(deserialised.location.height.value, original.location.height.value)


def test_altaz_serialization():
    class AltAzModel(SerialisableBaseModel):
        altaz: ac.AltAz

    original = AltAzModel(altaz=ac.AltAz(az=10 * au.deg, alt=20 * au.deg,
                                         location=ac.EarthLocation.of_site('vla'),
                                         obstime=at.Time.now()))

    # Serialise the object to JSON
    serialised = original.json()
    print(serialised)

    # Deserialise the object from JSON
    deserialised = AltAzModel.parse_raw(serialised)

    # Validate the deserialisation
    assert np.isclose(deserialised.altaz.az.deg, original.altaz.az.deg)
    assert np.isclose(deserialised.altaz.alt.deg, original.altaz.alt.deg)
    assert np.isclose(deserialised.altaz.location.lat.deg, original.altaz.location.lat.deg)
    assert np.isclose(deserialised.altaz.location.lon.deg, original.altaz.location.lon.deg)
    assert np.isclose(deserialised.altaz.location.height.value, original.altaz.location.height.value)

    original = AltAzModel(altaz=ac.AltAz(az=[10, 20] * au.deg, alt=[20, 30] * au.deg,
                                         location=ac.EarthLocation.of_site('vla'),
                                         obstime=at.Time.now()))

    # Serialise the object to JSON
    serialised = original.json()

    # Deserialise the object from JSON
    deserialised = AltAzModel.parse_raw(serialised)

    # Validate the deserialisation
    np.testing.assert_array_equal(deserialised.altaz.az.deg, original.altaz.az.deg)
    np.testing.assert_array_equal(deserialised.altaz.alt.deg, original.altaz.alt.deg)
    np.testing.assert_array_equal(deserialised.altaz.location.lat.deg, original.altaz.location.lat.deg)
    np.testing.assert_array_equal(deserialised.altaz.location.lon.deg, original.altaz.location.lon.deg)
    np.testing.assert_array_equal(deserialised.altaz.location.height.value, original.altaz.location.height.value)


def test_enu_serialization():
    class ENUModel(SerialisableBaseModel):
        enu: ENU

    original = ENUModel(
        enu=ENU(
            east=0. * au.m, north=0. * au.m, up=0. * au.m,
            location=ac.EarthLocation.of_site('vla'),
            obstime=at.Time.now()
        )
    )

    # Serialise the object to JSON
    serialised = original.json()
    print(serialised)

    # Deserialise the object from JSON
    deserialised = ENUModel.parse_raw(serialised)

    # Validate the deserialisation
    assert np.isclose(deserialised.enu.east.value, original.enu.east.value)
    assert np.isclose(deserialised.enu.north.value, original.enu.north.value)
    assert np.isclose(deserialised.enu.up.value, original.enu.up.value)

    original = ENUModel(
        enu=ENU(east=0., north=0., up=0.,
                location=ac.EarthLocation.of_site('vla'),
                obstime=at.Time.now())
    )

    # Serialise the object to JSON
    serialised = original.json()
    print(serialised)

    # Deserialise the object from JSON
    deserialised = ENUModel.parse_raw(serialised)

    # Validate the deserialisation
    assert np.isclose(deserialised.enu.east.value, original.enu.east.value)
    assert np.isclose(deserialised.enu.north.value, original.enu.north.value)
    assert np.isclose(deserialised.enu.up.value, original.enu.up.value)


def test_complex_ndarray_serialisation():
    class ComplexNumpyModel(SerialisableBaseModel):
        array: np.ndarray

    original = ComplexNumpyModel(array=np.array([1 + 1j, 2 + 2j, 3 + 3j]))

    # Serialise the object to JSON
    serialised = original.json()
    print(serialised)

    # Deserialise the object from JSON
    deserialised = ComplexNumpyModel.parse_raw(serialised)

    # Validate the deserialisation
    np.testing.assert_array_equal(deserialised.array, original.array)


def test_complex_quantity_serialisation():
    class ComplexQuantityModel(SerialisableBaseModel):
        quantity: au.Quantity

    original = ComplexQuantityModel(quantity=au.Quantity([1 + 1j, 2 + 2j, 3 + 3j], unit=au.m))

    # Serialise the object to JSON
    serialised = original.json()
    print(serialised)

    # Deserialise the object from JSON
    deserialised = ComplexQuantityModel.parse_raw(serialised)

    # Validate the deserialisation
    np.testing.assert_array_equal(deserialised.quantity, original.quantity)


def test_dimensionless_quantity_serialisation():
    class DimensionlessQuantityModel(SerialisableBaseModel):
        quantity: au.Quantity

    original = DimensionlessQuantityModel(quantity=[1, 2, 3] * au.dimensionless_unscaled)

    # Serialise the object to JSON
    serialised = original.json()
    print(serialised)

    # Deserialise the object from JSON
    deserialised = DimensionlessQuantityModel.parse_raw(serialised)

    # Validate the deserialisation
    np.testing.assert_array_equal(deserialised.quantity, original.quantity)

    original = DimensionlessQuantityModel(quantity=1. * au.dimensionless_unscaled)

    # Serialise the object to JSON
    serialised = original.json()
    print(serialised)

    # Deserialise the object from JSON
    deserialised = DimensionlessQuantityModel.parse_raw(serialised)

    # Validate the deserialisation
    np.testing.assert_array_equal(deserialised.quantity, original.quantity)


def test_nested_models_with_quantities():
    class NestedModel(SerialisableBaseModel):
        quantity: au.Quantity

    class OuterModel(SerialisableBaseModel):
        nested: NestedModel

    original = OuterModel(nested=NestedModel(quantity=[1, 2, 3] * au.m))

    # Serialise the object to JSON
    serialised = original.json()
    print(serialised)

    # Deserialise the object from JSON
    deserialised = OuterModel.parse_raw(serialised)

    # Validate the deserialisation
    np.testing.assert_array_equal(deserialised.nested.quantity, original.nested.quantity)


def test_interpolated_array():
    class Model(SerialisableBaseModel):
        x: InterpolatedArray

    original = Model(
        x=InterpolatedArray(
            x=np.arange(5),
            values=np.ones((5, 3)),
            axis=0,
            regular_grid=True
        )
    )

    # Serialise the object to JSON
    serialised = original.json(indent=2)
    print(serialised)

    # Deserialise the object from JSON
    deserialised = Model.parse_raw(serialised)
    np.testing.assert_allclose(deserialised.x.x, original.x.x)
    np.testing.assert_allclose(deserialised.x.values, original.x.values)
    np.testing.assert_allclose(deserialised.x.axis, original.x.axis)
    np.testing.assert_allclose(deserialised.x.regular_grid, original.x.regular_grid)


def test_parametric_delay_acf():
    acf = ParametricDelayACF(
        mu=np.asarray([1.0]),
        fwhp=np.asarray([1.0]),
        spectral_power=np.asarray([1.0]),
        channel_width=np.asarray([1.0]),
        resolution=1,
    )

    class Model(SerialisableBaseModel):
        acf: ParametricDelayACF

    original = Model(acf=acf)

    # Serialise the object to JSON
    serialised = original.json(indent=2)
    print(serialised)

    # Deserialise the object from JSON
    deserialised = Model.parse_raw(serialised)

    np.testing.assert_allclose(deserialised.acf.mu, original.acf.mu)
    np.testing.assert_allclose(deserialised.acf.fwhp, original.acf.fwhp)
    np.testing.assert_allclose(deserialised.acf.spectral_power, original.acf.spectral_power)
    np.testing.assert_allclose(deserialised.acf.channel_width, original.acf.channel_width)
    np.testing.assert_allclose(deserialised.acf.resolution, original.acf.resolution)

