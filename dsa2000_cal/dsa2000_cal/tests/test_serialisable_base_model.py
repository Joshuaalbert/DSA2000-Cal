import pickle

import astropy.coordinates as ac
import astropy.units as au
import numpy as np
import ujson

from dsa2000_cal.utils import SerialisableBaseModel


class TestModelInt(SerialisableBaseModel):
    value: int


def test_serialise_deserialise_model():
    model = TestModelInt(value=10)
    serialized_data = pickle.dumps(model)
    deserialized_model = pickle.loads(serialized_data)

    assert isinstance(deserialized_model, TestModelInt)
    assert deserialized_model.value == model.value


def test_config_values():
    assert TestModelInt.Config.validate_assignment is True
    assert TestModelInt.Config.arbitrary_types_allowed is True
    assert TestModelInt.Config.json_loads == ujson.loads
    # You can test for json_dumps once you decide on its implementation
    assert isinstance(TestModelInt.Config.json_encoders[np.ndarray], type(lambda x: x))


class TestModelNp(SerialisableBaseModel):
    array: np.ndarray


def test_numpy_array_json_serialization():
    model = TestModelNp(array=np.array([1, 2, 3]))
    serialized_data = model.json()

    expected_json = '{"array":[1,2,3]}'
    assert serialized_data == expected_json

    # Deserialize from the serialized data
    # deserialized_model = TestModelNp.model_validate_json(serialized_data)
    deserialized_model = TestModelNp.parse_raw(serialized_data)

    # Assert that the reconstructed numpy array is correct
    np.testing.assert_array_equal(deserialized_model.array, model.array)


def test_icrs_serialisation():
    class ICRSModel(SerialisableBaseModel):
        coord: ac.ICRS

    original = ICRSModel(coord=ac.ICRS(ra=10 * au.degree, dec=45 * au.degree))

    # Serialise the object to JSON
    serialised = original.json()

    # Deserialise the object from JSON
    deserialised = ICRSModel.parse_raw(serialised)

    # Validate the deserialisation
    assert np.isclose(deserialised.coord.ra.deg, original.coord.ra.deg)
    assert np.isclose(deserialised.coord.dec.deg, original.coord.dec.deg)


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
