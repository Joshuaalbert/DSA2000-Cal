import importlib
import inspect
from typing import TypeVar, Type, Dict, Any, List

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
import ujson
from pydantic import BaseModel

C = TypeVar('C')


def serialise_array_quantity(obj):
    if isinstance(obj, au.Quantity) and isinstance(obj.value, np.ndarray):
        return {
            'type': 'astropy.units.Quantity',
            'value': obj.value,
            'unit': str(obj.unit)
        }
    elif isinstance(obj, np.ndarray):
        return {
            'type': 'numpy.ndarray',
            'array': obj.tolist(),
            'dtype': str(obj.dtype)
        }
    return obj


class SerialisableBaseModel(BaseModel):
    """
    A pydantic BaseModel that can be serialised and deserialised using pickle, working well with Ray.
    """

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        json_loads = ujson.loads  # can use because ujson decodes NaN and Infinity
        json_dumps = ujson.dumps  #
        json_encoders = {
            np.ndarray: serialise_array_quantity,
            ac.ICRS: lambda x: {
                "type": 'astropy.coordinates.ICRS',
                "ra": x.ra.to_string(unit=au.hour, sep='hms', pad=True),
                "dec": x.dec.to_string(unit=au.deg, sep='dms', pad=True, alwayssign=True)
            },
            ac.ITRS: lambda x: {
                "type": 'astropy.coordinates.ITRS',
                "x": x.x.to(au.m).value,
                "y": x.y.to(au.m).value,
                "z": x.z.to(au.m).value,
            },
            at.Time: lambda x: {
                "type": 'astropy.time.Time',
                "value": x.isot,
                "scale": x.scale,
                "format": "isot"
            },
            ac.EarthLocation: lambda x: {
                "type": 'astropy.coordinates.EarthLocation',
                "x": x.get_itrs().x.to(au.m).value,
                "y": x.get_itrs().y.to(au.m).value,
                "z": x.get_itrs().z.to(au.m).value,
                'ellipsoid': x.ellipsoid
            },
            au.Quantity: lambda x: {
                "type": 'astropy.units.Quantity',
                "value": x.value,
                "unit": str(x.unit)
            }
        }

    @classmethod
    def _deserialise(cls, kwargs):
        """Required for this class's __reduce__ method to be picklable."""
        return cls(**kwargs)

    @classmethod
    def parse_obj(cls: Type[C], obj: Dict[str, Any]) -> C:
        model_fields = cls.__fields__  # get fields of the model

        # Convert all fields that are defined as np.ndarray
        for name, field in model_fields.items():
            # if isinstance(field.type_, type) and issubclass(field.type_, np.ndarray):
            #     if name in obj and isinstance(obj[name], dict):
            if field.type_ is np.ndarray and isinstance(obj.get(name), dict) and obj[name].get(
                    "type") == 'numpy.ndarray':
                array = obj[name]
                obj[name] = np.array(array["array"], dtype=array["dtype"])
                continue

            # Deserialise ICRS and ITRS
            if field.type_ is ac.ICRS and isinstance(obj.get(name), dict) and obj[name].get(
                    "type") == 'astropy.coordinates.ICRS':
                ra_dec = obj[name]
                if isinstance(ra_dec["ra"], dict) and ra_dec["ra"].get("type") == 'numpy.ndarray':
                    ra_dec["ra"] = np.array(ra_dec["ra"]["array"], dtype=ra_dec["ra"]["dtype"])
                    ra_dec["dec"] = np.array(ra_dec["dec"]["array"], dtype=ra_dec["dec"]["dtype"])
                obj[name] = ac.ICRS(ra=ac.Angle(ra_dec["ra"]), dec=ac.Angle(ra_dec["dec"]))
                continue

            if field.type_ is ac.ITRS and isinstance(obj.get(name), dict) and obj[name].get(
                    "type") == 'astropy.coordinates.ITRS':
                coords = obj[name]
                if isinstance(coords["x"], dict) and coords["x"].get("type") == 'numpy.ndarray':
                    # Convert to numpy array
                    coords["x"] = np.array(coords["x"]["array"], dtype=coords["x"]["dtype"])
                    coords["y"] = np.array(coords["y"]["array"], dtype=coords["y"]["dtype"])
                    coords["z"] = np.array(coords["z"]["array"], dtype=coords["z"]["dtype"])
                obj[name] = ac.ITRS(x=coords["x"] * au.m, y=coords["y"] * au.m, z=coords["z"] * au.m)
                continue

            if field.type_ is at.Time and isinstance(obj.get(name), dict) and obj[name].get(
                    "type") == 'astropy.time.Time':
                time = obj[name]
                if isinstance(time["value"], dict) and time["value"].get("type") == 'numpy.ndarray':
                    time["value"] = np.array(time["value"]["array"], dtype=time["value"]["dtype"])
                obj[name] = at.Time(time["value"], scale=time["scale"], format=time["format"])
                continue

            if field.type_ is ac.EarthLocation and isinstance(obj.get(name), dict) and obj[name].get(
                    "type") == 'astropy.coordinates.EarthLocation':
                coords = obj[name]
                if isinstance(coords["x"], dict) and coords["x"].get("type") == 'numpy.ndarray':
                    # Convert to numpy array
                    coords["x"] = np.array(coords["x"]["array"], dtype=coords["x"]["dtype"])
                    coords["y"] = np.array(coords["y"]["array"], dtype=coords["y"]["dtype"])
                    coords["z"] = np.array(coords["z"]["array"], dtype=coords["z"]["dtype"])
                obj[name] = ac.ITRS(x=coords["x"] * au.m, y=coords["y"] * au.m, z=coords["z"] * au.m).earth_location
                continue

            if field.type_ is au.Quantity and isinstance(obj.get(name), dict) and obj[name].get(
                    "type") == 'astropy.units.Quantity':
                quantity = obj[name]
                if isinstance(quantity["value"], dict) and quantity["value"].get("type") == 'numpy.ndarray':
                    quantity["value"] = np.array(quantity["value"]["array"], dtype=quantity["value"]["dtype"])
                obj[name] = au.Quantity(quantity["value"], unit=quantity["unit"])
                continue

        return super().parse_obj(obj)

    def __reduce__(self):
        # Uses the dict representation of the model to serialise and deserialise.
        # The efficiency of this depends on the efficiency of the dict representation serialisation.
        serialised_data = self.dict()
        return self.__class__._deserialise, (serialised_data,)


def example_from_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Generate example from schema and return as dict.

    Args:
        model: BaseModel

    Returns: dict of example
    """
    example = dict()
    properties = model.schema().get('properties', dict())
    for field in model.__fields__:
        # print(model, model.__fields__[field])
        if inspect.isclass(model.__fields__[field]):
            if issubclass(model.__fields__[field], BaseModel):
                example[field] = example_from_schema(model.__fields__[field])
                continue
            example[field] = None
        example[field] = properties[field].get('example', None)
        # print(field, example[field])
    return example


_T = TypeVar('_T')


def build_example(model: Type[_T]) -> _T:
    return model(**example_from_schema(model))


def build_serialiser(type_: type, keys: List[str]):
    def serialise(obj):
        if isinstance(obj, type_):
            class_name = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
            return {
                'type': 'obj',
                '__class__': class_name,
                '__data__': {k: getattr(obj, k) for k in keys}
            }
        return obj

    return serialise


def build_deserialiser(type_: type):
    def deserialise(obj):
        if isinstance(obj, dict) and obj.get('type') == 'obj' and obj.get(
                '__class__') == f"{type_.__module__}.{type_.__name__}":
            class_path = obj['__class__']
            module_name, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            return class_(**obj['__data__'])
        return obj

    return deserialise
