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
        if np.iscomplexobj(obj):
            # Store real and imaginary parts separately in array_real, array_imag
            return {
                'type': 'numpy.ndarray',
                'array_real': obj.real.tolist(),
                'array_imag': obj.imag.tolist(),
                'dtype': str(obj.dtype)
            }
        return {
            'type': 'numpy.ndarray',
            'array': obj.tolist(),
            'dtype': str(obj.dtype)
        }
    return obj


def deserialise_ndarray(obj):
    if 'array_real' in obj and 'array_imag' in obj:
        return np.array(obj["array_real"], dtype=obj["dtype"]) + 1j * np.array(obj["array_imag"], dtype=obj["dtype"])
    return np.array(obj["array"], dtype=obj["dtype"])


def deserialise_icrs(obj):
    if isinstance(obj["ra"], dict) and obj["ra"].get("type") == 'numpy.ndarray':
        obj["ra"] = deserialise_ndarray(obj["ra"])
        obj["dec"] = deserialise_ndarray(obj["dec"])
    return ac.ICRS(ra=ac.Angle(obj['ra']), dec=ac.Angle(obj["dec"]))


def deserialise_itrs(obj):
    if isinstance(obj["x"], dict) and obj["x"].get("type") == 'numpy.ndarray':
        obj["x"] = deserialise_ndarray(obj["x"])
        obj["y"] = deserialise_ndarray(obj["y"])
        obj["z"] = deserialise_ndarray(obj["z"])
    return ac.ITRS(x=obj["x"] * au.m, y=obj["y"] * au.m, z=obj["z"] * au.m)


def deserialise_earth_location(obj):
    if isinstance(obj["x"], dict) and obj["x"].get("type") == 'numpy.ndarray':
        obj["x"] = deserialise_ndarray(obj["x"])
        obj["y"] = deserialise_ndarray(obj["y"])
        obj["z"] = deserialise_ndarray(obj["z"])
    return ac.ITRS(x=obj["x"] * au.m, y=obj["y"] * au.m, z=obj["z"] * au.m).earth_location


def deserialise_time(obj):
    if isinstance(obj["value"], dict) and obj["value"].get("type") == 'numpy.ndarray':
        obj["value"] = deserialise_ndarray(obj["value"])
    return at.Time(obj["value"], scale=obj["scale"], format=obj["format"])


def deserialise_altaz(obj):
    if isinstance(obj["az"], dict) and obj["az"].get("type") == 'numpy.ndarray':
        obj["az"] = deserialise_ndarray(obj["az"])
        obj["alt"] = deserialise_ndarray(obj["alt"])
    return ac.AltAz(az=obj["az"] * au.deg, alt=obj["alt"] * au.deg,
                    location=deserialise_earth_location(obj["location"]), obstime=deserialise_time(obj["obstime"]))


def deserialise_quantity(obj):
    if isinstance(obj["value"], dict) and obj["value"].get("type") == 'numpy.ndarray':
        obj["value"] = deserialise_ndarray(obj["value"])
    return au.Quantity(obj["value"], unit=obj["unit"])


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
            ac.AltAz: lambda x: {
                "type": 'astropy.coordinates.AltAz',
                "az": x.az.to(au.deg).value,
                "alt": x.alt.to(au.deg).value,
                "location": x.location,
                "obstime": x.obstime
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

        for name, field in model_fields.items():
            # if isinstance(field.type_, type) and issubclass(field.type_, np.ndarray):
            #     if name in obj and isinstance(obj[name], dict):
            if field.type_ is np.ndarray and isinstance(obj.get(name), dict) and obj[name].get(
                    "type") == 'numpy.ndarray':
                obj[name] = deserialise_ndarray(obj[name])
                continue

            # Deserialise ICRS
            elif field.type_ is ac.ICRS and isinstance(obj.get(name), dict) and obj[name].get(
                    "type") == 'astropy.coordinates.ICRS':
                obj[name] = deserialise_icrs(obj[name])
                continue

            # Deserialise ITRS
            elif field.type_ is ac.ITRS and isinstance(obj.get(name), dict) and obj[name].get(
                    "type") == 'astropy.coordinates.ITRS':
                obj[name] = deserialise_itrs(obj[name])
                continue

            # Deserialise EarthLocation
            elif field.type_ is ac.EarthLocation and isinstance(obj.get(name), dict) and obj[name].get(
                    "type") == 'astropy.coordinates.EarthLocation':
                obj[name] = deserialise_earth_location(obj[name])
                continue

            # Deserialise Time
            elif field.type_ is at.Time and isinstance(obj.get(name), dict) and obj[name].get(
                    "type") == 'astropy.time.Time':
                obj[name] = deserialise_time(obj[name])
                continue

            # Deserialise AltAz
            elif field.type_ is ac.AltAz and isinstance(obj.get(name), dict) and obj[name].get(
                    "type") == 'astropy.coordinates.AltAz':
                obj[name] = deserialise_altaz(obj[name])
                continue

            # Deserialise Quantity
            elif field.type_ is au.Quantity and isinstance(obj.get(name), dict) and obj[name].get(
                    "type") == 'astropy.units.Quantity':
                obj[name] = deserialise_quantity(obj[name])
                continue

            # Deserialise nested models
            elif inspect.isclass(field.type_) and issubclass(field.type_, BaseModel):
                obj[name] = field.type_.parse_obj(obj[name])
                continue

            else:
                # print('No deserialisation for', name, field.type_, obj.get(name))
                pass
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
