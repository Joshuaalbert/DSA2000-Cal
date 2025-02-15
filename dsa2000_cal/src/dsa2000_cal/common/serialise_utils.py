import importlib
import inspect
from typing import TypeVar, Type, Dict, Any, List

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax.numpy as jnp
import numpy as np
import ujson
from pydantic import BaseModel
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_common.visibility_model.source_models.rfi.parametric_rfi_emitter import ParametricDelayACF

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


def deserialise_quantity(obj):
    if isinstance(obj["value"], dict) and obj["value"].get("type") == 'numpy.ndarray':
        obj["value"] = deserialise_ndarray(obj["value"])
    return au.Quantity(obj["value"], unit=obj["unit"])


def deserialise_ndarray(obj):
    if 'array_real' in obj and 'array_imag' in obj:
        return np.array(obj["array_real"], dtype=obj["dtype"]) + 1j * np.array(obj["array_imag"], dtype=obj["dtype"])
    return np.array(obj["array"], dtype=obj["dtype"])


def deserialise_icrs(obj):
    return ac.ICRS(ra=deserialise_quantity(obj['ra']), dec=deserialise_quantity(obj["dec"]))


def deserialise_itrs(obj):
    return ac.ITRS(x=deserialise_quantity(obj["x"]), y=deserialise_quantity(obj["y"]), z=deserialise_quantity(obj["z"]))


def deserialise_earth_location(obj):
    return ac.ITRS(x=deserialise_quantity(obj["x"]), y=deserialise_quantity(obj["y"]),
                   z=deserialise_quantity(obj["z"])).earth_location


def deserialise_time(obj):
    if isinstance(obj["value"], dict) and obj["value"].get("type") == 'numpy.ndarray':
        obj["value"] = deserialise_ndarray(obj["value"])
    return at.Time(obj["value"], scale=obj["scale"], format=obj["format"])


def deserialise_altaz(obj):
    return ac.AltAz(az=deserialise_quantity(obj["az"]), alt=deserialise_quantity(obj["alt"]),
                    location=deserialise_earth_location(obj["location"]), obstime=deserialise_time(obj["obstime"]))


def deserialise_enu(obj):
    return ENU(east=deserialise_quantity(obj["east"]),
               north=deserialise_quantity(obj["north"]),
               up=deserialise_quantity(obj["up"]),
               location=deserialise_earth_location(obj["location"]), obstime=deserialise_time(obj["obstime"]))


def deserialise_interpolated_array(obj):
    return InterpolatedArray(
        x=jnp.asarray(deserialise_ndarray(obj["x"])),
        values=jnp.asarray(deserialise_ndarray(obj["values"])),
        axis=obj["axis"],
        regular_grid=obj["regular_grid"],
    )


def deserialise_parametric_delay_acf(obj):
    return ParametricDelayACF(
        mu=np.asarray(deserialise_ndarray(obj["mu"])),
        fwhp=np.asarray(deserialise_ndarray(obj["fwhp"])),
        spectral_power=np.asarray(deserialise_ndarray(obj["spectral_power"])),
        channel_lower=np.asarray(deserialise_ndarray(obj["channel_lower"])),
        channel_upper=np.asarray(deserialise_ndarray(obj["channel_upper"])),
        resolution=obj["resolution"],
        convention=obj["convention"]
    )


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
                "ra": x.ra,
                "dec": x.dec
            },
            ENU: lambda x: {
                "type": 'tomographic_kernel.frames.ENU',
                "east": x.east,
                "north": x.north,
                "up": x.up,
                "location": x.location,
                "obstime": x.obstime
            },
            ac.ITRS: lambda x: {
                "type": 'astropy.coordinates.ITRS',
                "x": x.x,
                "y": x.y,
                "z": x.z,
            },
            at.Time: lambda x: {
                "type": 'astropy.time.Time',
                "value": x.isot,
                "scale": x.scale,
                "format": "isot"
            },
            ac.AltAz: lambda x: {
                "type": 'astropy.coordinates.AltAz',
                "az": x.az,
                "alt": x.alt,
                "location": x.location,
                "obstime": x.obstime
            },
            ac.EarthLocation: lambda x: {
                "type": 'astropy.coordinates.EarthLocation',
                "x": x.get_itrs().x,
                "y": x.get_itrs().y,
                "z": x.get_itrs().z,
                'ellipsoid': x.ellipsoid
            },
            au.Quantity: lambda x: {
                "type": 'astropy.units.Quantity',
                "value": x.value,
                "unit": str(x.unit)
            },
            InterpolatedArray: lambda x: {
                "type": 'dsa2000_cal.common.interp_utils.InterpolatedArray',
                "x": np.asarray(x.x),
                "values": np.asarray(x.values),
                "axis": x.axis,
                "regular_grid": x.regular_grid
            },
            ParametricDelayACF: lambda x: {
                "type": 'dsa2000_cal.visibility_model.source_models.rfi_parametric_rfi_emitter.ParametricDelayACF',
                "mu": np.asarray(x.mu),
                "fwhp": np.asarray(x.fwhp),
                "spectral_power": np.asarray(x.spectral_power),
                "channel_lower": np.asarray(x.channel_lower),
                "channel_upper": np.asarray(x.channel_upper),
                "resolution": x.resolution,
                "convention": x.convention
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

            # Deserialise ENU
            elif field.type_ is ENU and isinstance(obj.get(name), dict) and obj[name].get(
                    "type") == 'tomographic_kernel.frames.ENU':
                obj[name] = deserialise_enu(obj[name])
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

            # Deserialise InterpolatedArray
            elif field.type_ is InterpolatedArray and isinstance(obj.get(name), dict) and obj[name].get(
                    "type") == 'dsa2000_cal.common.interp_utils.InterpolatedArray':
                obj[name] = deserialise_interpolated_array(obj[name])
                continue

            # Deserialise ParametricDelayACF
            elif field.type_ is ParametricDelayACF and isinstance(obj.get(name), dict) and obj[name].get(
                    "type") == 'dsa2000_cal.visibility_model.source_models.rfi_parametric_rfi_emitter.ParametricDelayACF':
                obj[name] = deserialise_parametric_delay_acf(obj[name])
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
