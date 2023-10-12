import inspect
from typing import TypeVar, Type, Dict, Any

import astropy.coordinates as ac
import astropy.units as au
import numpy as np
import ujson
from pydantic import BaseModel

C = TypeVar('C')


class SerialisableBaseModel(BaseModel):
    """
    A pydantic BaseModel that can be serialised and deserialised using pickle, working well with Ray.
    """

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        json_loads = ujson.loads  # can use because ujson decodes NaN and Infinity
        json_dumps = ujson.dumps  # (currently not possible because ujson doesn't encode NaN and Infinity like json)
        # json_dumps = lambda *args, **kwargs: json.dumps(*args, **kwargs, separators=(',', ':'))
        json_encoders = {
            np.ndarray: lambda x: x.tolist(),
            ac.ICRS: lambda x: {
                "ra": x.ra.to_string(unit=au.hour, sep='hms', pad=True),
                "dec": x.dec.to_string(unit=au.deg, sep='dms', pad=True, alwayssign=True)
            },
            ac.ITRS: lambda x: {
                "x": x.x.to(au.m).value,
                "y": x.y.to(au.m).value,
                "z": x.z.to(au.m).value,
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
            if isinstance(field.type_, type) and issubclass(field.type_, np.ndarray):
                if name in obj and isinstance(obj[name], list):
                    obj[name] = np.array(obj[name])
                    continue

            # Deserialise ICRS and ITRS
            if field.type_ is ac.ICRS and isinstance(obj.get(name), dict):
                ra_dec = obj[name]
                obj[name] = ac.ICRS(ra=ac.Angle(ra_dec["ra"]), dec=ac.Angle(ra_dec["dec"]))
                continue

            if field.type_ is ac.ITRS and isinstance(obj.get(name), dict):
                coords = obj[name]
                obj[name] = ac.ITRS(x=coords["x"] * au.m, y=coords["y"] * au.m, z=coords["z"] * au.m)
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
