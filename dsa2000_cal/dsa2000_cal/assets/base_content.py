import hashlib
import os
import sys
import uuid


def deterministic_uuid(seed: str) -> uuid.UUID:
    """
    Generate a UUID using a deterministic hashing of a seed string.

    Args:
        seed: str, a string seed

    Returns:
        UUID

    """
    if not isinstance(seed, str):
        raise TypeError(f"Expected seed type `str`, got {type(seed)}.")
    m = hashlib.md5()
    m.update(seed.encode('utf-8'))
    new_uuid = uuid.UUID(m.hexdigest())
    return new_uuid


class BaseContent:
    """
    Represents an entity with UUID and content path.
    """

    def __init__(self, *, seed: str):
        self.content_path = self._content_path()
        self.seed = seed
        self.id = str(deterministic_uuid(seed))

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: 'BaseContent') -> bool:
        """
        Enables equality comparison of content by id.

        Args:
            other: a base content object

        Returns:
            True if the ids are equal, False otherwise.
        """
        if other is None:
            return False
        if not isinstance(other, BaseContent):
            raise TypeError(f"Expected 'BaseContent' inheritor, got {type(other)}")
        return self.id == other.id

    @classmethod
    def _content_path(cls):
        return os.path.split(os.path.abspath(sys.modules[cls.__module__].__file__))[:-1]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.id[:6]})"

    def __lt__(self, other: 'BaseContent') -> bool:
        """
        Enables sorting of content by id.

        Args:
            other: a base content object

        Returns:
            True if the id of the current object is less than the id of the other object, False otherwise.
        """
        if not isinstance(other, BaseContent):
            raise TypeError(f"Expected 'BaseContent' inheritor, got {type(other)}")
        return self.id < other.id
