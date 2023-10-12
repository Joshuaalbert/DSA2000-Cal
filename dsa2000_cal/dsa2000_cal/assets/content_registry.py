import importlib
import inspect
import pkgutil
from abc import abstractmethod
from functools import cached_property
from types import ModuleType
from typing import List, Dict, Callable, Any, TypeVar, Protocol, Type, Generic


from dsa2000_cal import assets
from dsa2000_cal.assets.base_content import BaseContent

__all__ = [
    'fill_registries',
    'NoMatchFound',
    'ContentRegistry',
    'AbstractContentFactory',
    'ContentMap'
]

_LOADED = False


def load_all_module_recursively(module: ModuleType):
    """
    Imports all sub-modules within a module path.
    Use this to ensure that global registries are fully updated.
    Beware of unintended consequences.

    Args:
        module: the module to start at an import recursively from.
    """
    # The description of this code is:
    # https://stackoverflow.com/questions/3365740/how-to-import-all-submodules
    if inspect.ismodule(module) and hasattr(module, '__path__'):
        for _, mod_name, is_pkg in pkgutil.walk_packages(module.__path__):
            full_mod_name = module.__name__ + '.' + mod_name
            try:
                submodule = importlib.import_module(full_mod_name)
            except ImportError:
                continue

            if inspect.ismodule(submodule):
                load_all_module_recursively(submodule)


def fill_registries():
    """
    Fill the registries with content.
    """
    global _LOADED

    load_all_module_recursively(module=assets)
    _LOADED = True


TemplateType = TypeVar('TemplateType')
MatchPatternType = TypeVar('MatchPatternType')


class NoMatchFound(Exception):
    """
    Exception raised when no match is found.
    """

    def __init__(self, match_pattern: MatchPatternType | None = None):
        super().__init__(f"Could not match {match_pattern}")


class SupportsLessThan(Protocol):
    """
    Protocol for supporting less than.
    """

    def __lt__(self, other: Any) -> bool:
        pass


ALWAYS_MATCH = object()

T = TypeVar('T')


class AbstractContentFactory:
    """
    Abstract factory for content objects.
    """

    @abstractmethod
    def __call__(self, content_cls: Type[BaseContent]) -> BaseContent:
        """
        Constructs a content object from the class.

        Args:
            content_cls: class to construct from

        Returns:
            constructed content object
        """
        ...


class NoArgInit(Protocol):
    """
    Protocol for classes that have no arguments to __init__.
    """

    def __init__(self) -> None: ...


class BaseContentFactory(AbstractContentFactory, Generic[T]):
    """
    Factory for content objects, with no arguments needed to construct.
    """

    def __call__(self, content_cls: Type[BaseContent | NoArgInit | T]) -> T:
        """
        Constructs a content object from the class.

        Args:
            content_cls: class to construct from

        Returns:
            constructed content object
        """
        return content_cls()


class ContentRegistry(Generic[T]):
    """
    Registry for content. Content is registered with a template. The template is used to match against a pattern.
    """

    def __init__(self, match_func: Callable[[MatchPatternType, TemplateType], bool],
                 sort_key_func: Callable[[MatchPatternType, TemplateType], SupportsLessThan] | None = None,
                 content_factory: AbstractContentFactory | Callable[..., AbstractContentFactory] = BaseContentFactory()
                 ):
        self.entries: Dict[Type[T], List[TemplateType]] = {}
        self.match_func = match_func
        self.sort_key_func = sort_key_func
        self.content_factory = content_factory

    def register_module(self, module: ModuleType):
        """
        Registers a module (and all sub-modules), by importing them an collection all global registrations.

        Args:
            module: module to recursively import
        """
        load_all_module_recursively(module)

    def register(self, obj: Type[T], template: TemplateType = ALWAYS_MATCH):
        """
        Stores the object with the template.

        Args:
            obj: object to store
            template: template to match against
        """
        if obj not in self.entries:
            self.entries[obj] = []
        self.entries[obj].append(template)

    def __call__(self, template: TemplateType = ALWAYS_MATCH):
        """
        Descriptor that registers the function with template.

        Args:
            template: template to match against, or None for always match.

        Returns:
            wrapper that registers
        """

        def decorator(obj: Type[T]):
            self.register(obj=obj, template=template)
            return obj

        return decorator

    def get_instance(self, content_cls: Type[T]) -> T:
        """
        Gets an instance of the content object.

        Args:
            obj: object to get instance of

        Returns:
            instance of the object
        """
        if content_cls not in self.entries:
            raise ValueError(f"obj {content_cls} not in entries")

        if isinstance(self.content_factory, AbstractContentFactory):
            content = self.content_factory(content_cls)
        else:
            if not callable(self.content_factory):
                raise ValueError(f"content_factory {self.content_factory} not callable")
            templates = self.entries[content_cls]
            factory = self.content_factory(templates)
            content = factory(content_cls)
        return content

    def get_match(self, match_pattern: MatchPatternType) -> Type[T]:
        """
        Get first request script that matches the template.

        Args:
            match_pattern: pattern to match for

        Returns:
            request script

        Raises:
            NoMatchFound if no match found.
        """
        matches = self.get_all(match_pattern)
        if len(matches) == 0:
            # fill_registries()  # TODO: This is a hack to fix the issue of the registries not being filled
            matches = self.get_all(match_pattern)
            if len(matches) == 0:  # Still no matches
                raise NoMatchFound(match_pattern=match_pattern)
        return matches[0]

    def get_all(self, match_pattern: MatchPatternType = ALWAYS_MATCH) -> List[Type[T]]:
        """
        Get all matches ordered from closest match to furthest. Closest is when exact match.

        Args:
            match_pattern: pattern to match for, by default all unsorted.

        Returns:
            the list of functions ordered from best to worst match.
        """
        if match_pattern is ALWAYS_MATCH:  # Get all
            matched, templates = list(zip(*self.entries.items()))
            return list(matched)  # unsorted

        matched: List[Type[T]] = []
        templates: List[TemplateType] = []
        for obj in self.entries:
            for template in self.entries[obj]:
                if self.match_func(match_pattern, template) or (template is ALWAYS_MATCH):
                    matched.append(obj)
                    templates.append(template)
                    break

        if self.sort_key_func is None:
            return matched

        def _dist(template: TemplateType):
            return self.sort_key_func(match_pattern, template)

        if len(matched) == 0:
            return []

        matched, _ = list(zip(*sorted(zip(matched, templates), key=lambda item: _dist(item[1]))))
        return matched


class ContentNotFoundError(Exception):
    """
    Exception raised when content is not found.
    """
    pass


class ContentMap(Generic[T]):
    """
    Map of content objects, with the id as the key.
    """

    def __init__(self, content_registry: ContentRegistry):
        """
        Initialize the content map, with the content registry and content factory.

        Args:
            content_registry: content registry
            content_factory: content factory, defaults to BaseContentFactory which constructs with no arguments.
        """
        self.content_registry = content_registry

    @cached_property
    def content_map(self) -> Dict[str, T]:
        """
        Lazily constructs to content map, using the content factory and content registry.

        Returns:
            the content map
        """
        content_map = {}
        # print(self.content_registry.entries)
        for content_cls, templates in self.content_registry.entries.items():
            content = self.content_registry.get_instance(content_cls=content_cls)
            if content.id in content_map:
                raise ValueError(f"content {content} with id {content.id} already in map")
            content_map[content.id] = content
        return content_map

    def __getitem__(self, item: str) -> T:
        """
        Gets the content object with the given id.

        Args:
            item: id of the content object

        Returns:
            the content object
        """
        if not isinstance(item, str):
            raise ValueError("item must be a string")
        if item not in self.content_map:
            raise ContentNotFoundError(f"Could not find content with id {item}")
        return self.content_map[item]


class SetKwargsFactory(AbstractContentFactory, Generic[T]):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, content_cls: Type[BaseContent | T]) -> T:
        return content_cls(**self.kwargs)
