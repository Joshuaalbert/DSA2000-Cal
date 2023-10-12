import re
from typing import List

from dsa2000_cal.assets.arrays.array import AbstractArray
from dsa2000_cal.assets.content_registry import ContentRegistry, ContentMap, AbstractContentFactory, SetKwargsFactory


def match_func(match_pattern: str, template: str) -> bool:
    """
    Match function for the dimension registry.

    Args:
        match_pattern: pattern to match
        template: template to match against

    Returns:
        True if the pattern matches the template, False otherwise.
    """
    # use re to look for match in templates
    if re.search(match_pattern, template):
        return True
    return False


def sort_key_func(match_pattern: str, template: str) -> int:
    """
    Sort key function for the dimension registry.

    Args:
        match_pattern: pattern to match
        template: template to match against

    Returns:
        The negative number of elements in the intersection of the pattern and the template.
    """
    # use the size of match. Find the largest match of match_pattern in template, and return negative this value
    return -len(max(re.findall(match_pattern, template), default='', key=len))


def create_factory_from_templates(templates: List[str]) -> AbstractContentFactory:
    seed = ", ".join(templates)
    return SetKwargsFactory(seed=seed)


array_registry = ContentRegistry[AbstractArray](
    match_func=match_func,
    sort_key_func=sort_key_func,
    content_factory=create_factory_from_templates
)
array_map = ContentMap[AbstractArray](content_registry=array_registry)
