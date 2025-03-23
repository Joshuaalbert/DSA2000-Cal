import dataclasses

from dsa2000_common.common.pytree import Pytree

@dataclasses.dataclass(eq=False)
class Smearing(Pytree):
    def time_smearing(self):
        ...

    def freq_smearing(self):
        ...
