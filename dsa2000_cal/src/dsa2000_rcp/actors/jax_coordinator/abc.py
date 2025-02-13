from abc import ABC, abstractmethod


class AbstractJaxCoordinator(ABC):
    """
    Abstract class for the JaxCoordinator actor.
    """

    @abstractmethod
    def get_coordinator_address(self):
        """
        Get the coordinator address, e.g. "localhost:6379".
        """
        ...
