from abc import ABC, abstractmethod


class AbstractEventBus(ABC):
    """
    Abstract EventBus PubSub.
    """

    @abstractmethod
    async def shutdown(self):
        """
        Shutdown the EventBus.
        """
        ...
