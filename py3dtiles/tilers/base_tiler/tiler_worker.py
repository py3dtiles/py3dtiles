from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Generic, TypeVar

from py3dtiles.tilers.base_tiler.shared_metadata import SharedMetadata

_SharedMetadataT = TypeVar("_SharedMetadataT", bound=SharedMetadata)


class TilerWorker(ABC, Generic[_SharedMetadataT]):
    def __init__(self, shared_metadata: _SharedMetadataT):
        # The attribute shared_metadata must not be modified by any tiler worker
        self.shared_metadata = shared_metadata

    def initialize(self) -> None:
        """
        This method will be called once at the start of the process in the subprocess, not in the main thread

        Useful to initialize non-pickable objects on windows, for instance
        """

    @abstractmethod
    def execute(
        self, command: bytes, content: list[bytes]
    ) -> Iterator[Sequence[bytes]]:
        """
        Executes a command sent by the tiler. Each sequence of bytes returned by the returned Iterator will be sent back to the corresponding tiler.

        The easiest way to do so is to ``yield`` the Sequence to be sent. This
        will automatically turn the method into a generator, which is
        incidentally an Iterator. This has the good additional side-effect of
        passing back control to the calling method, which will in turn send the
        message immediately, without waiting for this method to be executed.

        Implementing classes can use any message format they like provided it
        is a Sequence of bytes. The only restriction is that the first element
        of the sequence should **not** be a value in the
        :class:`py3dtiles.tilers.base_tiler.message_type.WorkerMessageType`
        enum, as those are used internally by the convert process to manage
        the lifecycles of the different entities.
        """
