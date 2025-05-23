from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Generic, TypeVar

from py3dtiles.tilers.base_tiler.shared_metadata import SharedMetadata

_SharedMetadataT = TypeVar("_SharedMetadataT", bound=SharedMetadata)


class TilerWorker(ABC, Generic[_SharedMetadataT]):
    def __init__(self, shared_metadata: _SharedMetadataT):
        # The attribute shared_metadata must not be modified by any tiler worker
        self.shared_metadata = shared_metadata

    @abstractmethod
    def execute(
        self, command: bytes, content: list[bytes]
    ) -> Iterator[Sequence[bytes]]:
        """
        Executes a command sent by the tiler. The method should yield all the
        messages that it wants to be sent back to the tiler or the main
        process.

        Implementing classes can use any message format they like provided it
        is a Sequence of bytes. The only restriction is that the first element
        of the sequence should **not** be a value in the :class:`py3dtiles.tilers.base_tiler.message_type.WorkerMessageType`
        enum, as  those are used internally by the convert process to manage
        the lifecycles of the different entities.
        """
