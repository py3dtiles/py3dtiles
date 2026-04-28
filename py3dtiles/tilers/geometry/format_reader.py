from abc import abstractmethod
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from py3dtiles.tilers.base_tiler import SharedMetadata
from py3dtiles.tilers.geometry.geometry_message_type import GeometryWorkerMessage


class FormatReader:

    def __init__(self, shared_metadata: SharedMetadata):
        self.shared_metadata = shared_metadata

    """
    Basic class to implement format reader
    """

    @abstractmethod
    def read(
        self, filename: Path
    ) -> Iterator[tuple[GeometryWorkerMessage, Sequence[Any]]]:
        """
        A reader must implement this read method, obviously!

        It MUST emit those two messages to the tiler:

        - one `GeometryWorkerMessage.METADATA_READ` message, whose value must
          be a `FilenameAndOffset` instance. This does not need to be the first
          message
        - one `GeometryWorkerMessage.FILE_READ` with the filename as payload
          when the file is completely read. This must be the last message.

        It *may* emit any number of `GeometryWorkerMessage.TILE_PARSED`, with
        payload of [filename, tile_id, tile]`, where tile is an instance of
        `FeatureGroup`.

        **About the return type**: although it's not a hard requirement, reader
        are **strongly** encouraged to be *generators*. The tiler worker will
        actually send a message back to the main thread for each returned
        element. Implementing this as a generator allows the message to be sent
        as soon as it is ready, instead of waiting for this function to
        complete, which can take a long time.

        """
