from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Generic, TypeVar

from .shared_metadata import SharedMetadata
from .tiler_worker import TilerWorker

_SharedMetadataT = TypeVar("_SharedMetadataT", bound=SharedMetadata)
_TilerWorkerT = TypeVar("_TilerWorkerT", bound=TilerWorker[Any])


class Tiler(ABC, Generic[_SharedMetadataT, _TilerWorkerT]):
    """
    This class is the superclass for all tilers in py3dtilers. It is responsible to instanciate the workers it will use and generate new tasks according to the current state of the conversion.

    It will receive messages both from its workers and the main process (see `process_message`).

    Its role is to organize tasks to be dispatched to the worker it has constructed and later, to write the tileset corresponding to the hierarchy of tiles it created.

    **Implementor notes**

    - the `name` class attribute should be changed by suclasses.
    - `__init__` should not read any files on the disk, `initialization` on the other hand is espected to gather metadata for input files.

    This class will organize the different tasks and their order of dispatch to the TilerWorker instances. When creating a subclass of Tiler, you're supposed to subclass SharedMetadata and TilerWorker as well.
    """

    name = b""
    shared_metadata: _SharedMetadataT

    @abstractmethod
    def supports(self, file: Path) -> bool:
        """
        This function tells the main process if this tiler supports this file or not.

        The main process will use the first supporting tiler it finds for each file.

        Implementation should not require to read the whole file to determine
        if this tiler supports it. In other word, the execution time should be
        a constant regardless of the file size.
        """

    @abstractmethod
    def initialization(
        self, files: list[Path], working_dir: Path, out_folder: Path
    ) -> None:
        """
        The __init__ method must only set attributes without any action.
        It is in this method that this work must be done (and the initialization of shared_metadata).

        The method will be called before all others.
        """

    @abstractmethod
    def get_worker(self) -> _TilerWorkerT:
        """
        Returns an instantiated tiler worker.
        """

    @abstractmethod
    def get_tasks(self) -> Iterator[tuple[bytes, list[bytes]]]:
        """
        Yields tasks to be sent to workers.

        This methods will get called by the main convert function each time it wants new tasks to be fed to workers.
        Implementors should each time returns the task that has the biggest priority.


        """

    @abstractmethod
    def process_message(self, return_type: bytes, content: list[bytes]) -> None:
        """
        Updates the state of the tiler in function of the return type and the returned data
        """

    @abstractmethod
    def write_tileset(self, use_process_pool: bool = True) -> None:
        """
        Writes the tileset file once the binary data written

        :param use_process_pool: allow the use of a process pool. Process pools can cause issues in environment lacking shared memory.
        """

    def validate(self) -> None:
        """
        Checks if the state of the tiler or the binary data written is correct.
        This method is called after the end of the conversion of this tiler (but before writing the_tileset). Overwrite this method if you wish to run some validation code for the generated tileset.
        """

    def memory_control(self) -> None:
        """
        Method called at the end of each loop of the convert method.
        Checks if there is no too much memory used by the tiler and do actions in function
        """

    def print_summary(self) -> None:
        """
        Prints the summary of the tiler before the start of the conversion.
        """
        ...

    def benchmark(self, benchmark_id: str, startup: float) -> None:
        """
        Prints benchmark info at the end of the conversion of this tiler and the writing of the tileset.
        """
