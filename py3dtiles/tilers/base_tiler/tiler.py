from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Generic, TypeVar

from pyproj import CRS

from py3dtiles.constants import CPU_COUNT, DEFAULT_CACHE_SIZE
from py3dtiles.tileset.tile import Tile

from .shared_metadata import SharedMetadata
from .tiler_worker import TilerWorker

_SharedMetadataT = TypeVar("_SharedMetadataT", bound=SharedMetadata)
_TilerWorkerT = TypeVar("_TilerWorkerT", bound=TilerWorker[Any])


class Tiler(ABC, Generic[_SharedMetadataT, _TilerWorkerT]):
    """
    This class is the superclass for all tilers in py3dtilers. It is
    responsible to instantiate the workers it will use and generate new tasks
    according to the current state of the conversion.

    It will receive messages both from its workers and the main process (see `process_message`).

    Its role is to organize tasks to be dispatched to the worker it has
    constructed and later, to write the tileset corresponding to the hierarchy
    of tiles it created.

    **Implementation notice**:

    - the `name` class attribute should be overwritten by subclasses and should be unique
    - `__init__` should not read any files on the disk, `initialize` on the
      other hand is expected to gather metadata for input files
    - as this class is generic over the type of SharedMetadata and TilerWorker,
      subclassing these 2 classes is also needed when creating a Tiler
    - modifications to the SharedMetadata instance will *not* be transmitted to
      other processes, initialize it in `initialize` and **don't mutate it
      afterwards**
    - all mutable data and parameters **must** be passed as messages between tilers and workers. Workers will send messages by using ``yield`` (see :class:`py3dtiles.tilers.base_tiler.tiler_worker.TilerWorker`)
    - the constructor of a Tiler is not expected to do any real work. The ``initialize`` method on the other hand, should gather metadata from input files

    **important note about `out_folder`**

    `out_folder` will be given as parameter when the base process calls
    `initialize`. This folder should contain all the written tiles by this tiler. How this folder is organized is left to the tiler's
    implementor. The only constraint is that all the `content_uri` should be
    relative to the **parent** folder of this folder. `out_folder` given to each tiler
    is actually a subfolder (of the name self.name) of `out_folder` given to
    the main process. The latter will actually contain the tileset.json, that's
    why all the content_uri should be relative to the parent folder of the
    given out_folder.

    For instance, if the `out_folder` given to the convert process is `.`, and self.name is "points", the parameter `out_folder` given to `initialize` would be `./points`, and the hierarchy at the end of the process should be::

        .
        ├── points
        │   ├── r0.pnts
        │   ├── r2.pnts
        │   ├── ...
        │   ├── r6.pnts
        └── tileset.json

    The `content_uri` of the different tiles should therefore be relative to `.` and not `./points`.

    This class will organize the different tasks and their order of dispatch to
    the TilerWorker instances. When creating a subclass of Tiler, you're
    supposed to subclass SharedMetadata and TilerWorker as well.
    """

    name = ""
    shared_metadata: _SharedMetadataT
    crs_in: CRS | None
    crs_out: CRS | None
    force_crs_in: bool
    pyproj_always_xy: bool
    cache_size: int
    verbosity: int
    number_of_jobs: int

    def __init__(
        self,
        crs_in: CRS | None = None,
        crs_out: CRS | None = None,
        force_crs_in: bool = False,
        pyproj_always_xy: bool = False,
        cache_size: int = DEFAULT_CACHE_SIZE,
        verbosity: int = 0,
        number_of_jobs: int = CPU_COUNT,
    ):
        super().__init__()
        self.crs_in = crs_in
        self.crs_out = crs_out
        self.force_crs_in = force_crs_in
        self.pyproj_always_xy = pyproj_always_xy

        self.cache_size = cache_size

        self.verbosity = verbosity
        self.number_of_jobs = number_of_jobs

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
    def initialize(
        self, files: list[Path], working_dir: Path, out_folder: Path
    ) -> None:
        """
        This method will be called first by convert to initialize the conversion
        process. Tilers will receive all the paths information as argument to
        this method. Only files supported by this tiler will be in the files
        argument.  Tilers are expected to gather metadata from those input
        files so that subsequent call to `get_tasks` can generate some
        conversion work to do by workers.

        This method is probably a good place to init the SharedMetadata
        subclass instance as well.

        :param working_dir: a temporary directory where this tiler can store intermediate result
        :param out_folder: the output folder for this tiler. Please see the note about this folder in the general documentation for this class.
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

        This methods will get called by the main convert function each time it wants new tasks to be
        fed to workers. Implementors should each time returns the task that has the biggest
        priority.

        py3dtiles will iterate until the returned iterator is exhausted before continuing the
        process. It will call it as many times as needed during the execution. It is therefore not
        necessary to generate all the tasks in one go.

        Once this function returns an empty list and all the workers are idle, the conversion
        process stops.

        If generating the tasks is somewhat expensive, do return a Generator instead. Tasks will be
        sent to workers as soon as they are yielded.
        """

    @abstractmethod
    def process_message(self, message_type: bytes, message: list[bytes]) -> None:
        """
        This method is called with each message sent by workers. Its role is to process those
        messages, to update the internal state of the tiler, so that new tasks or a tileset writing
        could proceed.
        """

    @abstractmethod
    def get_root_tile(self, use_process_pool: bool = True) -> Tile:
        """
        Get the tileset file once the binary data are written.

        This function will be called once by convert after this tiler has stopped generating tasks and all
        the workers are idle.

        Tilers are expected to returns one root tile which will be the root of all their hierarchy. Tilers are expected to set the `content_uri` of every tile

        :param use_process_pool: allow the use of a process pool. Process pools can cause issues in
        environment lacking shared memory.
        """

    def validate(self) -> None:
        """
        Checks if the state of the tiler or the binary data written is correct.
        This method is called after the end of the conversion of this tiler (but before writing the tileset). Overwrite this method if you wish to run some validation code for the generated tileset.
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
