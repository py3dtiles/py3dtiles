import argparse
import os
import pickle
import shutil
import sys
import tempfile
import time
import traceback
from multiprocessing import Process
from pathlib import Path
from time import sleep
from typing import Any

import zmq
from pyproj import CRS

from py3dtiles.constants import CPU_COUNT, DEFAULT_CACHE_SIZE, EXIT_CODES
from py3dtiles.exceptions import (
    Py3dtilesException,
    SrsInMissingException,
    TilerException,
    TilerNotFoundException,
    WorkerException,
)
from py3dtiles.merger import create_tileset_from_root_tiles
from py3dtiles.tilers.base_tiler import Tiler
from py3dtiles.tilers.base_tiler.message_type import ManagerMessage, WorkerMessageType
from py3dtiles.tilers.base_tiler.tiler_worker import TilerWorker
from py3dtiles.tilers.point.point_tiler import PointTiler
from py3dtiles.utils import mkdir_or_raise, str_to_CRS

try:
    from py3dtiles.tilers.ifc.ifc_tiler import IfcTiler

    HAS_IFC_SUPPORT = True
except ImportError as e:
    if e.name == "ifcopenshell":
        HAS_IFC_SUPPORT = False
    else:
        raise

# IPC protocol is not supported on Windows
if os.name == "nt":
    URI = "tcp://127.0.0.1:0"
else:
    # Generate a unique name for this socket
    tmpdir = tempfile.TemporaryDirectory()
    URI = f"ipc://{tmpdir.name}/py3dtiles.sock"


META_TILER_NAME = b"meta"


def _worker_target(
    worker_tilers: dict[str, TilerWorker[Any]],
    verbosity: int,
    uri: bytes,
) -> None:
    return _WorkerDispatcher(
        worker_tilers,
        verbosity,
        uri,
    ).run()


class _WorkerDispatcher:
    """
    This class waits from jobs commands from the Zmq socket.
    """

    skt: zmq.Socket[bytes]

    def __init__(
        self,
        worker_tilers: dict[str, TilerWorker[Any]],
        verbosity: int,
        uri: bytes,
    ) -> None:
        self.worker_tilers = worker_tilers
        self.verbosity = verbosity
        self.uri = uri

        # Socket to receive messages on
        self.context = zmq.Context()

    def run(self) -> None:
        self.skt = self.context.socket(zmq.DEALER)

        self.skt.connect(self.uri)  # type: ignore [arg-type]

        startup_time = time.time()
        idle_time = 0.0

        for worker in self.worker_tilers.values():
            worker.initialize()

        # notify we're ready
        self.skt.send_multipart([WorkerMessageType.REGISTER.value])

        while True:
            try:
                before = time.time() - startup_time
                self.skt.poll()
                after = time.time() - startup_time

                idle_time += after - before

                message = self.skt.recv_multipart()
                tiler_name = message[1].decode()
                command = message[2]
                content = message[3:]

                delta = time.time() - pickle.loads(message[0])
                if delta > 0.01 and self.verbosity >= 1:
                    print(
                        f"{os.getpid()} / {round(after, 2)} : Delta time: {round(delta, 3)}"
                    )

                if command == ManagerMessage.SHUTDOWN.value:
                    break  # ack
                else:
                    for answer in self.worker_tilers[tiler_name].execute(
                        command, content
                    ):
                        self.skt.send_multipart(answer, copy=False)

                # notify we're idle
                self.skt.send_multipart([WorkerMessageType.IDLE.value])
            except Exception as e:
                traceback.print_exc()
                error_message = f"{e.__class__.__module__}.{e.__class__.__name__}: {e}"
                self.skt.send_multipart(
                    [WorkerMessageType.ERROR.value, error_message.encode()]
                )
                # we still print it for stacktraces

        if self.verbosity >= 1:
            print(
                "total: {} sec, idle: {}".format(
                    round(time.time() - startup_time, 1), round(idle_time, 1)
                )
            )

        self.skt.send_multipart([WorkerMessageType.HALTED.value])


# Manager
class _ZmqManager:
    """
    This class sends messages to the workers.
    We can also request general status.
    """

    def __init__(
        self,
        number_of_jobs: int,
        worker_tilers: dict[str, TilerWorker[Any]],
        verbosity: int,
    ) -> None:
        """
        For the process_args argument, see the init method of Worker
        to get the list of needed parameters.
        """
        self.context = zmq.Context()

        self.number_of_jobs = number_of_jobs

        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(URI)
        # Useful only when TCP is used to get the URI with the opened port
        self.uri = self.socket.getsockopt(zmq.LAST_ENDPOINT)
        if not isinstance(self.uri, bytes):
            raise RuntimeError(
                "The uri returned by self.socket.getsockopt should be bytes."
            )

        self.processes = [
            Process(
                target=_worker_target,
                args=(worker_tilers, verbosity, self.uri),
            )
            for _ in range(number_of_jobs)
        ]
        for p in self.processes:
            p.start()

        self.activities = [p.pid for p in self.processes]
        self.clients: set[bytes] = set()
        self.idle_clients: set[bytes] = set()

        self.killing_processes = False
        self.number_processes_killed = 0
        self.time_waiting_an_idle_process = 0.0

    def all_clients_registered(self) -> bool:
        return len(self.clients) == self.number_of_jobs

    def send_to_process(self, message: list[bytes]) -> None:
        if not self.idle_clients:
            raise ValueError("idle_clients is empty")
        self.socket.send_multipart(
            [self.idle_clients.pop(), pickle.dumps(time.time())] + message
        )

    def send_to_all_processes(self, message: list[bytes]) -> None:
        if len(self.clients) == 0:
            raise ValueError("No registered clients")
        for client in self.clients:
            self.socket.send_multipart([client, pickle.dumps(time.time())] + message)

    def send_to_all_idle_processes(self, message: list[bytes]) -> None:
        if not self.idle_clients:
            raise ValueError("idle_clients is empty")
        for client in self.idle_clients:
            self.socket.send_multipart([client, pickle.dumps(time.time())] + message)
        self.idle_clients.clear()

    def can_queue_more_jobs(self) -> bool:
        return len(self.idle_clients) != 0

    def register_client(self, client_id: bytes) -> None:
        if client_id in self.clients:
            print(f"Warning: {client_id!r} already registered")
        else:
            self.clients.add(client_id)
        self.add_idle_client(client_id)

    def add_idle_client(self, client_id: bytes) -> None:
        if client_id in self.idle_clients:
            raise ValueError(f"The client id {client_id!r} is already in idle_clients")
        self.idle_clients.add(client_id)

    def are_all_processes_idle(self) -> bool:
        return len(self.idle_clients) == self.number_of_jobs

    def are_all_processes_killed(self) -> bool:
        return self.number_processes_killed == self.number_of_jobs

    def shutdown_all_processes(self) -> None:
        self.send_to_all_processes([META_TILER_NAME, ManagerMessage.SHUTDOWN.value])
        self.killing_processes = True

    def join_all_processes(self) -> None:
        for p in self.processes:
            p.join()


def convert(
    files: list[str | Path] | str | Path,
    outfolder: str | Path = "./3dtiles",
    overwrite: bool = False,
    jobs: int = CPU_COUNT,
    cache_size: int = DEFAULT_CACHE_SIZE,
    crs_out: CRS | None = None,
    crs_in: CRS | None = None,
    force_crs_in: bool = False,
    pyproj_always_xy: bool = False,
    benchmark: str | None = None,
    rgb: bool = True,
    extra_fields: list[str] | None = None,
    color_scale: float | None = None,
    use_process_pool: bool = True,
    verbose: int = False,
) -> None:
    """
    Convert the input files into 3dtiles.

    :param files: Filenames to process. The file must use the .las, .laz, .xyz or .ply format.
    :param outfolder: The folder where the resulting tileset will be written.
    :param overwrite: Overwrite the ouput folder if it already exists.
    :param jobs: The number of parallel jobs to start. Default to the number of cpu.
    :param cache_size: Cache size in MB. Default to available memory / 10.
    :param crs_out: CRS to convert the output with
    :param crs_in: Set a default input CRS
    :param force_crs_in: Force every input CRS to be `crs_in`, even if not null
    :param pyproj_always_xy: When converting from a CRS to another, pass the `always_xy` flag to pyproj. This is useful if your data is in a CRS whose definition specifies an axis order other than easting/northing, but your data still have the easting component in the first field (often named X or longitude). See https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6 for more information.
    :param benchmark: Print summary at the end of the process
    :param rgb: Export rgb attributes.
    :param extra_fields: Extra fields names to include in this conversion. These field names should be present in each input files. Currently vlrs and evlrs are not supported for las files.
    :param color_scale: Scale the color with the specified amount. Useful to lighten or darken black pointclouds with only intensity.

    :raises SrsInMissingException: if py3dtiles couldn't find srs information in input files and srs_in is not specified
    :raises SrsInMixinException: if the input files have different CRS

    """

    files = [files] if isinstance(files, (str, Path)) else files
    paths = [Path(file) for file in files]
    tilers: list[Tiler[Any, Any]] = [
        PointTiler(
            crs_in,
            crs_out,
            force_crs_in,
            pyproj_always_xy,
            cache_size,
            verbose,
            jobs,
            rgb=rgb,
            color_scale=color_scale,
            extra_fields=extra_fields,
        ),
    ]
    if HAS_IFC_SUPPORT:
        tilers.append(
            IfcTiler(
                crs_in,
                crs_out,
                force_crs_in,
                pyproj_always_xy,
                cache_size,
                verbose,
                jobs,
            )
        )

    converter = Converter(
        tilers,
        overwrite=overwrite,
        jobs=jobs,
        cache_size=cache_size,
        crs_out=crs_out,
        crs_in=crs_in,
        force_crs_in=force_crs_in,
        pyproj_always_xy=pyproj_always_xy,
        benchmark=benchmark,
        use_process_pool=use_process_pool,
        verbose=verbose,
    )

    try:
        return converter.convert(paths, Path(outfolder), overwrite=overwrite)
    except TilerNotFoundException:
        print("ERROR: support not found for files", files)
        print(
            "Please check https://py3dtiles.org/v9.0.0/install.html#file-formats-support"
        )
        sys.exit(1)


class Converter:
    """
    The Converter class allows for fine-grained conversion process and custom Tilers.
    It is built with a list of tilers instead of files. Each tiler is responsible to generate a hierarchy of
    tiles. The process will then build a tileset that will regroup all the tilesets generated by individual
    tilers.

    :param jobs: The number of parallel jobs to start. Default to the number of cpu.
    :param cache_size: Cache size in MB. Default to available memory / 10.
    :param crs_out: CRS to convert the output with
    :param crs_in: Set a default input CRS
    :param force_crs_in: Force every input CRS to be `crs_in`, even if not null
    :param pyproj_always_xy: When converting from a CRS to another, pass the `always_xy` flag to pyproj. This is useful if your data is in a CRS whose definition specifies an axis order other than easting/northing, but your data still have the easting component in the first field (often named X or longitude). See https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6 for more information.
    :param benchmark: Print summary at the end of the process

    """

    def __init__(
        self,
        tilers: list[Tiler[Any, Any]],
        overwrite: bool = False,
        jobs: int = CPU_COUNT,
        cache_size: int = DEFAULT_CACHE_SIZE,
        crs_out: CRS | None = None,
        crs_in: CRS | None = None,
        force_crs_in: bool = False,
        pyproj_always_xy: bool = False,
        benchmark: str | None = None,
        use_process_pool: bool = True,
        verbose: int = False,
    ) -> None:
        # create folder

        self.tilers = tilers

        self.jobs = jobs

        self.verbose = verbose
        self.benchmark = benchmark
        self.use_process_pool = use_process_pool

    def _assign_file_to_tilers(self, files: list[Path]) -> dict[str, list[Path]]:
        files_by_tiler_names: dict[str, list[Path]] = {}
        tiler_not_found_files: list[Path] = []
        for file in files:
            for tiler in self.tilers:
                if tiler.supports(file):
                    if tiler.name not in files_by_tiler_names:
                        files_by_tiler_names[tiler.name] = []
                    files_by_tiler_names[tiler.name].append(file)
                    break
            else:
                tiler_not_found_files.append(file)

        if len(tiler_not_found_files) > 0:
            raise TilerNotFoundException(tiler_not_found_files)
        return files_by_tiler_names

    def convert(
        self,
        files: Path | list[Path],
        out_folder: Path,
        overwrite: bool = False,
    ) -> None:
        """
        Convert some files.

        :param files: Filenames to process. The file must use the .las, .laz, .xyz or .ply format.
        :param outfolder: The folder where the resulting tileset will be written.
        :param overwrite: Overwrite the ouput folder if it already exists.

        :raises SrsInMissingException: if py3dtiles couldn't find srs information in input files and srs_in is not specified
        :raises SrsInMixinException: if the input files have different CRS

        """
        mkdir_or_raise(out_folder, overwrite=overwrite)
        working_dir = out_folder / "tmp"
        working_dir.mkdir(parents=True)

        paths = [files] if isinstance(files, Path) else files

        paths_by_tiler_name = self._assign_file_to_tilers(paths)

        worker_tilers: dict[str, TilerWorker[Any]] = {}
        for tiler in self.tilers:
            # check if at least one file would use that tiler
            if tiler.name not in paths_by_tiler_name:
                continue

            if tiler.name in worker_tilers:
                raise TilerException("There are tilers with the same attribute name.")

            try:
                tiler_out_folder = Path(out_folder) / tiler.name
                tiler_out_folder.mkdir(exist_ok=True)
                tiler.initialize(
                    paths_by_tiler_name[tiler.name],
                    working_dir / str(tiler.name),
                    tiler_out_folder,
                )
            except Py3dtilesException as e:
                shutil.rmtree(out_folder)
                raise e

            worker_tilers[tiler.name] = tiler.get_worker()

        if self.verbose >= 1:
            for tiler in self.tilers:
                if tiler.name not in paths_by_tiler_name:
                    continue
                tiler.print_summary()

        self.zmq_manager = _ZmqManager(
            self.jobs,
            worker_tilers,
            self.verbose,
        )
        startup: float = time.time()

        try:
            root_tiles = []
            for tiler in self.tilers:
                if tiler.name not in paths_by_tiler_name:
                    continue

                while True:
                    if (
                        not self.zmq_manager.can_queue_more_jobs()
                        or self.zmq_manager.socket.poll(timeout=0, flags=zmq.POLLIN)
                    ):
                        self._process_message(tiler)

                    # we wait for all processes/threads to register
                    # if we don't there are tricky cases where an exception fires in a worker before all the workers registered, which means that not all workers will receive the shutdown signal
                    if not self.zmq_manager.all_clients_registered():
                        sleep(0.1)
                        continue

                    if self.zmq_manager.can_queue_more_jobs():
                        for command, data in tiler.get_tasks():
                            self.zmq_manager.send_to_process(
                                [tiler.name.encode("UTF-8"), command] + data
                            )
                            if not self.zmq_manager.can_queue_more_jobs():
                                break

                    # if at this point we have no work in progress => we're done
                    if self.zmq_manager.are_all_processes_idle():
                        break

                    tiler.memory_control()

                tiler.validate()

                if self.verbose >= 1:
                    print("Writing 3dtiles")

                root_tile = tiler.get_root_tile(use_process_pool=self.use_process_pool)
                root_tile.change_base(out_folder / tiler.name, out_folder)
                root_tiles.append(root_tile)

                if self.verbose >= 1:
                    print(f"Tiler {tiler.name!r} done")

                if self.benchmark:
                    tiler.benchmark(self.benchmark, startup)

            if self.verbose >= 1:
                print("Merging tilesets")
            for tile in root_tiles:
                # we need to make sure the contents are loaded for the merger
                if tile.has_content():
                    tile.get_or_fetch_content(out_folder)
            tileset = create_tileset_from_root_tiles(root_tiles)
            if tileset.root_tile.has_content():
                tileset.root_tile.write_content(out_folder)
            tileset.write_as_json(out_folder / "tileset.json")

        finally:
            self.zmq_manager.shutdown_all_processes()
            self.zmq_manager.join_all_processes()
            shutil.rmtree(working_dir, ignore_errors=True)

            if self.verbose >= 1:
                print(
                    "destroy", round(self.zmq_manager.time_waiting_an_idle_process, 2)
                )

            self.zmq_manager.context.destroy()

    def _process_message(self, tiler: Tiler[Any, Any]) -> None:
        # Blocking read but it's fine because either all our child processes are busy
        # or we know that there's something to read (zmq.POLLIN)
        start = time.time()
        message = self.zmq_manager.socket.recv_multipart()

        client_id = message[0]
        message_type = message[1]
        content = message[2:]

        if message_type == WorkerMessageType.REGISTER.value:
            self.zmq_manager.register_client(client_id)
        elif message_type == WorkerMessageType.IDLE.value:
            self.zmq_manager.add_idle_client(client_id)

            if not self.zmq_manager.can_queue_more_jobs():
                self.zmq_manager.time_waiting_an_idle_process += time.time() - start

        elif message_type == WorkerMessageType.HALTED.value:
            self.zmq_manager.number_processes_killed += 1

        elif message_type == WorkerMessageType.ERROR.value:
            raise WorkerException(
                f"An exception occurred in a worker: {content[0].decode()}"
            )

        else:
            tiler.process_message(message_type, content)


def _init_parser(
    subparser: "argparse._SubParsersAction[Any]",
) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = subparser.add_parser(
        "convert",
        help="Convert input 3D data to a 3dtiles tileset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Filenames to process. The file must use the .las, .laz (lastools must be installed), .xyz or .ply format.",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="The folder where the resulting tileset will be written.",
        default="./3dtiles",
    )
    parser.add_argument(
        "--overwrite",
        help="Delete and recreate the ouput folder if it already exists. WARNING: be careful, there will be no confirmation!",
        action="store_true",
    )
    parser.add_argument(
        "--jobs",
        help="The number of parallel jobs to start. Default to the number of cpu.",
        default=CPU_COUNT,
        type=int,
    )
    parser.add_argument(
        "--cache_size",
        help="Cache size in MB. Default to available memory / 10.",
        default=DEFAULT_CACHE_SIZE,
        type=int,
    )
    parser.add_argument(
        "--srs_out",
        help="SRS to convert the output with (numeric part of the EPSG code)",
        type=str,
    )
    parser.add_argument(
        "--srs_in", help="Override input SRS (numeric part of the EPSG code)", type=str
    )
    parser.add_argument(
        "--benchmark", help="Print summary at the end of the process", type=str
    )
    parser.add_argument(
        "--no-rgb", help="Don't export rgb attributes", action="store_true"
    )
    parser.add_argument(
        "--extra-fields",
        help="Extra field names present in source data to include in resulting tileset. All input files *must* have this fields, with the same data type.",
        action="append",
    )
    parser.add_argument("--color_scale", help="Force color scale", type=float)
    parser.add_argument(
        "--force-srs-in",
        help="Force the input srs even if the srs in the input files are different. CAUTION, use only if you know what you are doing.",
        action="store_true",
    )
    parser.add_argument(
        "--disable-processpool",
        help="Disables using a process pool when writing 3D tiles. Useful for running in environments lacking shared memory.",
        action="store_true",
    )
    parser.add_argument(
        "--pyproj-always-xy",
        help="When converting from a CRS to another, pass the `always_xy` flag to pyproj. This is useful if your data is in a CRS whose definition specifies an axis order other than easting/northing, but your data still have the easting component in the first field (often named X or longitude). See https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6 for more information. ",
        action="store_true",
    )

    return parser


def _main(args: argparse.Namespace) -> None:
    try:
        return convert(
            args.files,
            args.out,
            overwrite=args.overwrite,
            jobs=args.jobs,
            cache_size=args.cache_size,
            crs_out=str_to_CRS(args.srs_out),
            crs_in=str_to_CRS(args.srs_in),
            force_crs_in=args.force_srs_in,
            pyproj_always_xy=args.pyproj_always_xy,
            benchmark=args.benchmark,
            rgb=not args.no_rgb,
            extra_fields=[] if args.extra_fields is None else args.extra_fields,
            color_scale=args.color_scale,
            use_process_pool=not args.disable_processpool,
            verbose=args.verbose,
        )
    except SrsInMissingException:
        print(
            "No SRS information in input files, you should specify it with --srs_in",
            file=sys.stderr,
        )
        sys.exit(EXIT_CODES.MISSING_SRS_IN_FILE.value)
