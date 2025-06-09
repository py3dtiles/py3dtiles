import os
import pickle
import struct
import time
from collections.abc import Iterator, Sequence
from pathlib import PurePath
from typing import TYPE_CHECKING

import lz4.frame as gzip

from py3dtiles.tilers.base_tiler import TilerWorker
from py3dtiles.tilers.point.node import DummyNode
from py3dtiles.utils import READER_MAP

from .node import NodeCatalog, NodeProcess
from .pnts import pnts_writer
from .point_message_type import PointManagerMessage, PointWorkerMessageType
from .point_shared_metadata import PointSharedMetadata

if TYPE_CHECKING:
    from py3dtiles.tilers.point.node.node import DummyNodeDictType


class PointTilerWorker(TilerWorker[PointSharedMetadata]):
    def execute(
        self, command: bytes, content: list[bytes]
    ) -> Iterator[Sequence[bytes]]:
        if command == PointManagerMessage.READ_FILE.value:
            yield from self.execute_read_file(content)
        elif command == PointManagerMessage.PROCESS_JOBS.value:
            yield from self.execute_process_jobs(content)
        elif command == PointManagerMessage.WRITE_PNTS.value:
            yield from self.execute_write_pnts(content[1], content[0])
        else:
            raise NotImplementedError(f"Unknown command {command!r}")

    def execute_read_file(self, content: list[bytes]) -> Iterator[Sequence[bytes]]:
        parameters = pickle.loads(content[0])

        extension = PurePath(parameters["filename"]).suffix.lower()
        if extension in READER_MAP:
            reader = READER_MAP[extension]
        else:
            raise ValueError(
                f"The file with {extension} extension can't be read, "
                f"the available extensions are: {READER_MAP.keys()}"
            )

        reader_gen = reader.run(
            parameters["filename"],
            parameters["offset_scale"],
            parameters["portion"],
            self.shared_metadata.transformer,
            self.shared_metadata.color_scale,
            self.shared_metadata.write_rgb,
            self.shared_metadata.extra_fields_to_include,
        )
        for coords, colors, extra_fields in reader_gen:
            yield [
                PointWorkerMessageType.NEW_TASK.value,
                b"",
                pickle.dumps(
                    {"xyz": coords, "rgb": colors, "extra_fields": extra_fields}
                ),
                struct.pack(">I", len(coords)),
            ]

        yield [PointWorkerMessageType.READ.value]

    def execute_write_pnts(
        self, content: bytes, node_name: bytes
    ) -> Iterator[Sequence[bytes]]:
        # we can safely write the .pnts file
        if len(content) > 0:
            root = pickle.loads(gzip.decompress(content))
            total = 0
            for name in root:
                node_data: DummyNodeDictType = pickle.loads(root[name])
                node = DummyNode(node_data)
                total += pnts_writer.node_to_pnts(
                    name, node, self.shared_metadata.out_folder
                )
            yield [
                PointWorkerMessageType.PNTS_WRITTEN.value,
                struct.pack(">I", total),
                node_name,
            ]

    def execute_process_jobs(self, content: list[bytes]) -> Iterator[Sequence[bytes]]:
        begin = time.time()
        log_enabled = self.shared_metadata.verbosity >= 2
        if log_enabled:
            log_filename = f"py3dtiles-{os.getpid()}.log"
            log_file = open(log_filename, "a")
        else:
            log_file = None

        i = 0
        while i < len(content):
            name = content[i]
            node = content[i + 1]
            count = struct.unpack(">I", content[i + 2])[0]
            tasks = content[i + 3 : i + 3 + count]
            i += 3 + count

            node_catalog = NodeCatalog(
                node,
                name,
                self.shared_metadata.root_aabb,
                self.shared_metadata.root_spacing,
                self.shared_metadata.write_rgb,
                self.shared_metadata.extra_fields_to_include,
            )

            node_process = NodeProcess(
                node_catalog,
                self.shared_metadata.scale[0],
                name,
                tasks,
                begin,
                log_file,
            )
            for proc_name, proc_data, proc_point_count in node_process.run():
                yield [
                    PointWorkerMessageType.NEW_TASK.value,
                    proc_name,
                    proc_data,
                    struct.pack(">I", proc_point_count),
                ]

            if log_enabled:
                print(f"save on disk {name!r} [{time.time() - begin}]", file=log_file)

            # save node state on disk
            if len(name) > 0:
                data = node_catalog.dump(name, node_process.infer_depth_from_name() - 1)
            else:
                data = b""

            if log_enabled:
                print(f"saved on disk [{time.time() - begin}]", file=log_file)

            yield [
                PointWorkerMessageType.PROCESSED.value,
                pickle.dumps(
                    {
                        "name": name,
                        "total": node_process.total_point_count,
                        "data": data,
                    }
                ),
            ]

        if log_enabled:
            print(
                "[<] return result [{} sec] [{}]".format(
                    round(time.time() - begin, 2), time.time() - begin
                ),
                file=log_file,
                flush=True,
            )
            if log_file is not None:
                log_file.close()
