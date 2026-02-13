from collections.abc import Iterator, Sequence
from pathlib import Path

from py3dtiles.constants import SpecVersion
from py3dtiles.tilers.base_tiler.shared_metadata import SharedMetadata
from py3dtiles.tilers.base_tiler.tiler import Tiler
from py3dtiles.tilers.base_tiler.tiler_worker import TilerWorker
from py3dtiles.tileset.tile import Tile


def test_tiler_implementation() -> None:
    """
    This test aims at testing what is necessary to implement a Tiler, hence the
    fact we don't assert a lot of things. The fact that mypy passes and runtime
    execution succeeds is enough.
    """

    class Metadata(SharedMetadata):
        pass

    class TestTilerWorker(TilerWorker[Metadata]):
        def execute(
            self, command: bytes, content: list[bytes]
        ) -> Iterator[Sequence[bytes]]:
            yield [b"command"]

    class TestTiler(Tiler[Metadata]):
        def supports(self, file: Path) -> bool:
            return True

        def initialize(
            self, files: list[Path], working_dir: Path, out_folder: Path
        ) -> None:
            self.out_folder = out_folder
            self.shared_metadata = Metadata(
                spec_version=SpecVersion.V1_1, verbosity=1, out_folder=out_folder
            )

        def get_worker(self) -> TestTilerWorker:
            return TestTilerWorker(shared_metadata=self.shared_metadata)

        def get_tasks(self) -> Iterator[tuple[bytes, list[bytes]]]:
            yield b"foo", [b"bar"]

        def process_message(self, message_type: bytes, message: list[bytes]) -> None:
            pass

        def get_root_tile(self, use_process_pool: bool = True) -> Tile:
            return Tile()

    test_tiler = TestTiler()
    assert test_tiler is not None
