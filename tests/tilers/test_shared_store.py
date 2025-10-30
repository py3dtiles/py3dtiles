import shutil
import unittest
from pathlib import Path

from py3dtiles.tilers.shared_store import SharedStore


class TestSharedStore(unittest.TestCase):
    TMP_DIR = Path("tmp/")

    def test_remove_oldest_nodes(self) -> None:
        shared_store = SharedStore(TestSharedStore.TMP_DIR)

        self.assertEqual(len(shared_store.data), 0)
        self.assertEqual(len(shared_store.metadata), 0)

        shared_store.put(b"0", b"11111111")

        self.assertEqual(len(shared_store.data), 1)
        self.assertEqual(len(shared_store.metadata), 1)

        shared_store.remove_oldest_nodes()

        self.assertEqual(len(shared_store.data), 0)
        self.assertEqual(len(shared_store.metadata), 0)

        shutil.rmtree(self.TMP_DIR)
