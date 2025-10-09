import doctest
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest


def clean_test_artifacts() -> None:
    """Remove the output files and folders produced by API doc code examples."""
    # Remove files created by the tested files.
    Path("./mymodel.b3dm").unlink(missing_ok=True)
    Path("./mypoints.pnts").unlink(missing_ok=True)
    Path("./test.glb").unlink(missing_ok=True)
    shutil.rmtree("./3dtiles_output", ignore_errors=True)
    shutil.rmtree("./my3dtiles", ignore_errors=True)
    shutil.rmtree("./my3dtiles2", ignore_errors=True)


def identify_module(filepath: Path) -> str | None:
    """Build an identifier for the provided filepath, given that it is a module stored into the
    py3dtiles library source code folder.

    Parameters
    ----------
    filepath: pathlib.Path
        Path of a file of interest

    Returns
    -------
    str or None
        Simplified path towards the file, starting from the py3dtiles main folder. Returns None if
        the file is not in the py3dtiles source code folder

    """
    source_code_dir = Path.cwd() / "py3dtiles"
    if not filepath.is_relative_to(source_code_dir):
        return None
    return str(filepath.relative_to(source_code_dir))


@pytest.fixture()
def cleanup_api_rst_files() -> Iterator[None]:
    yield
    clean_test_artifacts()


def test_api_rst_file(cleanup_api_rst_files: Callable[[], Iterator[None]]) -> None:
    test_result = doctest.testfile("../docs/api.rst", optionflags=doctest.ELLIPSIS)
    assert test_result.failed == 0


def test_readme_rst_file() -> None:
    test_result = doctest.testfile("../README.rst", optionflags=doctest.ELLIPSIS)
    assert test_result.failed == 0


@pytest.mark.doctest
@pytest.mark.parametrize(
    "python_module",
    Path.cwd().glob("py3dtiles/**/*.py"),
    ids=identify_module,
)
def test_python_docstring(python_module: str) -> None:
    test_result = doctest.testfile(
        python_module, module_relative=False, optionflags=doctest.ELLIPSIS
    )
    assert test_result.failed == 0
