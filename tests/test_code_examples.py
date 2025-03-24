import doctest
import glob
import shutil
from pathlib import Path

if __name__ == "__main__":
    try:
        num_of_attempted = 0
        num_of_failed = 0
        # With doctest.ELLIPSIS, an ellipsis marker (...) in the expected output can match any substring in the actual output.
        # Useful to match strings like "<py3dtiles.tileset.content.pnts.PntsHeader object at 0x7f73f5530d90>"
        print("TESTING ./docs/api.rst")
        test_result = doctest.testfile("../docs/api.rst", optionflags=doctest.ELLIPSIS)
        num_of_failed += test_result.failed
        num_of_attempted += test_result.attempted
        print("TESTING ./README.rst")
        test_result = doctest.testfile("../README.rst", optionflags=doctest.ELLIPSIS)
        num_of_failed += test_result.failed
        num_of_attempted += test_result.attempted

        # test python files
        for pyfile in glob.glob("./py3dtiles/**/**.py", recursive=True):
            print("TESTING", pyfile)
            test_result = doctest.testfile(
                pyfile, module_relative=False, optionflags=doctest.ELLIPSIS
            )
            num_of_failed += test_result.failed
            num_of_attempted += test_result.attempted

    finally:
        # Remove files created by the tested files.
        Path("./mymodel.b3dm").unlink(missing_ok=True)
        Path("./mypoints.pnts").unlink(missing_ok=True)
        shutil.rmtree("./3dtiles_output", ignore_errors=True)
        shutil.rmtree("./my3dtiles", ignore_errors=True)
        shutil.rmtree("./my3dtiles2", ignore_errors=True)

    print(f"Summary: {num_of_attempted} tests attempted, {num_of_failed} tests failed")
    exit(num_of_failed != 0)
