from pathlib import Path

from py3dtiles.exceptions import Py3dtilesException


class IfcInvalidFile(Py3dtilesException):
    """
    Thrown when there is no project in this IfcFile
    """

    def __init__(self, filename: Path, message: str):
        self.filename = filename
        self.message = message
        # pass the arguments to avoid B042 flake8-bugbear error
        super().__init__(filename, message)

    def __str__(self) -> str:
        return f"Ifc file {self.filename} invalid: {self.message}"
