from pathlib import Path

from py3dtiles.exceptions import Py3dtilesException


class IfcInvalidFile(Py3dtilesException):
    """
    Thrown when there is no project in this IfcFile
    """

    def __init__(self, filename: Path, message: str):
        self.filename = filename
        super().__init__(f"Ifc file {self.filename} invalid: {message}")
