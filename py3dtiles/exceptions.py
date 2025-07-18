"""
This module contains all the exceptions that py3dtiles generates. They all derive from :py:class:`Py3dtilesException`.
"""

from pathlib import Path


class Py3dtilesException(Exception):
    """
    All exceptions thrown by py3dtiles code derives this class.

    Client code that wishes to catch all py3dtiles exception can use `except Py3dtilesException`.
    """


class TilerException(Py3dtilesException):
    """
    This exception will be thrown when there is an issue during a tiling task.
    """


class TilerNotFoundException(Py3dtilesException):
    """
    Exception raised when the convert process cannot assign a file to a tiler
    """

    def __init__(self, files: list[Path]):
        self.files = files
        message = ", ".join([str(f) for f in self.files])
        super().__init__(f"Cannot find tiler for the following files: {message}")


class WorkerException(TilerException):
    """
    This exception will be thrown by the conversion code if one exception occurs inside a worker.
    """


class SrsInMissingException(Py3dtilesException):
    """
    This exception will be thrown when an input srs is required but not provided.
    """


class SrsInMixinException(Py3dtilesException):
    """
    This exception will be thrown when among all input files, there is a mix of input srs.
    """


class Invalid3dtilesError(Py3dtilesException):
    """
    This exception will be thrown if the 3d tile specification isn't respected.
    """


class InvalidPntsError(Invalid3dtilesError):
    """
    This exception will be thrown if the point cloud format isn't respected.
    """


class InvalidB3dmError(Invalid3dtilesError):
    """
    This exception will be thrown if the batched 3D model format isn't respected.
    """


class InvalidTilesetError(Invalid3dtilesError):
    """
    This exception will be thrown if the tileset format isn't respected.
    """


class BoundingVolumeMissingException(InvalidTilesetError):
    """
    This exception will be thrown when a bounding volume is needed but not present.
    """


class TileContentMissingException(Py3dtilesException):
    """
    This exception will be thrown when py3dtiles expects a tile with content loaded, but found a tile with a content uri, but no content.

    This exception is raised when py3dtiles cannot load it itself. Indeed, to load a tile content, py3dtiles needs the full url, and often tiles only have the relative url to the tileset path. If you encounter this exception in your own code, load the tile when you can with ``tile.get_or_fetch_content``.


    .. property:: content_uri
       :type: Path

       the uri of the content that wasn't loaded
    """

    def __init__(self, tile_content_uri: Path):
        super().__init__(
            f"Content with uri {tile_content_uri} wasn't loaded. Please load it first with `Tile.get_or_fetch_content`."
        )
        self.content_uri: Path = tile_content_uri
