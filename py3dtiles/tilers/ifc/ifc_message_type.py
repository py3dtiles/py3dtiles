from enum import Enum


class IfcTilerMessage(Enum):
    READ_FILE = b"read_file"
    WRITE_TILE = b"write_tile"


class IfcWorkerMessage(Enum):
    TILE_PARSED = b"tile_parsed"  # sent when anybody wants something to get processed
    TILE_READY = b"tile_ready"  # sent when a b3dm has been written to disk and a tile has been created
    FILE_READ = b"file_read"  # sent when a file has been fully read
    METADATA_READ = b"metadata_read"  # sent when we have the metadata about a file
