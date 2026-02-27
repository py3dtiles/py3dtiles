from enum import Enum


class PointManagerMessage(Enum):
    READ_FILE = b"read_file"
    WRITE_CONTENT = b"write_content"
    PROCESS_JOBS = b"process_job"


class PointWorkerMessageType(Enum):
    READ = b"read"
    PROCESSED = b"processed"
    CONTENT_WRITTEN = b"content_written"
    NEW_TASK = b"new_task"
