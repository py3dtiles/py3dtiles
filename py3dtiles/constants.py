from enum import Enum
from multiprocessing import cpu_count

import psutil


class EXIT_CODES(Enum):
    SUCCESS = 0
    MISSING_DEPS = 1
    MISSING_ARGS = 2
    MISSING_SRS_IN_FILE = 10


class SpecVersion(Enum):
    V1_0 = "1.0"
    V1_1 = "1.1"


TOTAL_MEMORY_MB = int(psutil.virtual_memory().total / (1024 * 1024))
DEFAULT_CACHE_SIZE = int(TOTAL_MEMORY_MB / 10)
CPU_COUNT = cpu_count()
