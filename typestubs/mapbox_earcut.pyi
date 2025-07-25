import numpy
import numpy.typing as npt

__version__: str

def triangulate_float32(
    arg0: npt.NDArray[numpy.float32], arg1: npt.NDArray[numpy.uint32]
) -> npt.NDArray[numpy.uint32]: ...
def triangulate_float64(
    arg0: npt.NDArray[numpy.float64], arg1: npt.NDArray[numpy.uint32]
) -> npt.NDArray[numpy.uint32]: ...
def triangulate_int32(
    arg0: npt.NDArray[numpy.int32], arg1: npt.NDArray[numpy.uint32]
) -> npt.NDArray[numpy.uint32]: ...
def triangulate_int64(
    arg0: npt.NDArray[numpy.int64], arg1: npt.NDArray[numpy.uint32]
) -> npt.NDArray[numpy.uint32]: ...
