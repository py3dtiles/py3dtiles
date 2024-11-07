import numpy as np

DTYPE_TO_COMPONENT_TYPE_MAPPING = {
    np.dtype("int8"): "BYTE",
    np.dtype("uint8"): "UNSIGNED_BYTE",
    np.dtype("int16"): "SHORT",
    np.dtype("uint16"): "UNSIGNED_SHORT",
    np.dtype("int32"): "INT",
    np.dtype("uint32"): "UNSIGNED_INT",
    np.dtype("float32"): "FLOAT",
    np.dtype("float64"): "DOUBLE",
}

COMPONENT_TYPE_NUMPY_MAPPING = {
    "BYTE": np.int8,
    "UNSIGNED_BYTE": np.uint8,
    "SHORT": np.int16,
    "UNSIGNED_SHORT": np.uint16,
    "INT": np.int32,
    "UNSIGNED_INT": np.uint32,
    "FLOAT": np.float32,
    "DOUBLE": np.float64,
}
