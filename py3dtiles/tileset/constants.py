from typing import Any

import numpy as np

DTYPE_TO_COMPONENT_TYPE_MAPPING: dict[np.dtype[Any], str] = {
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
    value: key.type for key, value in DTYPE_TO_COMPONENT_TYPE_MAPPING.items()
}

# Warning: the literal types have changed with respect to the 1.0 version of 3DTiles standard.
DTYPE_TO_COMPONENT_TYPE_MAPPING_V11: dict[np.dtype[Any], str] = {
    np.dtype("int8"): "INT8",
    np.dtype("uint8"): "UINT8",
    np.dtype("int16"): "INT16",
    np.dtype("uint16"): "UINT16",
    np.dtype("int32"): "INT32",
    np.dtype("uint32"): "UINT32",
    np.dtype("int64"): "INT64",
    np.dtype("uint64"): "UINT64",
    np.dtype("float32"): "FLOAT32",
    np.dtype("float64"): "FLOAT64",
}

COMPONENT_TYPE_NUMPY_MAPPING_V11 = {
    value: key.type for key, value in DTYPE_TO_COMPONENT_TYPE_MAPPING_V11.items()
}
