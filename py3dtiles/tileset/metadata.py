"""Description of the 3DTiles metadata, as denoted in
https://docs.ogc.org/cs/22-025r4/22-025r4.html#toc40.

"""

import string
from typing import Any, Literal, Union

import numpy as np
import numpy.typing as npt

from py3dtiles.exceptions import InvalidIdentifierException

MetadataComponentTypeLiteral = Literal[
    "INT8",
    "UINT8",
    "INT16",
    "UINT16",
    "INT32",
    "UINT32",
    "INT64",
    "UINT64",
    "FLOAT32",
    "FLOAT64",
]

MetadataNumpyComponentType = Union[
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
    np.float32,
    np.float64,
]

MetadataNumpyEnumType = Union[
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
]

MetadataPropertyTypeLiteral = Literal[
    "SCALAR",
    "VEC2",
    "VEC3",
    "VEC4",
    "MAT2",
    "MAT3",
    "MAT4",
    "STRING",
    "BOOLEAN",
    "ENUM",
]


def is_identifier_character_valid(char: str, first_char: bool = True) -> bool:
    """Check if an identifier character is valid.

    A metadata identifier should contains (lowercase or uppercase) letters, underscores

    :param char: a character within a metadata identifier.
    :param first_char: True if this is the starting character, False otherwise.

    """
    return (char == "_" or char in string.ascii_letters) or (
        char in string.digits and not first_char
    )


def check_identifier_validity(identifier: str) -> str:
    """Check if an identifier is valid relatively to the 3DTiles standard.

    Identifiers are strings that match the regular expression ^[a-zA-Z_][a-zA-Z0-9_]*$: Strings
    that consist of upper- or lowercase letters, digits, or underscores, starting with either a
    letter or an underscore.

    :param identifier: metadata identifier.
    :returns: the identifier itself, if it is valid
    :raises InvalidIdentifierException: if the provided identifier does not respect the naming
    convention.

    """
    if not all(
        is_identifier_character_valid(char, idx == 0)
        for idx, char in enumerate(identifier)
    ):
        raise InvalidIdentifierException(
            f"The identifier '{identifier}' does not respect the naming convention."
        )
    return identifier


class MetadataEnum:
    """Define a 3DTile metadata enum."""

    def __init__(
        self,
        identifier: str,
        values: dict[str, MetadataNumpyEnumType],
        name: str | None = None,
        description: str | None = None,
    ):
        self.identifier = check_identifier_validity(identifier)
        self.values = values.copy()
        self.name = name
        self.description = description

    def to_json(self) -> dict[str, Any]:
        """Convert the metadata enum to a JSON-like dictionary.

        :returns: Dictionary version of the metadata enum.
        """
        enum_as_json: dict[str, Any] = {}
        if self.name is not None and self.name:
            enum_as_json["name"] = self.name
        if self.description is not None and self.description:
            enum_as_json["description"] = self.description
        enum_as_json["values"] = [
            {"name": key, "value": value} for key, value in self.values.items()
        ]
        return enum_as_json


class MetadataProperty:
    """Define a 3DTiles metadata property."""

    def __init__(
        self,
        identifier: str,
        property_type: MetadataPropertyTypeLiteral,
        name: str | None = None,
        description: str | None = None,
        component_type: MetadataComponentTypeLiteral | None = None,
        enum_type: str | None = None,
        array: bool = False,
        required: bool = False,
        offset: (
            MetadataNumpyComponentType | npt.NDArray[MetadataNumpyComponentType] | None
        ) = None,
        scale: (
            MetadataNumpyComponentType | npt.NDArray[MetadataNumpyComponentType] | None
        ) = None,
        minimum: (
            MetadataNumpyComponentType | npt.NDArray[MetadataNumpyComponentType] | None
        ) = None,
        maximum: (
            MetadataNumpyComponentType | npt.NDArray[MetadataNumpyComponentType] | None
        ) = None,
        nodata: (
            MetadataNumpyComponentType | npt.NDArray[MetadataNumpyComponentType] | None
        ) = None,
        default: (
            MetadataNumpyComponentType | npt.NDArray[MetadataNumpyComponentType] | None
        ) = None,
    ):
        self.identifier = check_identifier_validity(identifier)
        self.property_type = property_type
        self.name = name
        self.description = description
        if component_type is None and self.property_type not in (
            "STRING",
            "BOOLEAN",
            "ENUM",
        ):
            raise TypeError(
                "Missing 1 required positional argument: 'component_type'. "
                f"Hint: component_type is required when property type is {self.property_type}!"
            )
        self.component_type = component_type
        if enum_type is None and self.property_type == "ENUM":
            raise TypeError(
                "Missing 1 required positional argument: 'enum_type'. "
                f"Hint: enum_type is required when property type is {self.property_type}!"
            )
        self.enum_type = enum_type
        self.array = array
        self.required = required
        self.offset = offset
        self.scale = scale
        self.minimum = minimum
        self.maximum = maximum
        self.nodata = nodata
        self.default = default

    def to_json(self) -> dict[str, Any]:
        """Convert the property to a JSON-like dictionary.

        :returns: Dictionary version of the metadata property.
        """
        property_as_json: dict[str, Any] = {"type": self.property_type}
        if self.property_type in (
            "SCALAR",
            "VEC2",
            "VEC3",
            "VEC4",
            "MAT2",
            "MAT3",
            "MAT4",
        ):
            property_as_json["componentType"] = self.component_type
        elif self.property_type == "ENUM":
            property_as_json["enumType"] = self.enum_type
        property_as_json.update(
            {
                "noData" if key == "nodata" else key: value
                for key, value in vars(self).items()
                if key
                not in ("identifier", "property_type", "component_type", "enum_type")
                and value is not None
                and value
            }
        )
        return property_as_json


class MetadataClass:
    """Define a 3DTiles metadata class, composed of enums and properties."""

    def __init__(
        self, identifier: str, name: str | None = None, description: str | None = None
    ):
        self.identifier = check_identifier_validity(identifier)
        self.name = name
        self.description = description
        self.properties: dict[str, MetadataProperty] = {}

    def add_property(self, new_property: MetadataProperty) -> None:
        """Add a new property to the metadata class.

        :param new_property: Property to add.
        """
        self.properties[new_property.identifier] = new_property

    def to_json(self) -> dict[str, Any]:
        """Convert the metadata class to a JSON-like dictionary.

        :returns: Dictionary version of the metadata class.
        """
        class_as_json: dict[str, Any] = {}
        if self.name is not None and self.name:
            class_as_json["name"] = self.name
        if self.description is not None and self.description:
            class_as_json["description"] = self.description
        if len(self.properties) > 0:
            class_as_json["properties"] = {
                property_id: class_property.to_json()
                for property_id, class_property in self.properties.items()
            }
        return class_as_json


class MetadataSchema:
    """Contient un ensemble de classes et enums."""

    def __init__(
        self,
        identifier: str,
        name: str | None = None,
        version: str | None = None,
        description: str | None = None,
    ):
        self.identifier = check_identifier_validity(identifier)
        self.name = name
        self.version = version
        self.description = description
        self.enums: dict[str, MetadataEnum] = {}
        self.classes: dict[str, MetadataClass] = {}

    def add_enum(self, enum: MetadataEnum) -> None:
        self.enums[enum.identifier] = enum

    def add_class(self, cls: MetadataClass) -> None:
        for class_property in cls.properties.values():
            enum = class_property.enum_type
            if enum is not None and enum not in self.enums.keys():
                raise KeyError(
                    f"The {class_property.identifier} property in the provided class "
                    f"uses an unknown enum types ({enum})."
                )
        self.classes[cls.identifier] = cls

    def to_json(self) -> dict[str, Any]:
        schema_as_json: dict[str, Any] = {}
        if self.name is not None and self.name:
            schema_as_json["name"] = self.name
        if self.description is not None and self.description:
            schema_as_json["description"] = self.description
        if self.version is not None and self.version:
            schema_as_json["version"] = self.version
        if len(self.classes) > 0:
            schema_as_json["classes"] = {
                class_id: schema_class.to_json()
                for class_id, schema_class in self.classes.items()
            }
        if len(self.enums) > 0:
            schema_as_json["enums"] = {
                enum_id: schema_enum.to_json()
                for enum_id, schema_enum in self.enums.items()
            }
        return schema_as_json
