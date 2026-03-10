import numpy as np
import pytest

from py3dtiles.exceptions import InvalidIdentifierException
from py3dtiles.tileset import metadata


@pytest.mark.parametrize(
    "char,first_char,expected_validity",
    [
        ("_", False, True),
        ("_", True, True),
        ("a", False, True),
        ("b", True, True),
        ("B", False, True),
        ("A", True, True),
        ("1", False, True),
        ("2", True, False),
        ("-", False, False),
        ("-", True, False),
    ],
)
def test_is_identifier_character_valid(
    char: str, first_char: bool, expected_validity: bool
) -> None:
    """Check if a character is valid within a metadata identifier.

    According to the standard, the identifier should accept lowercase and uppercase letters, as
    well as underscores, and figures. The figures are not accepted at the first position.

    """
    assert metadata.is_identifier_character_valid(char, first_char) == expected_validity


@pytest.mark.parametrize(
    "identifier,expected_validity",
    [
        ("foo_", True),
        ("_FOO", True),
        ("a_b_c_122", True),
        ("B1A2C3", True),
        ("_Ba123", True),
        ("1foo", False),
        ("1_foo", False),
        ("foo-1-2-", False),
        ("=FOObar", False),
    ],
)
def test_check_identifier_validity(identifier: str, expected_validity: bool) -> None:
    """Check the metadata identifier validity.

    The identifier check returns the identifier itself if it is valid. Otherwise an error is
    raised.

    """
    if expected_validity:
        assert metadata.check_identifier_validity(identifier) == identifier
    else:
        with pytest.raises(
            InvalidIdentifierException, match="does not respect the naming convention"
        ):
            _ = metadata.check_identifier_validity(identifier)


def test_metadata_enum() -> None:
    """Test the MetadataEnum class."""
    with pytest.raises(
        InvalidIdentifierException, match="does not respect the naming convention"
    ):
        _ = metadata.MetadataEnum(
            "enum-with-forbidden-characters", values={"foo": np.int8(1)}
        )
    metadata_enum = metadata.MetadataEnum(
        "_enum",
        description="A dummy enum.",
        values={"foo": np.int8(1), "bar": np.int8(22), "wiz": np.int16(333)},
    )
    assert metadata_enum.to_json() == {
        "description": "A dummy enum.",
        "values": [
            {"name": "foo", "value": np.int8(1)},
            {"name": "bar", "value": np.int8(22)},
            {"name": "wiz", "value": np.int16(333)},
        ],
    }


def test_metadata_property() -> None:
    """Test the metadata property classes.

    SimpleMetadataProperty, EnumMetadataProperty and CompositeMetadataProperty inheritates from a
    generic MetadataProperty moter class.

    """
    with pytest.raises(
        InvalidIdentifierException, match="does not respect the naming convention"
    ):
        _ = metadata.SimpleMetadataProperty(
            "prop-with-forbidden-characters", property_type="STRING"
        )
    met_property = metadata.CompositeMetadataProperty(
        "prop", "VEC3", component_type="INT16"
    )
    assert met_property.to_json() == {
        "type": "VEC3",
        "componentType": "INT16",
    }
    met_property = metadata.CompositeMetadataProperty(
        "prop", "VEC3", name="test-property", component_type="INT16"
    )
    assert met_property.to_json() == {
        "type": "VEC3",
        "name": "test-property",
        "componentType": "INT16",
    }
    met_property = metadata.CompositeMetadataProperty(
        "prop", "VEC3", component_type="INT16", nodata=np.int16(-9999)
    )
    assert met_property.to_json() == {
        "type": "VEC3",
        "componentType": "INT16",
        "noData": np.int16(-9999),
    }


def test_metadata_class() -> None:
    """Test the MetadataClass class."""
    with pytest.raises(
        InvalidIdentifierException, match="does not respect the naming convention"
    ):
        _ = metadata.MetadataClass("cls-with-forbidden-characters")
    met_class = metadata.MetadataClass("CLS0")
    assert len(met_class.properties) == 0
    assert met_class.to_json() == {}
    met_property = metadata.CompositeMetadataProperty(
        "prop", "VEC3", component_type="INT16"
    )
    met_class.add_property(met_property)
    assert len(met_class.properties) == 1
    assert met_class.to_json() == {
        "properties": {
            "prop": {
                "type": "VEC3",
                "componentType": "INT16",
            }
        }
    }


def test_metadata_schema() -> None:
    """Test the MetadataSchema class."""
    with pytest.raises(
        InvalidIdentifierException, match="does not respect the naming convention"
    ):
        _ = metadata.MetadataSchema("schema-with-forbidden-characters")
    met_schema = metadata.MetadataSchema("SCH0", name="Schema0")
    assert met_schema.to_json() == {"name": "Schema0"}
    met_class = metadata.MetadataClass("CLS0")
    met_property = metadata.EnumMetadataProperty("prop", enum_type="testEnum")
    met_class.add_property(met_property)
    with pytest.raises(KeyError):
        met_schema.add_class(met_class)
    met_enum = metadata.MetadataEnum(
        "testEnum", values={"test1": np.int8(1), "test2": np.int8(2)}
    )
    met_schema.add_enum(met_enum)
    met_schema.add_class(met_class)
    assert met_schema.to_json() == {
        "name": "Schema0",
        "classes": {
            "CLS0": {
                "properties": {
                    "prop": {
                        "type": "ENUM",
                        "enumType": "testEnum",
                    }
                }
            }
        },
        "enums": {
            "testEnum": {
                "values": [
                    {"name": "test1", "value": np.int8(1)},
                    {"name": "test2", "value": np.int8(2)},
                ]
            }
        },
    }
