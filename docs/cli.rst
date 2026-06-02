Command line usage
------------------

info
~~~~

Prints information about a tile.

The info sub-command outputs information from a 3D Tiles file in the ``.pnts`` or ``.b3dm`` format.

The following example shows how to retrieve a basic information about a point cloud tile's binary content:

.. code-block:: shell

    $ py3dtiles info tests/pointCloudRGB.pnts
    Tile Header
    -----------
    Magic Value:  pnts
    Version:  1
    Tile byte length:  15176
    Feature table json byte length:  148
    Feature table bin byte length:  15000

    Feature Table Header
    --------------------
    {'POSITION': {'byteOffset': 0}, 'RGB': {'byteOffset': 12000}, 'POINTS_LENGTH': 1000, 'RTC_CENTER': [1215012.8828876738, -4736313.051199594, 4081605.22126042]}

    First point
    -----------
    {'Z': -0.17107764, 'Red': 44, 'X': 2.19396, 'Y': 4.4896851, 'Green': 243, 'Blue': 209}


convert
~~~~~~~

Converts one or more input files to 3D Tiles.

This commands also support CRS reprojection of the points (see ``py3dtiles convert --help`` for all the options).

.. code-block:: shell

    py3dtiles convert mypointcloud.las --out /tmp/destination

For each format documentation and the assumptions made for them (CSV/XYZ format, PLY property names etc.), please see `the documentation of the corresponding reader <./api/py3dtiles.reader.html>`_.


merge
~~~~~

Creates a meta-tileset from tilesets.

The merge feature is a special use case: it generates a `meta-tileset` from a group of existing tilesets. A meta-tileset simply references other tilesets without directly referencing tiles.

For example: with 6 input LAS files (``A.las``, ``B.las``, up to ``F.las``), there are 2 ways to vizualize them all in a 3D Tiles viewer:

* run ``py3dtiles convert A.las B.las ... F.las`` to generate a single tileset from the 6 LAS files, then diplay the resulting tileset,
* or run ``py3dtiles convert A.las``, then ``py3dtiles convert B.las``, ... and then run ``py3dtiles merge`` to generate one tileset per input file, then one meta-tileset that references the 6 tilesets.

The second approach makes it possible to update a subset of pointcloud easily, without having to rebuild every tile. For example, if ``B.las`` has been updated, the first approache will re-generate the entire tileset from scratch, while with the second approach, only the tilesets for `B.las` and the meta-tilsets have to be rebuilt.


export
~~~~~~

Two export modes are available, the database export or the directory export.
They both transform all the geometries provided in .b3dm files, along with a
tileset.json file which organizes them.

The directory export will use all the .wkb files in the provided directory.

Warning: the coordinates are read as floats, not doubles. Make sure to offset
the coordinates beforehand to reduce their size. Afterwards, you can indicate
in the command line the offset that needs to be applied to the tileset so it is
correctly placed. Usage example:

.. code-block:: shell

    $ py3dtiles export -d my_directory -o 10000 10000 0

The database export requires a connection info string, the name of the table and its
column that contains the geometry and (optionally) the name of the column that contains
the object's ID. Usage example:

.. code-block:: shell

    $ py3dtiles export -D "dbname=mydb user=me host=localhost port=5432" -t table -c geometry_column -i id


view
~~~~

Opens a web viewer to visualize a tileset.

.. code-block:: shell

    $ py3dtiles view 3dtiles/tileset.json

This command will launch a local webserver and opens an online demo pointing to that local URL.
