Install
-------

From pypi
~~~~~~~~~~~~

`py3dtiles` is published on pypi.org.

.. code-block:: shell

    pip install py3dtiles

Please read the section ":ref:`File formats support`" next.

From sources
~~~~~~~~~~~~

To use py3dtiles from sources:

.. code-block:: shell

    $ apt install git python3 python3-pip virtualenv
    $ git clone git@gitlab.com:py3dtiles/py3dtiles.git
    $ cd py3dtiles
    $ virtualenv -p python3 venv
    $ . venv/bin/activate
    (venv)$ pip install .

You might need to install specific format dependencies as described in the section "From pypi".

If you want to run unit tests:

.. code-block:: shell

    (venv)$ pip install -e .[dev]
    (venv)$ pytest

Please read the section ":ref:`File formats support`" next.

.. _File formats support:

File formats support
~~~~~~~~~~~~~~~~~~~~

By default, no specific format dependencies are installed. You should either install them separately, or use our `extra_requires` sections:

.. code-block:: shell

    # las support
    pip install py3dtiles[las]
    # ply
    pip install py3dtiles[ply]
    # postgres
    pip install py3dtiles[postgres]
    # ifc
    pip install py3dtiles[ifc]
    # everything at once
    pip install py3dtiles[postgres,ply,las,ifc]


To support laz files you need an external library and a laz backend for
laspy, see `this link <https://laspy.readthedocs.io/en/latest/installation.html#pip>`_. Short answer, for laszip, you need to follow these steps:

.. code-block:: shell

  $ # install liblaszip, for instance on ubuntu 22.04
  $ apt-get install -y liblaszip8

  $ # Install with LAZ support via laszip
  $ pip install laspy[laszip]


If you don't need waveform support, [laz-rs](https://github.com/laz-rs/laz-rs) is also a good option.

From docker
~~~~~~~~~~~~

We currently publish docker images on `docker hub <https://hub.docker.com/r/py3dtiles/py3dtiles>`_ and `gitlab registry <https://gitlab.com/py3dtiles/py3dtiles/container_registry>`_.

.. code-block:: shell

    docker run --rm py3dtiles/py3dtiles:<version> --help
    # or
    docker run --rm registry.gitlab.com/py3dtiles/py3dtiles:<version> --help

NOTE:

- the `--mount` option is necessary for docker to read your source data and to write the result. For instance, you can add `-mount type=bind,source="$(pwd)"/data,target=/app/data/` to your `docker run` command. This allows the docker container to read and write files in `./data`.
- If your uid is different from 1000, you should add `--volume /etc/passwd:/etc/passwd:ro --volume /etc/group:/etc/group:ro --user $(id -u):$(id -g)` to your `docker run` command
