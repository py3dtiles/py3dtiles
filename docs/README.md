# Doc of py3dtiles

The documentation is generated with sphinx.

## How to generate the doc?

### ApiDoc

First install dependencies in a python3 virtualenv:

```
pip install -e .
pip install -e .[doc]
```
The principle is the following:

- `sphinx-multiversion` checkouts each tags and executes `sphinx-build` on it.
- `spinxcontrib-apidoc` plugs `sphinx-apidoc` to each execution of `sphinx-build`, so that the api doc of the website is auto-generated.

To regenerate the doc for one version, from this folder:

```
sphinx-build -A current_version=HEAD -A "versions=[main]" -b html . ../_build/html
```
To generate the full doc as gitlab does it:

```
# this task just call sphinx-multiversion really
make apidoc
```

### Full website

A `Makefile` is used to generate the full site web:

```sh
make apidoc # will generate the apidoc, executes `sphinx-multiversion`
make static-site # will generate just the part that is not touched by sphinx
make site # generates the whole site, apidoc and static pages
make serve # convenience function over python3 -m http.server to serve the generated pages

```
You can customize the build dir with the BUILDDIR env variable.
