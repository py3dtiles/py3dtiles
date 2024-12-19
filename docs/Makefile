# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
current_version = $(shell git tag --list | grep -E "^v[0-9]+\.[0-9]+\.[0-9]+$$" | tail -n1)

apidoc:
	sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)"

static-site:
	mkdir -p "$(BUILDDIR)"
	cp -rv "$(SOURCEDIR)/index.html" "$(BUILDDIR)/index.html"
	cp -rv "$(SOURCEDIR)/../graphics/" "$(BUILDDIR)/"
	# NOTE the grep must reflect the smv_tag_whitelist and smv_released_pattern in docs/conf.py
	sed -i "s/VERSION/$(current_version)/g" "$(BUILDDIR)/index.html"

site: apidoc static-site

serve:
	python3 -m http.server --directory "$(BUILDDIR)"
