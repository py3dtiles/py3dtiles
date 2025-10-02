# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
current_version = $(shell git tag --list --sort=v:refname | grep -E "^v[0-9]+\.[0-9]+\.[0-9]+$$" | tail -n1)

"$(BUILDDIR)":
	mkdir -p "$(BUILDDIR)"

apidoc: $(BUILDDIR)/version_switcher.json
	sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)"

lastapidoc: "$(BUILDDIR)"
	sphinx-build -A current_version=HEAD -b html . "$(BUILDDIR)/main"


static-site: "$(BUILDDIR)"
	mkdir -p "$(BUILDDIR)"
	cp -rv "../site/index.html" "$(BUILDDIR)/"
	cp -rv "../site/css" "$(BUILDDIR)/"
	cp -rv "../graphics/" "$(BUILDDIR)/"
	# NOTE the grep must reflect the smv_tag_whitelist and smv_released_pattern in docs/conf.py
	sed -i "s/VERSION/$(current_version)/g" "$(BUILDDIR)/index.html"

site: apidoc static-site

serve:
	python3 -m http.server --bind 127.0.0.1 --directory "$(BUILDDIR)"

$(BUILDDIR)/version_switcher.json: "$(BUILDDIR)"
	python ./versions_switcher_builder.py "$(BUILDDIR)"

clean:
	rm -rf $(BUILDDIR)
