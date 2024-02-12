{ pkgs ? import <nixpkgs> { }, python }:
pkgs.mkShell rec {
  name = "py3dtiles: python ${python.version}";
  buildInputs = [
    python
  ];

  shellHook = ''
    set -h #remove "bash: hash: hashing disabled" warning !
    # https://nixos.org/manual/nixpkgs/stable/#python-setup.py-bdist_wheel-cannot-create-.whl
    SOURCE_DATE_EPOCH=$(date +%s)
    # Let's allow compiling stuff
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (with pkgs; [ zlib stdenv.cc.cc ])}":LD_LIBRARY_PATH;

    export VENVPATH=.venv${python.version}

    if ! [ -d $VENVPATH ]; then
      python -m venv $VENVPATH
    fi

    source $VENVPATH/bin/activate

    export TMPDIR=/tmp/pipcache

    if [ ! -x $VENVPATH/bin/py3dtiles ] > /dev/null 2>&1; then
      python -m pip install --cache-dir=$TMPDIR --upgrade pip
      python -m pip install --cache-dir="$TMPDIR" -e .\[postgres,las,ply,ifc,dev,doc,pack\]
      # keep this line after so that ipython deps doesn't conflict with other deps
      python -m pip install ipython debugpy
    fi
  '';
}
