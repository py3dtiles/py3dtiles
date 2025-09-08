{
  description = "py3dtiles' flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.05";
    flakeUtils = {
      type = "github";
      owner = "numtide";
      repo = "flake-utils";
      ref = "v1.0.0";
    };
  };

  outputs = { self, nixpkgs, flakeUtils }:
    flakeUtils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        supportedMinorVersions = [ "10" "11" "12" "13" ];
        defaultMinorVersion = "10";
      in
      rec {
        # generate one devshells per supported python version
        devShells =
          # map a [{ name = name1; value = value1;} { name = name2; value = value2; } ...] to an object of the form
          # { name1 = value1; name2 = value2; ... }
          pkgs.lib.listToAttrs
            # map the list of minor versions to a list of [{ name: "python3<minor>", value: <shell instance>}]
            (map
              # function that takes a minor version and returns:
              # { name: "python3<minor version>"; value = <the shell instance>; }
              (pythonVersion:
                let
                  # create the shell instance for the specific version
                  shell = import ./shell.nix { pkgs = pkgs; python = pkgs."python3${pythonVersion}"; };
                in
                {
                  name = "python3${pythonVersion}";
                  value = shell;
                })
              # all the supported minor versions
              supportedMinorVersions) // { default = devShells."python3${defaultMinorVersion}"; }
        ;
      }
    );
}
