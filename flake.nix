{
  inputs.flake-utils.url = "flake-utils";
  inputs.nixpkgs.url = "nixpkgs";
  inputs.poetry2nix = {
    url = "poetry2nix";
    inputs = {
      flake-utils.follows = "flake-utils";
      nixpkgs.follows = "nixpkgs";
      systems.follows = "flake-utils/systems";
    };
  };

  outputs = {
    self,
    flake-utils,
    nixpkgs,
    poetry2nix,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages."${system}";
      poetry-lib = poetry2nix.lib.mkPoetry2Nix {inherit pkgs;};

      python = pkgs.python312;

      python-with-deps = poetry-lib.mkPoetryEnv {
        python = python;
        projectDir = ./.;
        preferWheels = true;

        overrides = poetry-lib.overrides.withDefaults (final: prev: {
          pintax = prev.pintax.overridePythonAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ [final.poetry-core];
          });
        });

        extraPackages = ps: [
          ps.tkinter
        ];
      };
    in {
      packages.python = python;
      packages.env = python-with-deps;
      legacyPackages.pkgs = pkgs;
      legacyPackages.pypkgs = python-with-deps.pkgs;
    });
}
