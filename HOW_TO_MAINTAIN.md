# How to release

- before doing anything, just check if the CI is still passing on main ;-)
- edit the CHANGELOG.md. The best way is to start with commitizen for that:
```bash
cz bump --changelog
```
If commit messages have been correctly crafted, this will auto-detect which
type of upgrade (patch, minor or major) is necessary. If not, use the parameter
`--increment`. it's a good moment to quickly check the logs to see if a breaking change may have been forgotten.
- then edit the changelog to make it more user readable. Especially, the `BREAKING
CHANGE` needs to be reviewed carefully and often to be rewritten, including
migration guide for instance.
- edit the version in [pyproject.toml](pyproject.toml) if not done by `cz bump`
- edit the version in [sonar-project.properties](sonar-project.properties) (field `sonar.projectVersion`)
- create a merge request with these changes. Don't push the tag created with `cz bump`
- once it is merged, create a tagged release on gitlab.
- wait for the execution of the automatic deploy jobs:
    - `pages`: will update the documentation
    - `docker-build`: will push the docker image on docker hub and gitlab registry
    - `publish-to-pypi`: will push the package to pypi.org

What to check after the release:

- the gitlab registry container
- the docker hub page
- the pypi page
- py3dtiles.org and the new tag documentation

# How to support or drop a new python version

In both case, update the following files:
- [pyproject.toml](pyproject.toml)
  - modify `requires-python`
  - add or remove the python version in `classifiers` list
- [sonar-project.properties](sonar-project.properties)
  - edit the `sonar.python.version` variable
- [flake.nix](flake.nix): add or remove the minor version number to the `supportedMinorVersions` variable.

For dropped python version:

- [.pre-commit-config.yaml](.pre-commit-config.yaml): increase the python version in the `args` value for the hook `pyupgrade` and the hook `black`
- [.gitlab-ci.yml](.gitlab-ci.yml):
    - remove the old version in the python version matrix for the `test` job
    - change the python version for the mypy job (always the oldest supported)
    - change the python version for the windows job (always the oldest supported)
- [Dockerfile](docker%2FDockerfile)
  - check if the version of python is still supported by py3dtiles and change it if needed (always use the oldest supported)
  - if the python version in the Dockerfile is changed, regenerate [requirements.txt](requirements.txt). Be careful not to add unnecessary packages
- [flake.nix](flake.nix): increase `defaultMinorVersion` (alway the oldest supported)

NOTE: as of 11-2024, the windows and Docker jobs don't yet use the oldest supported version, but they will eventually.

For newly supported python version:

- [.gitlab-ci.yml](.gitlab-ci.yml)
  - change the python docker image version of the jobs
  - add the new version in the python version matrix for the `test` job

Then run `pre-commit run --all-files`, so that pyupgrade can do its thing.

For the commit message, use `feat`, don't use `chore` because the breaking change won't be displayed by commitizen in this case.
