#!/usr/bin/env python3
import json
import re
import subprocess
import sys
from pathlib import Path

# keep that in sync with smv_released_pattern in ./conf.py
released_pattern = r"^v(\d+\.)?(\d+\.)?(\*|\d+)$"

excluded_versions = ["v0.0.9", "v1.0.2"]


def main(builddir: Path):
    result = subprocess.run(
        ["git", "tag", "--list", "--sort=v:refname"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(
            f"git tag --list returned {result.returncode} with an error: {result.stderr}"
        )
    version_list_it = filter(
        lambda v: re.match(released_pattern, v) and v not in excluded_versions,
        result.stdout.split("\n"),
    )
    versions_dict = [{"name": v, "version": v, "url": f"/{v}"} for v in version_list_it]
    versions_dict.append({"name": "main", "version": "main", "url": "/main"})
    with open(builddir / "version_switcher.json", "w") as f:
        json.dump(versions_dict, f)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
