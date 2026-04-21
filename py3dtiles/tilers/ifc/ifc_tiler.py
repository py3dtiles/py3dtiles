from pathlib import Path

from pyproj import CRS

from py3dtiles.constants import CPU_COUNT, DEFAULT_CACHE_SIZE, SpecVersion
from py3dtiles.tilers.geometry.geometry_tiler import GeometryTiler

from .ifc_tiler_worker import IfcTilerWorker


class IfcTiler(GeometryTiler):
    name = "ifc"

    def __init__(
        self,
        crs_in: CRS | None = None,
        crs_out: CRS | None = None,
        force_crs_in: bool = False,
        pyproj_always_xy: bool = False,
        cache_size: int = DEFAULT_CACHE_SIZE,
        verbosity: int = 0,
        number_of_jobs: int = CPU_COUNT,
        spec_version: SpecVersion = SpecVersion.V1_0,
    ):
        super().__init__(
            crs_in=crs_in,
            crs_out=crs_out,
            force_crs_in=force_crs_in,
            pyproj_always_xy=pyproj_always_xy,
            cache_size=cache_size,
            verbosity=verbosity,
            number_of_jobs=number_of_jobs,
            spec_version=spec_version,
        )

        # NOTE: crs_in is always None currently because we don't yet support
        # crs parsing for individual ifc files.
        self.force_crs_in = True

    def supports(self, file: Path) -> bool:
        # ifcopenshell advertises to support xml, json, hdf5, but the only files I manage to work with are STEP files
        return file.suffix.lower() in [".ifc"]

    def get_worker(self) -> IfcTilerWorker:
        return IfcTilerWorker(self.shared_metadata)
