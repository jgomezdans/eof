"""Lazy, xarray-based access to Sentinel-2 and ECOSTRESS over an AOI.

    import eof.xr as eox

    data = eox.open_aoi("field.geojson", "2022-06-01", "2022-08-31")
    data["s2"]           # Sentinel-2 L2A surface reflectance
    data["ecostress"]    # ECOSTRESS L1CT TOA radiance (bands 2, 4, 5)
    data["ecostress_lste"]  # ECOSTRESS L2T land surface temperature

Everything is dask-backed; nothing is downloaded until ``.compute()``.
Use :func:`to_zarr_cache` to keep a local copy.
"""

import rioxarray  # noqa: F401  (registers the .rio accessor)

from eof.xr._aggregate import aggregate_to
from eof.xr._cache import open_cache, to_zarr_cache
from eof.xr._geometry import angles_at
from eof.xr._open import open_aoi
from eof.xr._schema import validate_obs_dataset

__all__ = [
    "open_aoi", "aggregate_to", "angles_at",
    "to_zarr_cache", "open_cache", "validate_obs_dataset",
]
