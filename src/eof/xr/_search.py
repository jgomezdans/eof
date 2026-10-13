"""STAC search, de-duplication and tile selection."""

from __future__ import annotations

import warnings
from collections import defaultdict

import pystac_client
from pyproj import CRS
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry

from eof._source_configs import AWS_CONFIG, PLANETARY_CONFIG
from eof.xr._assets import parse_eco_id
from eof.xr._grid import TileGrid, aoi_to_crs

CMR_STAC_URL = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
_S2_STAC = {"aws": AWS_CONFIG.stac_url, "planetary": PLANETARY_CONFIG.stac_url}
_TILE_KEYS = ("grid:code", "s2:mgrs_tile", "mgrs:tile")


def _datetime_range(start: str, end: str) -> str:
    return f"{start}T00:00:00Z/{end}T23:59:59Z"


# ---------------------------------------------------------------------------
# Sentinel-2
# ---------------------------------------------------------------------------

def s2_tile_of(item) -> str | None:
    """MGRS tile name ('14TQL') of an S2 STAC item."""
    for key in _TILE_KEYS:
        v = item.properties.get(key)
        if v:
            return str(v).replace("MGRS-", "").lstrip("T")
    # AWS ids: S2B_14TQL_20220708_0_L2A
    parts = item.id.split("_")
    if len(parts) >= 3 and len(parts[1]) == 5:
        return parts[1]
    return None


def search_s2(aoi: BaseGeometry, start: str, end: str, source: str = "aws",
              max_cloud: float = 100.0) -> list:
    """All L2A items touching the AOI, de-duplicated."""
    client = pystac_client.Client.open(_S2_STAC[source])
    kwargs = {}
    if max_cloud < 100:
        kwargs["query"] = {"eo:cloud_cover": {"lte": max_cloud}}
    search = client.search(
        collections=["sentinel-2-l2a"], intersects=mapping(aoi),
        datetime=_datetime_range(start, end), **kwargs)
    return dedupe_s2(list(search.items()), source)


def dedupe_s2(items: list, source: str) -> list:
    """Keep one item per acquisition and tile.

    AWS ids carry a reprocessing sequence (``_0_``, ``_1_``): keep the
    highest. Planetary can hold several processing generations of one
    datatake: keep the latest ``s2:generation_time``.
    """
    best: dict[tuple, object] = {}
    for it in items:
        tile = s2_tile_of(it)
        if source == "aws":
            platform = it.properties.get("platform")
            day = it.datetime.strftime("%Y%m%d")
            key = (platform, tile, day)
            rank = int(it.properties.get("s2:sequence", 0))
        else:
            key = (it.properties.get("s2:datatake_id", it.id), tile)
            rank = it.properties.get("s2:generation_time", "")
        if key not in best or rank > best[key][0]:
            best[key] = (rank, it)
    return sorted((v[1] for v in best.values()), key=lambda i: i.datetime)


def s2_grid(item, source: str, res: float = 10.0) -> TileGrid:
    """Native 10 m tile grid from the item's STAC ``proj:`` metadata."""
    key = "blue" if source == "aws" else "B02"
    asset = item.assets[key]
    p = item.properties
    code = (p.get("proj:code") or asset.extra_fields.get("proj:code")
            or (f"EPSG:{p['proj:epsg']}" if "proj:epsg" in p else None)
            or (f"EPSG:{asset.extra_fields['proj:epsg']}"
                if "proj:epsg" in asset.extra_fields else None))
    tr = asset.extra_fields["proj:transform"]
    ny, nx = asset.extra_fields["proj:shape"]
    return TileGrid(crs=CRS.from_user_input(code), x0=float(tr[2]),
                    y0=float(tr[5]), res=float(tr[0]), nx=int(nx), ny=int(ny))


# ---------------------------------------------------------------------------
# ECOSTRESS
# ---------------------------------------------------------------------------

def search_eco(aoi: BaseGeometry, start: str, end: str, product: str = "L1CT_RAD",
               version: str = "002", tile: str | None = None) -> list:
    """ECOSTRESS tiled granules (L1CT_RAD or L2T_LSTE) touching the AOI.

    Item properties carry no cloud/angle information, so the only filters
    are time, tile and footprint. Duplicate builds are collapsed.
    """
    client = pystac_client.Client.open(CMR_STAC_URL)
    search = client.search(
        collections=[f"ECO_{product}_{version}"], bbox=aoi.bounds,
        datetime=_datetime_range(start, end))
    best: dict[tuple, tuple] = {}
    for it in search.items():
        try:
            f = parse_eco_id(it.id)
        except ValueError:
            continue
        if tile and f["tile"] != tile:
            continue
        if it.geometry is not None and not shape(it.geometry).intersects(aoi):
            continue
        key = (f["orbit"], f["scene"], f["tile"])
        rank = (f["build"], f["rev"])
        if key not in best or rank > best[key][0]:
            best[key] = (rank, it)
    return sorted((v[1] for v in best.values()), key=lambda i: parse_eco_id(i.id)["start"])


# ---------------------------------------------------------------------------
# Tile selection
# ---------------------------------------------------------------------------

def choose_tile(aoi: BaseGeometry, grids: dict[str, TileGrid]) -> str:
    """Pick the tile that best contains the AOI.

    Score: fraction of the AOI inside the tile, then distance of the AOI
    centroid from the tile edge (most interior wins). Deterministic, so
    the same AOI always lands on the same grid.
    """
    if not grids:
        raise ValueError("No candidate tiles")
    scored = []
    for name, g in grids.items():
        proj = aoi_to_crs(aoi, g.crs)
        tile = box(*g.bounds)
        inter = proj.intersection(tile).area
        cover = inter / proj.area if proj.area > 0 else float(tile.contains(proj.centroid))
        interior = tile.exterior.distance(proj.centroid) if tile.contains(proj.centroid) else 0.0
        scored.append((round(cover, 6), interior, name))
    scored.sort(reverse=True)
    best = scored[0]
    if best[0] < 1.0 - 1e-6:
        warnings.warn(
            f"AOI is not fully inside any single tile; using {best[2]} "
            f"({best[0]:.0%} of the AOI). Pass tile= to choose another.",
            stacklevel=3)
    return best[2]


def group_by_tile(items: list, tile_fn) -> dict[str, list]:
    out = defaultdict(list)
    for it in items:
        out[tile_fn(it)].append(it)
    return dict(out)
