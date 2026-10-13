"""Sentinel-2 L2A as a lazy xarray Dataset on the native tile grid."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import dask.array as da
import numpy as np
import requests
import xarray as xr

from eof._types import SCL_MASK_VALUES
from eof.xr._assets import (
    S2_BAND_INFO, S2_BANDS, S2_MTD_BAND_ID, s2_asset_hrefs, s2_mtd_href,
)
from eof.xr._env import gdal_env_for, sign_planetary
from eof.xr._geometry import circular_mean_deg, nanmean_quiet, node_slice
from eof.xr._grid import Bounds, TileGrid, aoi_mask
from eof.xr._read import lazy_layer
from eof.xr._uncertainty import s2_sigma

QUANT = 1e-4  # DN -> reflectance


# ---------------------------------------------------------------------------
# DN conversion
# ---------------------------------------------------------------------------

def reflectance_offset(properties: dict, source: str) -> float:
    """Additive offset to apply to ``DN * 1e-4`` for one STAC item.

    * **aws** (Earth Search v1): the -1000 DN BOA offset is already in the
      COG values, so reflectance is ``DN / 1e4``. Checked against
      Planetary Computer for 2019-2025 (June-July, 14TQL): AWS DN equals
      PC DN minus 1000 for every processing-baseline >= 04.00 item,
      *including* those flagged ``earthsearch:boa_offset_applied: false``,
      and equals PC DN for older baselines. The flag is therefore
      ignored. (STAC ``raster:bands`` also still advertises offset -0.1
      on these items; do not use it.)
    * **planetary**: original ESA values, so baseline >= 04.00 needs
      -0.1 and older baselines 0.
    """
    if source == "aws":
        return 0.0
    baseline = str(properties.get("s2:processing_baseline", "")
                   or properties.get("processing:version", ""))
    try:
        major = int(baseline.split(".")[0])
    except ValueError:
        major = 5   # unknown: assume a recent baseline
    return -0.1 if major >= 4 else 0.0


# ---------------------------------------------------------------------------
# MTD_TL.xml angles
# ---------------------------------------------------------------------------

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first(node, name: str):
    """First descendant (or self) whose local tag is ``name``."""
    return next(e for e in node.iter() if _local(e.tag) == name)


def _values(node) -> np.ndarray:
    rows = [r.text.split() for r in node.iter() if _local(r.tag) == "VALUES"]
    return np.array(rows, dtype="float64")


def s2_angle_grids(xml_bytes: bytes, bands=S2_BANDS) -> dict:
    """Parse sun and (band-averaged) view angle grids from MTD_TL.xml.

    Returns a dict with ``sun_zen, sun_az, view_zen, view_az`` (23 x 23
    float arrays, NaN where no detector sees the node), ``ulx, uly`` (node
    (0, 0), the tile upper-left corner) and ``step`` [m].
    """
    root = ET.fromstring(xml_bytes)

    geo = _first(root, "Geoposition")
    ulx = float(_first(geo, "ULX").text)
    uly = float(_first(geo, "ULY").text)

    sun = _first(root, "Sun_Angles_Grid")
    step = float(_first(sun, "COL_STEP").text)
    sun_zen = _values(_first(sun, "Zenith"))
    sun_az = _values(_first(sun, "Azimuth"))

    want = {S2_MTD_BAND_ID[b] for b in bands}
    per_band_z: dict[int, list] = {}
    per_band_a: dict[int, list] = {}
    for g in root.iter():
        if _local(g.tag) != "Viewing_Incidence_Angles_Grids":
            continue
        b = int(g.get("bandId"))
        if b not in want:
            continue
        per_band_z.setdefault(b, []).append(_values(_first(g, "Zenith")))
        per_band_a.setdefault(b, []).append(_values(_first(g, "Azimuth")))

    # detectors -> band (nanmean / circular nanmean), then bands -> mean (D8)
    bz = [nanmean_quiet(np.stack(v), axis=0) for v in per_band_z.values()]
    ba = [circular_mean_deg(np.stack(v), axis=0) for v in per_band_a.values()]
    view_zen = nanmean_quiet(np.stack(bz), axis=0)
    view_az = circular_mean_deg(np.stack(ba), axis=0)
    return dict(sun_zen=sun_zen, sun_az=sun_az, view_zen=view_zen,
                view_az=view_az, ulx=ulx, uly=uly, step=step)


def _fetch_mtd(href: str, source: str) -> bytes:
    if source == "planetary":
        href = sign_planetary(href)
    r = requests.get(href, timeout=60)
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def open_s2(items: list, source: str, grid: TileGrid, bounds: Bounds,
            aoi, tile: str, bands=S2_BANDS, scl_mask=SCL_MASK_VALUES,
            sigma_abs: float = 0.005, rel_sigma: float = 0.05,
            chunk: int = 1024) -> xr.Dataset:
    """Build the lazy S2 Dataset for ``items`` (already on ``grid``)."""
    env = gdal_env_for(source)
    sign = "planetary" if source == "planetary" else None
    grid20 = TileGrid(grid.crs, grid.x0, grid.y0, 20.0,
                      grid.nx // 2, grid.ny // 2)
    x, y = bounds.coords(grid.res)

    # Geometry: read all MTD files up front, in parallel (small, eager)
    with ThreadPoolExecutor(max_workers=8) as pool:
        mtds = list(pool.map(
            lambda it: s2_angle_grids(
                _fetch_mtd(s2_mtd_href(it, source), source), bands), items))
    ang0 = mtds[0]
    n = ang0["sun_zen"].shape[0]
    isl = node_slice(ang0["ulx"], ang0["step"], bounds.xmin, bounds.xmax, n)
    jsl = node_slice(ang0["uly"], -ang0["step"], bounds.ymin, bounds.ymax, n)
    ax = ang0["ulx"] + ang0["step"] * np.arange(n)[isl]
    ay = ang0["uly"] - ang0["step"] * np.arange(n)[jsl]

    def stack(key):
        return np.stack([m[key][jsl, isl] for m in mtds]).astype("float32")

    obs_t, valid_t, scl_t = [], [], []
    for it in items:
        hrefs = s2_asset_hrefs(it, source, bands)
        off = reflectance_offset(it.properties, source)
        layers = []
        for b in bands:
            up = 2 if S2_BAND_INFO[b][1] == 20 else 1
            layers.append(lazy_layer(
                hrefs[b], grid if up == 1 else grid20, bounds, env,
                upsample=up, scale=QUANT, offset=off, sign=sign, chunk=chunk))
        obs_t.append(da.stack(layers))
        scl = lazy_layer(hrefs["SCL"], grid20, bounds, env, upsample=2,
                         sign=sign, raw=True, dtype=np.uint8, chunk=chunk)
        scl_t.append(scl)
        valid_t.append(~da.isin(scl, list(scl_mask)))

    obs = da.stack(obs_t)                                   # (t, b, y, x)
    unc = s2_sigma(obs, sigma_abs, rel_sigma).astype("float32")
    valid = da.stack(valid_t)
    scl = da.stack(scl_t)

    times = np.array([np.datetime64(it.datetime.replace(tzinfo=None), "ns")
                      for it in items])
    ds = xr.Dataset(
        data_vars=dict(
            obs=(("time", "band", "y", "x"), obs),
            obs_unc=(("time", "band", "y", "x"), unc),
            valid=(("time", "y", "x"), valid),
            scl=(("time", "y", "x"), scl),
            sza=(("time", "ay", "ax"), stack("sun_zen")),
            saa=(("time", "ay", "ax"), stack("sun_az")),
            vza=(("time", "ay", "ax"), stack("view_zen")),
            vaa=(("time", "ay", "ax"), stack("view_az")),
            inside_aoi=(("y", "x"), aoi_mask(aoi, bounds, grid.crs, grid.res)),
        ),
        coords=dict(
            time=times, band=list(bands), x=x, y=y, ax=ax, ay=ay,
            wl_centre=("band", np.array(
                [S2_BAND_INFO[b][0] for b in bands], dtype="float32")),
            platform=("time", [str(it.properties.get("platform", "sentinel-2"))
                               .lower().replace("sentinel-2", "sentinel-2")
                               for it in items]),
            item_id=("time", [it.id for it in items]),
        ),
        attrs=dict(sensor="sentinel-2", product="L2A", obs_kind="reflectance",
                   units="1", source=source, tile=tile,
                   resolution_m=grid.res),
    )
    ds.attrs["obs_unc_model"] = (
        f"sqrt({sigma_abs}**2 + ({rel_sigma}*rho)**2)")
    ds.attrs["view_geometry"] = "mean over bands used (D8)"
    return ds.rio.write_crs(grid.crs)
