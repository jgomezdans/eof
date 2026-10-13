"""GDAL environment and URL signing for the lazy xarray readers.

Everything here is passed *into* each ``rasterio.Env`` rather than set
process-wide, so reads are safe in dask worker threads and sources can't
leak configuration into each other.
"""

from __future__ import annotations

import os
import threading
import time

import requests

COMMON_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.xml",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
}

_COOKIE_FILE = os.path.expanduser("~/.eof/earthdata_cookies.txt")

PLANETARY_SAS_URL = (
    "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
)


def gdal_env_for(source: str) -> dict:
    """GDAL config options for ``source`` ('aws', 'planetary', 'earthdata')."""
    env = dict(COMMON_GDAL_ENV)
    if source == "aws":
        env["AWS_NO_SIGN_REQUEST"] = "YES"
    elif source == "earthdata":
        os.makedirs(os.path.dirname(_COOKIE_FILE), exist_ok=True)
        env.update(
            GDAL_HTTP_NETRC="YES",
            GDAL_HTTP_COOKIEFILE=_COOKIE_FILE,
            GDAL_HTTP_COOKIEJAR=_COOKIE_FILE,
        )
    elif source != "planetary":
        raise ValueError(f"Unknown source {source!r}")
    return env


def check_earthdata_login() -> None:
    """Raise a helpful error if no Earthdata credentials can be found."""
    netrc = os.path.expanduser("~/.netrc")
    has_netrc = False
    if os.path.exists(netrc):
        with open(netrc) as f:
            has_netrc = "urs.earthdata.nasa.gov" in f.read()
    if not has_netrc:
        raise EnvironmentError(
            "ECOSTRESS needs NASA Earthdata Login credentials. Add this to "
            "~/.netrc (chmod 600):\n"
            "  machine urs.earthdata.nasa.gov login <user> password <pass>\n"
            "Register for free at https://urs.earthdata.nasa.gov"
        )


# Planetary Computer SAS tokens, cached per collection until near expiry.
_token_lock = threading.Lock()
_tokens: dict[str, tuple[str, float]] = {}


def _planetary_token(collection: str = "sentinel-2-l2a") -> str:
    with _token_lock:
        tok = _tokens.get(collection)
        if tok and tok[1] - time.time() > 300:
            return tok[0]
        r = requests.get(PLANETARY_SAS_URL.format(collection=collection),
                         timeout=30)
        r.raise_for_status()
        j = r.json()
        # 'msft:expiry' is ISO-8601; fall back to 30 minutes if absent
        expiry = time.time() + 1800
        try:
            from datetime import datetime
            expiry = datetime.fromisoformat(
                j["msft:expiry"].replace("Z", "+00:00")).timestamp()
        except (KeyError, ValueError):
            pass
        _tokens[collection] = (j["token"], expiry)
        return j["token"]


def sign_planetary(href: str) -> str:
    """Append a SAS token to an Azure blob URL. Call at read time, not
    graph-build time: signed URLs expire."""
    return f"{href}?{_planetary_token()}"
