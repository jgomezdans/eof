"""Regression tests for the eager-path fixes (AWS BOA offset, SZA, grid)."""
import types

import numpy as np
import pytest

from eof._cache import get_cache_path
from eof._scl import dn_to_reflectance
from eof._source_configs import AWS_CONFIG, CDSE_CONFIG, PLANETARY_CONFIG


def test_dn_to_reflectance_offset_branches():
    dn = np.array([[[3000.0]]])
    # ESA original values, baseline >= 04.00: subtract 1000
    assert dn_to_reflectance(dn, "04.00")[0, 0, 0] == pytest.approx(0.2)
    # offset already in the DNs (AWS): no second subtraction
    assert dn_to_reflectance(dn, "04.00", offset_in_dn=True)[0, 0, 0] == pytest.approx(0.3)
    # old baselines never had an offset
    assert dn_to_reflectance(dn, "03.01")[0, 0, 0] == pytest.approx(0.3)


def test_only_aws_has_offset_in_dn():
    assert AWS_CONFIG.boa_offset_in_dn is True
    assert PLANETARY_CONFIG.boa_offset_in_dn is False
    assert CDSE_CONFIG.boa_offset_in_dn is False


def test_cache_path_is_versioned(tmp_path):
    p = get_cache_path("S2A_14TQL_20220718_0_L2A", str(tmp_path))
    assert p.endswith("_g2.npz")
    assert get_cache_path("x", str(tmp_path), "landsat").endswith("landsat_x_g2.npz")


def _reader(config):
    pytest.importorskip("mgrs")
    from eof._stac_reader import STACReader
    return STACReader(config)


def test_sza_planetary_uses_zenith_directly():
    item = types.SimpleNamespace(properties={
        "s2:mean_solar_zenith": 23.26, "s2:mean_solar_azimuth": 140.0})
    sza, _, _ = _reader(PLANETARY_CONFIG)._extract_angles(item)
    assert sza == pytest.approx(23.26)


def test_sza_aws_uses_sun_elevation():
    item = types.SimpleNamespace(properties={
        "view:sun_elevation": 68.15, "view:sun_azimuth": 143.0})
    sza, _, _ = _reader(AWS_CONFIG)._extract_angles(item)
    assert sza == pytest.approx(21.85)
