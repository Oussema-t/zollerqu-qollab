import json
import math
import os

import pytest

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "champaign.json")


@pytest.fixture(scope="module")
def d():
    with open(PATH, encoding="utf-8") as f:
        return json.load(f)


def test_file_size():
    assert os.path.exists(PATH)
    assert os.path.getsize(PATH) < 2 * 1024 * 1024


def test_keys(d):
    for k in ["county", "source", "blocks", "sites", "warehouses",
              "t_block_site", "t_site_site", "t_wh_site"]:
        assert k in d
    assert d["source"]["travel"]
    assert d["source"]["population"]


def test_blocks(d):
    b = d["blocks"]
    assert len(b) == 158
    assert sum(x["pop"] for x in b) == 205865
    assert [x["geoid"] for x in b] == sorted(x["geoid"] for x in b)
    for x in b:
        assert len(x["geoid"]) == 12
        assert 39.8 < x["lat"] < 40.5
        assert -88.5 < x["lon"] < -87.9
        assert x["pop"] >= 0
        assert x["area_m2"] > 0


def test_sites_and_warehouses(d):
    assert [s["id"] for s in d["sites"]] == list(range(12))
    assert len(d["warehouses"]) == 2
    geoids = {x["geoid"] for x in d["blocks"]}
    assert all(s["name"] in geoids for s in d["sites"])


def shape(m, r, c):
    assert len(m) == r
    for row in m:
        assert len(row) == c
        for v in row:
            assert math.isfinite(v) and v >= 0


def test_matrices(d):
    shape(d["t_block_site"], 158, 12)
    shape(d["t_site_site"], 12, 12)
    shape(d["t_wh_site"], 2, 12)
    t = d["t_site_site"]
    for i in range(12):
        assert t[i][i] == 0
        for j in range(12):
            assert t[i][j] == t[j][i]
            if i != j:
                assert t[i][j] >= 4.0


def test_positive_times(d):
    assert all(v > 0 for row in d["t_wh_site"] for v in row)
    assert all(v > 0 for row in d["t_block_site"] for v in row)
