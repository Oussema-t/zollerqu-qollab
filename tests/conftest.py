import numpy as np
import pytest


@pytest.fixture
def toy():
    """4 blocks, 3 sites, 2 warehouses. Hand-checked numbers in test_cost.py."""
    return {
        "blocks": [{"pop": 100}, {"pop": 200}, {"pop": 300}, {"pop": 400}],
        "sites": [{"id": 0}, {"id": 1}, {"id": 2}],
        "warehouses": [{"id": 0}, {"id": 1}],
        "t_block_site": [[2, 5, 9], [6, 3, 4], [8, 7, 1], [3, 9, 6]],
        "t_site_site": [[0, 5, 4], [5, 0, 8], [4, 8, 0]],
        "t_wh_site": [[3, 7, 2], [1, 1, 10]],
    }


TOY_PARAMS = {"k": 2, "c": 300, "staff_cost": 1.5, "tau": 6.0, "lam": 10.0, "wh_weight": 2.0}


def random_data(n_blocks, n_sites, seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 30, size=(n_blocks, 2))
    sites = rng.uniform(0, 30, size=(n_sites, 2))
    whs = rng.uniform(0, 30, size=(2, 2))

    def dist(a, b):
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)

    return {
        "blocks": [{"pop": int(p)} for p in rng.integers(200, 3000, n_blocks)],
        "sites": [{"id": i} for i in range(n_sites)],
        "warehouses": [{"id": 0}, {"id": 1}],
        "t_block_site": dist(pts, sites).tolist(),
        "t_site_site": dist(sites, sites).tolist(),
        "t_wh_site": dist(whs, sites).tolist(),
    }
