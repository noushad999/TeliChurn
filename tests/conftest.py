from pathlib import Path

import pytest

from churn import score, simulate, train
from churn.config import load_config


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory) -> dict:
    root: Path = tmp_path_factory.mktemp("churn")
    return load_config(
        overrides={
            "simulation": {"n_subscribers": 6000},
            "paths": {
                k: str(root / k)
                for k in ("raw_dir", "features_dir", "model_dir", "registry_dir", "reports_dir", "scores_dir")
            },
            "training": {"num_boost_round": 300},
        }
    )


@pytest.fixture(scope="session")
def tables(small_cfg):
    return simulate.run(small_cfg)


@pytest.fixture(scope="session")
def trained(small_cfg, tables):
    return train.run(small_cfg)


@pytest.fixture(scope="session")
def scores(small_cfg, trained):
    return score.run(small_cfg)
