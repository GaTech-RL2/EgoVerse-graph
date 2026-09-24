"""Audit declared data fields without constructing a model or fitting statistics."""

import hydra

from egomimic.scripts.viz_language import run_data_tool
from egomimic.utils.env import load_env


@hydra.main(
    version_base="1.3", config_path="../hydra_configs", config_name="check_data"
)
def main(config):
    load_env()
    print(run_data_tool(config) / "data-audit.json")


if __name__ == "__main__":
    main()
