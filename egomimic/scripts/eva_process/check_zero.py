"""Retained command; data checks and feature slices are now declared in YAML."""

import warnings

from egomimic.scripts.check_data import main

if __name__ == "__main__":
    warnings.warn(
        "Use python -m egomimic.scripts.check_data; alias owner: graph integration",
        FutureWarning,
    )
    main()
