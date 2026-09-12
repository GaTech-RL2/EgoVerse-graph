"""Collect demonstrations with the maintained Yam Quest teleop runtime.

The previous Eva collector is retained in eva/collect_demo_legacy.py.
"""
from egomimic.robot.yam.runtime import main

if __name__ == "__main__":
    main(default_mode="teleop")
