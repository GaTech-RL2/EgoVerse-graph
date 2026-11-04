"""Regenerate this checkout's Elmo + Aidan arcdur data from the frozen split."""

from recipe_builders import build_elmoaidan

if __name__ == "__main__":
    build_elmoaidan(("arcdur",))
