"""Regenerate the 16D velocity-profile data on the same frozen 218/11 split."""

from recipe_builders import build_elmoaidan

if __name__ == "__main__":
    build_elmoaidan(("arcvel",))
